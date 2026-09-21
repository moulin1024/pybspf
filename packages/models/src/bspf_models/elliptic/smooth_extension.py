"""Experimental BSPF smooth-extension Poisson system with exact spline normals.

Two independent fields u and xi share a homogeneous-Dirichlet background
space. Surface multipliers enforce u=g and matching normal jets. No exact
interior values or exact normal derivatives enter the solve. This is a modal
Galerkin, IBSE-inspired prototype, not the original delta-kernel IBSE code.
"""

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import scipy.linalg as la

from bspf_models._numerics.trial_spaces import _stream_line
from bspf_models._numerics.trial_spaces import stream_evaluate_line
from pybspf.tensor import tensor_product
from pybspf.tensor import tensor_elliptic_solve


@lru_cache(maxsize=8)
def extension_line(nodes=33, half_width=1.2, endpoint_points=12):
    return _stream_line(
        np.linspace(-half_width, half_width, nodes),
        clamped=False,
        dirichlet=True,
        endpoint_points=endpoint_points,
        chebyshev_modes=12,
    )


def factors(line, points):
    output = []
    for axis in range(2):
        unique, inverse = np.unique(points[:, axis], return_inverse=True)
        output.append([v[inverse] for v in stream_evaluate_line(line, unique)])
    return output


def basis_operators(line, points, normals=None):
    (x, dx, xx), (y, dy, yy) = factors(line, points)

    def pair(a, b):
        return tensor_product(a, b, paired=True)

    value = pair(x, y)
    lx, ly = pair(xx, y), pair(x, yy)
    laplace = -(lx + ly)
    if normals is None:
        return value, laplace
    nx, ny = normals[:, :1], normals[:, 1:]
    normal_first = nx * pair(dx, y) + ny * pair(x, dy)
    # Repeated differentiation along the fixed normal ray, not a derivative
    # of a separately interpolated normal field.
    normal_second = nx * nx * lx + 2 * nx * ny * pair(dx, dy) + ny * ny * ly
    return value, laplace, normal_first, normal_second


def evaluate_field(line, points, coefficient):
    return evaluate_factors(factors(line, points), coefficient)


def evaluate_factors(values, coefficient):
    (x, dx, xx), (y, dy, yy) = values
    c = coefficient.reshape(x.shape[1], y.shape[1])

    def apply(a, b):
        return np.sum((a @ c) * b, axis=1)

    return (
        apply(x, y),
        np.column_stack((apply(dx, y), apply(x, dy))),
        apply(xx, y) + apply(x, yy),
    )


def tensor_inverse(rhs, denominator):
    vector = rhs.ndim == 1
    a = rhs[:, None] if vector else rhs
    batch = a.T.reshape((-1,) + denominator.shape)
    solved = tensor_elliptic_solve(batch, denominator)
    result = solved.reshape(a.shape[1], -1).T
    return result[:, 0] if vector else result


@dataclass
class ExtensionGeometry:
    domain: object
    line: object
    nodes: int
    half_width: float
    volume_order: int
    boundary_order: int
    points: np.ndarray
    weights: np.ndarray
    value: np.ndarray
    laplace: np.ndarray
    boundary: np.ndarray
    boundary_weights: np.ndarray
    normals: np.ndarray
    traces: tuple
    poisson_denominator: np.ndarray
    exterior_laplace: np.ndarray


def assemble_geometry(
    domain, *, nodes=33, half_width=1.2, volume_order=16, boundary_order=4
):
    if np.max(abs(domain.controls)) >= half_width:
        raise ValueError("Spline domain must lie strictly inside the background square")
    line = extension_line(nodes, half_width)
    points, weights, _ = domain.volume_rule(volume_order)
    value, laplace = basis_operators(line, points)
    boundary, bw, normals = domain.boundary_rule(boundary_order)
    trace, _, dn, dnn = basis_operators(line, boundary, normals)
    lam = np.asarray(line.lam)
    denominator = lam[:, None] + lam[None, :]
    if denominator.min() <= 0:
        raise ValueError("Background Dirichlet Poisson matrix must be positive")
    interior = value.T @ (weights[:, None] * laplace)
    # For homogeneous Dirichlet factors, integration by parts on the full
    # rectangle gives K exactly in the discrete weak model. Subtract the
    # physical-domain strong volume form to obtain the exterior source map.
    exterior = np.diag(denominator.ravel()) - interior
    return ExtensionGeometry(
        domain,
        line,
        nodes,
        half_width,
        volume_order,
        boundary_order,
        points,
        weights,
        value,
        laplace,
        boundary,
        bw,
        normals,
        (trace, dn, dnn),
        denominator,
        exterior,
    )


def replace_boundary_rule(geometry, order):
    """Refine analytic interface sampling without rebuilding volume integration."""
    points, weights, normals = geometry.domain.boundary_rule(order)
    trace, _, dn, dnn = basis_operators(geometry.line, points, normals)
    return replace(
        geometry,
        boundary_order=order,
        boundary=points,
        boundary_weights=weights,
        normals=normals,
        traces=(trace, dn, dnn),
    )


@dataclass
class SmoothExtensionPlan:
    geometry: ExtensionGeometry
    matching_order: int
    extension_length: float
    trace_matrix: np.ndarray
    boundary_matrix: np.ndarray
    extension_response: np.ndarray
    solution_response: np.ndarray
    boundary_response: np.ndarray
    schur: np.ndarray
    row_scale: np.ndarray
    column_scale: np.ndarray

    def solve(self, forcing, boundary_data, *, svd_cutoff=1e-12):
        """forcing is sampled ONLY inside Omega; boundary_data only on Gamma."""
        geom = self.geometry
        f = np.asarray(forcing)
        g = np.asarray(boundary_data)
        if f.shape != (len(geom.points),) or g.shape != (len(geom.boundary),):
            raise ValueError("Incorrect physical forcing or boundary-data shape")
        load = geom.value.T @ (geom.weights * f)
        particular = tensor_inverse(load, geom.poisson_denominator)
        rhs = np.r_[
            self.trace_matrix @ particular,
            np.sqrt(geom.boundary_weights) * g - self.boundary_matrix @ particular,
        ]
        operator = self.schur / self.row_scale[:, None] / self.column_scale
        unknown, _, rank, singular = la.lstsq(
            operator, rhs / self.row_scale, cond=svd_cutoff
        )
        unknown /= self.column_scale
        count = self.trace_matrix.shape[0]
        mu, multiplier = unknown[:count], unknown[count:]
        extension = self.extension_response @ mu
        solution = (
            particular
            + self.solution_response @ mu
            + self.boundary_response @ multiplier
        )
        residual = self.schur @ unknown - rhs
        return (
            solution,
            extension,
            dict(
                matching_order=self.matching_order,
                extension_length=self.extension_length,
                surface_unknowns=len(unknown),
                retained_surface_rank=int(rank),
                svd_cutoff=svd_cutoff,
                smallest_relative_singular=float(singular[-1] / singular[0]),
                schur_relative_residual=float(
                    la.norm(residual) / max(la.norm(rhs), 1e-30)
                ),
                physical_data_only=True,
            ),
        )


def plan_smooth_extension(geometry, *, matching_order=1, extension_length=0.25):
    if matching_order not in (0, 1, 2) or extension_length <= 0:
        raise ValueError("Matching order must be 0, 1 or 2; length must be positive")
    geom = geometry
    root = np.sqrt(geom.boundary_weights)[:, None]
    spacing = 2 * geom.half_width / (geom.nodes - 1)
    traces = np.vstack(
        [root * spacing**j * geom.traces[j] for j in range(matching_order + 1)]
    )
    boundary = root * geom.traces[0]
    # H=(I+ell^2 K)^(k+1), a positive spectral high-order extension operator.
    h = (1 + extension_length**2 * geom.poisson_denominator) ** (matching_order + 1)
    extension = -tensor_inverse(traces.T, h)
    response = tensor_inverse(
        geom.exterior_laplace @ extension, geom.poisson_denominator
    )
    boundary_response = -tensor_inverse(boundary.T, geom.poisson_denominator)
    schur = np.block(
        [
            [traces @ (extension - response), -traces @ boundary_response],
            [boundary @ response, boundary @ boundary_response],
        ]
    )
    row_scale = np.maximum(la.norm(schur, axis=1), 1e-30)
    column_scale = np.maximum(la.norm(schur / row_scale[:, None], axis=0), 1e-30)
    return SmoothExtensionPlan(
        geom,
        matching_order,
        extension_length,
        traces,
        boundary,
        extension,
        response,
        boundary_response,
        schur,
        row_scale,
        column_scale,
    )
