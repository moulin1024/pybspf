"""BSPF restricted to a curved domain, with symmetric Nitsche Dirichlet data.

The enclosing square carries unrestricted BSPF factors; no PDE or boundary
condition is imposed outside the spline domain. Geometry is evaluated directly,
and volume integration uses all vertical intersections (no star-shaped map).
Dense assembly is deliberately used for this accuracy benchmark.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.linalg as la
from scipy.interpolate import BSpline, PPoly
from scipy.optimize import linprog, brentq
from scipy.special import roots_legendre, betainc

from ._flow_kernels import tensor_product
from .stream_navier_stokes import _stream_line, stream_evaluate_line


def _unique(values, tolerance=1e-11):
    values = np.sort(np.asarray(values))
    return values[np.r_[True, np.diff(values) > tolerance]]


@dataclass
class SplineDomain:
    """Regular, simple, counterclockwise periodic cubic B-spline boundary.

    Construction does not certify simplicity for arbitrary user-supplied controls.
    The bundled benchmark curves are simple rounded convex/C-shaped polygons.
    """

    controls: np.ndarray
    name: str = "spline"

    def __post_init__(self):
        self.controls = np.asarray(self.controls, dtype=float)
        if (
            self.controls.ndim != 2
            or self.controls.shape[1] != 2
            or len(self.controls) < 4
        ):
            raise ValueError("Expected at least four planar periodic control points")
        self.period = len(self.controls)
        self.curve = BSpline(
            np.arange(-3, self.period + 4, dtype=float),
            np.vstack((self.controls, self.controls[:3])),
            3,
            extrapolate="periodic",
        )
        self.xpoly = PPoly.from_spline(
            (self.curve.t, self.curve.c[:, 0], 3), extrapolate=False
        )
        roots = self.xpoly.derivative().roots(extrapolate=False)
        roots = roots[np.isfinite(roots) & (roots >= 0) & (roots <= self.period)]
        self.monotone_breaks = _unique(np.r_[np.arange(self.period + 1), roots])

    def normal(self, parameter):
        """Outward unit normal at arbitrary parameters of this CCW spline.

        Differentiate the original B-spline analytically; no polygonal or
        finite-difference normal is used. Scalar and array parameters work.
        """
        tangent = self.curve(parameter, 1)
        speed = np.linalg.norm(tangent, axis=-1)
        if np.any(speed < 1e-12):
            raise ValueError("Boundary is not regular")
        return np.stack((tangent[..., 1], -tangent[..., 0]), axis=-1) / speed[..., None]

    def boundary_rule(self, order=12):
        q, w = roots_legendre(order)
        t = (np.arange(self.period)[:, None] + (q + 1) / 2).ravel()
        tangent = self.curve(t, 1)
        speed = la.norm(tangent, axis=1)
        if np.min(speed) < 1e-12:
            raise ValueError("Boundary is not regular")
        normal = self.normal(t)
        return self.curve(t), np.tile(w / 2, self.period) * speed, normal

    def intersections(self, x):
        # Bracket on monotone cubic pieces. Polynomial root formulas can return
        # spurious roots when nearly straight spans have tiny cubic coefficients.
        roots = []
        for a, b in zip(self.monotone_breaks[:-1], self.monotone_breaks[1:]):
            xa, xb = self.curve([a, b])[:, 0]
            if abs(xa - xb) > 1e-14 and min(xa, xb) <= x <= max(xa, xb):
                roots.append(
                    brentq(lambda t: float(self.curve(t)[0]) - x, a, b, xtol=5e-15)
                )
        if not roots:
            return np.empty((0, 2))
        roots = _unique(np.mod(roots, self.period))
        y = np.sort(self.curve(roots)[:, 1])
        if len(y) % 2:
            raise ValueError("Odd slice intersection count; check geometry/tangencies")
        return y.reshape(-1, 2)

    def volume_rule(self, order=12):
        # Include spline knots and vertical tangencies. A sixth-power endpoint
        # substitution regularizes both square/cube-root inverse branches of a
        # cubic curve, including joins onto a straight vertical segment.
        roots = self.xpoly.derivative().roots(extrapolate=False)
        roots = roots[np.isfinite(roots) & (roots >= 0) & (roots <= self.period)]
        events = _unique(self.curve(np.r_[np.arange(self.period + 1), roots])[:, 0])
        events = np.concatenate(
            [
                np.linspace(a, b, max(1, int(np.ceil((b - a) / 0.25))) + 1)[:-1]
                for a, b in zip(events[:-1], events[1:])
            ]
            + [events[-1:]]
        )
        q, w = roots_legendre(order)
        unit = (q + 1) / 2
        mapped = betainc(6, 6, unit)
        derivative = 2772 * unit**5 * (1 - unit) ** 5
        points, weights = [], []
        maximum_intervals = 0
        for lo, hi in zip(events[:-1], events[1:]):
            x = lo + (hi - lo) * mapped
            wx = w * (hi - lo) / 2 * derivative
            for xi, wi in zip(x, wx):
                intervals = self.intersections(xi)
                maximum_intervals = max(maximum_intervals, len(intervals))
                for bottom, top in intervals:
                    breaks = np.linspace(
                        bottom, top, max(1, int(np.ceil((top - bottom) / 0.25))) + 1
                    )
                    for ya, yb in zip(breaks[:-1], breaks[1:]):
                        y = (ya + yb) / 2 + (yb - ya) / 2 * q
                        points.append(np.column_stack((np.full_like(y, xi), y)))
                        weights.append(wi * w * (yb - ya) / 2)
        return np.vstack(points), np.concatenate(weights), maximum_intervals

    def geometry_checks(self):
        p, w, n = self.boundary_rule(20)
        area = float(np.sum(w * np.sum(p * n, axis=1)) / 2)
        # A star center must lie in every inward tangent half-plane. Infeasibility
        # of even this FINITE subset is a witness of an empty visibility kernel.
        kernel = linprog(
            np.zeros(2),
            A_ub=n,
            b_ub=np.sum(n * p, axis=1),
            bounds=[(None, None)] * 2,
            method="highs",
        )
        t = np.linspace(0, self.period, 2001, endpoint=False)
        d, dd = self.curve(t, 1), self.curve(t, 2)
        curvature = (d[:, 0] * dd[:, 1] - d[:, 1] * dd[:, 0]) / la.norm(d, axis=1) ** 3
        return dict(
            area_boundary=area,
            kernel_lp_status=int(kernel.status),
            nonstar_witness=bool(kernel.status == 2),
            curvature_min=float(curvature.min()),
            curvature_max=float(curvature.max()),
        )


def benchmark_domains():
    t = np.arange(12) * 2 * np.pi / 12
    convex = SplineDomain(
        np.column_stack((0.9 * np.cos(t), 0.76 * np.sin(t))), "convex"
    )
    polygon = np.array(
        [
            [-0.9, -0.9],
            [0.9, -0.9],
            [0.9, 0.9],
            [0.35, 0.9],
            [0.35, -0.25],
            [-0.35, -0.25],
            [-0.35, 0.9],
            [-0.9, 0.9],
        ]
    )
    controls = np.concatenate(
        [
            a + np.arange(4)[:, None] / 4 * (b - a)
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0))
        ]
    )
    # Rotate the U so vertical slices actually exercise multiple disjoint pieces.
    controls = np.column_stack((-controls[:, 1], controls[:, 0]))
    return convex, SplineDomain(controls, "nonstar")


@lru_cache(maxsize=16)
def background_line(
    nodes=33,
    *,
    endpoint_method="chebyshev",
    endpoint_points=16,
    chebyshev_modes=12,
    endpoint_regularization=1e-12,
):
    return _stream_line(
        np.linspace(-1, 1, nodes),
        clamped=False,
        dirichlet=False,
        endpoint_method=endpoint_method,
        endpoint_points=endpoint_points,
        chebyshev_modes=chebyshev_modes,
        endpoint_regularization=endpoint_regularization,
    )


def basis_values(line, points, modes):
    """Direct MPFR BSPF evaluation, without an intermediate spline interpolant."""
    arrays = []
    for k in range(2):
        unique, inverse = np.unique(points[:, k], return_inverse=True)
        values = stream_evaluate_line(line, unique)
        arrays.append([v[inverse, :modes] for v in values[:2]])
    (x, dx), (y, dy) = arrays
    return tuple(
        tensor_product(a, b, paired=True) for a, b in ((x, y), (dx, y), (x, dy))
    )


def manufactured(points):
    x, y = np.asarray(points).T
    u = np.exp(x) * np.cos(y) + x * x + y * y
    gradient = np.column_stack(
        (np.exp(x) * np.cos(y) + 2 * x, -np.exp(x) * np.sin(y) + 2 * y)
    )
    return u, gradient, np.full_like(x, -4.0)


@dataclass
class EmbeddedSamples:
    domain: SplineDomain
    modes: int
    points: np.ndarray
    weights: np.ndarray
    basis: tuple
    boundary: np.ndarray
    boundary_weights: np.ndarray
    boundary_basis: np.ndarray
    normal_derivative: np.ndarray
    maximum_slice_intervals: int


def sample_domain(domain, modes=20, order=12, nodes=33):
    line = background_line(nodes)
    if not 2 <= modes <= line.b.shape[1]:
        raise ValueError("Invalid number of BSPF modes")
    p, w, intervals = domain.volume_rule(order)
    bp, bw, normal = domain.boundary_rule(order)
    basis = basis_values(line, p, modes)
    b, dx, dy = basis_values(line, bp, modes)
    dn = normal[:, :1] * dx + normal[:, 1:] * dy
    return EmbeddedSamples(domain, modes, p, w, basis, bp, bw, b, dn, intervals)


def solve_poisson(samples, modes, data=manufactured, mass_cutoff=1e-12):
    """Dense symmetric Nitsche solve; penalty from a discrete trace bound.

    Restricted-mass truncation is explicit and reported; it changes the trial
    space. The trace bound and positivity must also be checked under quadrature
    refinement. Constant functions belong to the unrestricted parent space.
    """
    if not 2 <= modes <= samples.modes:
        raise ValueError("Invalid mode truncation")
    idx = (np.arange(modes)[:, None] * samples.modes + np.arange(modes)).ravel()
    b, dx, dy = (a[:, idx] for a in samples.basis)
    w, bw = samples.weights, samples.boundary_weights
    root = np.sqrt(w)
    _, singular, vt = la.svd(root[:, None] * b, full_matrices=False)
    keep = singular**2 > mass_cutoff * singular[0] ** 2
    transform = vt[keep].T / singular[keep]
    b, dx, dy = (a @ transform for a in (b, dx, dy))
    boundary = samples.boundary_basis[:, idx] @ transform
    dn = samples.normal_derivative[:, idx] @ transform
    stiffness = dx.T @ (w[:, None] * dx) + dy.T @ (w[:, None] * dy)
    eigen, rotation = la.eigh(stiffness)
    positive = eigen > eigen[-1] * 1e-12
    inverse_root = rotation[:, positive] / np.sqrt(eigen[positive])
    trace = np.sqrt(bw[:, None]) * (dn @ inverse_root)
    trace_constant = float(la.svdvals(trace)[0] ** 2)
    penalty = 4 * trace_constant
    cross = boundary.T @ (bw[:, None] * dn)
    matrix = (
        stiffness - cross - cross.T + penalty * boundary.T @ (bw[:, None] * boundary)
    )
    matrix = (matrix + matrix.T) / 2
    _, _, f = data(samples.points)
    g = data(samples.boundary)[0]
    rhs = b.T @ (w * f) - dn.T @ (bw * g) + penalty * boundary.T @ (bw * g)
    solution = la.cho_solve(la.cho_factor(matrix), rhs)
    coefficients = np.zeros(samples.modes**2)
    coefficients[idx] = transform @ solution
    spectrum = la.eigvalsh(matrix)
    return coefficients, dict(
        modes=modes,
        raw_dofs=modes * modes,
        retained_dofs=int(keep.sum()),
        mass_cutoff=mass_cutoff,
        penalty=penalty,
        discrete_trace_constant=trace_constant,
        matrix_min_eigenvalue=float(spectrum[0]),
        condition_number=float(spectrum[-1] / spectrum[0]),
        linear_relative_residual=float(la.norm(matrix @ solution - rhs) / la.norm(rhs)),
        volume_points=len(w),
        boundary_points=len(bw),
        maximum_slice_intervals=samples.maximum_slice_intervals,
    )


def error_metrics(samples, coefficients, data=manufactured):
    exact, gradient, _ = data(samples.points)
    b, dx, dy = samples.basis
    error = b @ coefficients - exact
    grad_error = np.column_stack((dx @ coefficients, dy @ coefficients)) - gradient
    edge_error = samples.boundary_basis @ coefficients - data(samples.boundary)[0]
    w, bw = samples.weights, samples.boundary_weights
    return dict(
        relative_l2=float(np.sqrt(np.sum(w * error**2) / np.sum(w * exact**2))),
        relative_h1_seminorm=float(
            np.sqrt(
                np.sum(w[:, None] * grad_error**2) / np.sum(w[:, None] * gradient**2)
            )
        ),
        boundary_rms=float(np.sqrt(np.sum(bw * edge_error**2) / bw.sum())),
        sampled_linf=float(np.max(np.abs(error))),
        boundary_sampled_linf=float(np.max(np.abs(edge_error))),
    )
