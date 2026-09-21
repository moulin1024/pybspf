"""Poisson via Fourier extension, spectral particular solution and harmonic MFS.

Solves -Delta u=f in a smooth convex SplineDomain, u=g on its true boundary.
Only the forcing extension is Algorithm 1 of 1706.04848; the boundary solver
is an additional method of fundamental solutions (MFS), not from that paper.
CPU reference implementation with reusable factors and FFT volume operations.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.linalg as la

from .convex_poisson import ArcLengthBoundary, validate_convex
from .fourier_extension import FourierExtensionPlan


def spline_grid_mask(domain, grid_size, half_width):
    """Classify periodic grid points using exact spline slice intersections."""
    axis = np.linspace(-half_width, half_width, grid_size, endpoint=False)
    mask = np.zeros((grid_size, grid_size), bool)
    for i, x in enumerate(axis):
        for lower, upper in domain.intersections(x):
            mask[i] |= (axis > lower) & (axis < upper)
    return mask


def _harmonic_matrix(points, sources, derivative=(0, 0)):
    """2D Laplace log sources, augmented with a constant harmonic function."""
    d = np.asarray(points)[:, None, :] - sources[None, :, :]
    r2 = np.sum(d * d, axis=2)
    if np.any(r2 == 0):
        raise ValueError("Cannot evaluate the harmonic field at a source")
    a, b = derivative
    if (a, b) == (0, 0):
        kernel, constant = np.log(r2) / (4 * np.pi), 1.0
    elif a + b == 1:
        kernel, constant = d[:, :, b] / (2 * np.pi * r2), 0.0
    elif a + b == 2:
        if (a, b) == (1, 1):
            kernel = -2 * d[:, :, 0] * d[:, :, 1] / r2**2
        else:
            axis = int(b == 2)
            kernel = 1 / r2 - 2 * d[:, :, axis]**2 / r2**2
        kernel, constant = kernel / (2 * np.pi), 0.0
    else:
        raise ValueError("Harmonic derivatives are supported up to order two")
    return np.column_stack((np.full(len(points), constant), kernel))


class FourierPoissonPlan:
    """Reusable Dirichlet solver on convex periodic cubic spline domains.

    Complexity: FFT-based FE bulk solve plus dense boundary-only MFS factors.
    There is no full volume PDE matrix. Off-grid Fourier evaluation uses direct
    tensor sums. This is not a claim of O(volume log volume) end-to-end setup.
    """

    def __init__(self, domain, modes=33, *, grid_size=None, half_width=1.2,
                 boundary_count=256, source_count=128, source_distance=0.3,
                 cutoff=1e-12, boundary_cutoff=1e-13, seed=0,
                 extension_options=None):
        start = perf_counter()
        self.domain = domain
        self.geometry = validate_convex(domain)
        # The control polygon bounds the entire spline, a conservative box test.
        if np.max(np.abs(domain.controls)) >= half_width:
            raise ValueError("The spline control polygon must lie strictly inside the box")
        if source_count < 8 or source_count % 2 or boundary_count < 2 * source_count:
            raise ValueError("Require even source_count >=8 and boundary_count >=2*source_count")
        if not np.isfinite(source_distance) or source_distance <= 0:
            raise ValueError("source_distance must be positive and finite")
        if not 0 < boundary_cutoff < 1:
            raise ValueError("boundary_cutoff must lie in (0, 1)")
        # ~5.4 interior samples per coefficient on the bundled convex domain.
        # 2*modes undersamples the narrow boundary collar at tight cutoffs.
        grid_size = 4 * modes if grid_size is None else grid_size
        if not isinstance(grid_size, (int, np.integer)) or grid_size < modes:
            raise ValueError("grid_size must be an integer >= modes")
        mask = spline_grid_mask(domain, grid_size, half_width)
        self.extension = FourierExtensionPlan(
            mask, modes, half_width=half_width, cutoff=cutoff, seed=seed,
            **({} if extension_options is None else extension_options),
        )
        self.arc = ArcLengthBoundary(domain)
        self.boundary, _ = self.arc.sample(boundary_count)
        source_points, parameter = self.arc.sample(source_count, offset=0.5)
        # Convexity guarantees positive outward-normal offsets are exterior.
        self.sources = source_points + source_distance * domain.normal(parameter)
        h = _harmonic_matrix(self.boundary, self.sources)
        self.column_scale = la.norm(h, axis=0)
        u, s, vh = la.svd(h / self.column_scale, full_matrices=False)
        keep = s > boundary_cutoff * s[0]
        # Keep factored TSVD: an explicitly multiplied pseudoinverse loses many
        # digits when its large entries act on smooth, nearly compatible data.
        self.boundary_left = u[:, keep]
        self.boundary_singular = s[keep]
        self.boundary_right = vh[keep].conj().T
        self.boundary_matrix = h
        k = np.pi * self.extension.frequencies / half_width
        self.wave_squared = (k[:, None]**2 + k[None, :]**2).ravel()
        self.zero_index = (modes**2) // 2
        self.diagnostics = dict(
            extension=self.extension.diagnostics, boundary_count=boundary_count,
            source_count=source_count, source_distance=source_distance,
            boundary_rank=int(keep.sum()),
            boundary_factor_bytes=(self.boundary_left.nbytes + self.boundary_right.nbytes
                                   + self.boundary_singular.nbytes + h.nbytes),
            setup_seconds=perf_counter() - start,
        )

    def solve(self, forcing, boundary_data):
        """Inputs are callables(points)->values, or arrays at plan sample points."""
        start = perf_counter()
        f = forcing(self.extension.points) if callable(forcing) else forcing
        g = boundary_data(self.boundary) if callable(boundary_data) else boundary_data
        f, g = np.asarray(f), np.asarray(g)
        if f.ndim == 0:
            f = np.full(self.extension.samples, f)
        if g.ndim == 0:
            g = np.full(len(self.boundary), g)
        if g.shape != (len(self.boundary),) or not np.all(np.isfinite(g)):
            raise ValueError("Expected finite boundary values at plan.boundary")
        extension = self.extension.solve(f)
        c = extension.coefficients
        particular = np.zeros_like(c)
        nonzero = self.wave_squared > 0
        particular[nonzero] = c[nonzero] / self.wave_squared[nonzero]
        # -Delta[-c0*(x^2+y^2)/4]=c0. Do not discard the forcing mean:
        # the physical Dirichlet problem has no zero-mean compatibility condition.
        mean = c[self.zero_index]
        trace = self.extension.evaluate(particular, self.boundary)
        if extension.is_real:
            mean, trace = mean.real, trace.real
        trace -= mean * np.sum(self.boundary**2, axis=1) / 4
        harmonic = self.boundary_right @ ((self.boundary_left.conj().T @ (g - trace)) / self.boundary_singular)
        harmonic /= self.column_scale
        residual = trace + self.boundary_matrix @ harmonic - g
        return FourierPoissonSolution(self, particular, mean, harmonic, extension, dict(
            extension=extension.diagnostics,
            boundary_sample_max_residual=float(np.max(np.abs(residual))),
            harmonic_coefficient_norm=float(la.norm(harmonic)),
            solve_seconds=perf_counter() - start,
        ))


@dataclass
class FourierPoissonSolution:
    plan: FourierPoissonPlan
    particular: np.ndarray
    mean: complex
    harmonic: np.ndarray
    forcing_extension: object
    diagnostics: dict

    def derivative(self, points, derivative=(0, 0)):
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or not np.all(np.isfinite(p)):
            raise ValueError("Expected finite points of shape (count, 2)")
        if derivative not in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)):
            raise ValueError("Supported derivatives: value, gradient, Hessian")
        out = self.plan.extension.evaluate(self.particular, p, derivative)
        if derivative == (0, 0):
            out -= self.mean * np.sum(p**2, axis=1) / 4
        elif sum(derivative) == 1:
            out -= self.mean * p[:, derivative[1]] / 2
        elif derivative in ((2, 0), (0, 2)):
            out -= self.mean / 2
        for begin in range(0, len(p), 512):
            stop = begin + 512
            out[begin:stop] += _harmonic_matrix(p[begin:stop], self.plan.sources, derivative) @ self.harmonic
        if self.forcing_extension.is_real and np.isrealobj(self.harmonic):
            return out.real
        return out

    def evaluate(self, points):
        """Value, gradient and positive Laplacian; preserves complex inputs."""
        value = self.derivative(points)
        gradient = np.column_stack([self.derivative(points, d) for d in ((1, 0), (0, 1))])
        laplacian = self.derivative(points, (2, 0)) + self.derivative(points, (0, 2))
        return value, gradient, laplacian

    def hessian(self, points):
        xx, xy, yy = [self.derivative(points, d) for d in ((2, 0), (1, 1), (0, 2))]
        return np.stack((xx, xy, xy, yy), axis=1).reshape(-1, 2, 2)

    def validate(self, forcing, boundary_data, *, volume_order=20, boundary_count=1024):
        """Independent physical residuals using callbacks, without a known solution.

        Uses exact-geometry volume quadrature and shifted arclength samples.
        These are sampled diagnostics, not certified continuous error bounds.
        """
        from .convex_poisson import trace_transform

        points, weights, _ = self.plan.domain.volume_rule(volume_order)
        f = np.broadcast_to(np.asarray(forcing(points)), (len(points),))
        residual = self.forcing_extension.evaluate(points) - f
        edge, _ = self.plan.arc.sample(boundary_count, offset=0.371)
        g = np.broadcast_to(np.asarray(boundary_data(edge)), (len(edge),))
        error = self.derivative(edge) - g
        # trace_transform uses rfft and is real-valued; split complex data.
        boundary_h32_squared = sum(
            la.norm(trace_transform(part, self.plan.arc.length))**2
            for part in (error.real, error.imag)
        )
        return dict(
            forcing_residual_l2=float(np.sqrt(np.sum(weights * np.abs(residual)**2))),
            forcing_relative_l2=float(np.sqrt(np.sum(weights * np.abs(residual)**2) /
                                     max(np.sum(weights * np.abs(f)**2), 1e-300))),
            boundary_max=float(np.max(np.abs(error))),
            boundary_h32=float(np.sqrt(boundary_h32_squared)),
            volume_points=len(points), boundary_points=len(edge),
        )
