"""Smooth-field immersed Dirichlet Poisson on a rectangle with an elliptic hole.

The physical solution and the fictitious hole field are restrictions of one
BSPF tensor field. Only physical forcing and boundary values enter the fit.
The existing rectangular Dirichlet Poisson inverse right-preconditions the
oversampled strong equations. This is a dense accuracy prototype, not a
regularized-delta IB scheme or an implementation of the IBSE PDE formulation.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from time import perf_counter

import jax
import numpy as np
import scipy.linalg as la

from .convex_poisson import ArcLengthBoundary, trace_transform
from .smooth_extension import basis_operators, evaluate_factors, tensor_inverse
from .stream_navier_stokes import _stream_line, stream_evaluate_line
from ._flow_kernels import tensor_product


@dataclass(frozen=True)
class EllipticHole:
    center: tuple = (0.19, -0.13)
    axes: tuple = (0.31, 0.23)
    period: int = field(default=4, init=False)

    def __post_init__(self):
        for name in ("center", "axes"):
            vector = np.asarray(getattr(self, name), dtype=float)
            if vector.shape != (2,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} must contain two finite coordinates")
            object.__setattr__(self, name, tuple(vector))
        if min(self.axes) <= 0:
            raise ValueError("Hole axes must be positive")

    def curve(self, t, nu=0):
        phase = np.asarray(t) * np.pi / 2 + nu * np.pi / 2
        result = (
            np.asarray(self.axes)
            * (np.pi / 2) ** nu
            * np.stack((np.cos(phase), np.sin(phase)), axis=-1)
        )
        return result + np.asarray(self.center) if nu == 0 else result

    def level(self, points):
        return np.sum(((np.asarray(points) - self.center) / self.axes) ** 2, axis=-1)

    def normal(self, t):
        # Points from solid hole into fluid; reverse for fluid-domain outward.
        vector = (self.curve(t) - self.center) / np.asarray(self.axes) ** 2
        return vector / np.linalg.norm(vector, axis=-1)[..., None]


class ImmersedPoissonPlan:
    def __init__(
        self,
        *,
        nodes=33,
        hole=None,
        half_width=1.0,
        oversampling=1.5,
        boundary_count=None,
        rcond=1e-12,
        collar=True,
        endpoint_points=None,
        chebyshev_modes=None,
    ):
        start = perf_counter()
        if not jax.config.x64_enabled:
            raise ValueError("Enable jax_enable_x64 before constructing BSPF factors")
        self.hole = hole or EllipticHole()
        self.half_width = float(half_width)
        if not np.isfinite(half_width) or half_width <= 0:
            raise ValueError("half_width must be finite and positive")
        if np.any(np.abs(self.hole.center) + np.asarray(self.hole.axes) >= half_width):
            raise ValueError("Hole must lie strictly inside the rectangle")
        if not 0 < rcond < 1 or not np.isfinite(oversampling) or oversampling < 1:
            raise ValueError("Invalid hole or discretization parameters")
        if boundary_count is not None and (
            not isinstance(boundary_count, (int, np.integer)) or boundary_count < 8
        ):
            raise ValueError("boundary_count must be an integer >=8")
        self.nodes, self.rcond = nodes, rcond
        endpoint_points = min(nodes, 24) if endpoint_points is None else endpoint_points
        chebyshev_modes = (
            min(endpoint_points, 20) if chebyshev_modes is None else chebyshev_modes
        )
        self.endpoint_points, self.chebyshev_modes = endpoint_points, chebyshev_modes
        self.collar, self.oversampling = bool(collar), oversampling
        self.line = _stream_line(
            np.linspace(-half_width, half_width, nodes),
            clamped=False,
            dirichlet=True,
            endpoint_points=endpoint_points,
            chebyshev_modes=chebyshev_modes,
        )
        self.line_seconds = perf_counter() - start
        lam = np.asarray(self.line.lam)
        self.denominator = lam[:, None] + lam[None, :]
        self.ndofs = self.denominator.size
        self.scale = self.denominator.ravel()

        count = int(np.ceil(oversampling * nodes)) + 1
        axis = np.linspace(-half_width, half_width, count)
        xx, yy = np.meshgrid(axis, axis, indexing="ij")
        candidates = np.column_stack((xx.ravel(), yy.ravel()))
        self.physical = self.hole.level(candidates) > 1
        self.points = candidates[self.physical]
        self.spacing = 2 * half_width / (count - 1)
        endpoint_weight = np.ones(count)
        endpoint_weight[[0, -1]] = 0.5
        self.pde_weights = self.spacing * np.sqrt(
            np.outer(endpoint_weight, endpoint_weight).ravel()[self.physical]
        )
        # Tensor factors are reused: do not evaluate each paired point in MPFR.
        value, _, second = stream_evaluate_line(self.line, axis)
        full_laplace = -(
            np.einsum("ia,jb->ijab", second, value)
            + np.einsum("ia,jb->ijab", value, second)
        ).reshape(count**2, -1)
        operator = self.pde_weights[:, None] * full_laplace[self.physical] / self.scale
        del full_laplace
        self.arc = ArcLengthBoundary(self.hole)
        self.boundary_count = boundary_count or 4 * nodes
        if self.boundary_count % 2:
            self.boundary_count += 1
        self.boundary, _ = self.arc.sample(self.boundary_count)
        if collar:
            # Independent normal rays fill the unsampled strip between the
            # curved wall and Cartesian PDE points. This is a weighted sampling
            # norm, not a claim of high-order cut-cell volume quadrature.
            bp, bt = self.arc.sample(self.boundary_count, offset=0.5)
            gap = np.min(half_width - np.abs(self.hole.center) - self.hole.axes)
            width = min(self.spacing, gap / 3)
            offsets = width * np.array([1e-6, 0.25, 0.75])
            cp = (
                bp[None, :, :] + offsets[:, None, None] * self.hole.normal(bt)
            ).reshape(-1, 2)
            weight = np.sqrt(width * self.arc.length / len(cp))
            cop = weight * basis_operators(self.line, cp)[1] / self.scale
            operator = np.vstack((operator, cop))
            self.points = np.vstack((self.points, cp))
            self.pde_weights = np.r_[self.pde_weights, np.full(len(cp), weight)]
        trace = basis_operators(self.line, self.boundary)[0] / self.scale
        trace = trace_transform(trace, self.arc.length, 1.5)
        operator = np.vstack((operator, trace))
        self.assembly_seconds = perf_counter() - start - self.line_seconds
        factor_start = perf_counter()
        left, singular, right = la.svd(operator, full_matrices=False, overwrite_a=True)
        keep = singular > rcond * singular[0]
        self.left, self.singular, self.right = (
            left[:, keep],
            singular[keep],
            right[keep],
        )
        self.rank = int(keep.sum())
        self.svd_seconds = perf_counter() - factor_start
        self.setup_seconds = perf_counter() - start
        # Fixed output grids/points reuse only 1D factors, never dense 2D maps.
        line = self.line

        @lru_cache(maxsize=12)
        def cached_factors(coordinates):
            return stream_evaluate_line(
                line, np.frombuffer(coordinates, dtype=np.float64)
            )

        self._cached_factors = cached_factors

    def _line_values(self, coordinates):
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if coordinates.ndim != 1 or not np.all(np.isfinite(coordinates)):
            raise ValueError("Coordinates must be finite one-dimensional arrays")
        if np.any(abs(coordinates) > self.half_width + 1e-14):
            raise ValueError(
                "Evaluation points must lie inside the background rectangle"
            )
        return self._cached_factors(coordinates.tobytes())

    def solve(self, forcing, wall_data):
        """Only samples forcing in the fluid; homogeneous outer Dirichlet data."""
        start = perf_counter()
        f, g = np.asarray(forcing(self.points)), np.asarray(wall_data(self.boundary))
        if f.shape != (len(self.points),) or g.shape != (self.boundary_count,):
            raise ValueError("Callbacks must return one value per supplied point")
        if not np.all(np.isfinite(f)) or not np.all(np.isfinite(g)):
            raise ValueError("Nonfinite physical data")
        rhs = np.r_[self.pde_weights * f, trace_transform(g, self.arc.length, 1.5)]
        projected = self.left.T @ rhs
        a = self.right.T @ (projected / self.singular)
        # Actual reuse of the original rectangular tensor Poisson inverse.
        coefficient = tensor_inverse(a, self.denominator)
        diagnostics = dict(
            nodes=self.nodes,
            ndofs=self.ndofs,
            retained_rank=self.rank,
            rcond=self.rcond,
            oversampling=self.oversampling,
            collar=self.collar,
            endpoint_points=self.endpoint_points,
            chebyshev_modes=self.chebyshev_modes,
            retained_condition=float(self.singular[0] / self.singular[-1]),
            factor_bytes=self.left.nbytes + self.right.nbytes + self.singular.nbytes,
            physical_data_only=True,
            coefficient_norm=float(la.norm(coefficient)),
            physical_samples=len(self.points),
            boundary_samples=self.boundary_count,
            setup_seconds=self.setup_seconds,
            line_seconds=self.line_seconds,
            assembly_seconds=self.assembly_seconds,
            svd_seconds=self.svd_seconds,
            solve_seconds=perf_counter() - start,
            training_relative=float(
                la.norm(rhs - self.left @ projected) / max(la.norm(rhs), 1e-300)
            ),
        )
        return ImmersedPoissonSolution(self, coefficient, diagnostics)


@dataclass
class ImmersedPoissonSolution:
    plan: ImmersedPoissonPlan
    coefficients: np.ndarray
    diagnostics: dict

    def evaluate(self, points):
        """u, grad u, Delta u; inside the hole these describe the auxiliary field."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (count, 2)")
        factors = []
        for axis in range(2):
            coordinates, inverse = np.unique(points[:, axis], return_inverse=True)
            factors.append([v[inverse] for v in self.plan._line_values(coordinates)])
        return evaluate_factors(factors, self.coefficients)

    def grid(self, x, y):
        """Return arrays of shape (len(y), len(x)); hole values are auxiliary."""
        bx = self.plan._line_values(x)
        by = self.plan._line_values(y)
        c = self.coefficients.reshape(len(self.plan.line.lam), -1)
        value = tensor_product(bx[0], by[0], c).T
        gx = tensor_product(bx[1], by[0], c).T
        gy = tensor_product(bx[0], by[1], c).T
        laplace = (tensor_product(bx[2], by[0], c) + tensor_product(bx[0], by[2], c)).T
        return value, np.stack((gx, gy), axis=-1), laplace
