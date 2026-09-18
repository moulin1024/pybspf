"""Fixed-grid output for the convex-domain BSPF Poisson solver.

The dense reference remains the default backend. The experimental tensor backend
is opt-in and raises on failed iteration. Two cached one-dimensional factors
replace paired-point evaluation, and callers
receive grid values, an interior mask, and diagnostics rather than coefficients.
Arrays follow meshgrid(indexing="xy"): values[j, i] corresponds to (x[i], y[j]).
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .convex_poisson import ConvexPoissonPlan
from .convex_poisson_tensor import TensorConvexPoissonPlan
from .stream_navier_stokes import stream_evaluate_line


def _axis(values, bounds, name):
    a = np.array(values, dtype=float, copy=True)
    if a.ndim != 1 or not len(a) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional axis")
    if np.any(np.diff(a) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    if a[0] < bounds[0] or a[-1] > bounds[-1]:
        raise ValueError(f"{name} must lie in the auxiliary BSPF interval")
    a.setflags(write=False)
    return a


def _closed_domain_mask(domain, x, y):
    """Analytic spline/slice intersections, including the two x-tangencies."""
    roots = domain.xpoly.derivative().roots(extrapolate=False)
    roots = roots[np.isfinite(roots) & (roots >= 0) & (roots <= domain.period)]
    extrema = domain.curve(np.r_[np.arange(domain.period + 1), roots])
    lo, hi = extrema[:, 0].min(), extrema[:, 0].max()
    tol = 64 * np.finfo(float).eps * max(1.0, np.max(abs(extrema)))
    mask = np.zeros((len(y), len(x)), dtype=bool)
    for i, xx in enumerate(x):
        if xx < lo - tol or xx > hi + tol:
            continue
        # An isolated tangency has just one intersection, which the usual
        # even-intersection slice routine deliberately rejects. A vertical
        # boundary segment is covered by its extreme y coordinates as well.
        if abs(xx - lo) <= tol or abs(xx - hi) <= tol:
            edge = extrema[abs(extrema[:, 0] - xx) <= tol, 1]
            mask[:, i] = (y >= edge.min() - tol) & (y <= edge.max() + tol)
        else:
            for lower, upper in domain.intersections(xx):
                mask[:, i] |= (y >= lower - tol) & (y <= upper + tol)
    mask.setflags(write=False)
    return mask


class _TensorGridOutput:
    """Private cached value-only runtime; never assemble a 2D evaluation matrix."""

    def __init__(self, line, domain, x, y):
        bounds = np.asarray(line.x)[[0, -1]]
        self.x = _axis(x, bounds, "x")
        self.y = _axis(y, bounds, "y")
        self.inside = _closed_domain_mask(domain, self.x, self.y)
        if not self.inside.any():
            raise ValueError("The requested grid contains no points in the domain")
        self.bx = stream_evaluate_line(line, self.x)[0]
        self.by = (
            self.bx
            if np.array_equal(self.x, self.y)
            else stream_evaluate_line(line, self.y)[0]
        )

    @property
    def basis_storage_bytes(self):
        return self.bx.nbytes + (0 if self.by is self.bx else self.by.nbytes)

    def values(self, coefficients):
        c = np.asarray(coefficients).reshape(self.bx.shape[1], self.by.shape[1])
        # Contract the shorter target axis first to reduce the first product.
        if len(self.x) <= len(self.y):
            result = (self.bx @ c @ self.by.T).T
        else:
            result = self.by @ c.T @ self.bx.T
        result[~self.inside] = np.nan
        return result


@dataclass(frozen=True)
class ConvexPoissonGridResult:
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    inside: np.ndarray
    diagnostics: dict


class ConvexPoissonGridPlan:
    """Solve -Delta u=f, u|boundary=g and return only a specified Cartesian grid.

    ``nodes`` controls the internal approximation; ``x,y`` only select output
    samples. Forcing and boundary callbacks still supply physical data at the
    solver's integration/boundary points. There is no interpolation of data
    from the output grid. Geometry and callbacks follow ConvexPoissonPlan.

    ``backend="reference"`` uses the verified dense factorization. The optional
    ``backend="tensor"`` uses matrix-free regularized least squares and refuses
    unconverged iterates. Repeated solves reuse setup and output factors.
    Use ``from_plan`` to reuse an already constructed solver.
    """

    def __init__(self, domain, x, y, *, backend="reference", **solver_options):
        # Validate output axes before starting the costly solver construction.
        half_width = solver_options.get("half_width", 1.2)
        x = _axis(x, (-half_width, half_width), "x")
        y = _axis(y, (-half_width, half_width), "y")
        if not _closed_domain_mask(domain, x, y).any():
            raise ValueError("The requested grid contains no points in the domain")
        if backend == "reference":
            solver = ConvexPoissonPlan
        elif backend == "tensor":
            solver = TensorConvexPoissonPlan
        else:
            raise ValueError("backend must be 'reference' or 'tensor'")
        self._bind(solver(domain, **solver_options), x, y)

    @classmethod
    def from_plan(cls, plan, x, y):
        """Bind a target grid without rebuilding the existing solver."""
        result = cls.__new__(cls)
        result._bind(plan, x, y)
        return result

    def _bind(self, plan, x, y):
        if not isinstance(plan, (ConvexPoissonPlan, TensorConvexPoissonPlan)):
            raise TypeError("Expected a convex Poisson solver plan")
        self._solver = plan
        self._output = _TensorGridOutput(plan.line, plan.domain, x, y)

    def solve(self, forcing, boundary_data):
        start = perf_counter()
        solution = self._solver.solve(forcing, boundary_data)
        solved = perf_counter()
        values = self._output.values(solution.coefficients)
        elapsed = perf_counter() - solved
        diagnostics = dict(
            solution.diagnostics,
            output_backend="cached_tensor_product",
            output_shape=values.shape,
            output_inside_points=int(self._output.inside.sum()),
            output_basis_storage_bytes=self._output.basis_storage_bytes,
            solve_seconds=solved - start,
            output_seconds=elapsed,
        )
        diagnostics.setdefault("backend", "dense_reference_svd")
        return ConvexPoissonGridResult(
            self._output.x, self._output.y, values, self._output.inside, diagnostics
        )
