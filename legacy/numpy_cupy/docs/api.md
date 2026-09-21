# Public API

Import core types from `pybspf`: `BSPF1D`, `BSPF2D`, `PiecewiseBSPF1D`,
`Grid1D`, and `DerivativeResult`. Lowercase `bspf1d` and `bspf2d` remain aliases.

## BSPF1D

Construct with `BSPF1D.from_grid(degree, x, *, n_basis=None, knots=None,
domain=None, use_clustering=False, clustering_factor=2.0, order=None,
num_boundary_points=None, correction="spectral", use_gpu=False)`.

The defaults are `n_basis=4*(degree+1)`, `order=degree-1`, and
`num_boundary_points=degree`. Tune these for the smoothness and resolution of your
problem; a higher degree does not automatically improve conditioning.

| Method | Input shape | Return |
| --- | --- | --- |
| `differentiate(f, k=1, lam=0, *, neumann_bc=None)` | `(n,)` | `(derivative, spline)` |
| `derivatives(f, orders, lam=0, *, neumann_bc=None)` | `(n,)` | `DerivativeResult` |
| `derivatives_batched(f, orders, lam=0, *, neumann_bc=None)` | `(n, batch)` | `DerivativeResult` with the same shape |
| `fit_spline(f, lam=0, neumann_bc=None)` | `(n,)` | `(coefficients, spline, residual)` |
| `definite_integral(f, a=None, b=None, lam=0)` | `(n,)`, real | Python float |
| `antiderivative(f, order=1, *, left_value=0, match_right=None, lam=0)` | `(n,)`, real | `(antiderivative, spline)` |
| `interpolate(f, lam=0, use_fft=False)` | `(n,)`, real, CPU | `(x_new, f_new)` at original nodes and midpoints |
| `interpolate_split_mesh(f, refine_factor, lam=0, neumann_bc=None)` | `(n,)`, real, CPU | `(x_new, f_new, spline_new, residual_new)` |
| `enforced_zero_flux(f)` | `(n,)`, real, CPU | corrected left/right endpoint values |

`orders` is an integer or iterable of integers from 1 through 4. Duplicate orders
are computed once. `result[k]` retrieves a derivative; `result.spline` retrieves
the fitted spline. Empty batches have shape `(n, 0)`.

`neumann_bc=(left_flux, right_flux)` accepts `None` for unconstrained sides and
requires at least two endpoint constraints when a flux is supplied. For batches,
scalar fluxes apply to every column.

`antiderivative` supports orders 1 and 2. For order 1, `match_right`, if supplied,
shifts the integration constant and takes precedence over `left_value`. For
order 2, a linear adjustment can match both endpoints.

`grid`, `degree`, and `use_gpu` describe an operator. Basis matrices and KKT
caches are implementation details; do not mutate them or the underlying grid
coordinates after construction. Operator instances contain mutable caches and
are not promised to be thread-safe.

## BSPF2D

`BSPF2D.from_grids(x=x, y=y, degree_x=5, degree_y=5, use_gpu=False)` creates two
1D operators. Most 1D construction options have `_x` and `_y` variants.

Fields have shape `(len(y), len(x))`:

- `derivatives_axis(field, *, axis, orders, lam=0, neumann_bc=None)` returns a
  `DerivativeResult`.
- `differentiate_axis(field, *, axis, k=1, lam=0, neumann_bc=None)` returns a pair.
- `partial_x(field, *, order=1, ...)` and `partial_y(...)` return pairs.
- `laplacian(field, *, lam_x=0, lam_y=0, neumann_bc_x=None, neumann_bc_y=None)`
  returns one array.

Axis 0 is y and axis 1 is x.

## PiecewiseBSPF1D

`PiecewiseBSPF1D(degree, x, breakpoints=None, min_points_per_seg=16, **bspf_kwargs)`
creates independent operators for each segment. Breakpoints must be finite and
strictly inside the domain. Every sample belongs to a segment; undersized
segments cause a construction error.

`derivatives(f, orders, lam=0, neumann_bc_global=None)` returns stitched results.
Global flux constraints apply only to the outer boundaries.

## Grid1D

`Grid1D(x, *, atol=1e-13, use_gpu=False)` validates a uniform grid and exposes
`x`, `dx`, `a`, `b`, `n`, `omega` (rFFT angular frequencies), and `trap`
(trapezoid weights). Unlike the operator factories, direct GPU grid construction
requires CuPy coordinates.

## Specialized workflows

`Poisson1DDirichletSolver`, `Poisson2DDirichletSolver`, and the Neumann
precompute/apply functions remain available from `pybspf` and load lazily.
`integrate_rk4`, `BSplineValues`, `make_bspline_basis_values`, and the directional
KKT decomposition helpers retain their existing imports. These research-oriented
functions have separate contracts and are not all GPU-enabled. See
[assessment.md](assessment.md) for unresolved solver issues.
