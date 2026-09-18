# Masked pressure projection in JAX

The JAX backend now includes the 2D pressure core extracted from the 3D NS
benchmark, including the optional Chebyshev endpoint estimator. The default dense
backend uses JAX for numerical assembly and application, and NumPy for validation.
The optional DCT/HODLR backend uses SciPy during host compression setup; its
application is JAX. Neither backend depends on the NumPy solver or benchmark folder.

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import bspf_jax as b

x = jnp.linspace(0, 1, 64)
y = jnp.linspace(0, 1.5, 72)
plan = b.plan_pressure_poisson2d(
    x, y,
    endpoint_method="chebyshev",
    chebyshev_modes=14,
    baseline_points=18,
    endpoint_regularization=1e-12,
)
X, Y = jnp.meshgrid(x, y, indexing="ij")
exact = jnp.exp(X + 0.5*Y)
raw = jnp.stack([exact, 0.5*exact], axis=-1)
projected, result = jax.jit(b.project_pressure2d)(plan, raw)
assert bool(result.converged)
pressure_error = jnp.linalg.norm(result.pressure - b.pressure_remove_mean(plan, exact))
```

No Chebyshev nodes are needed: the fit uses the existing uniform-grid endpoint
samples. `endpoint_method="taylor"` remains the default. The other defaults are
`q=9`, `n_basis=32`, `degree=13`, `baseline_points=14`, and `chebyshev_modes=12`.
The endpoint-option names match the NumPy pressure solver. General JAX
differentiation plans instead use `boundary_points` and `chebyshev_alpha`;
both paths reuse the same augmented-QR Chebyshev estimator with penalty power 4.

## Shapes and equation

Follow JAX's physical-axis ordering: scalar arrays have shape `(nx, ny)` and
vectors have shape `(nx, ny, 2)`, with Cartesian components `(x, y)`. This differs
from the NumPy pressure API's `(ny, nx)` layout; swap axes 0 and 1 when comparing.
Use `jax.vmap` for a leading batch dimension.

```python
batched_project = jax.jit(jax.vmap(b.project_pressure2d, in_axes=(None, 0)))
vectors, results = batched_project(plan, jnp.stack([raw, 2*raw]))
assert bool(jnp.all(results.converged))
```

The operator is `S p = div(Q grad p)`, where Q is zero on all four walls and
one on strictly interior nodes. Projection solves `S p = div(Q raw)` and returns
`Q*(raw-grad p)`. In an NS RK stage, supply the non-pressure acceleration; there
is no time-step scaling. This is not a general prescribed-Neumann/Dirichlet
solver for the continuum Laplacian.

The public primitives are `pressure_gradient`, `pressure_divergence`,
`pressure_schur`, and `pressure_remove_mean`. They take `(plan, values)`.
They intentionally use the pressure plan's strong derivatives, not the general
calculus API's operators: the pressure core preserves the source's unregularized
QR spline fit and N-1-point FFT on endpoint-inclusive data.

## Solve, completion, and diagnostics

```python
rhs = b.pressure_divergence(plan, plan.mask[..., None]*raw)
result = jax.jit(b.solve_pressure_poisson2d)(plan, rhs, wall_gradient=raw)
```

The optional wall-gradient argument has full `(nx, ny, 2)` shape; only its
boundary entries are fitted. Seven nonconstant null modes are determined by
wall-gradient least squares, and a trapezoidal mean fixes the constant.
Without this argument, `solve_pressure_poisson2d` selects the zero-mean pressure
representative specified by the lifts, not homogeneous Neumann data.

`project_pressure2d` enables completion by default. To omit it under JIT:

```python
from functools import partial
project_without_completion = jax.jit(partial(b.project_pressure2d, completion=False))
```

Each result contains:

- `pressure`: the final pressure after completion and mean removal.
- `schur_residual_linf`, `schur_residual_l2`: the original unlifted equation residual.
- `wall_gradient_fit_linf`: the wall least-squares residual, or NaN without completion.
- `converged`: whether finite inputs and nonnegative finite tolerances produced a
  finite solution satisfying `norm(S p - rhs) <= atol + rtol*norm(rhs)`.

Default tolerances are `atol=1e-9`, `rtol=1e-10`. **Always check `converged`.**
Incompatible RHS or nonfinite numerical data returns a false flag, including
inside JIT; it does not raise a runtime Python exception or silently project the
RHS. This differs from the checked NumPy solve. Shape and complex-input errors
raise during tracing. A successful Schur solve does not imply zero wall fit error.

## JAX contract and validation

Build a plan outside JIT and reuse it. The plan is an immutable PyTree containing
JAX arrays. Numerical setup includes nonsymmetric eigendecompositions, so the
setup device must support those; CPU is the validated setup/execution backend.
Plans can be explicitly transferred with `jax.device_put` for other devices,
but accelerator execution has not been validated here. The package does not
enable x64 or change the selected device automatically.

Application kernels support JIT, batching with vmap, and reverse-mode derivatives
with respect to input fields. Geometry, spline choices, and endpoint parameters
are static setup inputs: autodiff through the checked factory is unsupported.
All fields are real; scientific float64 is required.

The default dense implementation uses 1D factorizations, tensor transforms,
two refinement steps, and a wall-by-seven QR factorization; it never builds a full 2D pressure
matrix. Tests compare NumPy/JAX pressures and projected fields on rectangular
grids for both endpoint methods, verify JIT/eager agreement, null completion,
idempotence, batched projection, reverse-mode directional derivatives, failure
flags, and pressure accuracy improvements at production spline order.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest -c jax/pyproject.toml jax/tests/test_pressure.py
```

## Experimental DCT/HODLR direct transforms

Install the optional setup dependency with `pip install -e 'jax[compression]'`.
Use `transform_backend="dct_hodlr"` in `plan_pressure_poisson2d`, or reuse a dense
plan without repeating its assembly:

```python
compressed = b.compress_pressure_plan(
    plan, tolerance=1e-12, leaf_size=16, protected_modes=8, layout="layered",
)
project = jax.jit(lambda model, raw: b.project_pressure2d(
    model, raw, completion=False, refinement_steps=0,
))
velocity, result = project(compressed, raw)
assert bool(result.converged)
```

The corresponding factory options are `compression_tolerance`,
`compression_leaf_size`, `protected_modes`, and `compression_layout`. Setup applies an orthonormal
DCT-II, matches modes, and factors off-diagonal blocks recursively. The default `layout="layered"` pads the coefficient matrix to equal-width leaves
and batches sibling blocks by tree level, using a shared maximum rank on each
level (rounded up to a multiple of four). The padding is internal to the matrix
application: the DCT still acts on the original grid. Reshape/reverse and batched
products replace block gather/scatter operations. The original implementation
is selectable with `layout="grouped"` (factory: `compression_layout="grouped"`).
Mode permutations still require gathers, outside the HODLR block application.
Dense leaves remain small; the full V or inverse is never reconstructed during
application. The compressed plan sets both `line.vectors` and
`line.inverse_vectors` to `None`; retained factors replace those arrays. The
original dense plan remains intact if the caller retains it. Only a few exact
columns/rows are retained for protected modes and the nullspace lift.

Fixed algebraic projections preserve the first `protected_modes` eigenmodes,
ordered by absolute eigenvalue. This includes both null modes. These corrections
are direct low-rank products, not iteration. Setup is host-side and not
JIT/differentiable; application supports JIT, vmap, and field autodiff. FFT/DCT
and factor application remain on the JAX device.

Compressed plans default to **zero refinement steps**, so one tensor inverse is
applied. Dense plans retain their previous two-step refinement default. An
explicit static `refinement_steps=0` makes dense/compressed comparisons fair.
There is no automatic retry, fallback solve, or tolerance relaxation. Always
check `result.converged`; block tolerance does not guarantee pressure accuracy.

This is an approximate direct method at the selected compression tolerance.
Small grids may not compress and may run slower. Dense eigendecomposition and
host SVD/matching are still required at setup. Actual timing and accuracy are
recorded by `scratch/benchmark_compressed_pressure.py`; see the accompanying
[layered-layout benchmark report](jax_layered_pressure.md). The original matrix
[compressibility analysis](pressure_transform_compressibility.md) does not by
itself establish a runtime speedup.
