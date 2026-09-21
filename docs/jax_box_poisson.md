# Shared GPU Poisson solver in 2D and 3D

`plan_box_poisson` and `solve_box_poisson` extend the shared fast-diagonalization
algorithm to three spatial dimensions. The existing `plan_rectangle_poisson`
and `solve_rectangle_poisson` API and two-dimensional PDE callers retain their
interfaces. Both constructors share axis validation and GPU factorization.

## Operator and API

For three axes the positive weak operator is

```text
A = shift Mx⊗My⊗Mz
    + wx Kx⊗My⊗Mz + wy Mx⊗Ky⊗Mz + wz Mx⊗My⊗Kz.
```

Each mass matrix must be real symmetric positive definite. Each stiffness
matrix must be real symmetric, and the resulting combined operator must be
numerically positive definite. As in the 2D API, the caller provides matrices
with homogeneous Dirichlet constraints already eliminated and integrated weak
loads. For nonzero boundary data, subtract the action of a boundary lift first.
Lengths and coordinate scaling belong in the axis matrices. Pure Neumann or
periodic nullspaces are rejected; no gauge or compatibility projection is
silently imposed. Defaults are zero shift and unit axis weights.

```python
import jax
from bspf_jax import plan_box_poisson, solve_box_poisson

jax.config.update("jax_enable_x64", True)
device = jax.devices("gpu")[0]
plan = plan_box_poisson(
    (Mx, My, Mz), (Kx, Ky, Kz), device=device,
)
load = jax.device_put(weak_load, device)  # (nx, ny, nz)
coefficients = solve_box_poisson(plan, load)
```

The coefficient layout follows `(x, y, z)` in C order. Input loads may have
leading batch dimensions, e.g. `(batch, nx, ny, nz)`. The number of spatial axes
comes from the plan, not the load rank. This avoids confusing a batch of 2D
problems with one 3D problem. Passing two axis pairs to `plan_box_poisson`
solves a 2D problem. Both plans support JIT, vmap and load differentiation.

`nx`, `ny`, `nz` count independent coefficients, not output-grid points or
spline-correction functions. For the benchmark's particular nodal BSPF
formulation, `n³` coefficients correspond to `(n+2)³` original sampling nodes
after removing boundary values. That relationship need not hold for other bases.

## Shared implementation and cost

1. Cholesky-whiten each axis mass matrix and diagonalize its symmetric stiffness
   operator on the selected device. Reuse these factors for subsequent loads.
2. Apply the three generalized eigenvector transforms along the spatial axes.
3. Divide by `shift + wx*λx + wy*λy + wz*λz`.
4. Apply the three inverse transforms to recover the coefficients.

The plan stores the axis rotations, weighted eigenvalue vectors and shift.
It does not retain a full 3D denominator array. Broadcasting constructs the
modal denominator inside the compiled solve. Identical mass/stiffness input
object pairs share their decomposition and rotation storage within a plan.

For a cubic coefficient array of width `n`, setup uses `O(n³)` work on the
axis matrices; each solve uses `O(n⁴)` dense tensor-transform work. Plan storage
is `O(n²)`, while fields and solve workspace scale as `O(n³)`. No `n³ × n³`
global matrix is built. This is a dense fast-diagonalization algorithm, not an
FFT-complexity claim. Reported plan bytes are not peak GPU memory.

The specialized `pressure3d.py` solver retains its separate face elimination,
nullspace lifts and optional compressed transforms. This new SPD box solver
is not a replacement for those pressure boundary operators or for a
variable-coefficient/curved-domain discretization.

## Validation and benchmark

`jax/tests/test_box_poisson.py` covers non-cubic 3D systems with nonidentity,
nondiagonal masses against an independent full Kronecker matrix; anisotropic
weights and shifts; multiple leading batch axes; JIT/vmap; autodiff; GPU
transfer guards; 2D equivalence; nullspace rejection; and shared-axis storage.
A continuous non-cubic sine MMS verifies second-order FD convergence. A second,
non-eigenmode manufactured forcing is independently checked with autodiff.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src \
  python -m pytest jax/tests/test_box_poisson.py jax/tests/test_rectangle_poisson.py -q

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src \
  python scratch/benchmark_box_poisson.py
```

The benchmark solves `-Δu=f` on `[0,2] × [0,3] × [0,1.5]` with zero Dirichlet
boundaries. Let `t=x/2`, `s=y/3`, `r=z/1.5`:

```text
u = t(1-t)s(1-s)r(1-r) *
    [exp(0.7t-0.4s-0.4r) + 0.2 sin(5πt) cos(3πs) cos(3πr)]
f = -∂xx u - ∂yy u - ∂zz u.
```

The analytic forcing is independent of the discrete operator. BSPF uses degree
5, 16 spline-correction functions per axis, seven-point endpoint fits and
order-8 split Gauss quadrature. FD uses the seven-point stencil. Relative L2
errors are estimated on an independent 257³ Gauss grid using BSPF's continuous
interpolant or trilinear FD reconstruction. This is separate from the algebraic
residual, recomputed using the original mass/stiffness matrices or FD stencil.

Each resolution runs in a fresh process. Cold problem preparation (including
basis/load and error-evaluation tables), factor setup, RHS upload, first solve
and nine synchronized warm solves are recorded separately. Warm solves exclude
transfers and validation, and execute under a transfer guard. Every case must
satisfy an independently recomputed relative residual of at most `2e-10`.

Measured GPU timings, L2 errors and plan storage are in the
[3D benchmark report](jax_box_poisson_benchmark.md). Generate that report from
saved measurements with `python scratch/report_box_poisson.py`.

Actual process GPU peaks, allocator high-water marks and compiled buffers are
reported in the [GPU memory profile](jax_box_poisson_memory.md).
