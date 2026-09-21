# Shared rectangular Poisson inverse

The shared algorithm also supports 3D boxes through `plan_box_poisson` and
`solve_box_poisson`; see the [3D API and validation](jax_box_poisson.md).

`bspf_jax.rectangle_poisson` owns the tensor elliptic inverse previously in
`_flow_kernels`. It also exposes a reusable device-resident plan whose setup
uses JAX Cholesky, triangular solves, and symmetric eigendecompositions.
Selecting a GPU places both factorization and subsequent solves on that GPU.
The package does not change precision or device configuration at import.

## Operator and boundary contract

The solver returns coefficient arrays `U` satisfying

```
shift * Mx @ U @ My.T
  + weights[0] * Kx @ U @ My.T
  + weights[1] * Mx @ U @ Ky.T = load
```

`Mx, My` are real symmetric positive-definite one-dimensional mass matrices;
`Kx, Ky` are real symmetric stiffness matrices. The combined operator must be
numerically positive definite. Homogeneous Dirichlet boundary DOFs must already
be eliminated. Lengths and coordinate scaling belong in those matrices. The
load is an integrated weak load, not raw forcing samples. For nonzero boundary
data, the PDE caller subtracts the action of its boundary lift first.

Default `shift=0, weights=(1, 1)` gives the positive Poisson operator `-Delta`.
Positive shifts also support reaction/diffusion and implicit Helmholtz steps.
Pure Neumann/periodic Poisson nullspaces are rejected: compatibility and gauge
handling are not silently replaced by denominator clipping.

## Usage

```python
import jax
from bspf_jax import plan_rectangle_poisson, solve_rectangle_poisson

jax.config.update("jax_enable_x64", True)
device = jax.devices("gpu")[0]

# E.g. boundary-reduced mass/stiffness arrays from Galerkin1D on each axis.
plan = plan_rectangle_poisson(
    gx.mass, gx.stiffness, gy.mass, gy.stiffness, device=device,
)
load = jax.device_put(weak_load, device)  # (nx, ny) or (..., nx, ny)
coefficients = solve_rectangle_poisson(plan, load)
coefficients.block_until_ready()        # synchronize only when needed
```

Build the plan once per fixed operator, outside the PDE time loop. The plan is a
JAX pytree, and the compiled solve supports leading RHS batch dimensions,
`vmap`, and differentiation with respect to the load. CPU devices are supported
for independent verification. Setup performs scalar validation synchronizations;
the warm solve has no host callback, SciPy call, or implicit device transfer
when factors and loads already reside on the same device.

## Cost and reuse

Setup diagonalizes two one-dimensional generalized eigenproblems, using
Cholesky whitening rather than explicit matrix inverses. Each solve applies
four matrix products and one modal division. Storage is
`O(nx² + ny² + nx*ny)`, with solve work
`O(nx²*ny + nx*ny²)` per RHS. No `(nx*ny)²` global matrix is built. This is a
dense fast-diagonalization implementation, not an FFT-complexity claim.

PDEs with existing normalized modal factors use `tensor_elliptic_solve` directly
and avoid refactorization or extra nested compilation:

- Streamfunction NS inertia and its dynamic-boundary variant.
- Weak NS scalar Helmholtz/diffusion solves.
- Smooth-extension rectangular inverse, also used by immersed Poisson.
- Tensor Poisson preconditioners and axisymmetric equilibrium solves.

The small tensor kernel retains NumPy compatibility for existing host assembly.
Reusing it does **not** move the entire immersed-Poisson dense SVD or the
NumPy PCG loop onto GPU. `_flow_kernels.tensor_elliptic_solve` remains a
compatibility import pointing to the same implementation.

The nonsymmetric pressure Schur solver retains its boundary elimination,
nullspace lifts, and compressed transforms. Mapped/variable-coefficient domains
are not separable; the rectangular inverse can serve as a preconditioner there,
but replacing those operators outright would change the equation.

## Verification and timing

`jax/tests/test_rectangle_poisson.py` runs on CPU and GPU (GPU cases skip if
unavailable). It checks a non-square generalized operator against an independent
full Kronecker solve, batched RHS, gradients, JIT/vmap, device residency under a
transfer guard, rejected singular operators, and second-order convergence to a
continuous sine solution on a 2-by-3 rectangle.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src \
  python scratch/benchmark_rectangle_poisson.py --backend gpu
```

The benchmark records synchronized setup, first-call compilation/solve, and
nine warm solves separately, alongside factor storage and independently
recomputed residual/error. Warm times exclude load assembly and transfers.
Results are written to `build/rectangle_poisson/results.json`.

Measured on 2026-09-19 with JAX 0.10.0, FP64, NVIDIA A100-SXM4-40GB:

| Coefficient grid | Setup including setup compilation | First solve including compilation | Warm median | Relative residual |
| --- | ---: | ---: | ---: | ---: |
| 128 × 96 | 2.488 s | 73.96 ms | 0.0921 ms | 2.05e-15 |
| 256 × 192 | 1.949 s | 69.90 ms | 0.1048 ms | 2.45e-15 |
| 512 × 384 | 2.137 s | 70.07 ms | 0.1648 ms | 3.12e-15 |

These are inverse-only timings on centered-difference axis matrices with
identity masses, using random known coefficients to construct a discrete load.
They measure the shared dense tensor algorithm, not BSPF basis construction,
physical discretization error, immersed-domain fitting, or a complete PDE step.
The CPU/GPU unit tests separately exercise nonidentity, nondiagonal mass matrices.
The extraction preserves the previous modal algorithm; these measurements do
not establish a speedup over its previous location in `_flow_kernels`.

Validation: the new CPU/GPU tests and selected stream NS, weak NS, immersed
Poisson, smooth-extension, and tokamak regressions passed (37 tests). The new
files also pass Ruff and the changes pass `git diff --check`.

For continuous manufactured-solution accuracy and comparisons against cuSOLVER,
PyAMGX-PCG and GPU CG, see the [MMS benchmark](jax_rectangle_mms_benchmark.md).
