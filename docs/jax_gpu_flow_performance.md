# Rational-obstacle GPU profiling and optimization

Measured on NVIDIA A100-SXM4-40GB, driver 580.178.04, JAX/jaxlib 0.10.0,
float64, with one OpenBLAS/OpenMP thread. The physical case is the unchanged
73×33 BSPF grid, Re=20, dt=0.02, final time 20, rational wall correction,
quadrature factor 2.5, and 41 snapshots on a 401×161 output grid.

## GPU rational recurrences and optional Arnoldi construction (latest)

Rational basis values, first/second derivatives, stream rows, and coefficient
products now execute on the GPU whenever `RationalStokesExtension.evaluate`
receives `device`. The CPU evaluator remains an independent reference. Blocks
share one padded complex128 recurrence loop; a padded final point batch avoids
recompiling every kernel for a partial batch. Small recurrence tables are cached
per device. No host basis evaluation or large row upload occurs in this path.

Measured on the same 17,881 volume points and 185 coefficient columns:

| Measurement | Previous host rows + GPU products | GPU rows + GPU products |
| --- | ---: | ---: |
| Warm rational evaluation, median of 3 synchronized calls | 2.067 s | 0.151 s |
| Cold volume operators and lift evaluation | 2.962 s | 3.016 s |
| Cold complete isolated setup | 20.130 s | 20.317 s |

The warm operation is **13.7× faster**. Cold setup is essentially unchanged;
compilation absorbs the execution saving. This is not a demonstrated cold-start
speedup. The measurements include all GPU work and synchronization, not just
kernel dispatch. Artifacts: `build/immersed_flow/profile/gpu_rational/` contains
`profile_setup.py`, `decomposition.json`, and full-run comparisons.

The initial twice-modified-Gram–Schmidt Arnoldi construction is also implemented
on GPU, preserving both ordered orthogonalization passes. Enable it with:

```python
plan = ImmersedFlowPlan(
    assembly_device=jax.devices("gpu")[0],
    basis_precision="float64", wall_method="rational",
    rational_options={"basis_construction": "gpu"},
)
```

Or add `--rational-basis-construction gpu` to the GPU rational-flow example.
It remains **opt-in**: the complete t=20 run with GPU Arnoldi took **22.675 s
setup**, compared with about 20.3 s using CPU Arnoldi. A representative
96/64-degree construction microbenchmark also showed slower warm GPU construction;
the ordered reductions are small and sequential. Small recurrence tables return
to the host to support the CPU reference and boundary setup. AAA pole selection,
initial boundary-row/SVD-input assembly, and reference/output evaluation remain
host-side; this change does not make the entire geometry setup GPU-resident.

With GPU Arnoldi enabled, all 41 snapshots agree with the preceding host-Arnoldi
run to max absolute 4.819e-12 (u), 4.426e-12 (v), and 1.788e-13 (psi). Rational
rank 1,091, trace rank 184, and 2,059 retained modes are unchanged. Independent
wall speed is 3.963e-11, outer Dirichlet error 1.797e-10, and relative flux error
4.024e-12. Vorticity is more sensitive to recurrence roundoff at the lightning
corners: max change 2.575e-6 at (-1,-1), versus 1.146e-9 on the interior output
grid (excluding outer edges). Global relative L2 vorticity change is 2.320e-9.
This sensitivity is another reason to retain CPU Arnoldi as the default.

Validation includes GPU-resident construction and differentiated evaluation under
transfer guards, independent CPU recurrence comparisons, vector/matrix products,
empty/partial batches, and a check that GPU evaluation cannot invoke the host
basis evaluator. The manufactured-solution suite also exercises GPU Arnoldi.
Final regression result: **49 passed, 1 skipped in 212.50 s**. The skip is
the GPU parity check on the CPU-only reference fixture. Existing accuracy
thresholds are unchanged.

### Investigation of the corner-vorticity discrepancy

Controlled experiments use the same geometry, poles, boundary samples, rank
cutoff, and boundary data (the negative channel lift on the hole), before time
integration. Both construction variants use the same GPU SVD unless stated
otherwise. This isolates the initial t=0 discrepancy exactly: at (-1,-1), GPU
Arnoldi changes the lift vorticity by -1.94377e-7, matching the full run's initial
-1.94377e-7 difference. The larger 2.57439e-6 maximum occurs later at t=0.5.

The evidence separates three effects:

* **Evaluation is accurate.** With a fixed coefficient vector, changing from
  CPU-built to GPU-built recurrences changes corner vorticity by only 4.98e-15.
  Direct FP64 vorticity agrees with 160-bit MPFR evaluation within 3.61e-15 over
  all probe points. Increasing MPFR evaluation to 256 bits leaves the reported
  CPU corner values unchanged. GPU evaluation versus host evaluation differs by
  at most 1.05e-13 in the same experiment; subtracting the two differentiated
  velocity components contributes at most 9.84e-14. These cannot explain the
  observed 1.94e-7 difference.
* **The clustered Arnoldi blocks amplify construction roundoff.** Polynomial,
  Laurent, and AAA blocks differ at roughly 1e-15 relative; in each 32-pole
  lightning block, relative differences grow from roundoff in early columns to
  about 1e-10–2e-10 in the final columns. The resulting column-normalized
  boundary matrices differ by 1.214e-10 in relative Frobenius norm. The poles,
  samples, degrees, and precision selection have not changed.
* **The boundary fit is sensitive to this perturbation.** Its retained condition
  number is about 4.4e12. Re-solving identical data changes coefficients by up
  to 9.37e-7. Applying that coefficient difference to the CPU boundary matrix
  changes boundary data by only 6.26e-14, while corner vorticity changes by
  1.94e-7. Most of the derivative difference comes from singular directions
  with relative singular values between 1e-6 and 1e-11. Both boundary fits have
  max residual 9.33e-12, and both retain rank 1,091.

This is sensitivity of the nearly rank-deficient boundary fit, not a failure of
GPU recurrence evaluation or time integration. It is also not unique to GPU
Arnoldi: switching only the SVD backend, while keeping the CPU basis, changes
the same corner vorticity by -3.11107e-7. The CPU baseline itself is not a
high-precision derivative reference at the corners.

The discrepancy is highly localized. In the controlled lift problem, the
CPU/GPU difference falls from 1.94e-7 at the corner to 2.78e-10 at an inward
offset of 1e-6 in each coordinate, and to 7.83e-14 at an offset of 1e-2. For the
complete flow, max interior output-grid vorticity difference is 1.146e-9.

A rank-cutoff sweep does not justify simply discarding more modes:

| rcond | Retained rank | CPU corner lift vorticity magnitude | Boundary max residual |
| --- | ---: | ---: | ---: |
| 1e-10 | 1,031 | 2.78e-4 | 2.37e-9 |
| 1e-11 | 1,051 | 1.71e-4 | 6.00e-10 |
| 1e-12 | 1,077 | 1.95e-5 | 1.38e-11 |
| 1e-13 (unchanged default) | 1,091 | 1.44e-6 | 9.33e-12 |

The homogeneous velocity correction vanishes along both inlet-adjacent walls;
its compatible smooth corner vorticity is zero. Stronger truncation therefore
worsens the absolute corner error even if it reduces CPU/GPU disagreement.
Higher-precision evaluation alone cannot fix coefficient sensitivity. Improving
corner derivative accuracy would require stabilizing the basis/boundary fit or
adding consistent derivative control, with the manufactured-solution checks
repeated. No cutoff, constraint, or output value was changed to hide the error.

Reproducible diagnostics and data:
`build/immersed_flow/profile/gpu_rational/isolate_arnoldi.py`,
`arnoldi_isolation.json`, `corner_lift.json`, and `comparison.json`.

## GPU volume-operator construction (preceding change)

The synchronized volume-operator/lift stage now takes **2.962 s**, versus
**6.295 s**: **53.0% less time**, or 2.13× faster. The complete isolated setup
profile measured 20.130 s. A separate full FP64-basis t=20 run measured
**20.395 s setup**, down from 23.834 s, with setup plus evolution/output falling
from **40.019 s to 36.654 s**. Resolution, precision selection, time step, output
frames, and all rank cutoffs are unchanged.

The targeted CPU profile (`gpu_volume/before.prof`) attributed 2.095 s to applying
rational corrections, 1.291 s to rational row/coefficient products, and 0.658 s to
tensor products. These computations now execute on the GPU:

* Rational stream rows are constructed in bounded host batches and their
  coefficient products execute on device, preserving the existing two-stage
  response/mapping order.
* Tensor-product operators and rational corrections are combined on device.
* Lift application and column scaling (or the full constraint transform) use
  resident operators. Only the small lift vectors return to the host here.
* Unnormalized volume matrices remain resident for Gram assembly. They are no
  longer materialized on the host and uploaded again. The Gram-stage wall time
  decreased from 1.159 s to 0.792 s in the isolated profiles.

The new volume-stage measurement explicitly waits for all output operators to
be GPU-ready; it does not hide asynchronous execution in the next stage. It
includes 0.685 s of backend compilation. Remaining host work includes the
rational Arnoldi recurrences and construction of their stream rows. The
factored-wall geometry path retains its existing host construction; GPU volume
construction is used for rational, sampled-SVD, and unobstructed channel plans.
Both MPFR and FP64 line-basis options are supported. CPU assembly remains the
independent reference, and the default public `operators()` return is unchanged.

Across all 41 saved frames, maximum absolute changes from the preceding GPU-SVD
run are 1.453e-12 (u), 1.261e-12 (v), 3.347e-10 (vorticity), and 4.375e-14
(streamfunction). Relative L2 changes are 7.121e-14, 6.365e-13, 2.668e-12, and
9.392e-15 respectively. Rational rank 1,091, trace rank 184, and retained volume
mode count 2,059 are unchanged. Independent wall speed is 3.961e-11, outer
Dirichlet error 1.798e-10, and relative flux error 4.054e-12.

Validation: **41 passed, 1 skipped in 205.17 s**. The skip is the GPU parity
check on the CPU-only reference fixture. Tests include independent tensor-field
contractions for diagonal/full constraints with and without rational corrections,
transfer guards and device-residency checks, vector/matrix rational evaluation
with partial batches, host/GPU operator parity, and all selected existing
manufactured-flow, boundary/flux, GPU-stepper, decomposition, and immersed-flow
regressions. Physical thresholds were unchanged.

Evidence is under `build/immersed_flow/profile/gpu_volume/`:
`before.prof`, `decomposition.json`, `full/{summary.json,fields.npz}`, and
`comparison.json`. The full run and isolated profile finished before regression
testing began. Phase sums exclude post-run independent checks and file writing.

## GPU setup SVDs and energy eigendecomposition


With a GPU `assembly_device` (the example's `--backend gpu`), the rational
boundary-extension SVD, obstacle-trace SVD, and volume energy eigendecomposition
now execute on that device. The full-matrix SVD used by `wall_method="svd"` also
uses the GPU. CPU setup remains the default when no assembly device is selected.
This applies to both MPFR and FP64 basis evaluation.

The implementation uses explicit QR-based `jax.lax.linalg.svd` / cuSOLVER rather
than forming normal equations, and a symmetric GPU eigendecomposition. Rank
cutoffs, volume-mode cutoffs, and subsequent SVD response ordering are unchanged.
There is no CPU fallback: non-finite factors raise an error. Finished factors
return to the host geometry plan; the evolution runtime remains GPU-resident.
Small line-space CPU factorizations, AAA pole selection, and host geometry work
remain separate setup stages.

An isolated cold profile on the same A100 measured:

| Decomposition | Previous CPU computation | GPU trace/lower | GPU compile | GPU execute/synchronize |
|---|---:|---:|---:|---:|
| Rational boundary SVD | 2.537 s | 0.008 s | 0.420 s | 0.189 s |
| Obstacle-trace SVD | 2.098 s | 0.007 s | 0.129 s | 0.331 s |
| Volume energy eigendecomposition | 1.926 s | 0.010 s | 0.333 s | 0.060 s |

GPU columns exclude factor upload/download and host response construction.
Total profiled setup was 23.944 s. The separate full FP64-basis t=20 run measured
**23.834 s setup**, down from **29.124 s** (18.2% reduction). Setup plus
evolution/output fell from **45.578 s to 40.019 s**. The full run completed before
regression testing; the isolated profile also ran without concurrent tests.

All 41 physical frames were compared to the preceding FP64-basis run with CPU
decompositions. Maximum absolute changes were 6.846e-12 (u), 4.290e-12 (v),
3.035e-13 (streamfunction), and 1.262e-6 (vorticity). The vorticity maximum occurs
at the inlet/bottom corner at t=0.5; its global relative L2 difference is
5.108e-10. This is not bitwise equivalence across CPU/GPU factorizations.
Rational rank 1,091, trace rank 184, and all 2,059 retained volume modes agree.
Independent wall speed is 3.961e-11, outer Dirichlet error 1.798e-10, and relative
flux error 4.054e-12. No physical accuracy threshold was relaxed.

Validation: **35 tests passed** in two runs: 34 decomposition/GPU-basis/GPU-stepper/
hybrid/immersed-flow regressions in 192.24 s, plus the sampled-wall GPU SVD integration
test in 21.74 s. Tests cover tall/wide and full/thin SVD reconstruction, orthogonality,
near-null rank cutoffs, eigen-residuals, mode cutoffs, device residency under transfer
guards, and errors for non-finite factors. The sampled-wall test blocks large SciPy
SVD/eigh calls and retains the existing wall-accuracy threshold. Existing hybrid
manufactured-solution and boundary criteria pass with both MPFR and FP64 bases.

Evidence is under `build/immersed_flow/profile/gpu_decompositions/`:
`full/{summary.json,fields.npz}`, `comparison.json`, and `decomposition.json`.

## Opt-in FP64 GPU basis evaluation

`--basis-precision float64` explicitly replaces 113-bit MPFR basis evaluation
with GPU FP64, including cancellation, normalization transforms, and subsequent
field reconstruction. The default remains `mpfr`. The option requires
`--backend gpu`; it does not use the MPFR worker pool or fall back to MPFR.
Library usage is `ImmersedFlowPlan(..., assembly_device=device,
basis_precision="float64")`.

```bash
python examples/pde/immersed_channel_flow.py --backend gpu \
  --wall-method rational --basis-precision float64 \
  --out build/immersed_flow/fp64
```

With the same A100, resolution, time step, and 41 frames through t=20:

| Phase | MPFR basis | FP64 GPU basis |
|---|---:|---:|
| Cold setup | 35.697 s | 29.124 s |
| Evolution and diagnostics | 6.294 s | 6.312 s |
| Reconstruction | 14.841 s | 10.142 s |
| Phase sum | 56.831 s | 45.578 s |

FP64 reduces setup by 18.4% and the phase sum by 19.8% in these individual runs.
An initial implementation took 47.594 s to set up because varying point counts
triggered repeated compilation. The final implementation uses a compact loop
for the spline recurrence and fixed 256-point batches to reuse compiled kernels.
This removes that compilation penalty without introducing a persistent cache.

This is a precision tradeoff, unlike the preceding MPFR-preserving optimizations.
Across all frames, maximum absolute differences from the latest MPFR run are
1.396e-9 (u), 1.643e-9 (v), 6.152e-7 (vorticity), and 7.671e-11 (streamfunction).
Relative L2 differences are respectively 1.006e-10, 9.218e-10, 3.144e-9, and
1.430e-11. Independent wall speed is 4.000e-11, versus 9.273e-13 with MPFR;
outer Dirichlet error is 1.798e-10 and relative flux error is 4.120e-12.
Accuracy on other grids and more sensitive enrichments is not implied by this run.

Validation: **34 tests passed** in two runs: four FP64 evaluator/selector tests
in 14.40 s, then 30 existing/extended flow tests in 206.29 s. The hybrid fixture
now includes FP64 GPU assembly and retains the original manufactured-solution,
boundary/flux, and batched reconstruction thresholds. The evaluator tests verify
GPU float64 execution under a transfer guard, agreement with MPFR on well-conditioned
inputs, and no MPFR fallback during FP64 plan construction or reconstruction.
The original MPFR stream and immersed-flow regressions also passed.

Evidence: `build/immersed_flow/profile/fp64_basis/batched/{summary.json,fields.npz}`
and `build/immersed_flow/profile/fp64_basis/comparison.json`. The intermediate
compilation-heavy implementation is retained in `fp64_basis/full/` for comparison.

## Latest optimization retaining MPFR


Fresh-process setup is now **35.697 s**, down from **44.061 s**: an additional
**19.0% reduction** (1.23× faster). Setup plus evolution/output is **56.831 s**,
down from 65.721 s. These runs use the same A100, four MPFR workers, single-threaded
BLAS, 73×33 grid, 113-bit basis evaluation, float64 operators, Re=20, dt=0.02,
and all 41 snapshots through t=20. No setup cache is required for this gain.

| Phase | Previous | Latest |
|---|---:|---:|
| Setup | 44.061 s | 35.697 s |
| Evolution and diagnostics | 6.356 s | 6.294 s |
| Output reconstruction | 15.304 s | 14.841 s |
| Phase sum | 65.721 s | 56.831 s |

The next profile exposed a 4.67 s host sponge product, roughly 1.05 s of host
reduced mass/stiffness products, and 1.91 s of repeated lift geometry evaluation.
The implementation now:

* Keeps normalized quadrature operators on the GPU while assembling reduced
  mass, stiffness, and sponge matrices, before returning the host-plan results.
* Evaluates rational modes and the fixed lift together at volume/outlet points,
  sharing rational basis rows and their derivatives.
* Computes only values for the first line-normalization QR. Previously that
  pass also evaluated unused derivatives and nodal traces. Sensitive transformed
  derivatives are still evaluated at 113 bits afterward.

The intermediate dense-product/geometry changes measured 36.626 s in the
component benchmark; eliminating unused basis work gives the final full-run
setup above. The component benchmark retained 3.945 ms per warm GPU step.
The value-only path produces bit-for-bit identical basis values in the even/odd,
near-node, enriched/transformed precision regression.

**25 tests passed in 184.10 s**, using the five-file GPU regression command in
the following section. Independent energy-form tests now include reduced mass,
stiffness, and sponge quadratic forms. Host/GPU manufactured-solution, wall,
flux, streamfunction, parallel basis, and transfer-guard tests also passed.
Tests finished before the final timed full run.

Across all 41 frames, differences from the previous implementation are:

| Physical field | Maximum absolute difference | Relative L2 difference |
|---|---:|---:|
| u | 3.76e-13 | 1.46e-14 |
| v | 3.56e-13 | 1.03e-13 |
| Vorticity | 9.57e-11 | 4.18e-13 |
| Streamfunction | 3.74e-14 | 3.96e-15 |

Independent wall speed is 9.273e-13, outer Dirichlet error 1.797e-10, and
relative flux error 4.124e-12. Numerical thresholds were unchanged.
Evidence is saved under `build/immersed_flow/profile/setup_optimization_2/`:
`components/timings.json`, `full/{summary.json,fields.npz}`, and `comparison.json`.

Remaining setup includes MPFR basis evaluation, rational boundary SVD, volume
normalization, and first-use JAX compilation. This is a dense globally coupled
Galerkin discretization: 2,059 retained modes at 17,881 volume quadrature points,
with five roughly 281 MiB float64 quadrature operators. GPU-resident time stepping
is unchanged; geometry setup still contains explicit host stages. Phase sums
exclude independent post-run checks, archive compression, and PNG rendering.

## Previous setup optimization


The preceding cold setup was **44.1 s**, versus **104.5 s** after the first round
below: **2.37× faster setup**. The full phase sum is now **65.7 s** rather than
126.8 s, with the same Re, resolution, precision, dt, and all 41 output frames.

| Setup variant | Isolated benchmark |
|---|---:|
| Previous optimized setup | 103.553 s |
| Diagonal scaling, GPU volume products, MPFR constant reuse; one worker | 65.487 s |
| Same plus four MPFR workers | 44.233 s |

The four workers use four of the 18 allocated CPU cores, with one BLAS thread
per process. `--basis-workers 1` disables multiprocessing; the example and
component benchmark default to four. The library plan defaults to serial CPU
setup, with explicit opt-in:

```python
plan = ImmersedFlowPlan(
    nx=73, ny=33, wall_method="rational",
    assembly_device=jax.devices("gpu")[0], basis_workers=4,
)
```

Use a spawn-safe `if __name__ == "__main__":` entry point when using multiple
workers in a standalone script. Pool lifetime is bounded to plan construction;
workers are closed even if construction fails. Postprocessing does not retain
or reuse a closed executor.

The profile attributed 11.29 s to multiplying by a diagonal constraint matrix,
11.78 s to Gram products, 11.29 s to dense basis transforms, and 39.70 s to MPFR
basis evaluation (profiling adds overhead). The changes:

* Replace diagonal matrix products with column scaling where mathematically
  identical; retain full matrix products for the SVD constraint case.
* Upload raw quadrature operators once for GPU Gram and normalization products;
  return completed operators to the host plan for compatibility. Time stepping
  continues to use its existing GPU-resident runtime.
* Reuse MPFR sine/cosine values, node fractions, and derivative scale constants.
  Precision remains 113 bits, including the cancellation-sensitive transforms.
* Evaluate independent MPFR point rows in a bounded spawn pool. Serial and
  four-worker benchmark fields are bit-for-bit identical.

GPU dense products change reduction order in the ill-conditioned volume space.
Across the full t=20 run, maximum changes versus the previous CPU assembly are
1.58e-11 (u), 1.44e-11 (v), 5.48e-13 (streamfunction), and 4.19e-9 (vorticity).
Vorticity relative L2 difference is 2.19e-11. A strict initial 1e-9 absolute
vorticity comparison was exceeded; no discretization or manufactured-solution
accuracy criterion was relaxed. Independent wall speed remains 9.273e-13,
outer Dirichlet error 1.797e-10, and relative flux error 4.124e-12. Reduced
coefficient vectors are not compared directly across differently normalized
bases; comparisons use physical fields and independent PDE tests.

Evidence: `build/immersed_flow/profile/setup_optimization/before.prof`,
`after/timings.json` (one worker), `parallel/timings.json` (four workers), and
`full/{summary.json,fields.npz}` under that same directory. The full validation
measured 44.061 s setup and 21.660 s evolution/output. Warm stepping is unchanged
at 3.943 ms/step.

Latest regression validation: **25 passed in 195.51 s** on the GPU, covering
GPU-resident stepping, independent assembly energy forms, exact serial/parallel
basis agreement, host/GPU hybrid manufactured solutions, and existing stream
and immersed-flow cases. With the CUDA preload and environment below configured:

```bash
python -m pytest -q jax/tests/test_immersed_flow_gpu.py \
  jax/tests/test_parallel_basis.py jax/tests/test_hybrid_flow.py \
  jax/tests/test_stream_navier_stokes.py jax/tests/test_immersed_flow.py
```

The following sections retain the first-round measurements for comparison.

## Measurements

| Component | Initial GPU workflow | Optimized workflow |
|---|---:|---:|
| Full-case setup | 183.501 s | 104.482 s |
| Full-case evolution plus output reconstruction | 500.905 s | 22.342 s |
| Sum of these phases | 684.406 s | 126.824 s |
| Basis-line setup (isolated component benchmark) | 94.657 s | 18.653 s |
| Upload plus GPU factorization | 1.456 s | 1.434 s |
| First GPU step, including compilation | 1.003 s | 1.021 s |
| Warm GPU step (100 synchronized steps) | 3.950 ms | 3.942 ms |
| Warm GPU diagnostic reduction | 2.074 ms | 2.069 ms |
| One warm-grid snapshot | 12.185 s | 7.721 s |
| 41 snapshots together, warm grid | not available | 9.995 s |

The full-case speedup is **5.4×**, and evolution/output is **22.4×** faster.
The reported phase sum excludes independent post-run checks, archive compression,
and the separate Matplotlib PNG renderer. In the final full run, evolution plus
diagnostics took 6.565 s and reconstruction took 15.777 s (including first-use
high-precision output-grid evaluation). The final validation run overlapped with
regression tests; the isolated component timings are the cleaner kernel comparison.
These measurements are individual runs, not statistical performance guarantees.

A final isolated repeat (`profile/final_components/timings.json`) measured
103.553 s setup, 18.096 s basis setup, 0.449 s matrix/state upload, 0.562 s GPU
factorization including its initial compilation, 1.010 s first-step compilation
and execution, 3.946 ms/warm step, and 2.074 ms/diagnostic. Total runtime-object
construction was 1.505 s; the difference from upload+factorization includes
scalar placement and factor validation. Forty-one warm-grid snapshots took
10.293 s. Separate rendering of both PNGs took 2.972 s.

The final GPU regression command below passed **25 tests**.

## Changes supported by the profiles

* `RationalStokesExtension.stream_rows` evaluated the same rational basis twice
  per point batch. The reconstruction profile attributed 7.112 s per frame to
  those evaluations. Streamfunction and velocity rows now share the values.
* Output reconstruction was repeated independently for all 41 times. `grid_many`
  evaluates fixed spatial rows once per snapshot batch and applies the existing
  multiple-RHS rational evaluator. The default batch of 64 bounds temporary memory;
  callers can choose smaller batches. All original frames are retained.
* A 73-node setup profile spent 55.255 s compiling 539 separate JAX kernels.
  `_line_projector` now compiles the constrained spline/jet construction together.
  Streamfunction setup uses this projector directly rather than constructing an
  unused pressure eigensystem. GPU QR/Cholesky remain on the GPU.

The GPU runtime owns all matrices and state used in evolution. Both midpoint
stages, nonlinear volume loads, outlet backflow, solves, and diagnostic reductions
stay on the selected device. Host geometry/MPFR/rational assembly and output
reconstruction remain explicit setup/output stages. No precision, resolution,
timestep, or boundary formulation was reduced to obtain the speedup.

## Accuracy evidence

The complete t=20 optimized run has an exactly identical reduced final state to
the original GPU baseline. Maximum absolute difference over all saved fields is
1.125e-11 (batched versus individual matrix products). Independent final checks:

* obstacle wall maximum speed: 9.273e-13;
* outer Dirichlet maximum velocity error: 1.797e-10;
* maximum relative flux error across independent stations: 4.123e-12.

`test_immersed_flow_gpu.py` compares five GPU steps and diagnostics against SciPy,
asserts GPU placement, and disallows implicit transfers while executing.
`test_hybrid_flow.py` checks manufactured time convergence, physical boundaries,
flux, and batched reconstruction with batch sizes 1, 2, and 64. Pressure and
streamfunction regressions cover the shared compiled projector.

## Reproduction

The local environment combines pip cuBLAS 13.4.1.1 and system CUDA 13.0.1 cuSolver.
Preloading the system cuBLAS resolves the independently reproduced GPU QR failure;
this is a local environment requirement, not a generic CUDA configuration.
Install the `rational-flow` extra; for this session `gmpy2` was installed only into
`/tmp/pybspf-gpu-deps` to avoid modifying the shared environment.

```bash
export LD_PRELOAD=/mpcdf/soft/SLE_15/packages/x86_64/cuda/13.0.1/lib64/libcublas.so.13
export JAX_PLATFORM_NAME=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=jax/src:/tmp/pybspf-gpu-deps
python examples/pde/profile_immersed_flow.py --out build/immersed_flow/profile/repeat
python examples/pde/immersed_channel_flow.py --backend gpu --wall-method rational \
  --out build/immersed_flow/profile/repeat_full
python -m pytest -q jax/tests/test_immersed_flow_gpu.py jax/tests/test_hybrid_flow.py \
  jax/tests/test_pressure.py jax/tests/test_stream_navier_stokes.py
MPLCONFIGDIR=/tmp/pybspf-mpl python examples/pde/render_immersed_flow.py \
  --out build/immersed_flow/profile/repeat_full
```

Local evidence: `build/immersed_flow/profile/baseline/{timings.json,grid.prof,line_setup.prof}`,
`build/immersed_flow/profile/optimized/{timings.json,fields.npz}`,
`build/immersed_flow/hybrid/channel/{summary.json,fields.npz}` (original full run),
and `build/immersed_flow/profile/final_full/{summary.json,fields.npz}` (final full run).
