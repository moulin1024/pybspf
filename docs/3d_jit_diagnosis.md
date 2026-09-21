# CPU BLAS concurrency failure: diagnosis and verified configuration

The failure was isolated below JAX to the installed **OpenMP OpenBLAS** build.
SciPy alone reproduces it on well-conditioned systems. Launching a fresh
process with `OMP_NUM_THREADS=1` eliminates the observed errors while retaining
outer JIT. The 3D notebook has been restored to JIT-compiled vector operators.

Update: a [corrected project-local OpenBLAS build](corrected_blas_build.md)
now passes the concurrent native BLAS and JAX/SciPy solve probes at four and
eight OpenMP threads. The original conda build remains unchanged; the
single-thread workaround below still applies when not using the local launcher.

## Runtime evidence

- macOS arm64, Python 3.13, JAX/jaxlib 0.10.0, SciPy 1.17.1.
- SciPy's BLAS/LAPACK libraries resolve to `libopenblas.0.dylib`.
- The library reports `OpenBLAS 0.3.32 NO_AFFINITY USE_OPENMP VORTEX MAX_THREADS=128`.
- It reports ten BLAS threads by default and one with `OMP_NUM_THREADS=1`.
- JAX's CPU triangular-solve lowering calls BLAS TRSM through its LAPACK FFI.

A standalone probe used three 34-by-34 matrices with condition numbers between
1.46 and 1.49, known solutions, and 6,027/4,851/4,059 right-hand sides. JAX
executed the three LU solves in one JIT call (30 repetitions). SciPy executed
three solves in a thread pool (15 repetitions). A run fails if maximum absolute
solution error exceeds 1e-10 or is nonfinite.

| Fresh-process configuration | BLAS threads | JAX failed repetitions | SciPy failed repetitions |
|---|---:|---:|---:|
| Default | 10 | 22 / 30 | 14 / 15 |
| `OPENBLAS_NUM_THREADS=1` only | 10 | 17 / 30 | 15 / 15 |
| `OMP_NUM_THREADS=1` only | 1 | 0 / 30 | 0 / 15 |

The successful configuration had maximum absolute error 4.0e-15 in both
implementations. Thus neither BSPF conditioning nor JAX outer JIT is necessary
to trigger the failure. This evidence identifies a concurrency problem in this
installed BLAS stack, not a universal defect in all OpenBLAS builds.

## Supported launch configuration

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m jupyterlab
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests packages/models/tests packages/sim/tests
```

Start a fresh kernel/process. These variables are read during numerical-library
initialization; changing them later in a notebook is not a reliable fix.
`OMP_NUM_THREADS` controls OpenMP builds; `OPENBLAS_NUM_THREADS` covers pthread
builds. This matches the [OpenBLAS runtime documentation](https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/).
The OpenMP setting may also limit other OpenMP code in the process. JAX outer
JIT and its independent task scheduling remain enabled.

The package must not silently change process-wide thread settings on import.
The notebook test runner supplies them before launching Python, and
`tests/test_cpu_solve_runtime.py` repeats the independent JAX/SciPy probe
in a fresh, explicitly configured CPU process.

For a long-term environment repair, use a validated BLAS build suitable for
concurrent callers and rerun this probe before permitting multithreaded BLAS.
A clean project environment is preferable to changing the shared conda base.
No dependencies were replaced and no global environment settings were changed
in this investigation. Further binary inspection identifies an unsafe shared
scratch-buffer reservation in this installed build, detailed below.

## Direct BLAS reproduction and internal race

The standalone [TRSM probe](diagnostics/openblas_trsm_probe.py) calls the installed
OpenBLAS `dtrsm_` symbol through `ctypes.CDLL`, which releases the Python GIL.
It uses three distinct 34-by-34 triangular matrices with diagonals near 40,
known solutions, and 6,027/4,851/4,059 RHS columns. Every call gets its own output
buffer. No JAX or SciPy solver is involved.

```sh
OMP_NUM_THREADS=4 python docs/diagnostics/openblas_trsm_probe.py /Users/moulin/miniforge3/lib/libopenblas.0.dylib
OMP_NUM_THREADS=1 python docs/diagnostics/openblas_trsm_probe.py /Users/moulin/miniforge3/lib/libopenblas.0.dylib
```

| OpenMP threads | Caller execution | Failed repetitions | Maximum absolute error |
|---|---|---:|---:|
| 4 | Sequential | 0 / 50 | 1.33e-15 |
| 4 | Three concurrent callers | 50 / 50 | 15.90 |
| 4 | Three callers, lock around BLAS call | 0 / 50 | 1.33e-15 |
| 1 | Three concurrent callers | 0 / 50 | 1.33e-15 |

All original factors and RHS arrays remained unchanged. Failure counts are
timing-dependent. An initial probe through SciPy's `blas.dtrsm` wrapper passed;
that wrapper result does not establish simultaneous native execution. Direct
CDLL calls make the concurrency explicit and reproduce the corruption.

The [OpenBLAS 0.3.32 OpenMP dispatcher source](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.32/driver/others/blas_server_omp.c#L398-L413)
reserves shared scratch-buffer sets using an atomic compare/exchange when
`HAVE_C11` is enabled, but a plain boolean check/set otherwise. Worker scratch
storage is indexed by buffer-set number and OpenMP thread number. Concurrent
teams must not reserve the same set.

Inspection with `nm` and `otool -tvV` of this installed binary found:

```text
_blas_thread_buffer  0x00c47770
_blas_buffer_inuse   0x00c47b70
_exec_blas:
  0x181030  mov  w8, #1
  0x181034  adrp x9, ...        ; page 0xc47000
  0x181038  strb w8, [x9,#0xb70]
  0x18103c  str  xzr, [sp,#0x20] ; buffer-set index = 0
```

There is no atomic reservation or even a load/check of the in-use byte on this
path. The dispatcher's release also uses a plain byte store. Its worker routine
loads scratch pointers from `_blas_thread_buffer` using the buffer-set and
thread indices. This is consistent with the non-C11 branch being optimized
without thread synchronization; simultaneous calls can overwrite the same
scratch workspace. The binary defect and the direct-call/lock controls strongly
identify this mechanism. The subsequent corrected build enables the atomic
branch and passes these concurrency probes; see the linked build report.

The 3D outer JIT exposes the bug by executing independent axis solves at once.
It is not inherently specific to three dimensions; any overlapping calls that
enter this dispatcher can be affected. Single-axis timing can therefore show
valid OpenMP speedup while the combined 3D result is wrong.

Keep the verified single-thread BLAS launch configuration when using the original
library, or use the corrected local build. A Python lock is a
diagnostic control, not a lock around native calls inside a compiled JAX graph.

## Original BSPF reproduction and controls

The field is sampled on a 33-by-41-by-49 grid over [0, 3*pi/2]^3 with degree 9,
18 basis functions, eight endpoint constraints per side, and the 12-mode,
16-point Chebyshev endpoint estimator. Each KKT factor is 34-by-34.

Observed controls:

- Individual x/y/z derivatives agree with the analytic Jacobian, with relative
  L2 errors about 2.10e-10, 1.63e-11 and 3.34e-12, respectively. Individual
  derivatives also agree between eager execution and a single-axis JIT.
- The eager divergence maximum is about 5.93e-10.
- Repeated outer-JIT gradient calls on identical inputs ranged from small
  errors to errors above 1e22. Fused curl and Laplacian calls also failed.
- A JIT returning the three spline decompositions already exhibits changing
  coefficients, before any Fourier differentiation.
- A JIT containing only the three FFTs of fixed residual arrays is stable.
- A JIT containing only three `jax.scipy.linalg.lu_solve` calls with fixed LU
  factors and fixed RHS arrays reproduces the inconsistency. Sequential solves
  provide the reference. Endpoint extraction alone is stable under outer JIT.

The isolated solve test can be constructed from the notebook's `plan` and
`field` variables as follows:

```python
import jax.scipy.linalg as jl

rhs = []
for axis, p in enumerate(plan.axes):
    values = jnp.moveaxis(field, axis, 0)
    jets = bspf.endpoint_jets(plan, field, axis=axis)
    jets = jets.reshape((2*p.constraint_order,) + values.shape[1:])
    data = 2*jnp.tensordot(p.weighted_basis, values, axes=(1, 0))
    joined = jnp.concatenate((data, jets))
    rhs.append(joined.reshape(joined.shape[0], -1))
factors = tuple(p.lu for p in plan.axes)
reference = tuple(jl.lu_solve(lu, r) for lu, r in zip(factors, rhs))
solve_all = jax.jit(lambda factors, rhs:
    tuple(jl.lu_solve(lu, r) for lu, r in zip(factors, rhs)))
for _ in range(5):
    result = solve_all(factors, tuple(rhs))
    print([float(jnp.max(jnp.abs(a-b))) for a, b in zip(result, reference)])
```

With the verified launch configuration, repeated outer-JIT gradient, curl and
Laplacian calls returned identical observed errors (1.29781e-10, 1.21264e-10 and
2.56830e-9 respectively), matching eager execution. The divergence maximum was
5.93365e-10 in both modes. No solver rewrite or tolerance relaxation is needed.
