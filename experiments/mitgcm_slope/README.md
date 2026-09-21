# MITgcm / BSPF ISW slope timing comparison

This experiment uses the supplied `examples/pde/isw_slope/reference/shared_initial.npz`.
The MITgcm checkout is `build/MITgcm-source`, upstream commit
`d861cd501f21303825de860eb3caa0a8a7ae22f8`.

The fluid domain has 321 nonuniform horizontal columns with the same x mapping,
one periodic transverse cell, and 161 vertical z levels over the 100 m maximum
depth. Two additional dry columns close the x boundaries. The 1800 m slope
profile, initial streamfunction/buoyancy, viscosity 0.01 m²/s and diffusivity
0.003 m²/s match the BSPF case. Temperature represents total buoyancy through a
linear EOS, B = g alpha T, with alpha = 0.001/K. There is no rotation or forcing.

`prepare.py` samples total buoyancy at tracer centers and computes layer-mean U
from streamfunction differences. Each face has zero net transport; MITgcm
initializes W from discrete continuity. Partial-cell factors used for this
conversion were checked against MITgcm's written hFacC (maximum difference
4.35e-13); there are 33,480 wet tracer cells. The source initial SHA256 matches
the BSPF timing run.

MITgcm uses nonhydrostatic pressure projection, AB2 with abEps=0.1, explicit
viscosity/diffusion, and centered second-order tracer advection. Both pressure
solvers request normalized residual 1e-11. The 3-D-named solver is the generic
nonhydrostatic solver; with Ny=1 it solves a two-dimensional x-z problem.
CG2D is the horizontal barotropic problem and cannot replace the vertical
nonhydrostatic pressure solve. CG3D is allowed 10,000 iterations; every timestep
is monitored and the benchmark requires both pressure solves to meet tolerance. Side/bottom walls are no-slip. The
local `mom_[uv]_rviscflux.F` overrides add the dissipative upper rigid-lid flux
(-nu u rkSign/drC times face area) to enforce no-slip there as in BSPF. No
upstream file is modified. Only this constant-viscosity, explicit-viscosity,
flux-form configuration is intended for these overrides.

The initial 2 s timestep produced nonfinite output and is **not** a valid timing
result. The stable-step trial uses dt=0.5 s (100 steps to 50 s). A first finite-field dt=0.5 trial hit the CG3D 1000-iteration cap and missed
pressure tolerance, so its 48.01 s wall time is also rejected. The accepted run
uses a raised cap and checks convergence at every step. Raw failed runs
are retained separately for diagnosis.

This is a **same nominal resolution** timing, not an equal-error or equal-DOF
comparison: BSPF uses a terrain-following spectral basis and RK4, whereas MITgcm
uses z-level partial cells and AB2/projection. MITgcm runs serially with GNU
Fortran 15.2, -O3, no MPI or OpenMP; BSPF uses the existing CPU JAX configuration.
Both wall timers include executable/Python startup, runtime initialization,
time integration and native output. MITgcm compilation and input conversion are
excluded; BSPF's per-process JIT is included in the earlier total-process time.

## Reproduce

Install the repository packages first (`python -m pip install -e . -e './jax[slope]'`),
or prefix the commands below with `PYTHONPATH=src:jax/src` from the repository root.
The canonical BSPF case is now [examples/pde/isw_slope](../../examples/pde/isw_slope/README.md).

From the repository root, obtain the source if needed:

```sh
git clone --depth 1 --filter=blob:none --sparse https://github.com/MITgcm/MITgcm.git build/MITgcm-source
git -C build/MITgcm-source sparse-checkout set model eesupp pkg tools
mkdir -p build/mitgcm-slope-321x161/compile
cd build/mitgcm-slope-321x161/compile
../../MITgcm-source/tools/genmake2 -rootdir ../../MITgcm-source -mods ../../../experiments/mitgcm_slope/code -of ../../../experiments/mitgcm_slope/build_options
make depend
make -j4
cd ../../..
python experiments/mitgcm_slope/prepare.py --out build/mitgcm-slope-321x161/new-run --dt 0.5
python experiments/mitgcm_slope/benchmark.py --executable build/mitgcm-slope-321x161/compile/mitgcmuv --run build/mitgcm-slope-321x161/new-run --source build/MITgcm-source
```

The benchmark checks both pressure residuals at every step, normal termination,
and finite final U/V/W/T output rather
than trusting Fortran's exit code alone. `wall_time.json` stores configuration,
source/executable identity, elapsed time, field statistics and pressure logs.
The initial compiler/dependency timing logs are in the compile directory.

Reference: [official MITgcm configuration and initialization documentation](https://mitgcm.readthedocs.io/en/latest/getting_started/getting_started.html).

## Accepted measurement

50 simulated seconds, same machine, one complete process per timing:

| Solver | Nominal grid | dt (s) | Steps | Process wall time (s) |
| --- | --- | ---: | ---: | ---: |
| MITgcm serial, nonhydrostatic x-z | 321 × 161, Ny=1 | 0.5 | 100 | 69.337 |
| BSPF JAX CPU | 321 × 161 | 2.0 | 25 | 8.690 |

All 100 CG2D and 100 CG3D solves satisfy the requested normalized residual
1e-11; maximum reported residual is 9.98492e-12.
CG3D uses 1616–1719 iterations per step. Final U/V/W/T are finite, and the
transverse velocity magnitude is 4.328e-17 m/s,
consistent with the intended two-dimensional configuration. MITgcm final
max U is 0.278758 m/s, versus BSPF 0.278990 m/s; agreement of this single
observable is not an accuracy certificate.

Accepted evidence: `build/mitgcm-slope-321x161/run-50s-converged/wall_time.json`.
Comparison: `build/mitgcm-slope-321x161/comparison.json`.
MITgcm printed model time was independently checked to reach 50 s.
The ratio 7.98 is specific to these settings and
includes the different stable timesteps and algorithms; it is not an intrinsic
solver speedup or equal-accuracy benchmark.

## dt = 1 s trial

With the executable, grid, initial state and pressure settings unchanged,
`run-50s-dt1` attempted 50 steps. Velocity extrema began visibly growing and
alternating around 21–22 simulated seconds (max U = 0.781 m/s at 22 s and
2.307 m/s at 24 s). MITgcm stopped at step 35 / 35 s due to extreme temperature.
The failed process took 24.125 s; this is not a completed 50 s benchmark.
A 1 s timestep is unstable for this explicit-viscosity AB2 configuration.
The accepted 0.5 s result above remains the valid timing.

Evidence: `build/mitgcm-slope-321x161/run-50s-dt1/wall_time.json` and `STDERR.0000`.

## dt = 0.75 s trial

Because 0.75 does not divide 50, this trial used 67 steps to 50.25 s. Wall time
was 46.789 s. Fields remained finite and all pressure solves reached tolerance,
but the last steps show growing alternating U minima: -0.371 m/s at 47.25 s,
-0.532 at 48.75 s and -0.763 at 50.25 s, alternating with approximately
-0.171 m/s. This is evidence of temporal instability despite normal termination;
the trial is not accepted as a stable benchmark. The dt=0.5 s reference has
Umin=-0.170596 m/s at 50 s. Evidence is in `run-50p25s-dt075/wall_time.json`.

## Grid visualization

Run `python experiments/mitgcm_slope/plot_grids.py` to generate PNG/SVG in
`build/mitgcm-slope-321x161/figures`. BSPF construction coordinates come from the
actual basis convention and physical mapping; MITgcm edges and wet fractions
are read from the actual XG/RF/hFacC outputs. Full-domain panels show every eighth
construction/grid line; slope close-ups show all lines. The graphic distinguishes
BSPF construction points (321×161) from its Gauss quadrature (642×322) and from
MITgcm wet finite-volume cells (33,480).

## Eight-worker CPU runs

MITgcm uses OpenMP (`-omp=-fopenmp`), `nTx=8`, `nTy=1`, and a local
`MAX_NO_THREADS=8` header. `code_omp8/SIZE.h` has eight x tiles of width 41:
328 allocated columns = 321 unchanged wet columns + 7 dry wall columns.
Wet-column initial U and temperature are byte-identical to the serial inputs.
Common physics/configuration overrides are symlinked from `code`, not copied.

```sh
mkdir -p build/mitgcm-slope-omp8/compile
cd build/mitgcm-slope-omp8/compile
../../MITgcm-source/tools/genmake2 -rootdir ../../MITgcm-source -mods ../../../experiments/mitgcm_slope/code_omp8 -of ../../../experiments/mitgcm_slope/build_options -omp=-fopenmp
make depend
make -j4
cd ../../..
python experiments/mitgcm_slope/prepare.py --out build/mitgcm-slope-omp8/new-run --dt 0.5 --threads 8
python experiments/mitgcm_slope/benchmark.py --executable build/mitgcm-slope-omp8/compile/mitgcmuv --run build/mitgcm-slope-omp8/new-run --source build/MITgcm-source
python experiments/mitgcm_slope/run_bspf_threads.py --threads 8 --out build/bspf-slope-threads8-new
```

The MITgcm launcher sets OMP_NUM_THREADS=8, OMP_DYNAMIC=FALSE,
OMP_MAX_ACTIVE_LEVELS=1, OMP_STACKSIZE=400M, and validates `nThreads=8` in model
output. The JAX launcher sets PJRT_NPROC=8 and enables multithreaded Eigen;
BLAS/OMP/MKL/Accelerate thread limits are also set to 8 before imports. It keeps
one JAX CPU device; creating eight fake devices is not a threading control.

The installed JAX 0.10.0 library contains PJRT_NPROC support. In an isolated
probe with BLAS restricted to one thread, PJRT_NPROC=1 yielded 11 total OS
threads after CPU-client creation, whereas PJRT_NPROC=8 yielded 25, consistent
with two XLA pools growing from 1 to 8 workers. Total OS threads include support
threads and need not equal the compute worker count. The launcher records its
own CPU-client initialization too. macOS schedules workers onto CPUs; these
settings do not pin eight physical cores or guarantee full utilization.

`process_timing.py` records wall and child-process CPU time. External per-thread
sampling can be denied by the macOS sandbox; this is recorded as AccessDenied,
not treated as evidence of a one-thread run. CPU time / wall time reports mean
CPU use, not requested thread count. The two solver timings run sequentially.

Implementation references: [MITgcm threading configuration](https://mitgcm.readthedocs.io/en/latest/getting_started/getting_started.html#parallel-execution)
and [XLA DefaultThreadPoolSize](https://github.com/openxla/xla/blob/main/xla/pjrt/utils.cc)
/ [CPU thread pools](https://github.com/openxla/xla/blob/main/xla/pjrt/cpu/cpu_client.cc).
