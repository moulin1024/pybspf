# Shared 3D GPU box solver: continuous MMS benchmark

Measured on 2026-09-19 using JAX 0.10.0, FP64 and NVIDIA A100-SXM4-40GB.

The manufactured problem, boundary conditions, discretizations and public API are described in [the 3D solver documentation](jax_box_poisson.md). These are independent coefficient counts: `n³` unknowns on `(n+2)³` sampling nodes for this particular nodal formulation. The L2 error is evaluated against the continuous exact solution on an independent 257³ Gauss grid.

Each case runs in a fresh process. Warm timings are medians of 9 synchronized solves under a transfer guard, reusing the fixed plan and GPU-resident RHS. They exclude setup, uploads and error evaluation. BSPF and FD are two discretizations using the same production tensor solver; this table does not compare against a different linear solver.

| Discretization | Coefficient shape | Unknowns | Warm solve [ms] | Relative L2 error | Relative algebraic residual |
|---|---:|---:|---:|---:|---:|
| BSPF | 15³ | 3,375 | 0.113 | 1.191e-04 | 6.275e-14 |
| BSPF | 31³ | 29,791 | 0.116 | 2.548e-06 | 1.623e-13 |
| BSPF | 63³ | 250,047 | 0.152 | 1.980e-08 | 8.633e-13 |
| BSPF | 127³ | 2,048,383 | 0.525 | 5.551e-10 | 4.433e-12 |
| FD | 31³ | 29,791 | 0.119 | 3.397e-03 | 9.181e-14 |
| FD | 63³ | 250,047 | 0.156 | 8.483e-04 | 3.321e-13 |
| FD | 127³ | 2,048,383 | 0.526 | 2.129e-04 | 1.768e-12 |
| FD | 255³ | 16,581,375 | 4.656 | 5.447e-05 | 1.093e-11 |

All cases passed the independent relative residual limit of `2e-10`, recomputed from the original weak mass/stiffness matrices or seven-point FD stencil. No tolerance relaxation or residual refinement was applied.

| Discretization | Coefficient shape | Cold preparation [s] | Factor setup [s] | First solve [ms] | Plan storage [MiB] |
|---|---:|---:|---:|---:|---:|
| BSPF | 15³ | 14.838 | 1.462 | 152.204 | 0.0055 |
| BSPF | 31³ | 15.185 | 1.660 | 233.901 | 0.0227 |
| BSPF | 63³ | 15.415 | 1.668 | 263.686 | 0.0923 |
| BSPF | 127³ | 15.347 | 1.639 | 375.458 | 0.3721 |
| FD | 31³ | 0.006 | 1.696 | 272.368 | 0.0227 |
| FD | 63³ | 0.015 | 1.649 | 290.642 | 0.0923 |
| FD | 127³ | 0.071 | 1.680 | 373.378 | 0.3721 |
| FD | 255³ | 0.721 | 1.856 | 293.792 | 1.4941 |

Cold preparation includes axis/load construction and error-evaluation tables; BSPF also incurs first-use JAX compilation. Factor setup is measured separately from RHS upload. The first solve includes solve compilation. Plan storage counts only axis rotations, spectra and shift, not input/output volumes or temporary GPU workspace. The denominator is not retained as a full volume in the plan.

The implementation applies to positive-definite separable operators on boxes. Different geometry or nonseparable coefficients need their own operator treatment. CPU/GPU independent-matrix, non-cubic, batching, autodiff, transfer-guard and existing PDE regressions passed: **46 tests**.

[Raw timings, errors and storage measurements](data/box_poisson/results.json)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src \
  python scratch/benchmark_box_poisson.py
python scratch/report_box_poisson.py
```
