# Endpoint estimator performance

These measurements are the baseline **before compact endpoint-block storage**.
The current implementation stores `(2, q, boundary_points)` weights rather than
a dense `(2*q, N)` matrix, so the storage totals and timings below describe the
earlier implementation. At N=513 and q=8, current Chebyshev endpoint weights
use 2,048 bytes per axis (16-point windows), versus 65,664 bytes previously.
The remaining plan arrays are unchanged.

Measured on macOS arm64, CPU, JAX 0.10.0, float64.

The current smooth 2D differentiation field was evaluated with degree 9, 18 basis functions, constraint order 8, and spline regularization 1e-6. FD uses 9 boundary points. Chebyshev uses 12 modes, 16 boundary points, and local regularization 1e-12. Only the endpoint estimator changes.

## Warm differentiation and accuracy

Each time is the median of 50 synchronized compiled calls, alternating estimator order. Plan and input arrays are already resident. This measures a single x derivative over the full N-by-N field; it excludes plotting, reference generation and compilation.

| N | FD ms | Chebyshev ms | FD relative L2 | Chebyshev relative L2 |
|---:|---:|---:|---:|---:|
| 129 | 0.681 | 0.674 | 1.026e-09 | 6.785e-11 |
| 193 | 1.718 | 1.727 | 2.298e-11 | 9.172e-13 |
| 257 | 2.921 | 2.919 | 1.524e-12 | 8.178e-14 |
| 513 | 5.936 | 5.893 | 1.323e-13 | 1.221e-13 |

## Construction and compilation

Cold setup clears JAX caches before constructing the 2D plan. It includes primitive compilation, but excludes initial JAX process initialization. Warm setup is the median of three subsequent constructions. Compilation measures `jit(differentiate).lower(plan, field).compile()` separately from its first execution.

| N | FD cold setup s | Cheb cold setup s | FD warm setup ms | Cheb warm setup ms | FD compile s | Cheb compile s |
|---:|---:|---:|---:|---:|---:|---:|
| 129 | 6.422 | 7.043 | 97.81 | 107.59 | 0.105 | 0.107 |
| 193 | 6.395 | 6.698 | 104.60 | 101.10 | 0.114 | 0.104 |
| 257 | 6.312 | 6.584 | 101.37 | 104.90 | 0.108 | 0.105 |
| 513 | 6.236 | 6.619 | 101.52 | 107.88 | 0.108 | 0.101 |

The warm runtime difference is under 1.1% in this run, within measured variability. Both methods store the same-shaped endpoint map and execute the same application operations; QR is performed only during construction. Numerical plan-array storage is identical (1,514,672 bytes per 2D plan at N=513), excluding compiler executables, temporaries and process RSS.

At N=129, 193 and 257, Chebyshev improves relative L2 error by approximately 15x, 25x and 19x. At N=513 both are near 1e-13; that small difference is not evidence of a robust accuracy advantage. Results are CPU-only and from one benchmark run. Reuse plans and compiled kernels across time steps.

Raw timing samples are summarized with p10/p90 in the accompanying local JSON output. No implementation or example parameters were changed for this benchmark.
