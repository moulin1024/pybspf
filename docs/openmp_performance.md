# OpenMP scaling on the development CPU

CPU float64, JAX/jaxlib 0.10.0, installed OpenMP OpenBLAS 0.3.32. Each thread count ran in a fresh process, sequentially rather than concurrently with other benchmark processes. The reported BLAS thread count matched the requested OMP_NUM_THREADS in every case.

The BSPF workload is the shifted Taylor–Green field on a 65³ grid with three velocity components, degree 9, 18 basis functions, and compact Chebyshev endpoint blocks (12 modes, 16 samples). Timings are medians of 30 synchronized calls after compilation and three warmups; plan construction, compilation, reference generation, and correctness checks are excluded. A separate representative LU workload uses a well-conditioned 34-by-34 matrix and 12,675 right-hand sides.

| OpenMP threads | Single LU ms | Single-axis BSPF derivative ms | Full-gradient failed calls / 30 |
|---:|---:|---:|---:|
| 1 | 4.479 | 16.071 | 0 |
| 2 | 2.641 | 14.034 | 30 |
| 4 | 1.667 | 12.914 | 28 |
| 8 | 1.509 | 13.155 | 26 |

The isolated LU solve speeds up approximately 2.97x at eight threads. The best single-axis BSPF time occurs at four threads (1.24x speedup). All single-axis and isolated-LU calls pass correctness checks. Their maximum relative L2 errors were about 7.2e-14 and 2.8e-16 respectively.

The full outer-JIT 3D gradient takes 30.07 ms with one BLAS thread and passes all 30 checks. With 2, 4, and 8 BLAS threads it fails 30, 28, and 26 checks respectively, with relative errors reaching about 1e22. Those executions cannot be counted as useful speedups. Failure counts can vary because the failure is nondeterministic. The error threshold used here is relative L2 <= 1e-9.

These results support one BLAS thread for reliable concurrent multi-axis execution on this installation. The isolated-LU speedup shows useful internal OpenMP parallelism is possible, but it does not establish safe concurrency between independent BLAS calls. Most of the complete derivative cost is outside that small solve, limiting end-to-end scaling. Thread-count differences near a few percent should not be overinterpreted: these are local measurements, not a controlled cross-machine performance study.

See [the runtime diagnosis](3d_jit_diagnosis.md) for launch configuration and the independent SciPy reproduction. No library or global environment settings were changed by this benchmark.
