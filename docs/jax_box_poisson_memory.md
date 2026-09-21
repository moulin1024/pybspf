# Measured GPU memory: 3D BSPF Poisson

Measured on 2026-09-19 on NVIDIA A100-SXM4-40GB, driver 580.178.04, JAX 0.10.0, FP64. Each size runs in a fresh process with GPU preallocation disabled and the default JAX BFC allocator.

This profiles the actual BSPF assembly, factor setup, RHS upload, first solve, repeated solves and MMS validation from `benchmark_box_poisson.py`. It retains the assembly/evaluation objects as that benchmark does; it is not a minimal solver-only program.

## Actual high-water measurements

| Independent coefficients | Full-process sampled peak [MiB] | Warm-solve process peak [MiB] | JAX live-buffer peak [MiB] | JAX allocator pool peak [MiB] |
|---:|---:|---:|---:|---:|
| 31³ | 610.0 | 610.0 | 69.6 | 130.0 |
| 63³ | 610.0 | 610.0 | 73.1 | 130.0 |
| 127³ | 610.0 | 610.0 | 114.6 | 130.0 |

NVML measures device memory charged to the worker PID, including runtime/library memory and allocator-reserved memory. JAX `peak_bytes_in_use` is the allocator high-water counter for live tracked allocations across the workflow, including setup and first-use compilation/autotuning; it does not include every CUDA/driver allocation. `peak_pool_bytes` measures the retained BFC pool. These columns overlap and must not be added together.

The observed baseline immediately after JAX GPU initialization was 424.0 MiB (31³), 424.0 MiB (63³), 424.0 MiB (127³). The identical process peaks at these sizes reflect the substantial fixed runtime footprint and cached pool, not size-independent asymptotic memory usage.

## Plan and compiled single-call buffers

| Independent coefficients | Stored plan [MiB] | Arguments [MiB] | Output [MiB] | Compiler temporary buffers [MiB] | Compiler buffer total [MiB] |
|---:|---:|---:|---:|---:|---:|
| 31³ | 0.0227 | 0.250 | 0.227 | 4.456 | 4.933 |
| 63³ | 0.0923 | 2.000 | 1.908 | 7.817 | 11.725 |
| 127³ | 0.3721 | 16.000 | 15.628 | 31.257 | 62.885 |

Compiler buffer totals are `arguments + output + temporaries - aliases` from `Compiled.memory_analysis()`. They describe one executable call, not measured process peaks. They exclude CUDA context/library overhead, previously retained assembly arrays, allocator caching, and any prior output still alive while dispatching a new solve. Compiler inspection runs after the measured workflow and is excluded from its peaks.

At 127³, the 0.372 MiB figure describes only the plan. The measured full-process peak is 610.0 MiB; the live JAX allocator peak is 114.6 MiB. Thus neither plan storage nor RHS-plus-output storage is a complete execution-memory figure.

## Sampling and validation

The parent process requests one NVML sample every 2 ms and monitors the worker PID. Each worker performs 2000 synchronized warm solves to expose steady repeated-solve behavior. Phase boundaries and allocator counters are recorded by the worker using a shared monotonic clock.

| Coefficients | Actual mean interval [ms] | Largest sampling gap [ms] | Samples | Relative algebraic residual |
|---:|---:|---:|---:|---:|
| 31³ | 2.327 | 92.997 | 8586 | 1.623e-13 |
| 63³ | 2.251 | 98.647 | 8914 | 8.633e-13 |
| 127³ | 2.302 | 114.047 | 9188 | 4.433e-12 |

NVML values are sampled high-water observations, not an exact trace of every transient allocation. Short-lived peaks can fall between samples, especially during the reported scheduling gaps. The JAX allocator high-water counters supplement this sampling but do not cover external CUDA allocations. Raw per-phase maxima and allocator snapshots are in [results.json](data/box_memory/results.json); timestamped samples and worker events remain under `build/box_memory/`.

All measured runs passed the unchanged independent MMS residual criterion (`2e-10`). A dedicated test verifies that the report excludes post-workflow analysis from its peaks. The production solver was not changed for profiling.

```bash
python -m pip install nvidia-ml-py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src \
  python scratch/profile_box_memory.py
python scratch/report_box_memory.py
python -m pytest jax/tests/test_box_memory_profile.py -q
```

[3D solver API](jax_box_poisson.md) · [Accuracy and performance benchmark](jax_box_poisson_benchmark.md)
