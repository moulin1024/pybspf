"""Archive the phase-aware 3D BSPF GPU-memory measurements."""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("build/box_memory/results.json")
    )
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args()
    result = json.loads(args.input.read_text())
    cases = result["cases"]
    largest = max(cases, key=lambda c: c["unknowns"])
    destination = args.docs / "data/box_memory"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.input, destination / "results.json")
    mib = lambda value: value / 2**20
    lines = [
        "# Measured GPU memory: 3D BSPF Poisson",
        "",
        f"Measured on 2026-09-19 on {result['device']}, driver {result['driver']}, JAX 0.10.0, FP64. Each size runs in a fresh process with GPU preallocation disabled and the default JAX BFC allocator.",
        "",
        "This profiles the actual BSPF assembly, factor setup, RHS upload, first solve, repeated solves and MMS validation from `benchmark_box_poisson.py`. It retains the assembly/evaluation objects as that benchmark does; it is not a minimal solver-only program.",
        "",
        "## Actual high-water measurements",
        "",
        "| Independent coefficients | Full-process sampled peak [MiB] | Warm-solve process peak [MiB] | JAX live-buffer peak [MiB] | JAX allocator pool peak [MiB] |",
        "|---:|---:|---:|---:|---:|",
    ]
    for c in cases:
        lines.append(
            f"| {c['n']}³ | {mib(c['process_peak_bytes']):.1f} | {mib(c['phases']['warm_solves']['process_peak_bytes']):.1f} | {mib(c['allocator_peak_bytes']):.1f} | {mib(c['allocator_pool_peak_bytes']):.1f} |"
        )
    lines += [
        "",
        "NVML measures device memory charged to the worker PID, including runtime/library memory and allocator-reserved memory. JAX `peak_bytes_in_use` is the allocator high-water counter for live tracked allocations across the workflow, including setup and first-use compilation/autotuning; it does not include every CUDA/driver allocation. `peak_pool_bytes` measures the retained BFC pool. These columns overlap and must not be added together.",
        "",
        "The observed baseline immediately after JAX GPU initialization was "
        + ", ".join(
            f"{mib(c['jax_context_baseline_bytes']):.1f} MiB ({c['n']}³)"
            if c["jax_context_baseline_bytes"] is not None
            else f"unavailable ({c['n']}³)"
            for c in cases
        )
        + ". The identical process peaks at these sizes reflect the substantial fixed runtime footprint and cached pool, not size-independent asymptotic memory usage.",
        "",
        "## Plan and compiled single-call buffers",
        "",
        "| Independent coefficients | Stored plan [MiB] | Arguments [MiB] | Output [MiB] | Compiler temporary buffers [MiB] | Compiler buffer total [MiB] |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cases:
        a = c["compiled_buffers"]
        lines.append(
            f"| {c['n']}³ | {mib(c['plan_bytes']):.4f} | {mib(a['argument_size_in_bytes']):.3f} | {mib(a['output_size_in_bytes']):.3f} | {mib(a['temp_size_in_bytes']):.3f} | {mib(a['total_buffer_bytes']):.3f} |"
        )
    lines += [
        "",
        "Compiler buffer totals are `arguments + output + temporaries - aliases` from `Compiled.memory_analysis()`. They describe one executable call, not measured process peaks. They exclude CUDA context/library overhead, previously retained assembly arrays, allocator caching, and any prior output still alive while dispatching a new solve. Compiler inspection runs after the measured workflow and is excluded from its peaks.",
        "",
        f"At {largest['n']}³, the {mib(largest['plan_bytes']):.3f} MiB figure describes only the plan. The measured full-process peak is {mib(largest['process_peak_bytes']):.1f} MiB; the live JAX allocator peak is {mib(largest['allocator_peak_bytes']):.1f} MiB. Thus neither plan storage nor RHS-plus-output storage is a complete execution-memory figure.",
        "",
        "## Sampling and validation",
        "",
        f"The parent process requests one NVML sample every {result['interval_requested_ms']:g} ms and monitors the worker PID. Each worker performs {result['warm_repetitions']} synchronized warm solves to expose steady repeated-solve behavior. Phase boundaries and allocator counters are recorded by the worker using a shared monotonic clock.",
        "",
        "| Coefficients | Actual mean interval [ms] | Largest sampling gap [ms] | Samples | Relative algebraic residual |",
        "|---:|---:|---:|---:|---:|",
    ]
    for c in cases:
        lines.append(
            f"| {c['n']}³ | {c['sampling_mean_ms']:.3f} | {c['sampling_max_ms']:.3f} | {c['sample_count']} | {c['relative_residual']:.3e} |"
        )
    lines += [
        "",
        "NVML values are sampled high-water observations, not an exact trace of every transient allocation. Short-lived peaks can fall between samples, especially during the reported scheduling gaps. The JAX allocator high-water counters supplement this sampling but do not cover external CUDA allocations. Raw per-phase maxima and allocator snapshots are in [results.json](data/box_memory/results.json); timestamped samples and worker events remain under `build/box_memory/`.",
        "",
        "All measured runs passed the unchanged independent MMS residual criterion (`2e-10`). A dedicated test verifies that the report excludes post-workflow analysis from its peaks. The production solver was not changed for profiling.",
        "",
        "```bash",
        "python -m pip install nvidia-ml-py",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src \\",
        "  python scratch/profile_box_memory.py",
        "python scratch/report_box_memory.py",
        "python -m pytest jax/tests/test_box_memory_profile.py -q",
        "```",
        "",
        "[3D solver API](jax_box_poisson.md) · [Accuracy and performance benchmark](jax_box_poisson_benchmark.md)",
        "",
    ]
    (args.docs / "jax_box_poisson_memory.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
