"""Per-process NVML and JAX allocator peak memory for the 3D BSPF MMS.

Requires nvidia-ml-py in the monitoring interpreter. Production numerical
kernels and the benchmark data/validation remain unchanged.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from bisect import bisect_right
from pathlib import Path
from time import monotonic_ns, sleep
from types import SimpleNamespace

ACTIVE_PHASES = (
    "initialization",
    "preparation",
    "factor_setup",
    "rhs_upload",
    "first_solve",
    "warm_solves",
    "validation",
)


def worker(args):
    events_path = args.out / f"{args.n}_events.jsonl"
    memory_path = args.out / f"{args.n}_jax.json"
    saved = {}

    def event(phase, device=None, **data):
        saved.update(data)
        stats = device.memory_stats() if device is not None else None
        record = {"time_ns": monotonic_ns(), "phase": phase, "allocator": stats}
        with events_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    event("initialization")
    import benchmark_box_poisson

    benchmark_box_poisson.worker(
        SimpleNamespace(
            n=args.n,
            discretization="bspf",
            repeats=args.repeats,
            result=args.out / f"{args.n}_accuracy.json",
        ),
        observer=event,
    )
    # Inspect compiler-assigned buffers only after the real workflow completes,
    # so this extra analysis cannot inflate its reported process-memory peak.
    from bspf_jax import solve_box_poisson

    import jax

    device = jax.devices("gpu")[0]
    event("analysis", device)
    compiled = solve_box_poisson.lower(saved["plan"], saved["load"]).compile()
    stats = compiled.memory_analysis()
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
        "generated_code_size_in_bytes",
    )
    analysis = {name: getattr(stats, name) for name in names}
    analysis["total_buffer_bytes"] = (
        analysis["argument_size_in_bytes"]
        + analysis["output_size_in_bytes"]
        + analysis["temp_size_in_bytes"]
        - analysis["alias_size_in_bytes"]
    )
    memory_path.write_text(json.dumps(analysis, indent=2) + "\n")
    event("finished", device)


def summarize(events, samples, analysis, accuracy):
    event_times = [e["time_ns"] for e in events]
    by_phase = {name: [] for name in ACTIVE_PHASES}
    for sample in samples:
        index = bisect_right(event_times, sample["time_ns"]) - 1
        if index >= 0:
            phase = events[index]["phase"]
            if phase in by_phase:
                by_phase[phase].append(sample)
    phases = {}
    for i, e in enumerate(events[:-1]):
        phase = e["phase"]
        if phase not in by_phase:
            continue
        values = by_phase[phase]
        if not values:
            continue
        stats = events[i + 1]["allocator"]
        phases[phase] = {
            "duration_seconds": (events[i + 1]["time_ns"] - e["time_ns"]) / 1e9,
            "samples": len(values),
            "process_peak_bytes": max(s["process_bytes"] for s in values),
            "device_peak_bytes": max(s["device_bytes"] for s in values),
            "allocator_at_end": stats,
        }
    active = [s for name in ACTIVE_PHASES for s in by_phase[name]]
    # Baseline is the sample closest to the first post-JAX-initialization event.
    prep = next(e for e in events if e["phase"] == "preparation")
    before = [s for s in samples if s["time_ns"] <= prep["time_ns"]]
    baseline = before[-1]["process_bytes"] if before else None
    intervals = [
        (b["time_ns"] - a["time_ns"]) / 1e6 for a, b in itertools.pairwise(samples)
    ]
    return {
        "n": accuracy["shape"][0],
        "unknowns": accuracy["unknowns"],
        "process_peak_bytes": max(s["process_bytes"] for s in active),
        "jax_context_baseline_bytes": baseline,
        "allocator_peak_bytes": max(
            e["allocator"].get("peak_bytes_in_use", 0)
            for e in events
            if e["allocator"] is not None and e["phase"] in (*ACTIVE_PHASES, "complete")
        ),
        "allocator_pool_peak_bytes": max(
            e["allocator"].get("peak_pool_bytes", 0)
            for e in events
            if e["allocator"] is not None and e["phase"] in (*ACTIVE_PHASES, "complete")
        ),
        "sample_count": len(samples),
        "sampling_mean_ms": sum(intervals) / len(intervals),
        "sampling_max_ms": max(intervals),
        "phases": phases,
        "compiled_buffers": analysis,
        "plan_bytes": accuracy["plan_bytes"],
        "relative_residual": accuracy["relative_residual"],
        "relative_l2_error": accuracy["relative_l2_error"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("build/box_memory"))
    parser.add_argument("--sizes", nargs="+", type=int, default=[31, 63, 127])
    parser.add_argument("--interval-ms", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--n", type=int)
    args = parser.parse_args()
    if args.interval_ms <= 0 or args.repeats < 1:
        parser.error("Need positive sampling interval and solve repetitions")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    import pynvml as nvml

    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    env = os.environ.copy()
    env.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    env.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root / "jax/src") + os.pathsep + env.get("PYTHONPATH", "")
    results = {
        "device": nvml.nvmlDeviceGetName(handle),
        "driver": nvml.nvmlSystemGetDriverVersion(),
        "preallocate": False,
        "allocator": "JAX default BFC",
        "interval_requested_ms": args.interval_ms,
        "warm_repetitions": args.repeats,
        "cases": [],
    }
    try:
        for n in args.sizes:
            print(f"Profiling BSPF {n}^3", flush=True)
            event_file = args.out / f"{n}_events.jsonl"
            event_file.write_text("")
            samples = []
            with (args.out / f"{n}.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--n",
                        str(n),
                        "--out",
                        str(args.out),
                        "--repeats",
                        str(args.repeats),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    while process.poll() is None:
                        running = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
                        target = next(
                            (p for p in running if p.pid == process.pid), None
                        )
                        if target is not None:
                            used = int(target.usedGpuMemory)
                            if 0 <= used < 2**63:
                                samples.append(
                                    {
                                        "time_ns": monotonic_ns(),
                                        "process_bytes": used,
                                        "device_bytes": int(
                                            nvml.nvmlDeviceGetMemoryInfo(handle).used
                                        ),
                                    }
                                )
                        sleep(args.interval_ms / 1000)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=10)
                if process.returncode:
                    raise RuntimeError(f"Memory worker failed: {args.out / f'{n}.log'}")
            events = [json.loads(line) for line in event_file.read_text().splitlines()]
            analysis = json.loads((args.out / f"{n}_jax.json").read_text())
            accuracy = json.loads((args.out / f"{n}_accuracy.json").read_text())
            case = summarize(events, samples, analysis, accuracy)
            results["cases"].append(case)
            (args.out / f"{n}_samples.json").write_text(json.dumps(samples) + "\n")
            (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(
                f"  process peak {case['process_peak_bytes'] / 2**20:.1f} MiB; JAX live peak {case['allocator_peak_bytes'] / 2**20:.1f} MiB; compiled buffers {analysis['total_buffer_bytes'] / 2**20:.1f} MiB",
                flush=True,
            )
    finally:
        nvml.nvmlShutdown()


if __name__ == "__main__":
    main()
