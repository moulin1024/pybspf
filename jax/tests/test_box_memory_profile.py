"""Memory reports must exclude post-workflow compiler inspection from peaks."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "box_memory_profile",
    Path(__file__).resolve().parents[2] / "scratch/profile_box_memory.py",
)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def test_profile_phase_boundaries_and_late_analysis_exclusion():
    def stats(peak):
        return {"peak_bytes_in_use": peak, "peak_pool_bytes": 2 * peak}

    events = [
        {"time_ns": 0, "phase": "initialization", "allocator": None},
        {"time_ns": 10, "phase": "preparation", "allocator": stats(0)},
        {"time_ns": 20, "phase": "first_solve", "allocator": stats(100)},
        {"time_ns": 30, "phase": "warm_solves", "allocator": stats(120)},
        {"time_ns": 40, "phase": "validation", "allocator": stats(150)},
        {"time_ns": 50, "phase": "complete", "allocator": stats(200)},
        {"time_ns": 60, "phase": "analysis", "allocator": stats(3000)},
        {"time_ns": 70, "phase": "finished", "allocator": stats(3000)},
    ]
    samples = [
        {"time_ns": t, "process_bytes": v, "device_bytes": v + 10}
        for t, v in [
            (2, 5),
            (12, 20),
            (22, 40),
            (32, 30),
            (42, 35),
            (52, 45),
            (62, 1000000),
        ]
    ]
    accuracy = {
        "shape": [31] * 3,
        "unknowns": 31**3,
        "plan_bytes": 100,
        "relative_residual": 1e-12,
        "relative_l2_error": 1e-6,
    }
    report = profile.summarize(events, samples, {"total_buffer_bytes": 43}, accuracy)
    assert report["process_peak_bytes"] == 40
    assert report["jax_context_baseline_bytes"] == 5
    assert report["allocator_peak_bytes"] == 200
    assert report["allocator_pool_peak_bytes"] == 400
    assert report["phases"]["warm_solves"]["process_peak_bytes"] == 30
    assert (
        report["phases"]["first_solve"]["allocator_at_end"]["peak_bytes_in_use"] == 120
    )
    assert "analysis" not in report["phases"]
