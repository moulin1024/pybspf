"""! @file test_performance_regression.py
@brief Lightweight performance regression guards against the legacy baseline.
"""

from __future__ import annotations

import math
import time
from statistics import median

import numpy as np

from pybspf import BSPF1D
from bspf1d import bspf1d as LegacyBSPF1D


def _make_signal(x: np.ndarray) -> np.ndarray:
    """Build a smooth real-valued test signal."""
    return np.sin(x / (1.01 + np.cos(x))) + 0.15 * np.cos(3.0 * x)


def _build_ops(n: int, degree: int = 5):
    """Construct matching package and legacy operators."""
    x = np.linspace(0.0, 2.0 * np.pi, n)
    kwargs = dict(
        degree=degree,
        x=x,
        n_basis=4 * degree,
        domain=(0.0, 2.0 * np.pi),
        use_clustering=True,
        clustering_factor=2.0,
        order=degree,
        num_boundary_points=degree + 3,
        correction="spectral",
    )
    return x, BSPF1D.from_grid(**kwargs), LegacyBSPF1D.from_grid(**kwargs)


def _measure_apply_time(
    op,
    f: np.ndarray,
    *,
    warmup: int = 3,
    repeats: int = 7,
    inner: int = 10,
) -> tuple[float, list[float]]:
    """Measure median steady-state apply time for shared first/second derivatives."""
    for _ in range(warmup):
        if hasattr(op, "derivatives"):
            op.derivatives(f, orders=(1, 2), lam=1.0e-8)
        else:
            op.differentiate_1_2(f, lam=1.0e-8)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(inner):
            if hasattr(op, "derivatives"):
                op.derivatives(f, orders=(1, 2), lam=1.0e-8)
            else:
                op.differentiate_1_2(f, lam=1.0e-8)
        samples.append((time.perf_counter() - start) / inner)
    return median(samples), samples


def test_package_time_complexity_tracks_legacy():
    """The package implementation should scale similarly to the legacy path."""
    sizes = (10001,20001)

    package_times = []
    legacy_times = []
    package_samples = []
    legacy_samples = []

    for n in sizes:
        x, package_op, legacy_op = _build_ops(n)
        f = _make_signal(x)
        package_time, package_runs = _measure_apply_time(package_op, f)
        legacy_time, legacy_runs = _measure_apply_time(legacy_op, f)
        package_times.append(package_time)
        legacy_times.append(legacy_time)
        package_samples.append(package_runs)
        legacy_samples.append(legacy_runs)

    package_growth = package_times[1] / package_times[0]
    legacy_growth = legacy_times[1] / legacy_times[0]

    package_exponent = math.log(package_growth, 2.0)
    legacy_exponent = math.log(legacy_growth, 2.0)

    print("performance regression timing summary")
    for idx, n in enumerate(sizes):
        print(
            f"  n={n} package median={package_times[idx]:.6e}s "
            f"samples={[f'{sample:.6e}' for sample in package_samples[idx]]}"
        )
        print(
            f"  n={n} legacy  median={legacy_times[idx]:.6e}s "
            f"samples={[f'{sample:.6e}' for sample in legacy_samples[idx]]}"
        )
    print(
        "  growth "
        f"package={package_growth:.3f} legacy={legacy_growth:.3f} "
        f"package_exponent={package_exponent:.3f} legacy_exponent={legacy_exponent:.3f}"
    )

    assert package_exponent <= legacy_exponent + 0.75, (
        "Package shared first/second-derivative path scales materially worse than legacy. "
        f"package_times={package_times}, legacy_times={legacy_times}, "
        f"package_exponent={package_exponent:.3f}, legacy_exponent={legacy_exponent:.3f}"
    )

    assert package_times[1] <= 3.0 * legacy_times[1], (
        "Package shared first/second-derivative path is too slow at the larger regression size. "
        f"package_times={package_times}, legacy_times={legacy_times}"
    )
