"""Plot saved BSPF vs FD/PyAMG measurements, keeping accuracy notions separate."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

root = Path("build/pressure3d_pyamg")
amg = json.loads((root / "results.json").read_text())
cg = json.loads(Path("build/pressure3d_pyamg_cg/results.json").read_text())
bspf = json.loads(Path("build/pressure3d_benchmark/results.json").read_text())
fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
for backend, label, color in [
    ("dense", "BSPF dense", "#2463ac"),
    ("compressed", "BSPF compressed", "#db7432"),
]:
    data = [r for r in bspf if r["backend"] == backend]
    ns = [r["n"] for r in data]
    axes[0, 0].loglog(
        ns,
        [r["timing"]["median_seconds"] for r in data],
        "o-",
        label=label,
        color=color,
    )
    axes[0, 1].plot(
        ns, [r["peak_rss_bytes"] / 2**30 for r in data], "o-", label=label, color=color
    )
ns = np.array([r["n"] for r in amg])
axes[0, 0].loglog(
    ns,
    [r["cases"]["smooth_discrete"]["median_seconds"] for r in amg],
    "o-",
    label="FD + PyAMG V-cycle",
    color="#31835b",
)
axes[0, 1].plot(
    ns,
    [r["peak_rss_bytes"] / 2**30 for r in amg],
    "o-",
    label="FD + PyAMG V-cycle",
    color="#31835b",
)
axes[0, 0].loglog(
    ns,
    [r["cases"]["smooth_discrete"]["median_seconds"] for r in cg],
    "s--",
    label="FD + AMG-preconditioned CG",
    color="#77539c",
)
axes[0, 1].plot(
    ns,
    [r["peak_rss_bytes"] / 2**30 for r in cg],
    "s--",
    label="FD + AMG-preconditioned CG",
    color="#77539c",
)
for r in amg:
    hist = r["cases"]["smooth_discrete"]["residual_histories"][-1]
    axes[1, 0].semilogy(range(len(hist)), hist, "o-", ms=3, label=f"V / N={r['n']}")
for r in cg:
    hist = r["cases"]["smooth_discrete"]["residual_histories"][-1]
    axes[1, 0].semilogy(range(len(hist)), hist, "s--", ms=3, label=f"CG / N={r['n']}")
axes[1, 0].axhline(1e-10, color="gray", ls="--", label="Relative target ~1e-10")
e = np.array([r["cases"]["smooth_continuous"]["error_linf"] for r in amg])
axes[1, 1].loglog(
    ns, e, "o-", label="FD: continuous manufactured solution", color="#31835b"
)
axes[1, 1].loglog(
    ns, e[0] * ((ns[0] - 1) / (ns - 1)) ** 2, "--", label="O(h²)", color="gray"
)
for ax, title, ylabel in zip(
    axes.flat,
    [
        "Warm solve time: smooth discrete RHS",
        "Worker peak RSS (includes validation)",
        "PyAMG convergence: smooth discrete RHS",
        "FD discretization error (separate experiment)",
    ],
    ["Seconds", "GiB", "Relative L2 residual", "Maximum pressure error"],
):
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_xlabel("N per axis")
    if ax is not axes[1, 0]:
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.xaxis.set_minor_locator(NullLocator())
axes[1, 0].set_xlabel("V-cycle or preconditioned CG iteration")
fig.suptitle(
    "BSPF direct core vs 7-point FD + PyAMG / CPU float64\nDifferent discrete operators and boundary treatment; BSPF uses zero refinement",
    fontsize=13,
)
fig.savefig(root / "comparison.png", dpi=180)
