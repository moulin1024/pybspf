"""Render the saved 3D direct-core benchmark (no solver reruns)."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

root = Path("build/pressure3d_benchmark")
rows = json.loads((root / "results.json").read_text())
fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
for backend, label, color in [
    ("dense", "Dense", "#2463ac"),
    ("compressed", "DCT + layered blocks", "#db7432"),
]:
    data = [r for r in rows if r["backend"] == backend]
    x = np.array([r["n"] for r in data])
    axes[0, 0].loglog(
        x,
        [r["timing"]["median_seconds"] * 1e3 for r in data],
        "o-",
        label=label,
        color=color,
    )
    axes[0, 1].plot(
        x,
        [r["transform_storage"]["stored_bytes"] / 2**20 for r in data],
        "o-",
        label=label,
        color=color,
    )
    axes[1, 0].semilogy(
        x,
        [r["accuracy"]["smooth"]["error_linf"] for r in data],
        "o-",
        label=label + " / smooth",
        color=color,
    )
    axes[1, 0].semilogy(
        x,
        [r["accuracy"]["random"]["error_linf"] for r in data],
        "s--",
        label=label + " / random",
        color=color,
    )
    axes[1, 1].plot(
        x, [r["peak_rss_bytes"] / 2**30 for r in data], "o-", label=label, color=color
    )
for ax, title, ylabel in zip(
    axes.flat,
    [
        "Direct application: 7-run median",
        "Shared 1D transform factors",
        "Lifted equation: pressure recovery",
        "Worker peak RSS (includes validation)",
    ],
    ["Time (ms)", "Storage (MiB)", "Maximum absolute error", "Memory (GiB)"],
):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Points per axis N (total N³)")
    ax.set_xticks([64, 128, 256], ["64", "128", "256"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
fig.suptitle(
    "3D BSPF pressure direct core — CPU / float64 / ZERO refinement", fontsize=14
)
fig.savefig(root / "comparison.png", dpi=180)
plt.close(fig)
