"""Saved-data dynamic outflow comparison; no new simulation."""

from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from report_kh_stream import render_movie

root = Path("build/kh_stream")
paths = ["open96", "dynamic96_d1", "extended160_s4"]
labels = [
    "Original open boundary",
    "Dynamic outflow D0=1, no buffer",
    "External absorbing extension (cropped)",
]
fig, axes = plt.subplots(3, 2, figsize=(13, 9), layout="constrained")
metrics = {}
for row, (directory, label) in enumerate(zip(paths, labels)):
    f = np.load(root / directory / "frames.npz")
    metrics[directory] = {}
    for col, t in enumerate([6, 12]):
        i = np.argmin(abs(f["times"] - t))
        om = f["vorticity"][i]
        region = (
            (abs(f["x"][:, None]) < 3.0001)
            & (abs(f["x"][:, None]) > 2.8)
            & (abs(f["y"][None, :]) < 0.9)
        )
        metrics[directory][str(t)] = {
            "full_nodal_vorticity_max": float(abs(om).max()),
            "near_interest_edge_vorticity_max": float(abs(om)[region].max()),
        }
        ax = axes[row, col]
        im = ax.imshow(
            om.T,
            origin="lower",
            extent=[f["x"][0], f["x"][-1], -1, 1],
            aspect="auto",
            cmap="RdBu_r",
            vmin=-12,
            vmax=12,
            interpolation="nearest",
        )
        ax.set(
            xlim=(-3, 3),
            title=f"{label}\nt={t}; full-domain max |omega|={abs(om).max():.1f}",
            xlabel="x",
            ylabel="y",
        )
        fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle(
    "BSPF KH | unchanged spatial order, viscosity and interior seed formula\nVorticity colors clipped at +/-12; fixed horizontal boundaries retained"
)
fig.savefig(root / "dynamic_comparison.png", dpi=150)
plt.close(fig)
(root / "dynamic_comparison.json").write_text(json.dumps(metrics, indent=2))
probe = json.loads((root / "dynamic_vortex/diagnostics.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
for label in probe[0]["cases"]:
    for ax, key, title in zip(
        axes,
        ["full_absolute_l2", "upstream_linf"],
        ["Velocity L2 difference, original domain", "Velocity max difference, x < 2.5"],
    ):
        ax.semilogy(
            [r["t"] for r in probe],
            [r["cases"][label][key] for r in probe],
            label=label,
        )
        ax.set(xlabel="t", ylabel="Difference", title=title)
        ax.grid(alpha=0.2)
axes[0].legend(fontsize=8)
fig.suptitle(
    "Small vortex crossing: differences from a longer-domain calculation\nThe reference is numerical, not an exact infinite-domain solution"
)
fig.savefig(root / "dynamic_vortex/comparison.png", dpi=150)
plt.close(fig)
print(json.dumps(metrics, indent=2))
render_movie(root / "dynamic96_d1")
