"""Independent rational Stokes reference and fixed-grid BSPF error maps."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

root = Path("build/immersed_flow/stokes_comparison")
r = np.load(root / "reference.npz")
x, y = r["x"], r["y"]
rf = r["fields"]
xx, yy = np.meshgrid(x, y)
mask = ~np.isfinite(rf[0])
rows = json.loads((root / "comparison.json").read_text())
rows = {a["method"]: a for a in rows}
fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")


def panel(ax, value, title, cmap, lo, hi):
    im = ax.pcolormesh(
        x, y, np.ma.array(value, mask=mask), cmap=cmap, vmin=lo, vmax=hi, shading="auto"
    )
    ax.add_patch(
        Ellipse((0.19, -0.13), 0.62, 0.46, facecolor=".7", edgecolor="k", lw=1)
    )
    ax.set(
        xlim=(-1, 2), ylim=(-1, 1), xlabel="x", ylabel="y", title=title, aspect="equal"
    )
    fig.colorbar(im, ax=ax, shrink=0.82)


panel(
    axes[0, 0],
    np.hypot(rf[0], rf[1]),
    "AAA-lightning reference: speed",
    "viridis",
    0,
    1.5,
)
panel(axes[0, 1], rf[3], "AAA-lightning reference: vorticity", "RdBu_r", -11, 11)
for k, (name, label) in enumerate(
    [("svd_strict", "Original SVD"), ("factor", "Analytic wall factor")], start=1
):
    f = np.load(root / f"{name}_pde.npz")
    row = rows[name]["pde"]
    panel(
        axes[k, 0],
        np.hypot(f["u"] - rf[0], f["v"] - rf[1]),
        f"{label}: velocity error (rel L2 {100 * row['velocity_relative_l2']:.3f}%)",
        "magma",
        0,
        0.028,
    )
    panel(
        axes[k, 1],
        f["vorticity"] - rf[3],
        f"{label}: vorticity error (rel L2 {100 * row['vorticity_relative_l2']:.2f}%)",
        "RdBu_r",
        -2.1,
        2.1,
    )
fig.suptitle(
    "Fixed BSPF 73 x 33: identical Stokes geometry and boundary conditions\nNear-obstacle view; error norms cover the entire fluid rectangle; sponge disabled"
)
fig.savefig(root / "comparison.png", dpi=155)
fig2, ax = plt.subplots(figsize=(8, 4.7), layout="constrained")
names = ["svd_strict", "svd", "factor"]
labels = ["SVD 1e-10", "SVD 1e-5", "Wall factor"]
coords = np.arange(3)
ax.bar(
    coords - 0.18,
    [100 * rows[n]["pde"]["velocity_relative_h1"] for n in names],
    width=0.36,
    label="Stokes solve",
)
ax.bar(
    coords + 0.18,
    [100 * rows[n]["best_h1_velocity"]["velocity_relative_h1"] for n in names],
    width=0.36,
    label="Best H1 projection",
)
ax.set(
    xticks=coords,
    xticklabels=labels,
    ylabel="Relative H1 velocity error (%)",
    title="Actual solve already nearly reaches the approximation-space limit",
)
ax.legend()
fig2.savefig(root / "best_approximation.png", dpi=155)
