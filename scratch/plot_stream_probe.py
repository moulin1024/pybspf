"""Display saved KH probes only; no simulation."""

from pathlib import Path
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("directories", nargs="+", type=Path)
p.add_argument("--out", type=Path, default=Path("build/kh_stream/comparison.png"))
args = p.parse_args()
fig, axes = plt.subplots(
    len(args.directories),
    2,
    figsize=(12, 3 * len(args.directories)),
    squeeze=False,
    layout="constrained",
)
for row, d in enumerate(args.directories):
    f = np.load(d / "frames.npz")
    extent = [f["x"][0], f["x"][-1], f["y"][0], f["y"][-1]]
    for col, (key, vmax) in enumerate(
        [("vorticity", 10), ("transverse_velocity", 0.8)]
    ):
        ax = axes[row, col]
        im = ax.imshow(
            f[key][-1].T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set(
            title=f"{d.parent.name}/{d.name} | {key} | t={f['times'][-1]:g}",
            xlabel="x",
            ylabel="y",
        )
        fig.colorbar(im, ax=ax, shrink=0.85)
fig.savefig(args.out, dpi=140)
print(args.out.resolve())
