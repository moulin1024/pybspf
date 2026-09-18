"""Compare saved open truncation and external-absorption KH simulations."""

from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from report_kh_stream import render_movie

root = Path("build/kh_stream")
a = np.load(root / "open96/frames.npz")
b = np.load(root / "extended160_s4/frames.npz")
fig, axes = plt.subplots(2, 2, figsize=(13, 6.5), layout="constrained")
for col, t in enumerate([6, 12]):
    for row, (f, label) in enumerate(
        [
            (a, "Open faces at x = +/-3"),
            (b, "Absorbing extension; crop to unchanged interest region"),
        ]
    ):
        i = np.argmin(abs(f["times"] - t))
        im = axes[row, col].imshow(
            f["vorticity"][i].T,
            origin="lower",
            extent=[f["x"][0], f["x"][-1], -1, 1],
            aspect="auto",
            cmap="RdBu_r",
            vmin=-12,
            vmax=12,
            interpolation="nearest",
        )
        axes[row, col].set(
            xlim=(-3, 3), xlabel="x", ylabel="y", title=f"{label}\nt={t}"
        )
        fig.colorbar(im, ax=axes[row, col], shrink=0.8)
fig.suptitle(
    "Same BSPF order and comparable spacing | vorticity colors clipped at +/-12\nAbsorption is strictly outside |x| <= 3; horizontal fixed boundaries retained"
)
fig.savefig(root / "extension_comparison.png", dpi=150)
plt.close(fig)
records = json.loads((root / "extended160_s4/diagnostics.json").read_text())
fig, axes = plt.subplots(2, 1, figsize=(11, 6), layout="constrained")
for ax, t in zip(axes, [6, 12]):
    i = np.argmin(abs(b["times"] - t))
    im = ax.imshow(
        b["vorticity"][i].T,
        origin="lower",
        extent=[-5, 5, -1, 1],
        aspect="auto",
        cmap="RdBu_r",
        vmin=-12,
        vmax=12,
        interpolation="nearest",
    )
    for x in [-3, 3]:
        ax.axvline(x, color="black", ls="--", lw=0.8)
    ax.axvspan(-5, -3, color="gray", alpha=0.1)
    ax.axvspan(3, 5, color="gray", alpha=0.1)
    ax.set(
        title=f"t={t}; full max |vorticity|={records[i]['max_vorticity']:.2f}; interest-region max={records[i]['roi_max_vorticity']:.2f}",
        xlabel="x",
        ylabel="y",
    )
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle(
    "Extended domain | dashed lines bound the unmodified region | colors clipped at +/-12"
)
fig.savefig(root / "extension_full.png", dpi=150)
plt.close(fig)
render_movie(root / "extended160_s4")
