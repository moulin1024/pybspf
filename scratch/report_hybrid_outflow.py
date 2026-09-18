"""Plots and a movie comparing the same interest region for three sponge widths."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from report_kh_stream import render_movie

root = Path("build/kh_stream")
names = ["extended160_s4", "hybrid_L1_s4_dynamic", "hybrid_L0.5_s4_dynamic"]
labels = [
    "Width 2 per side, ordinary outflow (reference)",
    "Width 1 per side, dynamic outflow",
    "Width 0.5 per side, dynamic outflow",
]
frames = [np.load(root / name / "frames.npz") for name in names]
metrics = json.loads((root / "hybrid_comparison.json").read_text())
fig, axes = plt.subplots(3, 2, figsize=(12, 9), layout="constrained")
for row, (f, name, label) in enumerate(zip(frames, names, labels)):
    for col, t in enumerate([6, 12]):
        i = np.argmin(abs(f["times"] - t))
        ax = axes[row, col]
        im = ax.imshow(
            f["vorticity"][i].T,
            origin="lower",
            extent=[f["x"][0], f["x"][-1], -1, 1],
            aspect="auto",
            cmap="RdBu_r",
            vmin=-12,
            vmax=12,
            interpolation="nearest",
        )
        ax.set(xlim=(-3, 3), xlabel="x", ylabel="y", title=f"{label}\nt={t}")
        fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle(
    "Identical region of interest | comparable spacing | damping peak = 4\nVorticity colors clipped at +/-12; absorption strictly outside |x| <= 3"
)
fig.savefig(root / "hybrid_comparison.png", dpi=150)
plt.close(fig)
# Plot differences from the same numerical reference; no exact-error claim.
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for ax, key, title in zip(
    axes,
    ["interest_relative_perturbation_l2", "center_relative_perturbation_l2"],
    ["Whole interest region", "Center: |x|<2, |y|<0.5"],
):
    for case, marker in [("open", "o"), ("dynamic", "s")]:
        vals = [100 * metrics[f"hybrid_L{L}_s4_{case}"][key] for L in ["0.5", "1"]]
        ax.plot([0.5, 1], vals, marker=marker, label=case)
    ax.set(
        xlabel="Absorption width per side",
        ylabel="Relative perturbation-velocity difference (%)",
        title=title,
        xticks=[0.5, 1],
    )
    ax.grid(alpha=0.2)
    ax.legend()
fig.suptitle("T=12 differences from width-2 ordinary-outflow reference")
fig.savefig(root / "hybrid_differences.png", dpi=150)
plt.close(fig)
if Path("/opt/homebrew/bin/ffmpeg").exists():
    matplotlib.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"
fig, axes = plt.subplots(3, 1, figsize=(11, 8), layout="constrained")
images = []
for ax, f, label in zip(axes, frames, labels):
    im = ax.imshow(
        f["vorticity"][0].T,
        origin="lower",
        extent=[f["x"][0], f["x"][-1], -1, 1],
        aspect="auto",
        cmap="RdBu_r",
        vmin=-12,
        vmax=12,
        interpolation="nearest",
    )
    ax.set(xlim=(-3, 3), xlabel="x", ylabel="y", title=label)
    fig.colorbar(im, ax=ax, shrink=0.8)
    images.append(im)
title = fig.suptitle("")
writer = FFMpegWriter(
    fps=5, codec="libx264", extra_args=["-crf", "18", "-pix_fmt", "yuv420p"]
)
with writer.saving(fig, str(root / "hybrid_comparison.mp4"), dpi=110):
    for i, t in enumerate(frames[0]["times"]):
        for f, im in zip(frames, images):
            if not np.isclose(f["times"][i], t):
                raise ValueError("Frame times differ")
            im.set_data(f["vorticity"][i].T)
        title.set_text(
            f"Same interest region, t={t:g} | all absorption is outside the displayed region\nVorticity colors clipped at +/-12 | fixed horizontal boundaries retained"
        )
        writer.grab_frame()
plt.close(fig)
render_movie(root / "hybrid_L1_s4_dynamic")
