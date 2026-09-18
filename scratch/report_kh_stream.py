"""Figures and movie from saved KH computations; no PDE solve."""

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


def render_movie(out):
    out = Path(out)
    f = np.load(out / "frames.npz")
    records = json.loads((out / "diagnostics.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    is_open = summary["boundary"].startswith(("Open", "Dynamic"))
    has_sponge = summary.get("buffer", False)
    label = (
        "BSPF | open vertical boundaries"
        if is_open
        else "BSPF + boundary-layer basis | fixed boundaries"
    )
    if summary["boundary"].startswith("Dynamic"):
        label = f"BSPF | dynamic outflow D0={summary['boundary_D0']:g}"
    if has_sponge:
        label = (
            f"BSPF | dynamic outflow D0={summary['boundary_D0']:g} + exterior absorption"
            if summary["boundary"].startswith("Dynamic")
            else "BSPF | external absorbing extension"
        )
    treatment = (
        "Exterior relaxation only | no filter" if has_sponge else "No buffer, no filter"
    )
    if Path("/opt/homebrew/bin/ffmpeg").exists():
        matplotlib.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"
    extent = [f["x"][0], f["x"][-1], f["y"][0], f["y"][-1]]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), layout="constrained")
    images = []
    for ax, key, limit, title in zip(
        axes,
        ["vorticity", "transverse_velocity"],
        [10, 0.8],
        [
            "Vorticity (color clipped at +/-10)",
            "Transverse velocity",
        ],
    ):
        im = ax.imshow(
            f[key][0].T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        if summary.get("extension", 0):
            for side in (-3, 3):
                ax.axvline(side, color="black", linestyle="--", linewidth=0.8)
            ax.axvspan(extent[0], -3, color="grey", alpha=0.1)
            ax.axvspan(3, extent[1], color="grey", alpha=0.1)
        images.append(im)
        ax.set(xlabel="x", ylabel="y", title=title)
        fig.colorbar(im, ax=ax, shrink=0.85)
    title = fig.suptitle("")
    writer = FFMpegWriter(
        fps=5,
        codec="libx264",
        extra_args=["-crf", "18", "-pix_fmt", "yuv420p"],
        metadata={
            "comment": f"Computed divergence-free BSPF KH; {summary['boundary']}; {treatment}."
        },
    )
    with writer.saving(fig, str(out / "kh_stream.mp4"), dpi=110):
        for i, t in enumerate(f["times"]):
            images[0].set_data(f["vorticity"][i].T)
            images[1].set_data(f["transverse_velocity"][i].T)
            title.set_text(
                f"{label} | {len(f['x'])} x {len(f['y'])} | t={t:g}\n{treatment} | max speed={records[i]['max_speed']:.3f}, max |vorticity|={records[i]['max_vorticity']:.1f}"
            )
            writer.grab_frame()
        fig.savefig(out / "kh_stream.png", dpi=150)
    plt.close(fig)


def report(root):
    root = Path(root)
    old = np.load(root.parent / "kh_weak/96x80/frames.npz")
    new = np.load(
        root
        / ("final96" if (root / "final96/frames.npz").exists() else "enriched96")
        / "frames.npz"
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for row, (f, label) in enumerate(
        [
            (old, "Previous weak projection"),
            (new, "Compatible BSPF + 6 layer functions"),
        ]
    ):
        extent = [f["x"][0], f["x"][-1], f["y"][0], f["y"][-1]]
        for col, (key, limit, title) in enumerate(
            [
                ("vorticity", 10, "Vorticity"),
                ("transverse_velocity", 0.8, "Transverse velocity"),
            ]
        ):
            ax = axes[row, col]
            im = ax.imshow(
                f[key][-1].T,
                extent=extent,
                origin="lower",
                aspect="auto",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
            ax.set(title=f"{label}\n{title}", xlabel="x", ylabel="y")
            fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    fig.suptitle(
        "Same 96 x 80 output grid, t=6, fixed original boundaries, no sponge\nVorticity colors clipped at +/-10; resolved wall layers reach about 403",
        fontsize=14,
    )
    fig.savefig(root / "boundary_fix.png", dpi=150)
    fig.savefig(root / "boundary_fix.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    base = np.load(root / "64x64/basis.npz")
    a0 = np.load(root / "64x64/checkpoint.npz")["a"]
    en = np.load(root / "enriched64/basis.npz")
    a1 = np.load(root / "enriched64/checkpoint.npz")["a"]
    # Same nearest y node in the two 64-point plans; actual basis evaluations in x.
    j = np.argmin(abs(base["y_x"] - 0.3))
    yy = base["y_x"][j]
    v0 = -base["x_g"] @ a0 @ base["y_bn"][j]
    v1 = -en["x_g"] @ a1 @ en["y_bn"][j]
    for ax in axes:
        ax.plot(base["x_points"], v0, label="Unenriched", color="#d97706")
        ax.plot(en["x_points"], v1, label="Enriched", color="#2563eb")
        ax.set(
            xlabel="x", ylabel="v", title=f"Physical reconstruction, y={yy:.3f}, t=6"
        )
        ax.grid(alpha=0.2)
    axes[0].set_xlim(-3, -1.5)
    axes[0].legend()
    axes[1].set_xlim(2.97, 3)
    axes[1].set_title("Resolved outflow layer (width ~0.002)")
    axes[1].legend()
    fig.savefig(root / "boundary_cuts.png", dpi=150)
    plt.close(fig)

    one = np.load(root / "layer_accuracy/curves.npz")
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for key, label in [
        ("original", "Original BSPF"),
        ("enriched", "BSPF + exact layer basis"),
        ("exact", "Analytic PDE solution"),
    ]:
        ax.plot(one["x"], one[key], label=label, ls="--" if key == "exact" else "-")
    ax.set(
        xlim=(0.7, 1),
        xlabel="x",
        ylabel="u",
        title="1D outflow-layer control: u\N{PRIME} - 0.002 u\N{DOUBLE PRIME} = 1, u(0)=u(1)=0",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(root / "layer_accuracy.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("build/kh_stream"))
    p.add_argument("--movie", type=Path)
    args = p.parse_args()
    if args.movie:
        render_movie(args.movie)
    else:
        report(args.root)
