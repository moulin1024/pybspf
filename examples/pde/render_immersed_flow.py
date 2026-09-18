"""Actual BSPF channel solution and measured buffer attenuation."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("build/immersed_flow/main"))
    ap.add_argument("--movie", action="store_true")
    args = ap.parse_args()
    out = args.out
    d = np.load(out / "fields.npz")
    summary = json.loads((out / "summary.json").read_text())
    x, y, t = d["x"], d["y"], d["t"]
    xx, yy = np.meshgrid(x, y)
    fluid = ((xx - d["center"][0]) / d["axes"][0]) ** 2 + (
        (yy - d["center"][1]) / d["axes"][1]
    ) ** 2 >= 1
    f = d["fields"]
    u, v, omega, psi = f[-1]
    baseomega = 2 * summary["peak_inlet"] * yy / summary["bounds"][2] ** 2
    speed = np.where(fluid, np.hypot(u, v), np.nan)
    perturbation = np.where(fluid, omega - baseomega, np.nan)
    extent = [x[0], x[-1], y[0], y[-1]]
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    a = axs[0].imshow(speed, origin="lower", extent=extent, cmap="viridis", vmin=0)
    fig.colorbar(a, ax=axs[0], label="Speed")
    axs[0].streamplot(
        x,
        y,
        np.ma.masked_where(~fluid, u),
        np.ma.masked_where(~fluid, v),
        density=(2, 0.8),
        color="white",
        linewidth=0.5,
        arrowsize=0.7,
    )
    axs[0].set_title(
        f"Velocity and streamlines | Re={summary['reynolds']:g}, t={t[-1]:g}, BSPF {int(d['nx'])} x {int(d['ny'])}"
    )
    limit = np.nanmax(abs(perturbation))
    a = axs[1].imshow(
        perturbation,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    fig.colorbar(a, ax=axs[1], label="Vorticity minus Poiseuille shear")
    axs[1].set_title("Wake vorticity perturbation (baseline wall shear retained)")

    def geometry(ax):
        ax.add_patch(
            Ellipse(
                d["center"],
                *(2 * d["axes"]),
                facecolor=".65",
                edgecolor="black",
                lw=1,
                zorder=8,
            )
        )
        ax.axvline(float(d["buffer_start"]), color="black", ls="--", lw=1)
        ax.axvspan(float(d["buffer_start"]), x[-1], facecolor="white", alpha=0.13)
        ax.text(
            (float(d["buffer_start"]) + x[-1]) / 2,
            0.87,
            "Buffer: sigma 0 -> " + str(summary["buffer_strength"]),
            ha="center",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])

    for ax in axs:
        geometry(ax)
    fig.savefig(out / "flow.png", dpi=170)
    plt.close(fig)
    h = json.loads((out / "history.json").read_text())
    b = summary["checks"]["buffer_sections"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for key, label in [
        ("perturbation_speed_l2", "Velocity perturbation"),
        ("perturbation_vorticity_l2", "Vorticity perturbation"),
    ]:
        axs[0].semilogy(
            [a["x"] for a in b], [a[key] for a in b], "o-", ms=3, label=label
        )
    axs[0].set_title("Cross-section L2 norms through buffer")
    axs[0].set_xlabel("x")
    axs[0].legend(fontsize=8)
    axs[1].plot(x, d["sigma"])
    axs[1].axvline(float(d["buffer_start"]), color="0.5", ls="--")
    axs[1].set_title("Damping: zero in the physical domain")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("sigma")
    axs[2].semilogy([a["t"] for a in h], [a["acceleration_l2"] for a in h])
    axs[2].set_title("Approach to a steady flow")
    axs[2].set_xlabel("time")
    axs[2].set_ylabel("||du/dt|| L2")
    for ax in axs:
        ax.grid(alpha=0.2)
    fig.savefig(out / "diagnostics.png", dpi=170)
    plt.close(fig)
    if args.movie:
        # Prefer the self-contained encoder when available; a system ffmpeg
        # may depend on unavailable shared libraries in the plotting runtime.
        try:
            import imageio_ffmpeg

            matplotlib.rcParams["animation.ffmpeg_path"] = (
                imageio_ffmpeg.get_ffmpeg_exe()
            )
        except ImportError:
            pass
        fig, axs = plt.subplots(2, 1, figsize=(10, 6.4), layout="constrained")
        images = []
        maxspeed = np.max(np.hypot(f[:, 0][:, fluid], f[:, 1][:, fluid]))
        lim = np.max(abs((f[:, 2] - baseomega)[:, fluid]))
        for ax, cmap, vmin, vmax in [
            (axs[0], "viridis", 0, maxspeed),
            (axs[1], "RdBu_r", -lim, lim),
        ]:
            images.append(
                ax.imshow(
                    np.zeros_like(xx),
                    origin="lower",
                    extent=extent,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
            )
            geometry(ax)
            fig.colorbar(images[-1], ax=ax)
        title = fig.suptitle("")
        axs[0].set_title("Speed")
        axs[1].set_title("Vorticity perturbation")

        def update(i):
            images[0].set_data(np.where(fluid, np.hypot(f[i, 0], f[i, 1]), np.nan))
            images[1].set_data(np.where(fluid, f[i, 2] - baseomega, np.nan))
            title.set_text(
                f"BSPF immersed channel | Re={summary['reynolds']:g} | t={t[i]:.2f}"
            )
            return images + [title]

        animation = FuncAnimation(fig, update, frames=len(t), interval=125)
        animation.save(out / "flow.mp4", writer=FFMpegWriter(fps=8), dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()
