"""Plot closed flux surfaces, q profile, and stable versus unstable displacement."""

import argparse
import json
import pickle
from pathlib import Path
import jax
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def render(out, movie=True):
    jax.config.update("jax_enable_x64", True)
    m = pickle.loads((out / "model.pkl").read_bytes())
    s = json.loads((out / "summary.json").read_text())
    profile = json.loads((out / "q_profile.json").read_text())
    data = np.load(out / "evolution.npz")
    control = np.load(out / "far_wall.npz")
    surfaces = np.load(out / "equilibrium_surfaces.npz")["surfaces"]
    fig = plt.figure(figsize=(13, 7.4), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=(1.2, 1, 1))
    ax = fig.add_subplot(grid[:, 0])
    axq = fig.add_subplot(grid[0, 1])
    axs = fig.add_subplot(grid[0, 2])
    axt = fig.add_subplot(grid[1, 1:])
    title = fig.suptitle("")
    wall = m.vacuum.points[m.vacuum.outer]
    wall = np.vstack((wall, wall[0]))
    ax.fill(wall[:, 0], wall[:, 1], color="#e7eef7", label="Vacuum gap")
    ax.plot(
        wall[:, 0], wall[:, 1], color="#795548", lw=3, label="Ideal conducting shell"
    )
    ax.fill(m.boundary[:, 0], m.boundary[:, 1], color="white")
    for surf in surfaces[1::3]:
        surf = np.vstack((surf, surf[0]))
        ax.plot(surf[:, 0], surf[:, 1], color="#29948a", lw=0.8)
    boundary = np.vstack((m.boundary, m.boundary[0]))
    ax.plot(
        boundary[:, 0],
        boundary[:, 1],
        "--",
        color=".4",
        label="Equilibrium plasma edge",
    )
    (line,) = ax.plot(
        boundary[:, 0],
        boundary[:, 1],
        color="black",
        lw=1.5,
        label="Moving plasma edge",
    )
    ax.plot(*profile["axis"], "+", color="#b71c1c", ms=10, label="Magnetic axis")
    ax.set(
        xlabel="R",
        ylabel="Z",
        xlim=(1, 3),
        ylim=(-1.4, 1.4),
        aspect="equal",
        title=f"Closed flux surfaces | elongation {s['elongation']:.2f}",
    )
    ax.legend(loc="lower left", fontsize=8)
    ax.text(
        0.03,
        0.96,
        "Fixed external coil currents\nNo fluid in vacuum\nDisplacement shown at true scale",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    axq.plot(
        np.r_[0, profile["normalized_poloidal_flux"]],
        np.r_[profile["q_axis"], profile["q"]],
        color="#1565c0",
    )
    axq.axhline(1, color=".4", ls="--", lw=1)
    axq.set(
        xlabel="Normalized poloidal flux",
        ylabel="q",
        title=f"Safety factor: axis {s['q_axis']:.2f}, q95 {s['q95']:.2f}",
    )
    axq.grid(alpha=0.2)
    scan = s["wall_scan"]
    axs.plot(
        [v["wall_scale"] - 1 for v in scan],
        [v["dominant_vertical_omega_squared"] for v in scan],
        "o-",
        color="#00897b",
    )
    axs.axhline(0, color=".4", lw=1)
    axs.set(
        xlabel="Radial wall gap / local plasma radius",
        ylabel=r"Dominant vertical $\omega^2$",
        title="Passive wall restoring effect",
    )
    axs.grid(alpha=0.2)
    a = s["minor_radius"]
    axt.plot(
        data["time"],
        data["centroid_z"] / a,
        color="#1565c0",
        label="Close shaped wall: bounded oscillation",
    )
    axt.plot(
        control["time"],
        control["centroid_z"] / a,
        color="#e65100",
        label="Distant rectangular wall: instability",
    )
    (marker,) = axt.plot([0], [data["centroid_z"][0] / a], "o", color="black")
    axt.set(
        xlabel="Time (normalized)",
        ylabel=r"$\langle\xi_Z\rangle/a$",
        title="Same initial vertical perturbation; no mechanical driving",
    )
    axt.grid(alpha=0.2)
    axt.legend(fontsize=8)
    axt.text(
        0.99,
        0.60,
        "Ideal, incompressible, linear n=0 vertical sector.\nNo resistive wall, feedback, heat transport or fusion power.",
        transform=axt.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    def update(i):
        b = data["boundary"][i]
        b = np.vstack((b, b[0]))
        line.set_data(b[:, 0], b[:, 1])
        marker.set_data([data["time"][i]], [data["centroid_z"][i] / a])
        title.set_text(
            f"Tokamak passive stabilization benchmark | t = {data['time'][i]:.1f}"
        )
        return line, marker, title

    update(0)
    fig.savefig(out / "tokamak_confined.png", dpi=150)
    if movie:
        import imageio_ffmpeg

        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        animation = FuncAnimation(
            fig,
            update,
            frames=np.linspace(0, len(data["time"]) - 1, 160).astype(int),
            blit=False,
        )
        animation.save(
            out / "tokamak_confined.mp4",
            writer=FFMpegWriter(
                fps=20, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]
            ),
            dpi=100,
        )
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("build/tokamak_confined"))
    ap.add_argument("--no-movie", action="store_true")
    args = ap.parse_args()
    render(args.out, not args.no_movie)
