"""Render time-advanced BSPF linear tokamak states without geometric magnification."""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import label, binary_dilation
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("build/tokamak_axisymmetric_n49"),
    )
    ap.add_argument("--no-movie", action="store_true")
    args = ap.parse_args()
    root = args.directory
    s = json.loads((root / "summary.json").read_text())
    d = np.load(root / "evolution.npz")
    r, z, t = d["R"], d["Z"], d["time"]
    psi0 = d["psi0"]
    flux = d["delta_psi"]
    u = d["velocity"]
    a = s["minor_radius"]
    gamma = s["gamma_eigenvalue"]
    coils = d["coils"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(
        2, 2, figsize=(12.8, 9), gridspec_kw={"height_ratios": [1.45, 1]}
    )
    fig.subplots_adjust(
        left=0.075, right=0.945, top=0.84, bottom=0.11, hspace=0.35, wspace=0.30
    )
    fig.suptitle(
        "Axisymmetric tokamak: linear vertical instability",
        fontsize=20,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.075,
        0.913,
        f"BSPF {s['n']} × {s['n']}  |  elongation ≈ {s['elongation']:.2f}  |  fixed coils  |  gamma = {gamma:.5f}",
    )
    clock = fig.text(0.945, 0.913, "", ha="right", fontweight="bold")
    fig.text(
        0.075,
        0.875,
        "Incompressible full-vector n=0 model; finite-density resistive exterior; fixed outer magnetic boundary",
        fontsize=10,
    )
    ax, fx = axes[0]
    ax.add_patch(
        Rectangle(
            (r[0], z[0]), r[-1] - r[0], z[-1] - z[0], fill=False, lw=1.4, ec="#505050"
        )
    )
    ax.scatter(
        coils[:, 0],
        coils[:, 1],
        s=45,
        c=np.where(coils[:, 2] > 0, "#cf493f", "#3676b6"),
        marker="s",
        zorder=4,
    )
    labels, _ = label(psi0 > 0)
    core = labels == labels[np.argmin(abs(r - 2.0)), np.argmin(abs(z))]
    contour_window = binary_dilation(core, iterations=3)

    def plasma_flux(values):
        # Other psi=0 branches in the vacuum are NOT the plasma boundary.
        return np.ma.array(values.T, mask=~contour_window.T)

    levels = np.linspace(0, psi0[core].max(), 12)[:-1]
    contours = ax.contour(
        r, z, plasma_flux(psi0), levels=levels, colors="#333333", linewidths=0.7
    )
    ax.contour(
        r,
        z,
        plasma_flux(psi0),
        levels=[0],
        colors="#b9b9b9",
        linestyles="--",
        linewidths=2,
    )
    edge = ax.contour(
        r, z, plasma_flux(psi0 + flux[0]), levels=[0], colors="#ed782c", linewidths=1.8
    )
    ax.set(
        xlim=(0.4, 3.7),
        ylim=(-2.2, 2.2),
        xlabel="R",
        ylabel="Z",
        aspect="equal",
        title="Fixed coils, wall and free plasma contour",
    )
    limit = np.max(abs(flux))
    image = fx.pcolormesh(
        r, z, flux[0].T, shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    fx.contour(r, z, plasma_flux(psi0), levels=[0], colors="#222222", linewidths=1)
    fig.colorbar(image, ax=fx, pad=0.02, label="poloidal flux perturbation")
    fx.set(
        xlim=(r[0], r[-1]),
        ylim=(z[0], z[-1]),
        xlabel="R",
        ylabel="Z",
        aspect="equal",
        title="Evolving magnetic perturbation",
    )
    history, flow = axes[1]
    history.semilogy(
        t,
        abs(d["centroid_z"]) / a,
        color="#cf493f",
        lw=2,
        label="Implicit time advance",
    )
    history.semilogy(
        t,
        abs(d["centroid_z"][0]) / a * np.exp(gamma * t),
        "--",
        color="#303030",
        lw=1,
        label="Eigenvalue prediction",
    )
    history.set(
        xlabel="normalized time",
        ylabel="|vertical displacement| / a",
        title="Small perturbation grows without applied motion",
        xlim=(0, t[-1]),
    )
    history.legend(fontsize=8, frameon=False)
    history.grid(alpha=0.2)
    cursor = history.axvline(0, color="gray", lw=1)
    umax = np.max(abs(u[..., 1]))
    ui = flow.pcolormesh(
        r, z, u[0, ..., 1].T, shading="auto", cmap="RdBu_r", vmin=-umax, vmax=umax
    )
    flow.contour(r, z, plasma_flux(psi0), levels=[0], colors="#222222", linewidths=1)
    skip = max(1, len(r) // 15)
    ar = flow.quiver(
        r[::skip],
        z[::skip],
        u[0, ::skip, ::skip, 0].T,
        u[0, ::skip, ::skip, 1].T,
        color="#333333",
        scale=20 * np.max(np.linalg.norm(u[..., :2], axis=-1)),
        width=0.003,
    )
    fig.colorbar(ui, ax=flow, pad=0.02, label="vertical velocity")
    flow.set(
        xlim=(1.4, 2.95),
        ylim=(-1.15, 1.15),
        xlabel="R",
        ylabel="Z",
        aspect="equal",
        title="Internal poloidal flow and deformation",
    )
    footer = fig.text(0.075, 0.035, "", fontsize=9)

    def draw(i):
        nonlocal edge, contours
        edge.remove()
        contours.remove()
        contours = ax.contour(
            r,
            z,
            plasma_flux(psi0 + flux[i]),
            levels=levels,
            colors="#333333",
            linewidths=0.7,
        )
        edge = ax.contour(
            r,
            z,
            plasma_flux(psi0 + flux[i]),
            levels=[0],
            colors="#ed782c",
            linewidths=1.8,
        )
        image.set_array(flux[i].T.ravel())
        ui.set_array(u[i, ..., 1].T.ravel())
        ar.set_UVC(u[i, ::skip, ::skip, 0].T, u[i, ::skip, ::skip, 1].T)
        cursor.set_xdata([t[i], t[i]])
        clock.set_text(f"t = {t[i]:.2f}")
        footer.set_text(
            f"Linear regime only  |  centroid displacement / a = {d['centroid_z'][i] / a:.4f}  |  actual contour (no magnification); fixed color scales"
        )

    draw(len(t) - 1)
    fig.savefig(root / "tokamak_vertical.png", dpi=150)
    if not args.no_movie:
        import imageio_ffmpeg

        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = FFMpegWriter(
            fps=20,
            codec="libx264",
            extra_args=["-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        with writer.saving(fig, str(root / "tokamak_vertical.mp4"), dpi=100):
            for i in range(len(t)):
                draw(i)
                writer.grab_frame()
    plt.close(fig)
    print(root / "tokamak_vertical.mp4")


if __name__ == "__main__":
    main()
