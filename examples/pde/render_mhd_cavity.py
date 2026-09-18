"""Render computed MHD states; fixed color scales throughout the animation."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path, nargs="?", default=Path("build/mhd_cavity"))
    ap.add_argument("--no-movie", action="store_true")
    args = ap.parse_args()
    d = args.directory
    z = np.load(d / "evolution.npz")
    s = json.loads((d / "summary.json").read_text())
    ts, x, vel, flux, current = [
        z[k] for k in ("time", "x", "velocity", "flux", "current")
    ]
    speed = np.linalg.norm(vel, axis=-1)
    h = s["history"]
    kinetic = np.array([r["kinetic_energy"] for r in h])
    magnetic = np.array([r["magnetic_energy"] for r in h])
    center = np.array([r["center_max_speed"] for r in h])
    control = np.array([r["control_center_max_speed"] for r in h])
    peak = int(np.argmax([r["center_max_velocity_change"] for r in h]))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(12.8, 8), gridspec_kw={"height_ratios": [1.5, 1]}
    )
    fig.subplots_adjust(
        left=0.08, right=0.95, bottom=0.1, top=0.85, hspace=0.38, wspace=0.3
    )
    fig.suptitle(
        "Magnetic energy release inside a cavity",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.08,
        0.92,
        f"BSPF incompressible MHD  |  Re={s['reynolds']:g}, Rm={s['magnetic_reynolds']:g}  |  {s['n']} × {s['n']} nodes  |  initial max |B|={s['peak_initial_field']:g}",
    )
    clock = fig.text(0.95, 0.92, "", ha="right", fontweight="bold")
    a, b = axes[0]
    im = a.pcolormesh(
        x, x, speed[0].T, shading="auto", cmap="magma", vmin=0, vmax=speed.max()
    )
    fig.colorbar(im, ax=a, pad=0.02, label="speed / lid peak speed")
    skip = max(1, len(x) // 16)
    arrow = a.quiver(
        x[::skip],
        x[::skip],
        vel[0, ::skip, ::skip, 0].T,
        vel[0, ::skip, ::skip, 1].T,
        color="white",
        scale=max(4, 12 * speed.max()),
        width=0.003,
    )
    a.set(
        title="Fluid speed and velocity vectors", xlabel="x", ylabel="y", aspect="equal"
    )
    lim = np.max(abs(current))
    jm = b.pcolormesh(
        x, x, current[0].T, shading="auto", cmap="RdBu_r", vmin=-lim, vmax=lim
    )
    fig.colorbar(jm, ax=b, pad=0.02, label="out-of-plane current j")
    levels = np.linspace(0, flux.max(), 17)[1:-1]
    contour = b.contour(
        x, x, flux[0].T, levels=levels, colors="#262626", linewidths=0.7
    )
    b.set(
        title="Current and magnetic flux contours",
        xlabel="x",
        ylabel="y",
        aspect="equal",
    )
    e, v = axes[1]
    e.plot(ts, magnetic, color="#3970b5", lw=2, label="Magnetic energy")
    e.plot(ts, kinetic, color="#d35732", lw=2, label="Kinetic energy")
    e.plot(
        ts, kinetic + magnetic, color="#737373", ls="--", lw=1.3, label="Total energy"
    )
    e.set(
        xlabel="time since magnetic release",
        ylabel="energy",
        title="Magnetic energy conversion and dissipation",
        xlim=(ts[0], ts[-1]),
    )
    e.legend(fontsize=8, frameon=False)
    v.plot(ts, center, color="#d35732", lw=2, label="MHD")
    v.plot(
        ts,
        control,
        color="#3970b5",
        lw=2,
        ls="--",
        label="NS control: no magnetic field",
    )
    v.set(
        xlabel="time since magnetic release",
        ylabel="max speed in central square",
        title="Central region: [0.25, 0.75]²",
        xlim=(ts[0], ts[-1]),
    )
    v.legend(fontsize=8, frameon=False)
    cursor1 = e.axvline(0, color="#202020", lw=1, alpha=0.6)
    cursor2 = v.axvline(0, color="#202020", lw=1, alpha=0.6)
    for ax in (e, v):
        ax.grid(alpha=0.15)
    footer = fig.text(0.08, 0.025, "", fontsize=9)

    def draw(i):
        nonlocal contour
        im.set_array(speed[i].T.ravel())
        jm.set_array(current[i].T.ravel())
        arrow.set_UVC(vel[i, ::skip, ::skip, 0].T, vel[i, ::skip, ::skip, 1].T)
        contour.remove()
        contour = b.contour(
            x, x, flux[i].T, levels=levels, colors="#262626", linewidths=0.7
        )
        cursor1.set_xdata([ts[i], ts[i]])
        cursor2.set_xdata([ts[i], ts[i]])
        clock.set_text(f"t = {ts[i]:.3f}")
        footer.set_text(
            f"Computed fields; fixed color scales  |  max div u = {h[i]['div_u_linf']:.1e}, max div B = {h[i]['div_b_linf']:.1e}  |  conducting, nonperiodic walls"
        )

    draw(peak)
    fig.savefig(d / "mhd_cavity.png", dpi=150)
    if not args.no_movie:
        import imageio_ffmpeg

        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = FFMpegWriter(
            fps=20,
            codec="libx264",
            extra_args=["-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        with writer.saving(fig, str(d / "mhd_cavity.mp4"), dpi=100):
            for i in range(len(ts)):
                draw(i)
                writer.grab_frame()
    plt.close(fig)
    print(f"Rendered {d}, snapshot at t={ts[peak]:g}")


if __name__ == "__main__":
    main()
