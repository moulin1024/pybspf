"""Display velocity only in plasma; display magnetic response in the vacuum."""

import argparse
import json
import pickle
from pathlib import Path
import jax
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.interpolate import CubicSpline


def render(folder, movie=True):
    jax.config.update("jax_enable_x64", True)
    # Only read locally generated, trusted caches.
    model = pickle.loads((folder / "model.pkl").read_bytes())
    data = np.load(folder / "evolution.npz")
    summary = json.loads((folder / "summary.json").read_text())
    rr, zz = np.meshgrid(
        np.linspace(1, 3, 101), np.linspace(-1.6, 1.6, 151), indexing="ij"
    )
    points = np.column_stack((rr.ravel(), zz.ravel()))
    theta = np.mod(np.arctan2(points[:, 1], points[:, 0] - 2), 2 * np.pi)
    radius = CubicSpline(
        np.r_[model.theta, 2 * np.pi],
        np.r_[model.radius, model.radius[0]],
        bc_type="periodic",
    )(theta)
    core = np.linalg.norm(points - [2, 0], axis=1) < radius
    plasma_points = points[core]
    fields = model.basis.evaluate(plasma_points)
    operators = [fields[k] @ model.transform for k in ("xr", "xz", "xp")]
    velocities = np.array(
        [
            np.column_stack([a @ v for a in operators])
            for v in data["velocity_coefficients"]
        ]
    )
    # NaN is deliberate: velocity is undefined in vacuum, not numerically zero.
    uz = np.full((len(data["time"]), len(points)), np.nan)
    uz[:, core] = velocities[:, :, 1]
    uz = uz.reshape(len(data["time"]), *rr.shape)
    np.savez_compressed(
        folder / "display_fields.npz",
        R=rr[:, 0],
        Z=zz[0],
        plasma_mask=core.reshape(rr.shape),
        time=data["time"],
        uZ=uz,
        velocity_plasma=velocities,
        plasma_points=plasma_points,
        note="Eulerian linear fields on equilibrium plasma; vacuum velocity is undefined (NaN)",
    )
    tri = mtri.Triangulation(
        model.vacuum.points[:, 0], model.vacuum.points[:, 1], model.vacuum.triangles
    )
    limit = max(abs(uz).max(where=np.isfinite(uz), initial=0), 1e-15)
    fluxlim = np.max(abs(data["vacuum_flux"]))
    fig = plt.figure(figsize=(13.6, 7.2), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 1.05))
    axv = fig.add_subplot(grid[:, 0])
    axp = fig.add_subplot(grid[:, 1])
    axg = fig.add_subplot(grid[0, 2])
    axt = fig.add_subplot(grid[1, 2])
    title = fig.suptitle("")
    vacuum = axv.tripcolor(
        tri,
        data["vacuum_flux"][-1],
        shading="gouraud",
        cmap="RdBu_r",
        vmin=-fluxlim,
        vmax=fluxlim,
    )
    fig.colorbar(vacuum, ax=axv, shrink=0.7, label=r"Vacuum $\delta\psi$")
    image = axp.pcolormesh(
        rr, zz, uz[-1], cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto"
    )
    fig.colorbar(image, ax=axp, shrink=0.7, label=r"Plasma $u_Z$")
    boundaries = []
    initial = np.vstack((model.boundary, model.boundary[0]))
    for ax in (axv, axp):
        ax.plot(
            initial[:, 0],
            initial[:, 1],
            "--",
            color=".4",
            lw=1,
            label="Equilibrium interface",
        )
        (line,) = ax.plot(
            initial[:, 0],
            initial[:, 1],
            color="black",
            lw=1.8,
            label="Displaced interface",
        )
        boundaries.append(line)
        ax.set(xlim=(1, 3), ylim=(-1.6, 1.6), xlabel="R", ylabel="Z", aspect="equal")
        ax.set_facecolor("#f1f1f1")
    axv.plot([1, 3, 3, 1, 1], [-1.6, -1.6, 1.6, 1.6, -1.6], color="#795548", lw=2)
    axv.set_title(
        "Vacuum magnetic response\nFixed-flux rectangular outer wall", fontsize=11
    )
    axv.legend(loc="upper left", fontsize=8)
    axp.set_title(
        "Velocity exists only in plasma\nGrey exterior: vacuum, no fluid", fontsize=11
    )
    # Use a sparse, evenly spaced physical grid for arrows.
    sparse = np.zeros(rr.shape, dtype=bool)
    sparse[::8, ::8] = True
    select = np.flatnonzero(sparse.ravel()[core])
    u = velocities[-1]
    quiver = axp.quiver(
        plasma_points[select, 0],
        plasma_points[select, 1],
        u[select, 0],
        u[select, 1],
        color="#333333",
        scale=limit * 8,
        width=0.005,
    )
    times = data["time"]
    centroid = abs(data["centroid_z"]) / summary["minor_radius"]
    axg.semilogy(times, centroid, color="#1565c0", label="Time-integrated displacement")
    axg.semilogy(
        times,
        centroid[0] * np.exp(summary["gamma"] * times),
        "--",
        color="#e67e22",
        label="Eigenvalue prediction",
    )
    (marker,) = axg.plot([times[-1]], [centroid[-1]], "o", color="black")
    axg.set(
        xlabel="Time",
        ylabel=r"$|\langle\xi_Z\rangle|/a$",
        title="Linear vertical instability",
    )
    axg.grid(alpha=0.2)
    axg.legend(fontsize=8)
    axt.axis("off")
    axt.text(
        0,
        1,
        (
            "BSPF plasma + mapped BSPF vacuum\n\n"
            if summary.get("vacuum_method") == "mapped_bspf"
            else "BSPF plasma + fitted FEM vacuum\n\n"
        )
        + r"$\Delta^*\delta\psi_v=0$ in vacuum"
        + "\n"
        r"$\delta\psi_v=-\boldsymbol{\xi}\cdot\nabla\psi_0$ at interface" + "\n\n"
        f"Growth rate: {summary['gamma']:.6f}\n"
        f"Vacuum fluid DOFs: {summary['exterior_velocity_dofs']}\n"
        f"Final max displacement: {summary['max_final_displacement_over_minor_radius']:.2%} of a\n\n"
        "Fixed external coil currents; no lid driving.\n"
        "Axisymmetric, ideal, incompressible, linear.\n"
        "Fields evaluated on equilibrium domains.\n"
        "Interface displacement is not magnified.\n"
        "No wall contact or resistive vessel.",
        va="top",
        fontsize=9,
        linespacing=1.5,
    )

    def update(i):
        vacuum.set_array(data["vacuum_flux"][i])
        image.set_array(uz[i].ravel())
        for line in boundaries:
            b = data["boundary"][i]
            b = np.vstack((b, b[0]))
            line.set_data(b[:, 0], b[:, 1])
        quiver.set_UVC(velocities[i, select, 0], velocities[i, select, 1])
        marker.set_data([times[i]], [centroid[i]])
        title.set_text(f"Plasma–vacuum free-boundary mode | t = {times[i]:.2f}")
        return vacuum, image, quiver, marker, *boundaries, title

    update(len(times) - 1)
    fig.savefig(folder / "tokamak_vacuum.png", dpi=150)
    if movie:
        import imageio_ffmpeg

        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        animation = FuncAnimation(
            fig,
            update,
            frames=np.linspace(0, len(times) - 1, 120).astype(int),
            blit=False,
        )
        animation.save(
            folder / "tokamak_vacuum.mp4",
            writer=FFMpegWriter(
                fps=20, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]
            ),
            dpi=100,
        )
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("build/tokamak_vacuum"))
    ap.add_argument("--no-movie", action="store_true")
    args = ap.parse_args()
    render(args.out, not args.no_movie)
