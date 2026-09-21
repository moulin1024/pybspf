"""Analytic-vorticity movie and eddy-energy diagnostics of a saved shear run.

One movie frame per saved positive time; no temporal interpolation.
"""

import argparse
import json
from pathlib import Path
import jax
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from netCDF4 import Dataset
from bspf_sim.air_sea.platform import read_checkpoint
from bspf_sim.air_sea.platform import _plan
from bspf_models.fluids.stream_navier_stokes import stream_ns_velocity
from bspf_models.fluids.stream_navier_stokes import stream_ns_vorticity
from render_air_sea_mp4 import working_binary


def render(source, out, until_hours=None, fps=8):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_enable_x64", True)
    paths = sorted((source / "checkpoints").glob("step*.json"))
    if until_hours is not None:
        paths = [
            p
            for p in paths
            if json.loads(p.read_text())["time_seconds"] <= until_hours * 3600
        ]
    paths = [p for p in paths if (source / "fields" / (p.stem + ".nc")).exists()]
    if len(paths) < 2:
        raise ValueError("Need at least two saved field/checkpoint pairs")
    if (
        until_hours is not None
        and json.loads(paths[-1].read_text())["time_seconds"] != until_hours * 3600
    ):
        raise ValueError("Requested end time is not present in saved frames")
    first = read_checkpoint(paths[0])
    p = _plan(first.config)
    c = p.config
    frames, stats = [], []
    for path in paths:
        ck = read_checkpoint(path)
        s = ck.state
        u = np.asarray(stream_ns_velocity(p.air_flow, s.atmosphere))
        mean = np.einsum("i,ijc->jc", np.asarray(p.air_flow.x.weights), u)
        eddy = u - mean[None, :, :]
        omega = np.asarray(stream_ns_vorticity(p.air_flow, s.atmosphere)) / c.length
        stats.append(
            {
                "time_hours": ck.time_seconds / 3600,
                "air_eddy_energy_J_m2": float(
                    0.5
                    * c.air_mass
                    * np.sum(np.asarray(p.weight) * np.sum(eddy**2, axis=-1))
                ),
                "air_enstrophy_s_2": float(
                    0.5 * np.sum(np.asarray(p.weight) * omega**2)
                ),
            }
        )
        with Dataset(source / "fields" / f"step{ck.step:09d}.nc") as ds:
            frames.append(
                [
                    np.asarray(
                        stream_ns_vorticity(p.air_flow, s.atmosphere, nodes=True)
                    )
                    / c.length
                    * 1e5,
                    np.asarray(ds["humidity"][:]) * 1000,
                    np.asarray(ds["sst"][:]) - 273.15,
                    np.asarray(ds["total_heat"][:]),
                    np.asarray(ds["ocean_streamfunction"][:]),
                    np.asarray(ds["air_temperature"][:]) - 273.15,
                ]
            )
    (out / "eddy_diagnostics.json").write_text(json.dumps(stats, indent=2))
    data = np.asarray(frames)
    titles = [
        "Atmospheric vorticity (10⁻⁵ s⁻¹)",
        "Specific humidity (g/kg)",
        "Ocean temperature (°C)",
        "Upward total heat flux (W/m²)",
        "Ocean streamfunction (m²/s)",
        "Air temperature (°C)",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    x = np.asarray(p.scalar.x) * c.length / 1000
    meshes = []
    for k, ax in enumerate(axes.flat):
        lo, hi = data[:, k].min(), data[:, k].max()
        if k in (0, 4):
            hi = max(abs(lo), abs(hi))
            lo = -hi
        mesh = ax.pcolormesh(
            x,
            x,
            data[0, k].T,
            shading="auto",
            cmap="RdBu_r" if k in (0, 4) else "viridis",
            vmin=lo,
            vmax=hi,
        )
        meshes.append(mesh)
        fig.colorbar(mesh, ax=ax)
        ax.set(title=titles[k], xlabel="x (km)", ylabel="y (km)")
    title = fig.suptitle("")

    def draw(i):
        for k, mesh in enumerate(meshes):
            mesh.set_array(data[i, k].T.ravel())
        title.set_text(
            f"2D coupled shear instability · {stats[i]['time_hours']:.1f} h · EKE {stats[i]['air_eddy_energy_J_m2']:.0f} J/m²"
        )

    draw(0)
    fig.savefig(out / "initial.png", dpi=140)
    draw(len(frames) - 1)
    fig.savefig(out / "final.png", dpi=140)
    matplotlib.rcParams["animation.ffmpeg_path"] = working_binary("ffmpeg")
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(out / "shear_instability.mp4"), 100):
        for i in range(1, len(frames)):
            draw(i)
            writer.grab_frame()
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(
        [r["time_hours"] for r in stats], [r["air_eddy_energy_J_m2"] for r in stats]
    )
    ax.set(
        xlabel="Time (hours)",
        ylabel="Nonzonal kinetic energy (J/m²)",
        title="Growth and nonlinear evolution of the seeded disturbance",
    )
    fig.savefig(out / "eddy_energy.png", dpi=150)
    plt.close(fig)
    print(
        json.dumps(
            {"frames": len(frames) - 1, "initial": stats[0], "final": stats[-1]},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--until-hours", type=float)
    parser.add_argument("--fps", type=float, default=8)
    args = parser.parse_args()
    render(args.run, args.out, args.until_hours, args.fps)
