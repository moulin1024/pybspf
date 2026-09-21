"""Render simulated air-sea snapshots as a fixed-scale, six-panel MP4.

First run air_sea_double_gyre.py with --output-hours 0.5 for 48 positive-time
snapshots through 24 hours. No temporal interpolation is used by this renderer.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

from bspf_models.air_sea.air_sea import AirSeaConfig
from bspf_models.air_sea.air_sea import bulk_flux


def working_binary(name):
    """Skip broken binaries earlier on PATH (e.g. an unusable Conda ffmpeg)."""
    checked = set()
    for directory in os.get_exec_path():
        candidate = shutil.which(name, path=directory)
        if candidate is None or candidate in checked:
            continue
        checked.add(candidate)
        try:
            result = subprocess.run(
                [candidate, "-version"], capture_output=True, timeout=10, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return candidate
    raise RuntimeError(f"No working {name} found on PATH")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=Path("build/air_sea_double_gyre_48frames")
    )
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.frames < 2 or args.fps < 1:
        parser.error("frames must be >=2 and fps >=1")
    ffmpeg, ffprobe = working_binary("ffmpeg"), working_binary("ffprobe")
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg
    if (args.run / "fields").is_dir():
        from netCDF4 import Dataset

        configuration = json.loads((args.run / "config.json").read_text())
        config = AirSeaConfig(**configuration["physics"])
        paths, stamps = [], []
        for path in sorted((args.run / "fields").glob("step*.nc")):
            with Dataset(path) as ds:
                if ds.time_seconds > 0:
                    paths.append(path)
                    stamps.append(ds.time_seconds)
        if len(paths) < args.frames:
            parser.error(f"Only {len(paths)} positive-time NetCDF frames")
        indices = np.rint(np.linspace(0, len(paths) - 1, args.frames)).astype(int)
        times = np.asarray(stamps)[indices]
        keys = (
            "ocean_streamfunction",
            "ocean_velocity",
            "air_velocity",
            "sst",
            "air_temperature",
            "humidity",
            "total_heat",
        )
        frames = {k: [] for k in keys}
        for i in indices:
            with Dataset(paths[i]) as ds:
                x = np.asarray(ds["x"][:]) / 1000
                for k in keys:
                    frames[k].append(np.asarray(ds[k][:]))
        frames = {k: np.stack(v) for k, v in frames.items()}
        heat_flux = frames["total_heat"]
        summary = {"hours": float(times[-1] / 3600)}
        if not all(np.all(np.isfinite(v)) for v in frames.values()):
            parser.error("Nonfinite snapshots")
    else:
        summary = json.loads((args.run / "summary.json").read_text())
        config = AirSeaConfig(**summary["config"])
        with np.load(args.run / "snapshots.npz") as archive:
            times = archive["time_seconds"]
            available = np.flatnonzero(times > 0)
            if len(available) < args.frames:
                parser.error(
                    f"Only {len(available)} simulated positive-time snapshots; "
                    "rerun the solver with a shorter --output-hours interval"
                )
            if np.any(np.diff(times) <= 0):
                parser.error("Snapshot times must be strictly increasing")
            indices = available[
                np.rint(np.linspace(0, len(available) - 1, args.frames)).astype(int)
            ]
            frames = {
                key: archive[key][indices]
                for key in archive.files
                if key != "time_seconds"
            }
        times = times[indices]
        if not all(np.all(np.isfinite(v)) for v in frames.values()):
            parser.error("Nonfinite snapshots")
        with np.load(args.run / "state.npz") as state:
            x = state["x_m"] / 1000
        jax.config.update("jax_enable_x64", True)
        _, sensible, water = bulk_flux(
            config,
            frames["ocean_velocity"],
            frames["air_velocity"],
            frames["sst"],
            frames["air_temperature"],
            frames["humidity"],
        )
        heat_flux = np.asarray(sensible + config.latent_heat * water)
    panels = [
        (
            frames["ocean_streamfunction"],
            "Ocean circulation",
            "Streamfunction (m²/s)",
            "RdBu_r",
            True,
        ),
        (
            frames["air_velocity"][..., 0],
            "Atmospheric wind",
            "Zonal velocity (m/s)",
            "RdBu_r",
            True,
        ),
        (frames["sst"] - 273.15, "Sea-surface temperature", "°C", "coolwarm", False),
        (
            frames["air_temperature"] - 273.15,
            "Atmospheric temperature",
            "°C",
            "coolwarm",
            False,
        ),
        (
            1000 * frames["humidity"],
            "Atmospheric moisture",
            "Specific humidity (g/kg)",
            "YlGnBu",
            False,
        ),
        (
            heat_flux,
            "Upward ocean heat loss",
            "Sensible + latent flux (W/m²)",
            "magma",
            False,
        ),
    ]
    plt.rcParams.update({"font.size": 11, "axes.titleweight": "semibold"})
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=100)
    fig.subplots_adjust(
        left=0.055, right=0.965, bottom=0.085, top=0.865, wspace=0.34, hspace=0.29
    )
    fig.text(
        0.055,
        0.96,
        "AIR–SEA / DOUBLE GYRE",
        fontsize=21,
        weight="bold",
        color="#15324c",
    )
    fig.text(
        0.055,
        0.926,
        "Horizontal 2D · Vertically averaged ocean and atmosphere",
        fontsize=12,
        color="#526475",
    )
    clock = fig.text(
        0.965, 0.96, "", fontsize=18, ha="right", weight="bold", color="#15324c"
    )
    counter = fig.text(0.965, 0.926, "", fontsize=11, ha="right", color="#526475")
    fig.text(
        0.055,
        0.027,
        "Ocean: closed basin  |  Atmosphere: x periodic, y free-slip  |  "
        "Fixed color scales and arrow scales  |  Startup transient",
        fontsize=10,
        color="#526475",
    )
    maps, scales = [], []
    for ax, (data, title, unit, cmap, symmetric) in zip(axes.flat, panels):
        lo, hi = float(np.min(data)), float(np.max(data))
        if symmetric:
            hi = max(abs(lo), abs(hi), 1e-12)
            lo = -hi
        if hi - lo < 1e-12:
            hi = lo + 1e-12
        mesh = ax.pcolormesh(
            x,
            x,
            data[0].T,
            shading="nearest",
            cmap=cmap,
            vmin=lo,
            vmax=hi,
            rasterized=True,
        )
        ax.set(title=title, xlabel="East x (km)", ylabel="North y (km)", aspect="equal")
        ax.tick_params(labelsize=9)
        ax.set_xticks(np.linspace(x[0], x[-1], 5))
        ax.set_yticks(np.linspace(x[0], x[-1], 5))
        colorbar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.025)
        colorbar.set_label(unit, fontsize=10)
        colorbar.ax.tick_params(labelsize=9)
        maps.append(mesh)
        scales.append(dict(panel=title, minimum=lo, maximum=hi, units=unit))
    skip = max(1, (len(x) - 1) // 8)
    position = x[::skip]
    quivers = []
    for ax, key in zip(axes.flat[:2], ("ocean_velocity", "air_velocity")):
        data = frames[key]
        scale = max(float(np.linalg.norm(data, axis=-1).max()) * 12, 1e-9)
        quiver = ax.quiver(
            position,
            position,
            data[0, ::skip, ::skip, 0].T,
            data[0, ::skip, ::skip, 1].T,
            color="#182c3d",
            angles="xy",
            scale_units="width",
            scale=scale,
            width=0.004,
            headwidth=3.5,
            alpha=0.8,
        )
        quivers.append((quiver, data))

    def draw(k):
        for mesh, panel in zip(maps, panels):
            mesh.set_array(panel[0][k].T.ravel())
        for quiver, data in quivers:
            quiver.set_UVC(data[k, ::skip, ::skip, 0].T, data[k, ::skip, ::skip, 1].T)
        hours, minutes = divmod(round(float(times[k]) / 60), 60)
        clock.set_text(f"{hours:02d}:{minutes:02d} / {summary['hours']:g} h")
        counter.set_text(f"Frame {k + 1:02d} / {args.frames}  ·  Simulated snapshots")

    output = (
        args.out or args.run / f"air_sea_{summary['hours']:g}h_{args.frames}frames.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        extra_args=["-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={
            "title": "24-hour horizontal air-sea double-gyre coupling",
            "comment": "Simulated snapshots; fixed scales; no temporal interpolation",
        },
    )
    with writer.saving(fig, str(output), dpi=100):
        for k in range(args.frames):
            draw(k)
            writer.grab_frame(facecolor="white")
            if k in (0, args.frames - 1):
                fig.savefig(
                    output.with_name(
                        f"{output.stem}_{'first' if k == 0 else 'last'}.png"
                    ),
                    dpi=100,
                )
            if (k + 1) % 12 == 0:
                print(f"Rendered {k + 1}/{args.frames} frames", flush=True)
    plt.close(fig)
    probe = json.loads(
        subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration",
                "-of",
                "json",
                str(output),
            ],
            text=True,
        )
    )
    stream = probe["streams"][0]
    if int(stream["nb_read_frames"]) != args.frames:
        raise RuntimeError(f"Encoded frame count mismatch: {stream}")
    report = dict(
        video=str(output.resolve()),
        source=str(args.run.resolve()),
        source_indices=indices.tolist(),
        frame_times_seconds=times.tolist(),
        frames=args.frames,
        fps=args.fps,
        playback_seconds=args.frames / args.fps,
        temporal_interpolation=False,
        color_scales=scales,
        ffprobe=stream,
    )
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["ffprobe"], indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
