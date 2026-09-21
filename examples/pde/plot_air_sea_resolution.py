"""Plot recorded space/time differences and the finest 24-hour snapshot."""

import argparse
import json
from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bspf_models.air_sea.air_sea import AirSeaState
from bspf_models.air_sea.air_sea import plan_air_sea
from validate_air_sea_resolution import diagnostics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path("build/air_sea_resolution"))
    args = ap.parse_args()
    report = json.loads((args.run / "resolution.json").read_text())
    quantities = [("ocean_velocity", "Ocean velocity"),
                  ("sst_gradient", "SST gradient"),
                  ("air_temperature_gradient", "Air temperature gradient"),
                  ("humidity_gradient", "Humidity gradient"),
                  ("stress", "Interface stress"),
                  ("total_heat", "Upward total heat flux")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, quantities):
        for hour in ("6", "12", "24"):
            row = report["spatial"][hour]
            values = [100 * row[pair][key]["relative_rms"]
                      for pair in ("33_49", "49_65", "65_81")]
            ax.semilogy([49, 65, 81], values, "o-", label=f"{hour} h")
        temporal = report["temporal"]["24"]["81"]["300_150"][key]["rms"]
        scale = report["spatial"]["24"]["65_81"][key]["reference_rms"]
        ax.axhline(100 * temporal / max(scale, 1e-30), color="gray", ls=":",
                   label="Time difference at n=81, 24 h")
        ax.axhline(1, color="black", ls="--", lw=.8, label="1% RMS screen")
        ax.set(title=title, xlabel="Finer n of each adjacent grid pair",
               ylabel="Relative RMS difference (%)", xticks=[49, 65, 81])
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(args.run / "resolution.png", dpi=170)
    plt.close(fig)

    jax.config.update("jax_enable_x64", True)
    p = plan_air_sea(n=81, quadrature_order=32)
    data = np.load(args.run / "n81_H150.npz")
    state = AirSeaState(*(data[f"2_{key}"] for key in AirSeaState._fields))
    f = diagnostics(p, state)
    coordinate = np.asarray(p.scalar.points) * p.config.length / 1000
    panels = [(f["ocean_streamfunction"], "Ocean streamfunction (m²/s)", "RdBu_r"),
              (np.linalg.norm(f["sst_gradient"], axis=-1) * 1e5,
               "SST gradient (K / 100 km)", "magma"),
              (np.linalg.norm(f["air_temperature_gradient"], axis=-1) * 1e5,
               "Air temperature gradient (K / 100 km)", "magma"),
              (np.linalg.norm(f["humidity_gradient"], axis=-1) * 1e8,
               "Humidity gradient (g/kg / 100 km)", "magma"),
              (np.linalg.norm(f["stress"], axis=-1), "Stress magnitude (Pa)", "viridis"),
              (f["total_heat"], "Upward total heat flux (W/m²)", "viridis")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for ax, (value, title, cmap) in zip(axes.flat, panels):
        pc = ax.pcolormesh(coordinate, coordinate, value.T, shading="auto", cmap=cmap,
                           rasterized=True)
        fig.colorbar(pc, ax=ax, shrink=.8)
        ax.set(title=title, xlabel="x (km)", ylabel="y (km)", aspect="equal")
    fig.suptitle("Actual 24 h transient — n=81, H=150 s (finite-resolution result)")
    fig.savefig(args.run / "fields_24h.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
