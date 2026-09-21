"""Build separate read-only evidence reports from research run directories."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

from bspf_jax.air_sea_platform import report
from bspf_jax.air_sea_audit import PROCESSES, QUANTITIES


def create_report(run_directory, output):
    import jax

    jax.config.update("jax_enable_x64", True)
    source, output = Path(run_directory).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError(
            "Report output must be outside the immutable experiment directory"
        )
    output.mkdir(parents=True, exist_ok=False)
    summary = report(source)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    rows = (
        [json.loads(x) for x in (source / "diagnostics.jsonl").read_text().splitlines()]
        if (source / "diagnostics.jsonl").exists()
        else []
    )
    figures = []
    if rows:
        t = np.array([r["time_seconds"] for r in rows]) / 86400
        fig, ax = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for a, key, title in zip(
            ax.flat,
            (
                "heat_residual",
                "water_residual",
                "kinetic_residual",
                "spectral_tails",
                "max_RH",
                "internal_heat_leak",
            ),
            (
                "Heat residual (J/m²)",
                "Water residual (kg/m²)",
                "Kinetic residual (J/m²)",
                "High-mode energy fraction",
                "Maximum RH",
                "Internal heat leakage (J/m²)",
            ),
        ):
            a.plot(t, [r[key] for r in rows])
            a.set(title=title, xlabel="Simulated days")
            a.grid(alpha=0.2)
            if key == "kinetic_residual":
                a.legend(("ocean", "atmosphere"))
            if key == "spectral_tails":
                a.legend(("ocean", "atmosphere", "SST", "Ta", "q"), fontsize=8)
        fig.savefig(output / "budgets.png", dpi=150)
        plt.close(fig)
        figures.append("budgets.png")
        fig, ax = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for a, name in zip(ax[0], ("sst", "temperature", "humidity")):
            for stat in ("rms", "max"):
                a.plot(t, [r[name + "_gradient_" + stat] for r in rows], label=stat)
            a.set(title=name + " gradient", xlabel="Simulated days")
            a.legend()
        for a, name in zip(ax[1], ("stress", "sensible", "evaporation")):
            for stat in ("mean", "maxabs"):
                a.plot(t, [r[name + "_" + stat] for r in rows], label=stat)
            a.set(title=name, xlabel="Simulated days")
            a.legend()
        fig.savefig(output / "gradients_fluxes.png", dpi=150)
        plt.close(fig)
        figures.append("gradients_fluxes.png")
        ledger = np.asarray(rows[-1]["cumulative_by_process"])
        (output / "process_budgets.json").write_text(
            json.dumps(
                {
                    name: dict(zip(QUANTITIES, map(float, ledger[i])))
                    for i, name in enumerate(PROCESSES)
                },
                indent=2,
            )
        )
        fig, axes = plt.subplots(1, 3, figsize=(17, 7), constrained_layout=True)
        for a, indices, title in zip(
            axes,
            ((0, 1, 2), (3, 4), (13, 14)),
            (
                "Heat contributions (J/m²)",
                "Water contributions (kg/m²)",
                "Kinetic work (J/m²)",
            ),
        ):
            for j, k in enumerate(indices):
                a.barh(
                    np.arange(len(PROCESSES)) + (j - 0.5) * 0.22,
                    ledger[:, k],
                    height=0.2,
                    label=QUANTITIES[k],
                )
            a.set(yticks=np.arange(len(PROCESSES)), yticklabels=PROCESSES, title=title)
            a.legend(fontsize=8)
        fig.savefig(output / "process_budgets.png", dpi=150)
        plt.close(fig)
        figures.append("process_budgets.png")
    paths = sorted((source / "fields").glob("step*.nc"))
    if paths:
        with Dataset(paths[-1]) as ds:
            keys = [
                k
                for k in (
                    "ocean_streamfunction",
                    "sst",
                    "air_temperature",
                    "humidity",
                    "stress",
                    "total_heat",
                    "exchange_drag",
                    "exchange_heat",
                    "exchange_moisture",
                    "exchange_stability",
                    "exchange_friction_velocity",
                    "exchange_gustiness",
                )
                if k in ds.variables
            ]
            fig, axes = plt.subplots(
                (len(keys) + 2) // 3,
                3,
                figsize=(14, 3.8 * ((len(keys) + 2) // 3)),
                constrained_layout=True,
            )
            x, y = ds["x"][:] / 1000, ds["y"][:] / 1000
            for a, k in zip(axes.flat, keys):
                v = ds[k][:]
                v = np.linalg.norm(v, axis=-1) if v.ndim == 3 else v
                m = a.pcolormesh(x, y, v.T, shading="auto", cmap="viridis")
                fig.colorbar(m, ax=a, label=ds[k].units)
                a.set(title=k, xlabel="x (km)", ylabel="y (km)")
            for a in list(axes.flat)[len(keys) :]:
                a.set_visible(False)
            fig.savefig(output / "final_fields.png", dpi=150)
            plt.close(fig)
            figures.append("final_fields.png")
    if paths:
        with Dataset(paths[-1]) as ds:
            velocity = np.asarray(ds["ocean_velocity"][:])
            x, y = np.asarray(ds["x"][:]) / 1000, np.asarray(ds["y"][:]) / 1000
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            for fraction in (0.25, 0.75):
                j = int(np.argmin(abs(y - fraction * y[-1])))
                for a, mask, label in (
                    (axes[0], x <= 0.15 * x[-1], "Western boundary"),
                    (axes[1], x >= 0.85 * x[-1], "Eastern boundary"),
                ):
                    a.plot(
                        x[mask],
                        velocity[mask, j, 1],
                        marker=".",
                        label=f"y/L={fraction}",
                    )
                    a.set(
                        title=label,
                        xlabel="x (km)",
                        ylabel="Meridional ocean velocity (m/s)",
                    )
                    a.legend()
            fig.savefig(output / "boundary_sections.png", dpi=150)
            plt.close(fig)
            figures.append("boundary_sections.png")
    if paths:
        from bspf_jax.air_sea import saturation_specific_humidity

        configuration = json.loads((source / "config.json").read_text())
        pressure = configuration["physics"]["pressure"]
        node_ranges = []
        for path in paths:
            with Dataset(path) as ds:
                rh = np.asarray(ds["humidity"][:]) / np.asarray(
                    saturation_specific_humidity(
                        np.asarray(ds["air_temperature"][:]), pressure
                    )
                )
                node_ranges.append(
                    {
                        "time_seconds": float(ds.time_seconds),
                        "min_RH": float(rh.min()),
                        "max_RH": float(rh.max()),
                        "supersaturated_nodes": int(np.count_nonzero(rh > 1)),
                    }
                )
        (output / "nodal_humidity_ranges.json").write_text(
            json.dumps(node_ranges, indent=2)
        )
    latest = source / "latest_checkpoint.json"
    if latest.exists():
        import jax
        from bspf_jax.air_sea_platform import read_checkpoint, _plan

        jax.config.update("jax_enable_x64", True)
        checkpoint = read_checkpoint(
            source / json.loads(latest.read_text())["manifest"]
        )
        p = _plan(checkpoint.config)
        state = checkpoint.state
        fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for name, coeff, xlam, ylam in (
            ("ocean", state.ocean, p.flow.x.lam, p.flow.y.lam),
            ("atmosphere", state.atmosphere, p.air_flow.x.lam, p.air_flow.y.lam),
            ("SST", state.sst, p.scalar.lam, p.scalar.lam),
            ("Ta", state.air_temperature, p.air_flow.x.lam, p.air_scalar_y.lam),
            ("q", state.humidity, p.air_flow.x.lam, p.air_scalar_y.lam),
        ):
            lam = np.maximum(np.asarray(xlam)[:, None] + np.asarray(ylam)[None, :], 0)
            energy = lam * np.asarray(coeff) ** 2
            valid = lam > 1e-10
            bins = np.geomspace(
                np.sqrt(lam[valid].min()), np.sqrt(lam.max()) * (1 + 1e-12), 35
            )
            weights, _ = np.histogram(
                np.sqrt(lam[valid]), bins=bins, weights=energy[valid]
            )
            if weights.sum() > 0:
                axis.loglog(
                    np.sqrt(bins[:-1] * bins[1:]), weights / weights.sum(), label=name
                )
        axis.set(
            xlabel="Dimensionless stiffness wavenumber",
            ylabel="Fraction per logarithmic bin",
            title="Resolved velocity / scalar-gradient energy spectra",
        )
        axis.legend()
        axis.grid(alpha=0.2)
        fig.savefig(output / "spectra.png", dpi=150)
        plt.close(fig)
        figures.append("spectra.png")
    failure = {
        str(p.relative_to(source)): json.loads(p.read_text())
        for p in (source / "failure").glob("*.json")
    }
    (output / "failures.json").write_text(json.dumps(failure, indent=2))
    elapsed = (source / "status.json").stat().st_mtime - (
        source / "manifest.json"
    ).stat().st_mtime
    (output / "cost.json").write_text(
        json.dumps(
            {
                "elapsed_wall_seconds_including_pauses": elapsed,
                "last_invocation_seconds": summary["status"].get(
                    "invocation_runtime_seconds"
                ),
                "includes_setup": False,
            },
            indent=2,
        )
    )
    lines = [
        "# Air–sea research evidence",
        "",
        f"Source: `{source}`",
        "",
        f"Status: **{summary['status']['status']}**",
        "",
        "Horizontal, vertically averaged fixed-depth surface-proxy model. No condensation; 30 days is a reliability/adjustment experiment, not climate equilibrium.",
        "",
        f"Diagnostics: {summary['records']}. Maximum residuals: heat {summary['max_heat_residual']:.6g} J/m²; water {summary['max_water_residual']:.6g} kg/m²; kinetic {summary['max_kinetic_residual']:.6g} J/m².",
        "",
        "Full process values: process_budgets.json. Runtime and source metadata: original status.json/manifest.json. Spatial, temporal and quadrature differences belong to the parent validation.json; a single run cannot establish resolution.",
        "",
    ]
    for name in figures:
        lines += [f"![{name}]({name})", ""]
    lines += [f"Failure records: {len(failure)}. See failures.json.", ""]
    (output / "report.md").write_text("\n".join(lines))
    return output / "report.md"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(create_report(a.run, a.out))
