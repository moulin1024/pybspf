"""Run the horizontal 2D BSPF ocean / periodic-channel atmosphere example.

PYTHONPATH=jax/src MPLCONFIGDIR=/tmp/bspf-mpl python examples/pde/air_sea_double_gyre.py
All times are seconds internally; --hours is the elapsed physical duration.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import jax
import numpy as np

from bspf_jax.air_sea import (
    air_sea_fields,
    air_sea_step,
    bulk_flux,
    heat_content,
    initial_air_sea_state,
    plan_air_sea,
    plan_air_sea_stepper,
    saturation_specific_humidity,
    scalar_mean,
    scalar_values,
)
from bspf_jax.stream_navier_stokes import stream_ns_divergence


def render(path, x, field, flux, history, hours, latent_heat):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), constrained_layout=True)
    panels = [
        (
            field["ocean_streamfunction"],
            "Ocean streamfunction (transient)",
            "m²/s",
            "RdBu_r",
        ),
        (field["air_velocity"][..., 0], "Atmospheric zonal wind", "m/s", "RdBu_r"),
        (field["sst"] - 273.15, "Sea-surface temperature", "°C", "coolwarm"),
        (1000 * field["humidity"], "Atmospheric specific humidity", "g/kg", "YlGnBu"),
        (
            flux[1] + latent_heat * flux[2],
            "Upward sensible + latent heat flux",
            "W/m²",
            "magma",
        ),
    ]
    for ax, (values, title, unit, cmap) in zip(axes.flat, panels):
        kwargs = {}
        if cmap == "RdBu_r":
            limit = max(float(np.max(abs(values))), 1e-12)
            kwargs = dict(vmin=-limit, vmax=limit)
        im = ax.pcolormesh(x, x, values.T, shading="auto", cmap=cmap, **kwargs)
        ax.set(title=title, xlabel="East x (km)", ylabel="North y (km)", aspect="equal")
        fig.colorbar(im, ax=ax, label=unit)
    ax = axes.flat[-1]
    t = np.array([r["time_seconds"] for r in history]) / 3600
    ax.plot(
        t, [r["mean_sst_K"] - history[0]["mean_sst_K"] for r in history], label="SST"
    )
    ax.plot(
        t,
        [
            r["mean_air_temperature_K"] - history[0]["mean_air_temperature_K"]
            for r in history
        ],
        label="Air",
    )
    ax.set(
        xlabel="Time (hours)",
        ylabel="Domain-mean temperature change (K)",
        title="Two-way thermodynamic adjustment",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.suptitle(
        f"Horizontal 2D air–sea coupling | t={hours:g} h\n"
        "Ocean: closed beta-plane basin · Air: x-periodic, y free-slip · Vertically averaged layers",
        fontsize=13,
    )
    fig.savefig(path, dpi=155)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    for ax, key, title, unit in zip(
        axes,
        ("heat_budget_error", "water_budget_error", "air_zonal_momentum_error"),
        (
            "Thermodynamic budget residual",
            "Air + diagnostic ocean water residual",
            "Air zonal momentum residual",
        ),
        ("J/m²", "kg/m²", "kg/(m s)"),
    ):
        ax.plot(t, [r[key] for r in history])
        ax.set(xlabel="Time (hours)", ylabel=unit, title=title)
        ax.grid(alpha=0.2)
    fig.savefig(path.with_name("budgets.png"), dpi=155)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=33)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--dt-air", type=float, default=60.0)
    ap.add_argument("--dt-ocean", type=float, default=300.0)
    ap.add_argument("--window", type=float, default=600.0)
    ap.add_argument("--method", choices=("mri-gark4", "lagged"), default="mri-gark4")
    ap.add_argument("--output-hours", type=float, default=1.0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if args.out is None:
        suffix = "mri4" if args.method == "mri-gark4" else "lagged"
        args.out = Path(f"build/air_sea_double_gyre_{suffix}")
    if not np.isfinite(args.hours) or args.hours <= 0:
        ap.error("hours must be finite and positive")
    if not np.isfinite(args.output_hours) or args.output_hours <= 0:
        ap.error("output-hours must be finite and positive")
    try:
        stepper = plan_air_sea_stepper(
            dt_air=args.dt_air,
            dt_ocean=args.dt_ocean,
            window=args.window,
            method=args.method,
        )
    except ValueError as error:
        ap.error(str(error))
    windows = int(round(args.hours * 3600 / stepper.window))
    if windows < 1 or not np.isclose(
        windows * stepper.window, args.hours * 3600, rtol=1e-12, atol=0
    ):
        ap.error("hours must contain an integer number of coupling windows")
    if not np.isclose(
        args.output_hours * 3600 / stepper.window,
        round(args.output_hours * 3600 / stepper.window),
        rtol=0,
        atol=1e-10,
    ):
        ap.error("output-hours must align exactly with the window")
    stride = max(1, round(args.output_hours * 3600 / stepper.window))
    jax.config.update("jax_enable_x64", True)
    started = time.perf_counter()
    plan = plan_air_sea(n=args.n)
    config = plan.config
    state = initial_air_sea_state(plan)
    args.out.mkdir(parents=True, exist_ok=False)
    initial_heat = float(heat_content(plan, state))
    initial_water = float(config.air_mass * scalar_mean(plan, state.humidity, air=True))
    initial_mean_wind = float(state.air_mean_wind)
    external_heat = external_water = evaporation = impulse_x = external_momentum = 0.0
    heat_scale = 1.0
    history, times, snapshots = [], [], []
    print(
        f"setup={time.perf_counter() - started:.2f}s n={args.n}; "
        f"method={stepper.method} air={stepper.effective_dt_air:g}s "
        f"ocean-macro={stepper.dt_ocean:g}s sync-window={stepper.window:g}s",
        flush=True,
    )

    @jax.jit
    def advance(s):
        new, budget = air_sea_step(plan, stepper, s)
        means = jax.numpy.array(
            [
                budget.external_heat,
                budget.external_water,
                scalar_mean(plan, budget.exchange.water),
                budget.exchange.stress_mean[0],
                budget.external_air_zonal_momentum,
                abs(scalar_mean(plan, budget.exchange.sensible))
                + config.latent_heat * abs(scalar_mean(plan, budget.exchange.water)),
            ]
        )
        return new, means

    @jax.jit
    def diagnose(s):
        f = air_sea_fields(plan, s)
        qquad = scalar_values(plan, s.humidity, air=True)
        qnodes = f["humidity"]
        rel_humidity = qnodes / saturation_specific_humidity(
            f["air_temperature"], config.pressure
        )
        return f, jax.numpy.array(
            [
                heat_content(plan, s),
                config.air_mass * scalar_mean(plan, s.humidity, air=True),
                scalar_mean(plan, s.sst) + config.reference_temperature,
                scalar_mean(plan, s.air_temperature, air=True)
                + config.reference_temperature,
                jax.numpy.min(qquad),
                jax.numpy.max(qquad),
                jax.numpy.max(jax.numpy.linalg.norm(f["ocean_velocity"], axis=-1)),
                jax.numpy.max(jax.numpy.linalg.norm(f["air_velocity"], axis=-1)),
                jax.numpy.max(abs(stream_ns_divergence(plan.flow, s.ocean)))
                / config.length,
                jax.numpy.max(abs(stream_ns_divergence(plan.air_flow, s.atmosphere)))
                / config.length,
                jax.numpy.min(rel_humidity),
                jax.numpy.max(rel_humidity),
                jax.numpy.min(qnodes),
                jax.numpy.max(qnodes),
            ]
        )

    for k in range(windows + 1):
        if k:
            state, means = advance(state)
            means = np.asarray(means)
            if not np.all(np.isfinite(means)):
                raise RuntimeError(
                    f"Nonfinite window {k}; reduce steps and check resolution"
                )
            external_heat += means[0]
            external_water += means[1]
            evaporation += means[2]
            impulse_x += means[3]
            external_momentum += means[4]
            heat_scale += abs(means[0]) + means[5]
        if k % stride and k != windows:
            continue
        field, vals = jax.device_get(diagnose(state))
        if (
            not np.all(np.isfinite(vals))
            or min(vals[4], vals[12]) < 0
            or max(vals[5], vals[13]) > 1
        ):
            raise RuntimeError(
                f"Invalid state at window {k}: finite values and 0<=q<=1 required; no clipping is used"
            )
        row = dict(
            zip(
                (
                    "heat_content",
                    "air_water",
                    "mean_sst_K",
                    "mean_air_temperature_K",
                    "min_q",
                    "max_q",
                    "max_ocean_speed",
                    "max_air_speed",
                    "ocean_divergence",
                    "air_divergence",
                    "min_RH",
                    "max_RH",
                    "min_nodal_q",
                    "max_nodal_q",
                ),
                map(float, vals),
            )
        )
        row.update(
            time_seconds=k * stepper.window,
            heat_budget_error=float(vals[0] - initial_heat - external_heat),
            water_budget_error=float(
                vals[1] - initial_water - evaporation - external_water
            ),
            air_zonal_momentum_error=float(
                config.air_mass * (state.air_mean_wind - initial_mean_wind)
                + impulse_x
                - external_momentum
            ),
            cumulative_external_heat=float(external_heat),
            cumulative_external_water=float(external_water),
            diagnostic_ocean_water_change=float(-evaporation),
            heat_error_relative_to_exchanged=float(
                abs(vals[0] - initial_heat - external_heat) / heat_scale
            ),
        )
        if (
            row["heat_error_relative_to_exchanged"] > 1e-7
            or abs(row["water_budget_error"]) > 1e-8
        ):
            raise RuntimeError(f"Coupled budget failure: {row}")
        history.append(row)
        times.append(k * stepper.window)
        snapshots.append(field)
        print(
            f"t={k * stepper.window / 3600:6.2f}h Uo={vals[6]:.4e} Ua={vals[7]:.3f} "
            f"heat_err={row['heat_budget_error']:.2e} water_err={row['water_budget_error']:.2e} "
            f"RHmax={vals[11]:.3f}",
            flush=True,
        )
        # Progress checkpoint includes modal state and all accumulated budgets.
        np.savez_compressed(
            args.out / "state.npz",
            **{f"coeff_{name}": np.asarray(v) for name, v in state._asdict().items()},
            **field,
            x_m=np.asarray(plan.scalar.x) * config.length,
            time_seconds=k * stepper.window,
        )
        (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    flux = tuple(
        np.asarray(v)
        for v in bulk_flux(
            config,
            field["ocean_velocity"],
            field["air_velocity"],
            field["sst"],
            field["air_temperature"],
            field["humidity"],
        )
    )
    np.savez_compressed(
        args.out / "snapshots.npz",
        time_seconds=times,
        **{key: np.stack([s[key] for s in snapshots]) for key in snapshots[0]},
    )
    summary = dict(
        config=asdict(config),
        n=args.n,
        stepper=stepper._asdict(),
        effective_dt_air=stepper.effective_dt_air,
        hours=args.hours,
        runtime_seconds=time.perf_counter() - started,
        boundary="ocean: closed no-slip / zero scalar flux; atmosphere: x periodic, y impermeable free-slip / zero scalar flux",
        method=(
            "BSPF ocean + Fourier/sine/cosine atmospheric channel; conservative weak scalar transport; "
            + (
                "FOURTH-ORDER MRI-GARK-ERK45a / RK4; common-stage interface exchange"
                if stepper.method == "mri-gark4"
                else "FIRST-ORDER lagged ocean coupling / RK4"
            )
        ),
        scope="Horizontal vertically averaged transient; no resolved vertical turbulence, ocean baroclinicity, cloud condensation, or prognostic ocean freshwater/salinity",
        surface_wind="Target ua=-8*cos(2*pi*y/L), va=0; stress is computed from relative wind, never added twice",
        final=history[-1],
        max_heat_budget_error=max(abs(r["heat_budget_error"]) for r in history),
        max_water_budget_error=max(abs(r["water_budget_error"]) for r in history),
        max_air_zonal_momentum_error=max(
            abs(r["air_zonal_momentum_error"]) for r in history
        ),
        ocean_streamfunction_range=[
            float(np.min(field["ocean_streamfunction"])),
            float(np.max(field["ocean_streamfunction"])),
        ],
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.no_plot:
        render(
            args.out / "coupled.png",
            np.asarray(plan.scalar.x) * config.length / 1000,
            field,
            flux,
            history,
            args.hours,
            config.latent_heat,
        )
    print(f"Saved {args.out}; wall time {summary['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
