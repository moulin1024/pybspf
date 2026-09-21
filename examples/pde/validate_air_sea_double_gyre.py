"""Legacy first-order lagged coupling: window refinement and grid sensitivity.

Uses a deliberately faster-adjusting shallow mixed-layer case to expose lag
error. It does NOT certify a mature, grid-converged ocean gyre equilibrium.
PYTHONPATH=jax/src python examples/pde/validate_air_sea_double_gyre.py
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import jax
import numpy as np

from bspf_jax.air_sea import (
    air_sea_fields,
    air_sea_step,
    heat_content,
    initial_air_sea_state,
    plan_air_sea,
    plan_air_sea_stepper,
    scalar_mean,
)


def integrate(plan, stepper, seconds):
    count = round(seconds / stepper.window)
    if not np.isclose(count * stepper.window, seconds, rtol=1e-12):
        raise ValueError("Duration must be a multiple of the window")
    state = initial_air_sea_state(plan)
    h0 = float(heat_content(plan, state))
    q0 = float(plan.config.air_mass * scalar_mean(plan, state.humidity, air=True))
    external_heat, water = 0.0, 0.0

    @jax.jit
    def advance(s):
        s, b = air_sea_step(plan, stepper, s)
        return (
            s,
            b.external_heat,
            b.external_water + scalar_mean(plan, b.exchange.water),
        )

    start = time.perf_counter()
    for _ in range(count):
        state, h, q = advance(state)
        external_heat += float(h)
        water += float(q)
    error_h = float(heat_content(plan, state)) - h0 - external_heat
    error_q = (
        float(plan.config.air_mass * scalar_mean(plan, state.humidity, air=True))
        - q0
        - water
    )
    if not all(np.all(np.isfinite(a)) for a in state):
        raise RuntimeError("Nonfinite state")
    if abs(error_h) > 1e-4 or abs(error_q) > 1e-9:
        raise RuntimeError(f"Budget violation: heat={error_h}, water={error_q}")
    return state, dict(
        heat_error=error_h,
        water_error=error_q,
        runtime_seconds=time.perf_counter() - start,
    )


def distance(plan, a, b):
    """RMS on the common physical quadrature grid, plus a scaled combined norm."""
    weight = np.asarray(plan.weight)
    result = {}
    for key in ("ocean_velocity", "air_velocity", "sst", "air_temperature", "humidity"):
        delta = np.asarray(a[key]) - np.asarray(b[key])
        squared = np.sum(delta**2, axis=-1) if delta.ndim == 3 else delta**2
        result[key] = float(np.sqrt(np.sum(weight * squared)))
    scales = dict(
        ocean_velocity=0.1,
        air_velocity=8.0,
        sst=1.0,
        air_temperature=1.0,
        humidity=0.01,
    )
    result["scaled_combined"] = float(
        np.sqrt(sum((result[k] / scales[k]) ** 2 for k in scales))
    )
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("build/air_sea_validation"))
    ap.add_argument("--skip-grid", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    base = plan_air_sea(n=33)
    config = replace(
        base.config, ocean_depth=50.0, mixed_layer_depth=5.0, atmosphere_depth=300.0
    )
    p = replace(base, config=config)
    duration = 3600.0
    results, fields = {}, {}
    for window in (600.0, 300.0, 150.0, 75.0):
        stepper = plan_air_sea_stepper(
            method="lagged", dt_air=15.0, dt_ocean=75.0, window=window
        )
        state, budget = integrate(p, stepper, duration)
        fields[window] = jax.device_get(air_sea_fields(p, state, nodes=False))
        results[str(int(window))] = dict(stepper=stepper._asdict(), **budget)
        print(f"window={window:g}s: {budget}", flush=True)
    reference = fields[75.0]
    for window in (600.0, 300.0, 150.0):
        results[str(int(window))]["error_vs_75s"] = distance(
            p, fields[window], reference
        )
    successive = [
        distance(p, fields[a], fields[b])["scaled_combined"]
        for a, b in ((600.0, 300.0), (300.0, 150.0), (150.0, 75.0))
    ]
    orders = [float(np.log2(successive[k] / successive[k + 1])) for k in range(2)]
    if not all(a > b > 0 for a, b in zip(successive, successive[1:])):
        raise RuntimeError(f"Coupling-window refinement did not converge: {successive}")
    # Same coupling window, halved internal RK4 steps: distinguishes lag from
    # internal integration error instead of attributing everything to coupling.
    refined, budget = integrate(
        p,
        plan_air_sea_stepper(method="lagged", dt_air=7.5, dt_ocean=37.5, window=150.0),
        duration,
    )
    internal = distance(
        p, jax.device_get(air_sea_fields(p, refined, nodes=False)), fields[150.0]
    )
    if internal["scaled_combined"] >= successive[-1]:
        raise RuntimeError(
            "Internal stepping error dominates the coupling-window comparison"
        )
    report = dict(
        duration_seconds=duration,
        temporal_config=asdict(config),
        windows=results,
        successive_window_differences=successive,
        observed_window_orders=orders,
        internal_step_halving=dict(error=internal, budget=budget),
        scope="Transient convergence and conservation; no equilibrium or spatial asymptotic-order claim",
    )
    print(
        f"Observed coupling orders: {orders}; internal-step error: {internal}",
        flush=True,
    )

    if not args.skip_grid:
        grid_fields, grid_results = {}, {}
        for n in (33, 49, 65):
            gp = base if n == 33 else plan_air_sea(n=n)
            # Default physical depths, common well-overintegrated physical grid.
            if not np.allclose(gp.scalar.points, base.scalar.points):
                raise RuntimeError("Spatial comparison requires common quadrature")
            s, b = integrate(
                gp,
                plan_air_sea_stepper(
                    method="lagged", dt_air=30.0, dt_ocean=150.0, window=300.0
                ),
                duration,
            )
            grid_fields[n] = jax.device_get(air_sea_fields(gp, s, nodes=False))
            grid_results[str(n)] = b
            print(f"grid={n}: {b}", flush=True)
        e33 = distance(base, grid_fields[33], grid_fields[49])
        e49 = distance(base, grid_fields[49], grid_fields[65])
        report["grid_sensitivity"] = dict(
            config=asdict(base.config),
            runs=grid_results,
            difference_33_49=e33,
            difference_49_65=e49,
            decreasing=e49["scaled_combined"] < e33["scaled_combined"],
        )
        if not report["grid_sensitivity"]["decreasing"]:
            raise RuntimeError("Grid differences do not decrease")
    (args.out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validation passed; saved {args.out / 'validation.json'}", flush=True)


if __name__ == "__main__":
    main()
