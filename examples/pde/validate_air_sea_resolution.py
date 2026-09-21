"""Independent space/time resolution audit of the default 24-hour transient.

Stores actual coefficient snapshots at 6, 12, 24 hours. Comparisons use one
common overintegrated quadrature, analytic basis gradients, and raw bulk fluxes.
No interpolated images, fitted curves, or altered physical coefficients.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import jax
import numpy as np

from bspf_jax.air_sea import (
    AirSeaState, air_sea_fields, air_sea_step, bulk_flux, heat_content,
    initial_air_sea_state, plan_air_sea, plan_air_sea_stepper, scalar_mean,
)


def run(plan, n, h, out, fingerprint):
    path = out / f"n{n}_H{h}.npz"
    if path.exists():
        data = np.load(path)
        if str(data["fingerprint"]) != fingerprint:
            raise RuntimeError(f"Stale cache: {path}; use another output directory")
        return [AirSeaState(*(data[f"{i}_{k}"] for k in AirSeaState._fields))
                for i in range(3)], json.loads(str(data["metadata"]))
    stepper = plan_air_sea_stepper(dt_air=h / 5, dt_ocean=h, window=3600)

    @jax.jit
    def advance(s):
        s, b = air_sea_step(plan, stepper, s)
        return (s, b.external_heat,
                b.external_water + scalar_mean(plan, b.exchange.water))

    s = initial_air_sea_state(plan)
    heat0 = float(heat_content(plan, s))
    water0 = float(plan.config.air_mass * scalar_mean(plan, s.humidity, air=True))
    heat = water = 0.0
    states, budgets = [], []
    started = time.perf_counter()
    for hour in range(1, 25):
        s, dh, dq = advance(s)
        heat += float(dh)
        water += float(dq)
        if not all(np.all(np.isfinite(v)) for v in s):
            raise RuntimeError(f"Nonfinite state: n={n}, H={h}, hour={hour}")
        if hour in (6, 12, 24):
            states.append(jax.device_get(s))
            budgets.append(dict(
                hour=hour,
                heat_residual=float(heat_content(plan, s)) - heat0 - heat,
                water_residual=float(plan.config.air_mass * scalar_mean(
                    plan, s.humidity, air=True)) - water0 - water,
            ))
            print(f"n={n}, H={h}, hour={hour}: {budgets[-1]}", flush=True)
    if max(abs(b["heat_residual"]) for b in budgets) > 1e-4 or max(
        abs(b["water_residual"]) for b in budgets
    ) > 1e-9:
        raise RuntimeError("Budget check failed")
    metadata = dict(n=n, macro_step=h, fast_step=h / 5, budgets=budgets,
                    runtime_seconds=time.perf_counter() - started)
    arrays = {f"{i}_{k}": v for i, s in enumerate(states)
              for k, v in zip(AirSeaState._fields, s)}
    np.savez_compressed(path, **arrays, metadata=json.dumps(metadata),
                        fingerprint=fingerprint)
    return states, metadata


def diagnostics(p, state):
    f = {k: np.asarray(v) for k, v in air_sea_fields(p, state, nodes=False).items()}
    for name, coeff, air in (("sst", state.sst, False),
                              ("air_temperature", state.air_temperature, True),
                              ("humidity", state.humidity, True)):
        x, y = (p.air_flow.x, p.air_scalar_y) if air else (p.scalar, p.scalar)
        f[name + "_gradient"] = np.stack((
            np.asarray(x.g) @ coeff @ np.asarray(y.b).T,
            np.asarray(x.b) @ coeff @ np.asarray(y.g).T,
        ), axis=-1) / p.config.length
    stress, sensible, water = bulk_flux(
        p.config, f["ocean_velocity"], f["air_velocity"], f["sst"],
        f["air_temperature"], f["humidity"],
    )
    f.update(stress=np.asarray(stress), sensible_heat=np.asarray(sensible),
             evaporation=np.asarray(water),
             total_heat=np.asarray(sensible + p.config.latent_heat * water))
    return f


def compare(p, a, b):
    w = np.asarray(p.weight)
    result = {}
    for key in a:
        aa, bb = a[key], b[key]
        # Anomalies avoid normalizing a temperature error by a 290 K offset.
        if key in ("sst", "air_temperature"):
            aa = aa - np.sum(w * aa)
            bb = bb - np.sum(w * bb)
            # The error includes the mean: only the normalization is demeaned.
        diff = a[key] - b[key]
        norm = lambda v: np.sqrt(np.sum(v * v, axis=-1)) if v.ndim == 3 else abs(v)
        error, magnitude = norm(diff), norm(bb)
        rms = float(np.sqrt(np.sum(w * error**2)))
        ref = float(np.sqrt(np.sum(w * magnitude**2)))
        peak = float(np.max(magnitude))
        result[key] = dict(rms=rms, reference_rms=ref,
                           relative_rms=rms / max(ref, 1e-30),
                           linf=float(np.max(error)), reference_peak=peak,
                           relative_linf=float(np.max(error)) / max(peak, 1e-30),
                           mean_difference=(np.sum(w[..., None] * diff, axis=(0, 1)).tolist()
                                            if diff.ndim == 3 else float(np.sum(w * diff))))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("build/air_sea_resolution"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    grids = (33, 49, 65, 81)
    # All <=81 use exactly the same 19*32 Gauss nodes on each physical axis.
    plans = {n: plan_air_sea(n=n, quadrature_order=32) for n in grids}
    p = plans[81]
    for other in plans.values():
        np.testing.assert_array_equal(other.scalar.points, p.scalar.points)
    source = Path(__file__).resolve().parents[2] / "jax/src/bspf_jax"
    digest = hashlib.sha256()
    for file in (Path(__file__), source / "air_sea.py", source / "multirate.py"):
        digest.update(file.read_bytes())
    fingerprint = digest.hexdigest()
    # H=150 is the common spatial-comparison step. Both bounding grids receive
    # time refinement, so small-scale temporal error is checked at the fine end.
    cases = [(n, 150) for n in grids] + [(n, h) for n in (33, 81) for h in (600, 300)]
    states, metadata = {}, {}
    for n, h in cases:
        states[n, h], metadata[f"n{n}_H{h}"] = run(plans[n], n, h, args.out, fingerprint)
        jax.clear_caches()
    space, temporal = {}, {}
    for i, hour in enumerate((6, 12, 24)):
        fine = {n: diagnostics(plans[n], states[n, 150][i]) for n in grids}
        space[str(hour)] = {f"{a}_{b}": compare(p, fine[a], fine[b])
                            for a, b in zip(grids[:-1], grids[1:])}
        temporal[str(hour)] = {}
        for n in (33, 81):
            coarser = {h: diagnostics(plans[n], states[n, h][i]) for h in (600, 300)}
            coarser[150] = fine[n]
            temporal[str(hour)][str(n)] = {
                f"{a}_{b}": compare(p, coarser[a], coarser[b])
                for a, b in ((600, 300), (300, 150))
            }
        del fine
    report = dict(config=asdict(p.config), snapshots_hours=[6, 12, 24],
                  grids=grids, quadrature_order=32, fingerprint=fingerprint,
                  runs=metadata, spatial=space, temporal=temporal,
                  scope="Default 24-hour transient; finite-refinement differences, not exact errors or equilibrium certification",
                  screening_criteria=dict(relative_rms=0.01, relative_linf=0.05,
                                          time_to_space_ratio=0.1))
    report["assessment"] = {}
    for hour in space:
        report["assessment"][hour] = {}
        for key, last in space[hour]["65_81"].items():
            prior = space[hour]["49_65"][key]
            t = temporal[hour]["81"]["300_150"][key]
            t0 = temporal[hour]["81"]["600_300"][key]
            ratio = t["rms"] / max(last["rms"], 1e-30)
            report["assessment"][hour][key] = dict(
                spatial_rms=last["relative_rms"], spatial_peak=last["relative_linf"],
                decreasing=last["rms"] < prior["rms"], time_to_space=ratio,
                temporal_order=(float(np.log2(t0["rms"] / t["rms"]))
                                if t["rms"] > 1e-14 and t0["rms"] > 1e-14 else None),
                passes_screen=(last["relative_rms"] < .01 and last["relative_linf"] < .05
                               and last["rms"] < prior["rms"] and ratio < .1),
            )
    (args.out / "resolution.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["assessment"]["24"], indent=2), flush=True)


if __name__ == "__main__":
    main()
