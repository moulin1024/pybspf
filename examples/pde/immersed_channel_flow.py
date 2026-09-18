"""BSPF 2D incompressible flow around an eccentric ellipse, with outlet buffer."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
from scipy.special import roots_legendre

from bspf_jax.immersed_flow import ImmersedFlowPlan, channel_lift


def independent_checks(plan, state):
    boundary, _ = plan.arc.sample(512, offset=0.371)
    _, u, v, *_ = plan.evaluate(state, boundary)
    result = dict(hole_wall_max_speed=float(np.max(np.hypot(u, v))))
    x = np.linspace(plan.bounds[0], plan.bounds[1], 197)
    y = np.linspace(-plan.bounds[2], plan.bounds[2], 157)
    points = np.vstack(
        (
            np.column_stack((x, np.full_like(x, -plan.bounds[2]))),
            np.column_stack((x, np.full_like(x, plan.bounds[2]))),
            np.column_stack((np.full_like(y, plan.bounds[0]), y)),
        )
    )
    values = plan.evaluate(state, points)
    lift = channel_lift(points, plan.bounds[2], plan.peak)
    result["outer_dirichlet_max_error"] = float(
        np.max(np.hypot(values[1] - lift[1], values[2] - lift[2]))
    )
    q, w = roots_legendre(80)
    stations = [
        plan.bounds[0],
        -0.4,
        plan.hole.center[0],
        0.8,
        1.5,
        plan.buffer_start,
        (plan.buffer_start + plan.bounds[1]) / 2,
        plan.bounds[1],
    ]
    flux = []
    for pos in stations:
        cx, cy = plan.hole.center
        a, b = plan.hole.axes
        h = plan.bounds[2]
        if abs(pos - cx) < a:
            dy = b * np.sqrt(1 - ((pos - cx) / a) ** 2)
            intervals = [(-h, cy - dy), (cy + dy, h)]
        else:
            intervals = [(-h, h)]
        value = 0.0
        for lo, hi in intervals:
            yy = (lo + hi) / 2 + (hi - lo) / 2 * q
            points = np.column_stack((np.full_like(yy, pos), yy))
            value += (hi - lo) / 2 * (w @ plan.evaluate(state, points)[1])
        flux.append(value)
    result["flux_stations"] = stations
    result["flux_values"] = flux
    result["max_relative_flux_error"] = float(
        max(abs(np.array(flux) - 4 * plan.peak * h / 3)) / (4 * plan.peak * h / 3)
    )
    # Baseline wall shear vorticity is retained; measure wake perturbations.
    stations = np.linspace(plan.buffer_start, plan.bounds[1], 17)
    buffer = []
    for pos in stations:
        yy = h * q
        data = plan.evaluate(state, np.column_stack((np.full_like(yy, pos), yy)))
        reference = plan.peak * (1 - yy**2 / h**2)
        omega = data[5] - data[4] - 2 * plan.peak * yy / h**2
        buffer.append(
            dict(
                x=float(pos),
                sigma=float(plan.sponge_profile(np.array([pos]))[0]),
                perturbation_speed_l2=float(
                    np.sqrt(h * np.sum(w * ((data[1] - reference) ** 2 + data[2] ** 2)))
                ),
                perturbation_vorticity_l2=float(np.sqrt(h * np.sum(w * omega**2))),
            )
        )
    result["buffer_sections"] = buffer
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=73)
    ap.add_argument("--ny", type=int, default=33)
    ap.add_argument("--re", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--time", type=float, default=20.0)
    ap.add_argument("--buffer-length", type=float, default=2.0)
    ap.add_argument("--buffer-strength", type=float, default=3.0)
    ap.add_argument("--quadrature-factor", type=float, default=2.5)
    ap.add_argument("--wall-rcond", type=float, default=1e-10)
    ap.add_argument(
        "--wall-method", choices=("svd", "factor", "rational"), default="svd"
    )
    ap.add_argument("--wall-width", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=Path("build/immersed_flow"))
    args = ap.parse_args()
    jax.config.update("jax_enable_x64", True)
    args.out.mkdir(parents=True, exist_ok=True)
    plan = ImmersedFlowPlan(
        nx=args.nx,
        ny=args.ny,
        reynolds=args.re,
        buffer_length=args.buffer_length,
        buffer_strength=args.buffer_strength,
        quadrature_factor=args.quadrature_factor,
        wall_rcond=args.wall_rcond,
        wall_method=args.wall_method,
        wall_width=args.wall_width,
    )
    print(
        "SETUP",
        plan.setup_seconds,
        "DOFS",
        plan.ndofs,
        plan.constraint_rank,
        plan.dofs,
        "QUAD",
        len(plan.points),
        flush=True,
    )
    step = plan.stepper(args.dt)
    state = plan.stokes_state.copy()
    x, y = (
        np.linspace(plan.bounds[0], plan.bounds[1], 401),
        np.linspace(-plan.bounds[2], plan.bounds[2], 161),
    )
    frames, times, history = [], [], []
    total = round(args.time / args.dt)
    every = max(1, round(0.5 / args.dt))
    start = perf_counter()
    for k in range(total + 1):
        t = k * args.dt
        if k % every == 0 or k == total:
            diag = dict(t=t, **plan.diagnostics(state))
            history.append(diag)
            fields = plan.grid(state, x, y)
            frames.append(
                np.stack([fields[key] for key in ("u", "v", "vorticity", "psi")])
            )
            times.append(t)
            print(json.dumps(diag), flush=True)
            if not np.all(np.isfinite(state)) or diag["max_speed"] > 20:
                raise RuntimeError("Flow became unstable")
        if k < total:
            state = step.step(state, t)
    elapsed = perf_counter() - start
    checks = independent_checks(plan, state)
    summary = dict(
        nx=plan.nx,
        ny=plan.ny,
        wall_rcond=plan.wall_rcond,
        wall_method=plan.wall_method,
        wall_width=plan.wall_width,
        rational=None if plan.rational is None else plan.rational.info,
        energy_condition=plan.energy_condition,
        dt=args.dt,
        final_time=total * args.dt,
        reynolds=plan.reynolds,
        viscosity=plan.nu,
        peak_inlet=plan.peak,
        mean_inlet=plan.mean_speed,
        diameter=plan.diameter,
        bounds=plan.bounds,
        physical_xlim=plan.physical_xlim,
        buffer_length=plan.buffer_length,
        buffer_strength=plan.buffer_strength,
        setup_seconds=plan.setup_seconds,
        advance_and_output_seconds=elapsed,
        ndofs=plan.ndofs,
        wall_constraint_rank=plan.constraint_rank,
        retained_dofs=plan.dofs,
        quadrature_points=len(plan.points),
        quadrature_factor=args.quadrature_factor,
        final=history[-1],
        checks=checks,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    np.savez_compressed(
        args.out / "fields.npz",
        x=x,
        y=y,
        t=times,
        fields=frames,
        center=plan.hole.center,
        axes=plan.hole.axes,
        buffer_start=plan.buffer_start,
        sigma=plan.sponge_profile(x),
        state=state,
        coefficients=plan.coefficients(state),
        nx=plan.nx,
        ny=plan.ny,
    )
    print("CHECKS", json.dumps(checks), flush=True)


if __name__ == "__main__":
    main()
