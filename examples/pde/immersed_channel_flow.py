"""BSPF 2D incompressible flow around an eccentric ellipse, with outlet buffer."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
from scipy.special import roots_legendre

from bspf_models.fluids.immersed_flow import ImmersedFlowPlan
from bspf_models.fluids.immersed_flow import channel_lift


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
    ap.add_argument("--basis-precision", choices=("mpfr", "float64"), default="mpfr",
                        help="Basis arithmetic: mpfr reference or opt-in GPU float64")
    ap.add_argument("--rational-basis-construction", choices=("cpu", "gpu"), default="cpu",
                    help="Arnoldi recurrence construction; GPU is opt-in due to cold JIT cost")
    ap.add_argument("--rational-preprocessing", action=argparse.BooleanOptionalAction, default=False,
                    help="Experimental accuracy-gated pole sweep before volume assembly")
    ap.add_argument("--basis-workers", type=int, default=4)
    ap.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
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
    ap.add_argument("--initial-state", choices=("stokes", "compatible"), default="stokes",
                    help="Geometric compatible lift skips the Stokes solve and requires CPU factor space")
    ap.add_argument("--save-every", type=float, default=.5)
    ap.add_argument("--out", type=Path, default=Path("build/immersed_flow"))
    args = ap.parse_args()
    if args.initial_state == "compatible" and (args.wall_method != "factor" or args.backend != "cpu"):
        ap.error("--initial-state compatible requires --wall-method factor --backend cpu")
    if not np.isfinite(args.save_every) or args.save_every <= 0:
        ap.error("--save-every must be positive and finite")
    if args.basis_precision == "float64" and args.backend != "gpu":
        ap.error("--basis-precision float64 requires --backend gpu")
    if args.rational_basis_construction == "gpu" and (args.backend != "gpu" or args.wall_method != "rational"):
        ap.error("--rational-basis-construction gpu requires --backend gpu --wall-method rational")
    jax.config.update("jax_enable_x64", True)
    device = jax.devices("gpu")[0] if args.backend == "gpu" else None
    print("BACKEND", args.backend, device, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    plan = ImmersedFlowPlan(
        assembly_device=device,
        basis_workers=args.basis_workers,
        basis_precision=args.basis_precision,
        nx=args.nx,
        ny=args.ny,
        reynolds=args.re,
        buffer_length=args.buffer_length,
        buffer_strength=args.buffer_strength,
        quadrature_factor=args.quadrature_factor,
        wall_rcond=args.wall_rcond,
        wall_method=args.wall_method,
        wall_width=args.wall_width,
        rational_options=dict(basis_construction=args.rational_basis_construction),
        rational_preprocessing=args.rational_preprocessing,
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
    step = plan.stepper(args.dt, device=device)
    if args.initial_state == "compatible":
        state = plan.compatible_state()
        assert plan.rational is None and "stokes_state" not in plan.__dict__
    else:
        state = step.initial_state if device is not None else plan.stokes_state.copy()
    initial_state=np.asarray(jax.device_get(state)).copy()
    initial_checks=independent_checks(plan,initial_state)
    print("INITIAL",args.initial_state,json.dumps(initial_checks),flush=True)
    x, y = (
        np.linspace(plan.bounds[0], plan.bounds[1], 401),
        np.linspace(-plan.bounds[2], plan.bounds[2], 161),
    )
    snapshots, times, history = [], [], []
    total = round(args.time / args.dt)
    every = max(1, round(args.save_every / args.dt))
    start = perf_counter()
    for k in range(total + 1):
        t = k * args.dt
        if k % every == 0 or k == total:
            if device is not None:
                values = jax.device_get(step.diagnostics(state))
                diag = dict(t=t, **{key: float(value) for key, value in values.items()})
                output_state = jax.device_get(state)
            else:
                diag = dict(t=t, **plan.diagnostics(state))
                output_state = state
            history.append(diag)
            snapshots.append(output_state.copy())
            times.append(t)
            # Publish one complete checkpoint atomically for live inspection.
            # The currently running process must be restarted to pick up code edits.
            checkpoint_fields = plan.grid(output_state, x, y)
            checkpoint = args.out / "latest.tmp.npz"
            np.savez_compressed(
                checkpoint,
                x=x,
                y=y,
                t=t,
                fields=np.stack([
                    checkpoint_fields[key] for key in ("u", "v", "vorticity", "psi")
                ]),
                center=plan.hole.center,
                axes=plan.hole.axes,
                buffer_start=plan.buffer_start,
                sigma=plan.sponge_profile(x),
                state=output_state,
                coefficients=plan.coefficients(output_state),
                nx=plan.nx,
                ny=plan.ny,
            )
            checkpoint.replace(args.out / "latest.npz")
            history_path = args.out / "history.tmp.json"
            history_path.write_text(json.dumps(history, indent=2) + "\n")
            history_path.replace(args.out / "history.json")
            print(json.dumps(diag), flush=True)
            if not np.all(np.isfinite(output_state)) or diag["max_speed"] > 20:
                raise RuntimeError("Flow became unstable")
        if k < total:
            state = step.step(state, t)
    state = jax.device_get(state)
    evolution_seconds = perf_counter() - start
    reconstruction_start = perf_counter()
    frames = [
        np.stack([fields[key] for key in ("u", "v", "vorticity", "psi")])
        for fields in plan.grid_many(np.asarray(snapshots), x, y)
    ]
    reconstruction_seconds = perf_counter() - reconstruction_start
    elapsed = perf_counter() - start
    checks = independent_checks(plan, state)
    summary = dict(
        backend=args.backend,
        initial_state=args.initial_state,
        initial_checks=initial_checks,
        basis_workers=args.basis_workers if args.basis_precision == "mpfr" else 0,
        basis_precision=args.basis_precision,
        device=None if device is None else str(device),
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
        evolution_and_diagnostics_seconds=evolution_seconds,
        reconstruction_seconds=reconstruction_seconds,
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
        initial_state=initial_state,
        coefficients=plan.coefficients(state),
        nx=plan.nx,
        ny=plan.ny,
    )
    print("CHECKS", json.dumps(checks), flush=True)


if __name__ == "__main__":
    main()
