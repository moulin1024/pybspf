"""Run compatible BSPF KH with physical boundary-layer enrichment and checkpoints."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from bspf_jax.stream_navier_stokes import (
    stream_ns_divergence,
    plan_stream_sponge,
    stream_ns_inertia_apply,
    with_stream_dynamic_boundary,
)
from bspf_stream_ns import plan, velocity, vorticity, load, rhs, rk4, kh_seed
from validate_weak_ns_accuracy import fields


def parse_args(argv=None):
    pa = argparse.ArgumentParser()
    pa.add_argument(
        "--x-boundary",
        choices=["fixed", "open", "dynamic"],
        default=None,
        help="Default: open for KH, fixed for --accuracy",
    )
    pa.add_argument("--boundary-d0", type=float, default=1.0)
    pa.add_argument("--basis-from", type=Path)
    pa.add_argument(
        "--extension",
        type=float,
        default=None,
        help="External width per side (default: 1 for open KH)",
    )
    pa.add_argument(
        "--sponge-strength",
        type=float,
        default=None,
        help="Default: 4 with extension, otherwise 0",
    )
    pa.add_argument("--seed-cutoff-width", type=float)
    pa.add_argument(
        "--nx",
        type=int,
        default=None,
        help="Default: 128 for the default buffer domain",
    )
    pa.add_argument("--ny", type=int, default=80)
    pa.add_argument("--T", type=float, default=6)
    pa.add_argument("--dt", type=float, default=0.002)
    pa.add_argument("--accuracy", action="store_true")
    pa.add_argument("--resume", type=Path)
    pa.add_argument("--render", action="store_true")
    pa.add_argument("--layers", type=float, nargs="*", default=None)
    pa.add_argument("--out", type=Path, default=None)
    args = pa.parse_args(argv)
    if args.x_boundary is None:
        args.x_boundary = "fixed" if args.accuracy else "open"
    fixed = args.accuracy or args.x_boundary == "fixed"
    if args.extension is None:
        args.extension = 0.0 if fixed else 1.0
    if args.sponge_strength is None:
        args.sponge_strength = 4.0 if args.extension > 0 and not fixed else 0.0
    if args.nx is None:
        args.nx = 96 if fixed else int(np.ceil(32 * (3 + args.extension)))
    if args.layers is None:
        args.layers = [0.002, 0.008, 0.032] if fixed else []
    if args.out is None:
        args.out = Path(
            "build/kh_stream/accuracy_default"
            if args.accuracy
            else "build/kh_stream/default_buffer1"
        )
    return args


def main():
    jax.config.update("jax_enable_x64", True)
    args = parse_args()
    if (
        args.dt <= 0
        or args.T <= 0
        or not np.isclose(round(0.2 / args.dt) * args.dt, 0.2)
        or (not args.accuracy and not np.isclose(round(args.T / 0.2) * 0.2, args.T))
    ):
        raise ValueError(
            "Require positive dt dividing .2 and T a positive multiple of .2"
        )
    if args.extension < 0 or args.sponge_strength < 0:
        raise ValueError("extension and sponge strength must be nonnegative")
    if (args.extension or args.sponge_strength) and (
        args.accuracy or args.x_boundary == "fixed"
    ):
        raise ValueError("Extension/sponge requires open KH mode")
    if args.sponge_strength and not args.extension:
        raise ValueError("Sponge requires an external extension")
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    domain = (
        ((-1, 1), (-1, 1))
        if args.accuracy
        else ((-3 - args.extension, 3 + args.extension), (-1, 1))
    )
    if args.basis_from:
        from dynamic_boundary_common import load_open_plan

        if args.x_boundary == "fixed":
            raise ValueError("Cached open factors cannot impose fixed vertical faces")
        p = load_open_plan(args.basis_from)
        if (
            len(p.x.x) != args.nx
            or len(p.y.x) != args.ny
            or not np.allclose(p.x.x, np.linspace(*domain[0], args.nx))
            or not np.allclose(p.y.x, np.linspace(*domain[1], args.ny))
            or not np.array_equal(np.asarray(p.x.layers), np.asarray(args.layers))
        ):
            raise ValueError("Cached factors differ from requested grid or layers")
        if args.x_boundary == "dynamic":
            p = with_stream_dynamic_boundary(p, D0=args.boundary_d0)
    else:
        p = plan(
            args.nx,
            args.ny,
            domain=domain,
            layers=args.layers,
            x_boundary=args.x_boundary,
            boundary_D0=args.boundary_d0,
        )
    if args.accuracy and args.x_boundary != "fixed":
        raise ValueError("Use the dedicated open-boundary manufactured validation")
    if args.accuracy:
        u, lap, gradp, adv = fields(p.x.points, p.y.points)
        a = load(p, u) / p.denominator
        l1 = load(p, -u - 0.002 * lap + gradp)
        l2 = load(p, adv)
        exact = fields(p.x.x, p.y.x)[0]
        print(
            json.dumps(
                dict(
                    n=args.nx,
                    initial_error=float(
                        jnp.max(abs(velocity(p, a, nodes=True) - exact))
                    ),
                    pressure_gradient=float(
                        jnp.max(abs(load(p, gradp) / p.denominator))
                    ),
                )
            ),
            flush=True,
        )

        def advance(a, t, steps):
            def step(i, a):
                ti = t + i * args.dt

                def fun(a, t):
                    return rhs(p, a, jnp.exp(-t) * l1 + jnp.exp(-2 * t) * l2)

                k1 = fun(a, ti)
                k2 = fun(a + args.dt / 2 * k1, ti + args.dt / 2)
                k3 = fun(a + args.dt / 2 * k2, ti + args.dt / 2)
                k4 = fun(a + args.dt * k3, ti + args.dt)
                return a + args.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            return jax.lax.fori_loop(0, steps, step, a)

        advance = jax.jit(advance, static_argnums=2)
        a = advance(a, 0.0, round(args.T / args.dt))
        result = dict(
            n=args.nx,
            T=args.T,
            dt=args.dt,
            pde_linf=float(
                jnp.max(abs(velocity(p, a, nodes=True) - jnp.exp(-args.T) * exact))
            ),
            elapsed=time.perf_counter() - start,
        )
        print(json.dumps(result), flush=True)
        (args.out / "accuracy.json").write_text(json.dumps(result, indent=2))
        return
    sponge = (
        plan_stream_sponge(p, strength=args.sponge_strength)
        if args.sponge_strength
        else None
    )
    if args.seed_cutoff_width is not None and (
        not np.isfinite(args.seed_cutoff_width) or args.seed_cutoff_width <= 0
    ):
        raise ValueError("seed cutoff width must be finite and positive")
    cutoff_width = (
        args.seed_cutoff_width
        if args.seed_cutoff_width is not None
        else min(1.0, args.extension)
    )
    cutoff = (3.0, 3.0 + cutoff_width) if args.extension else None
    a = kh_seed(p, extension_cutoff=cutoff)
    initial = np.asarray(a)
    t0 = 0.0
    if args.resume:
        previous = np.load(args.resume)
        a = jnp.asarray(previous["a"])
        initial = previous["initial"]
        t0 = float(previous["t"])
        check = velocity(p, a, nodes=True, thickness=0.12)
        if np.max(abs(np.asarray(check) - previous["velocity"])) > 1e-9:
            raise ValueError("Checkpoint basis does not match")
    force = -stream_ns_inertia_apply(p, rhs(p, jnp.zeros_like(a), thickness=0.12))
    advance = jax.jit(
        lambda a: jax.lax.fori_loop(
            0,
            round(0.2 / args.dt),
            lambda i, a: rk4(p, a, args.dt, force, thickness=0.12, sponge=sponge),
            a,
        )
    )
    rec = []
    omegas = []
    vs = []
    for i in range(round((args.T - t0) / 0.2) + 1):
        if i:
            a = advance(a)
        vel = np.asarray(velocity(p, a, nodes=True, thickness=0.12))
        om = np.asarray(vorticity(p, a, nodes=True, thickness=0.12))
        center = (abs(np.asarray(p.x.x)[:, None]) < 2) & (
            abs(np.asarray(p.y.x)[None, :]) < 0.5
        )
        r = dict(
            t=t0 + 0.2 * i,
            max_speed=float(np.max(np.linalg.norm(vel, axis=-1))),
            central_v_rms=float(np.sqrt(np.mean(vel[..., 1][center] ** 2))),
            max_vorticity=float(abs(om).max()),
            roi_max_vorticity=float(abs(om[abs(np.asarray(p.x.x)) <= 3]).max()),
            pointwise_divergence_linf=float(jnp.max(abs(stream_ns_divergence(p, a)))),
            vertical_boundary_v_linf=float(abs(vel[[0, -1], :, 1]).max()),
            boundary_linf=float(
                max(
                    (
                        0
                        if args.x_boundary != "fixed"
                        else np.max(
                            abs(
                                vel[[0, -1], :, 0]
                                - np.tanh(np.asarray(p.y.x)[None, :] / 0.12)
                            )
                        )
                    ),
                    (
                        0
                        if args.x_boundary != "fixed"
                        else np.max(abs(vel[[0, -1], :, 1]))
                    ),
                    np.max(
                        abs(
                            vel[:, [0, -1], 0]
                            - np.tanh(np.asarray(p.y.x)[None, [0, -1]] / 0.12)
                        )
                    ),
                    np.max(abs(vel[:, [0, -1], 1])),
                )
            ),
            elapsed=time.perf_counter() - start,
        )
        rec.append(r)
        omegas.append(om)
        vs.append(vel[..., 1])
        print(json.dumps(r), flush=True)
        np.savez_compressed(
            args.out / "checkpoint.npz",
            a=a,
            initial=initial,
            velocity=vel,
            t=t0 + 0.2 * i,
            x=p.x.x,
            y=p.y.x,
        )
        (args.out / "diagnostics.json").write_text(json.dumps(rec, indent=2))
        if not np.all(np.isfinite(vel)) or r["max_speed"] > 10:
            raise RuntimeError("State failed")
    np.savez_compressed(
        args.out / "frames.npz",
        x=p.x.x,
        y=p.y.x,
        times=[r["t"] for r in rec],
        vorticity=omegas,
        transverse_velocity=vs,
    )
    np.savez_compressed(
        args.out / "basis.npz",
        **{"x_" + k: np.asarray(v) for k, v in zip(p.x._fields, p.x)},
        **{"y_" + k: np.asarray(v) for k, v in zip(p.y._fields, p.y)},
    )

    (args.out / "summary.json").write_text(
        json.dumps(
            dict(
                nx=args.nx,
                ny=args.ny,
                T=args.T,
                dt=args.dt,
                viscosity=0.002,
                layers=args.layers,
                boundary_D0=args.boundary_d0 if args.x_boundary == "dynamic" else 0.0,
                buffer=bool(args.sponge_strength),
                extension=args.extension,
                sponge_strength=args.sponge_strength,
                interest_region=[-3, 3],
                seed_extension_cutoff=cutoff,
                dx=float(p.x.x[1] - p.x.x[0]),
                filter=False,
                refinement_steps=0,
                max_speed=max(r["max_speed"] for r in rec),
                max_pointwise_divergence=max(
                    r["pointwise_divergence_linf"] for r in rec
                ),
                max_boundary_error=max(r["boundary_linf"] for r in rec),
                method="BSPF curl trial/test functions; optional exponential enrichment; direct tensor Poisson inertia solve; RK4",
                boundary=(
                    "Dynamic Dong traction with reference shear compensation; fixed horizontal faces"
                    if args.x_boundary == "dynamic"
                    else "Open vertical faces with incoming shear reservoir; fixed horizontal faces"
                    if args.x_boundary != "fixed"
                    else "Fixed (tanh(y/.12),0) on all four walls"
                ),
                forcing="Fixed load maintaining base shear",
            ),
            indent=2,
        )
    )
    if args.render:
        from report_kh_stream import render_movie

        render_movie(args.out)


if __name__ == "__main__":
    main()
