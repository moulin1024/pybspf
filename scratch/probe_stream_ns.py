"""Bounded tests of compatible BSPF streamfunction NS, with checkpoints."""

import argparse, json, time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from bspf_stream_ns import plan, velocity, vorticity, load, rhs, rk4, kh_seed
from validate_weak_ns_accuracy import fields


def main():
    jax.config.update("jax_enable_x64", True)
    pa = argparse.ArgumentParser()
    pa.add_argument("--nx", type=int, default=64)
    pa.add_argument("--ny", type=int, default=64)
    pa.add_argument("--T", type=float, default=6)
    pa.add_argument("--dt", type=float, default=0.004)
    pa.add_argument("--accuracy", action="store_true")
    pa.add_argument("--resume", type=Path)
    pa.add_argument("--layers", type=float, nargs="*", default=[])
    pa.add_argument("--out", type=Path, default=Path("build/kh_stream/64x64"))
    args = pa.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    p = plan(
        args.nx,
        args.ny,
        domain=((-1, 1), (-1, 1)) if args.accuracy else ((-3, 3), (-1, 1)),
        layers=args.layers,
    )
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
                fun = lambda a, t: rhs(p, a, jnp.exp(-t) * l1 + jnp.exp(-2 * t) * l2)
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
    a = kh_seed(p)
    initial = np.asarray(a)
    t0=0.
    if args.resume:
        previous=np.load(args.resume)
        a=jnp.asarray(previous['a']);initial=previous['initial'];t0=float(previous['t'])
        check=velocity(p,a,nodes=True,thickness=.12)
        if np.max(abs(np.asarray(check)-previous['velocity']))>1e-9:
            raise ValueError("Checkpoint basis does not match")
    force = -rhs(p, jnp.zeros_like(a), thickness=0.12) * p.denominator
    advance = jax.jit(
        lambda a: jax.lax.fori_loop(
            0,
            round(0.2 / args.dt),
            lambda i, a: rk4(p, a, args.dt, force, thickness=0.12),
            a,
        )
    )
    rec = []
    omegas = []
    vs = []
    for i in range(round((args.T-t0) / 0.2) + 1):
        if i:
            a = advance(a)
        vel = np.asarray(velocity(p, a, nodes=True, thickness=0.12))
        om = np.asarray(vorticity(p, a, nodes=True, thickness=0.12))
        center = (abs(np.asarray(p.x.x)[:, None]) < 2) & (
            abs(np.asarray(p.y.x)[None, :]) < 0.5
        )
        r = dict(
            t=t0+0.2 * i,
            max_speed=float(np.max(np.linalg.norm(vel, axis=-1))),
            central_v_rms=float(np.sqrt(np.mean(vel[..., 1][center] ** 2))),
            max_vorticity=float(abs(om).max()),
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
            t=t0+0.2 * i,
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


if __name__ == "__main__":
    main()
