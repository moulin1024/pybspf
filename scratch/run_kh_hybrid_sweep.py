"""Matched short-sponge experiments: ordinary vs dynamic exterior boundary."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from bspf_jax.stream_navier_stokes import (
    plan_stream_navier_stokes2d,
    with_stream_dynamic_boundary,
    plan_stream_sponge,
    stream_kh_initial,
    stream_ns_rhs,
    stream_ns_inertia_apply,
    stream_ns_rk4_step,
    stream_ns_velocity,
    stream_ns_vorticity,
    stream_ns_divergence,
)

jax.config.update("jax_enable_x64", True)
pa = argparse.ArgumentParser()
pa.add_argument("--extension", type=float, required=True)
pa.add_argument("--nx", type=int, required=True)
pa.add_argument("--ny", type=int, default=80)
pa.add_argument("--strength", type=float, default=4.0)
pa.add_argument("--D0", type=float, default=1.0)
pa.add_argument("--T", type=float, default=12.0)
pa.add_argument("--dt", type=float, default=0.002)
args = pa.parse_args()
if not args.extension > 0 or not args.strength >= 0 or args.dt <= 0 or args.T <= 0:
    raise ValueError("Require positive extension/dt/T and nonnegative strength")
if not np.isclose(round(0.2 / args.dt) * args.dt, 0.2) or not np.isclose(
    round(args.T / 0.2) * 0.2, args.T
):
    raise ValueError("Require dt dividing .2 and T a multiple of .2")
start = time.perf_counter()
p = plan_stream_navier_stokes2d(
    np.linspace(-3 - args.extension, 3 + args.extension, args.nx),
    np.linspace(-1, 1, args.ny),
    x_boundary="open",
)
plans = [
    with_stream_dynamic_boundary(p, D0=0, dong_backflow=False),
    with_stream_dynamic_boundary(p, D0=args.D0),
]
batch = jax.tree.map(lambda *v: jnp.stack(v), *plans)
sponge = plan_stream_sponge(p, strength=args.strength)
# Same physical initial data as the existing L=2 exterior reference, restricted
# to each domain. It may be nonzero at the new open exterior faces, which is valid.
initial = stream_kh_initial(p, extension_cutoff=(3.0, 4.0))
a = jnp.stack([initial, initial])
forces = jax.vmap(
    lambda plan, a: (
        -stream_ns_inertia_apply(
            plan, stream_ns_rhs(plan, jnp.zeros_like(a), thickness=0.12)
        )
    )
)(batch, a)
advance = jax.jit(
    lambda a: jax.lax.fori_loop(
        0,
        round(0.2 / args.dt),
        lambda i, a: jax.vmap(
            lambda plan, a, f: stream_ns_rk4_step(
                plan, a, args.dt, f, thickness=0.12, sponge=sponge
            )
        )(batch, a, forces),
        a,
    )
)
labels = ["open", "dynamic"]
dirs = [
    Path(f"build/kh_stream/hybrid_L{args.extension:g}_s{args.strength:g}_{label}")
    for label in labels
]
rec = [[], []]
vorts = [[], []]
vs = [[], []]
states = [[], []]
for directory in dirs:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "basis.npz",
        **{"x_" + k: np.asarray(v) for k, v in zip(p.x._fields, p.x)},
        **{"y_" + k: np.asarray(v) for k, v in zip(p.y._fields, p.y)},
    )
print(
    json.dumps(
        dict(
            event="setup_complete",
            extension=args.extension,
            nx=args.nx,
            seconds=time.perf_counter() - start,
        )
    ),
    flush=True,
)
xx, yy = np.asarray(p.x.x), np.asarray(p.y.x)
roi = abs(xx) <= 3
edge = (abs(xx[:, None]) > 2.8) & (abs(xx[:, None]) <= 3) & (abs(yy[None, :]) < 0.9)
center = (abs(xx[:, None]) < 2) & (abs(yy[None, :]) < 0.5)
for i in range(round(args.T / 0.2) + 1):
    if i:
        a = advance(a)
    vel = np.asarray(
        jax.vmap(lambda a: stream_ns_velocity(p, a, nodes=True, thickness=0.12))(a)
    )
    om = np.asarray(
        jax.vmap(lambda a: stream_ns_vorticity(p, a, nodes=True, thickness=0.12))(a)
    )
    div = np.asarray(jax.vmap(lambda a: stream_ns_divergence(p, a))(a))
    for k, label in enumerate(labels):
        row = dict(
            t=i * 0.2,
            max_speed=float(np.linalg.norm(vel[k], axis=-1).max()),
            max_vorticity=float(abs(om[k]).max()),
            roi_max_vorticity=float(abs(om[k, roi]).max()),
            interest_edge_vorticity=float(abs(om[k])[edge].max()),
            central_v_rms=float(np.sqrt(np.mean(vel[k, ..., 1][center] ** 2))),
            pointwise_divergence_linf=float(abs(div[k]).max()),
            vertical_boundary_v_linf=float(abs(vel[k, [0, -1], :, 1]).max()),
            boundary_linf=float(
                max(
                    abs(
                        vel[k, :, [0, -1], 0] - np.tanh(yy[[0, -1], None] / 0.12)
                    ).max(),
                    abs(vel[k, :, [0, -1], 1]).max(),
                )
            ),
            elapsed=time.perf_counter() - start,
        )
        rec[k].append(row)
        vorts[k].append(om[k])
        vs[k].append(vel[k, ..., 1])
        states[k].append(np.asarray(a[k]))
        (dirs[k] / "diagnostics.json").write_text(json.dumps(rec[k], indent=2))
        np.savez_compressed(
            dirs[k] / "checkpoint.npz",
            a=a[k],
            initial=initial,
            velocity=vel[k],
            t=i * 0.2,
            x=p.x.x,
            y=p.y.x,
        )
        if not np.all(np.isfinite(vel[k])) or row["max_speed"] > 10:
            raise RuntimeError(f"{label} state failed")
    if i % 5 == 0:
        print(
            json.dumps(
                dict(extension=args.extension, open=rec[0][-1], dynamic=rec[1][-1])
            ),
            flush=True,
        )
for k, label in enumerate(labels):
    np.savez_compressed(
        dirs[k] / "frames.npz",
        x=p.x.x,
        y=p.y.x,
        times=[r["t"] for r in rec[k]],
        vorticity=vorts[k],
        transverse_velocity=vs[k],
    )
    np.savez_compressed(
        dirs[k] / "states.npz", times=[r["t"] for r in rec[k]], a=states[k]
    )
    summary = dict(
        nx=args.nx,
        ny=args.ny,
        T=args.T,
        dt=args.dt,
        viscosity=0.002,
        layers=[],
        boundary_D0=args.D0 if k else 0.0,
        buffer=True,
        extension=args.extension,
        sponge_strength=args.strength,
        interest_region=[-3, 3],
        seed_extension_cutoff=[3, 4],
        dx=float(xx[1] - xx[0]),
        filter=False,
        refinement_steps=0,
        max_speed=max(r["max_speed"] for r in rec[k]),
        max_pointwise_divergence=max(r["pointwise_divergence_linf"] for r in rec[k]),
        max_boundary_error=max(r["boundary_linf"] for r in rec[k]),
        max_interest_edge_vorticity=max(r["interest_edge_vorticity"] for r in rec[k]),
        boundary="Dynamic Dong traction with reference shear compensation; fixed horizontal faces"
        if k
        else "Open vertical faces with incoming shear reservoir; fixed horizontal faces",
        forcing="Fixed base-balancing load plus external relaxation",
        method="BSPF curl Galerkin, direct inertia solve, RK4",
    )
    (dirs[k] / "summary.json").write_text(json.dumps(summary, indent=2))
print(
    json.dumps(
        dict(
            event="finished",
            directories=list(map(str, dirs)),
            seconds=time.perf_counter() - start,
        )
    ),
    flush=True,
)
