"""Repeat an extended KH run with another sponge strength using saved 1D factors."""

import argparse
from pathlib import Path
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
from bspf_models._numerics.trial_spaces import StreamLine
from bspf_models.fluids.stream_navier_stokes import StreamNavierStokes2DPlan
from bspf_models.fluids.stream_navier_stokes import plan_stream_sponge
from bspf_models.fluids.stream_navier_stokes import stream_ns_velocity
from bspf_models.fluids.stream_navier_stokes import stream_ns_vorticity
from bspf_models.fluids.stream_navier_stokes import stream_ns_rhs
from bspf_models.fluids.stream_navier_stokes import stream_ns_rk4_step
from bspf_models.fluids.stream_navier_stokes import stream_ns_divergence

jax.config.update("jax_enable_x64", True)
pa = argparse.ArgumentParser()
pa.add_argument(
    "--reference", type=Path, default=Path("build/kh_stream/extended160_s4")
)
pa.add_argument("--strength", type=float, default=2)
pa.add_argument("--out", type=Path, default=Path("build/kh_stream/extended160_s2"))
args = pa.parse_args()
start = time.perf_counter()
summary = json.loads((args.reference / "summary.json").read_text())
f = np.load(args.reference / "basis.npz")
lines = [
    StreamLine(*(jnp.asarray(f[axis + "_" + k]) for k in StreamLine._fields))
    for axis in ("x", "y")
]
p = StreamNavierStokes2DPlan(
    *lines, lines[0].lam[:, None] + lines[1].lam[None, :], jnp.asarray(summary["viscosity"]), 1.0
)
state = np.load(args.reference / "checkpoint.npz")
a = jnp.asarray(state["initial"])
initial = np.asarray(a)
sponge = plan_stream_sponge(p, strength=args.strength)
force = -stream_ns_rhs(p, jnp.zeros_like(a), thickness=0.12) * p.denominator
dt = summary["dt"]
advance = jax.jit(
    lambda a: jax.lax.fori_loop(
        0,
        round(0.2 / dt),
        lambda i, a: stream_ns_rk4_step(p, a, dt, force, thickness=0.12, sponge=sponge),
        a,
    )
)
args.out.mkdir(parents=True, exist_ok=True)
rec = []
oms = []
vs = []
for i in range(round(summary["T"] / 0.2) + 1):
    if i:
        a = advance(a)
    vel = np.asarray(stream_ns_velocity(p, a, nodes=True, thickness=0.12))
    om = np.asarray(stream_ns_vorticity(p, a, nodes=True, thickness=0.12))
    r = dict(
        t=0.2 * i,
        max_speed=float(np.linalg.norm(vel, axis=-1).max()),
        max_vorticity=float(abs(om).max()),
        roi_max_vorticity=float(abs(om[abs(np.asarray(p.x.x)) <= 3]).max()),
        pointwise_divergence_linf=float(abs(stream_ns_divergence(p, a)).max()),
        elapsed=time.perf_counter() - start,
    )
    if not np.all(np.isfinite(vel)) or r["max_speed"] > 10:
        raise RuntimeError("State failed")
    rec.append(r)
    oms.append(om)
    vs.append(vel[..., 1])
    if i % 5 == 0:
        print(json.dumps(r), flush=True)
    (args.out / "diagnostics.json").write_text(json.dumps(rec, indent=2))
np.savez_compressed(
    args.out / "frames.npz",
    x=p.x.x,
    y=p.y.x,
    times=[r["t"] for r in rec],
    vorticity=oms,
    transverse_velocity=vs,
)
np.savez_compressed(
    args.out / "checkpoint.npz",
    a=a,
    initial=initial,
    velocity=vel,
    t=summary["T"],
    x=p.x.x,
    y=p.y.x,
)
# Quantitative physical velocity differences against identical-grid reference.
a_ref = jnp.asarray(state["a"])
u = np.asarray(stream_ns_velocity(p, a))
u_ref = np.asarray(stream_ns_velocity(p, a_ref))
w = np.asarray(p.x.weights[:, None] * p.y.weights[None, :])
result = {}
for label, limit in [("interest", 3), ("center", 2)]:
    mask = abs(np.asarray(p.x.points[:, None])) < limit
    denom = np.sum(w[..., None] * mask[..., None] * u_ref**2)
    result[label + "_relative_perturbation_l2"] = float(
        np.sqrt(np.sum(w[..., None] * mask[..., None] * (u - u_ref) ** 2) / denom)
    )
    result[label + "_velocity_linf"] = float(np.max(abs(u - u_ref) * mask[..., None]))
summary.update(
    sponge_strength=args.strength,
    max_speed=max(r["max_speed"] for r in rec),
    max_pointwise_divergence=max(r["pointwise_divergence_linf"] for r in rec),
)
summary.pop("max_boundary_error", None)
(args.out / "summary.json").write_text(json.dumps(summary, indent=2))
(args.out / "sensitivity.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
