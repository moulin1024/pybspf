"""Small vortex crossing: static and dynamic OBC vs a longer no-sponge domain."""

import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from dynamic_boundary_common import load_open_plan
from bspf_jax.stream_navier_stokes import (
    with_stream_dynamic_boundary,
    stream_ns_load,
    stream_ns_rhs,
    stream_ns_rk4_step,
    stream_ns_inertia_apply,
    stream_ns_velocity,
    stream_ns_vorticity,
    stream_evaluate_line,
)

jax.config.update("jax_enable_x64", True)
start = time.perf_counter()
out = Path("build/kh_stream/dynamic_vortex")
out.mkdir(parents=True, exist_ok=True)
p = load_open_plan("build/kh_stream/open96")
r = load_open_plan("build/kh_stream/extended160_s4")
labels = ["relaxation_D0_0", "dong_D0_0", "dong_D0_1", "dong_D0_2"]
plans = [
    with_stream_dynamic_boundary(p, D0=d, dong_backflow=dong)
    for d, dong in [(0, False), (0, True), (1, True), (2, True)]
]
batch = jax.tree.map(lambda *v: jnp.stack(v), *plans)


def seed(p):
    x, y = p.x.points[:, None], p.y.points[None, :]
    radius = 0.18
    e = jnp.exp(-((x - 1.8) ** 2 + (y - 0.45) ** 2) / radius**2)
    g = (1 - y * y) ** 2
    u = 0.015 * e * (-2 * (y - 0.45) / radius**2 * g - 4 * y * (1 - y * y))
    v = 0.015 * e * 2 * (x - 1.8) / radius**2 * g
    return stream_ns_load(p, jnp.stack((u, v), axis=-1)) / p.denominator


a = jnp.stack([seed(p)] * len(plans))
ar = seed(r)
forces = jax.vmap(
    lambda p, a: (
        -stream_ns_inertia_apply(p, stream_ns_rhs(p, jnp.zeros_like(a), thickness=0.12))
    )
)(batch, a)
fr = -stream_ns_inertia_apply(r, stream_ns_rhs(r, jnp.zeros_like(ar), thickness=0.12))
advance = jax.jit(
    lambda a: jax.lax.fori_loop(
        0,
        100,
        lambda i, a: jax.vmap(
            lambda p, a, f: stream_ns_rk4_step(p, a, 0.002, f, thickness=0.12)
        )(batch, a, forces),
        a,
    )
)
advance_ref = jax.jit(
    lambda a: jax.lax.fori_loop(
        0, 100, lambda i, a: stream_ns_rk4_step(r, a, 0.002, fr, thickness=0.12), a
    )
)
bx, gx, _ = map(jnp.asarray, stream_evaluate_line(r.x, np.asarray(p.x.points)))
bxn, gxn, _ = map(jnp.asarray, stream_evaluate_line(r.x, np.asarray(p.x.x)))
w = np.asarray(p.x.weights[:, None] * p.y.weights[None, :])
records = []
frames = []
reference_frames = []
for i in range(13):
    if i:
        a = advance(a)
        ar = advance_ref(ar)
    va = np.asarray(jax.vmap(lambda a: stream_ns_velocity(p, a))(a))
    vr = np.asarray(jnp.stack((bx @ ar @ r.y.g.T, -gx @ ar @ r.y.b.T), axis=-1))
    diag = {"t": i * 0.2, "elapsed": time.perf_counter() - start, "cases": {}}
    for k, label in enumerate(labels):
        diff = va[k] - vr
        diag["cases"][label] = {}
        for region, mask in [
            ("full", np.ones_like(w)),
            ("upstream", np.asarray(p.x.points[:, None]) < 2.5),
        ]:
            weight = w * mask
            diag["cases"][label][region + "_absolute_l2"] = float(
                np.sqrt(np.sum(weight[..., None] * diff**2))
            )
            diag["cases"][label][region + "_linf"] = float(
                np.max(abs(diff) * mask[..., None])
            )
        diag["cases"][label]["max_speed"] = float(
            np.linalg.norm(
                va[k]
                + np.stack(
                    (
                        np.broadcast_to(
                            np.tanh(np.asarray(p.y.points)[None, :] / 0.12),
                            va[k].shape[:2],
                        ),
                        np.zeros(va[k].shape[:2]),
                    ),
                    axis=-1,
                ),
                axis=-1,
            ).max()
        )
    records.append(diag)
    frames.append(
        np.asarray(
            jax.vmap(lambda a: stream_ns_vorticity(p, a, nodes=True, thickness=0.12))(a)
        )
    )
    # Transverse velocity reference is enough to show actual vortex passage.
    reference_frames.append(np.asarray(-gxn @ ar @ r.y.bn.T))
    print(json.dumps(diag), flush=True)
    (out / "diagnostics.json").write_text(json.dumps(records, indent=2))
    if not np.all(np.isfinite(va)):
        raise RuntimeError("Nonfinite state")
np.savez_compressed(
    out / "frames.npz",
    times=np.arange(13) * 0.2,
    x=p.x.x,
    y=p.y.x,
    vorticity=frames,
    reference_v=reference_frames,
    labels=labels,
)
summary = {
    label: {
        "integrated_full_l2": float(
            np.trapezoid(
                [v["cases"][label]["full_absolute_l2"] for v in records], dx=0.2
            )
        ),
        "peak_upstream_linf": max(v["cases"][label]["upstream_linf"] for v in records),
    }
    for label in labels
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
