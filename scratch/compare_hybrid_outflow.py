"""Common-quadrature comparison of narrow hybrid layers to saved L=2 reference."""

import json
from pathlib import Path
import numpy as np
import jax

from dynamic_boundary_common import load_open_plan
from bspf_jax.stream_navier_stokes import stream_evaluate_line

jax.config.update("jax_enable_x64", True)
root = Path("build/kh_stream")
names = [
    "extended160_s4",
    "hybrid_L0.5_s4_open",
    "hybrid_L0.5_s4_dynamic",
    "hybrid_L1_s4_open",
    "hybrid_L1_s4_dynamic",
]
common = load_open_plan(root / "open96")
x = np.asarray(common.x.points)
y = np.asarray(common.y.points)
w = np.asarray(common.x.weights[:, None] * common.y.weights[None, :])
reference = None
initial_reference = None
results = {}
for name in names:
    p = load_open_plan(root / name)
    state = np.load(root / name / "checkpoint.npz")
    bx, gx, _ = stream_evaluate_line(p.x, x)
    # All cases use the identical y interval, node count and basis construction.
    if not np.allclose(p.y.x, common.y.x) or not np.allclose(
        p.y.b, common.y.b, atol=1e-12, rtol=1e-12
    ):
        raise ValueError("y bases differ")
    by, gy = np.asarray(p.y.b), np.asarray(p.y.g)

    def evaluate(a):
        return np.stack((bx @ a @ gy.T, -gx @ a @ by.T), axis=-1)

    velocity = evaluate(state["a"])
    initial = evaluate(state["initial"])
    if reference is None:
        reference = velocity
        initial_reference = initial
    summary = json.loads((root / name / "summary.json").read_text())
    records = json.loads((root / name / "diagnostics.json").read_text())
    result = {
        key: summary[key]
        for key in [
            "nx",
            "ny",
            "extension",
            "sponge_strength",
            "max_speed",
            "max_pointwise_divergence",
            "max_boundary_error",
        ]
    }
    result["initial_velocity_linf_difference"] = float(
        abs(initial - initial_reference).max()
    )
    for label, mask in [
        ("interest", np.ones_like(w)),
        ("center", (abs(x[:, None]) < 2) & (abs(y[None, :]) < 0.5)),
    ]:
        weight = w * mask
        result[label + "_relative_perturbation_l2"] = float(
            np.sqrt(
                np.sum(weight[..., None] * (velocity - reference) ** 2)
                / np.sum(weight[..., None] * reference**2)
            )
        )
        result[label + "_absolute_velocity_l2"] = float(
            np.sqrt(np.sum(weight[..., None] * (velocity - reference) ** 2))
        )
        result[label + "_velocity_linf"] = float(
            np.max(abs(velocity - reference) * mask[..., None])
        )
    f = np.load(root / name / "frames.npz")
    mask = (
        (abs(f["x"][:, None]) > 2.8)
        & (abs(f["x"][:, None]) <= 3)
        & (abs(f["y"][None, :]) < 0.9)
    )
    result["run_interest_edge_vorticity_peak"] = float(
        abs(f["vorticity"][:, mask]).max()
    )
    result["final_interest_edge_vorticity_peak"] = float(
        abs(f["vorticity"][-1, mask]).max()
    )
    result["final_roi_max_vorticity"] = records[-1]["roi_max_vorticity"]
    results[name] = result
    print(json.dumps({name: result}), flush=True)
(root / "hybrid_comparison.json").write_text(json.dumps(results, indent=2))
