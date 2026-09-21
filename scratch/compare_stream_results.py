"""Compare saved physical reconstructions, including resolved thin layers."""

import argparse
import json
from pathlib import Path
import numpy as np
from bspf_models._numerics.trial_spaces import StreamLine
from bspf_models._numerics.trial_spaces import stream_evaluate_line

parser = argparse.ArgumentParser()
parser.add_argument("--coarse", default="enriched64")
parser.add_argument("--fine", default="enriched96")
parser.add_argument("--output", default="grid_comparison")
args = parser.parse_args()
root = Path("build/kh_stream")
f = np.load(root / args.fine / "basis.npz")
fine = {
    axis: StreamLine(
        *(
            f[axis + "_" + k] if axis + "_" + k in f else None
            for k in StreamLine._fields
        )
    )
    for axis in ("x", "y")
}
af = np.load(root / args.fine / "checkpoint.npz")["a"]
c = np.load(root / args.coarse / "basis.npz")
ac = np.load(root / args.coarse / "checkpoint.npz")["a"]
bx, gx, hx = stream_evaluate_line(fine["x"], c["x_points"])
by, gy, hy = stream_evaluate_line(fine["y"], c["y_points"])
vf = np.stack((bx @ af @ gy.T, -gx @ af @ by.T), axis=-1)
vc = np.stack((c["x_b"] @ ac @ c["y_g"].T, -c["x_g"] @ ac @ c["y_b"].T), axis=-1)
w = c["x_weights"][:, None] * c["y_weights"][None, :]
center = (abs(c["x_points"][:, None]) < 2) & (abs(c["y_points"][None, :]) < 0.5)
res = {}
for name, mask in [("full", np.ones_like(w)), ("center", center)]:
    res[name + "_relative_velocity_l2"] = float(
        np.sqrt(
            np.sum(w[..., None] * mask[..., None] * (vf - vc) ** 2)
            / np.sum(w[..., None] * mask[..., None] * vf**2)
        )
    )
    res[name + "_velocity_linf"] = float(np.max(abs(vf - vc) * mask[..., None]))
res["fine_pointwise_divergence"] = float(abs((gx @ af) @ gy.T - gx @ (af @ gy.T)).max())
res["coarse_pointwise_divergence"] = float(
    abs((c["x_g"] @ ac) @ c["y_g"].T - c["x_g"] @ (ac @ c["y_g"].T)).max()
)
np.savez_compressed(
    root / (args.output + ".npz"),
    x=c["x_points"],
    y=c["y_points"],
    coarse=vc,
    fine=vf,
    weights=w,
)
(root / (args.output + ".json")).write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
