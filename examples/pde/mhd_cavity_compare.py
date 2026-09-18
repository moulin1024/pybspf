"""Compare saved MHD runs on shared times/nodes without field interpolation."""

import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("first", type=Path)
    ap.add_argument("second", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    a, b = (np.load(path / "evolution.npz") for path in (args.first, args.second))
    sa, sb = (
        json.loads((path / "summary.json").read_text())
        for path in (args.first, args.second)
    )
    for key in (
        "reynolds",
        "magnetic_reynolds",
        "peak_initial_field",
        "fluid_time_offset",
    ):
        if not np.isclose(sa[key], sb[key]):
            raise ValueError(f"Physical configuration mismatch: {key}")
    common = np.intersect1d(a["x"], b["x"])
    if len(common) < 3:
        raise ValueError("Need at least 3 exactly shared nodes")
    ia, ib = np.searchsorted(a["x"], common), np.searchsorted(b["x"], common)
    time_pairs = []
    for i, t in enumerate(a["time"]):
        j = np.argmin(abs(b["time"] - t))
        if abs(b["time"][j] - t) < 1e-12:
            time_pairs.append((i, j))
    if len(time_pairs) < 2:
        raise ValueError("Need at least two matching output times")
    result = dict(
        first=str(args.first),
        second=str(args.second),
        shared_nodes_per_axis=len(common),
        final_shared_time=float(a["time"][time_pairs[-1][0]]),
        shared_frames=len(time_pairs),
        note="Node-sampled differences, not full-domain or exact-solution error bounds.",
    )
    for field in ("velocity", "magnetic"):
        differences = []
        for i, j in time_pairs:
            va = a[field][i][ia[:, None], ia[None, :]]
            vb = b[field][j][ib[:, None], ib[None, :]]
            differences.append(float(np.max(np.linalg.norm(va - vb, axis=-1))))
        result[field + "_max_shared_node_vector_difference"] = max(differences)
    for energy in ("kinetic_energy", "magnetic_energy"):
        result[energy + "_max_difference"] = max(
            abs(sa["history"][i][energy] - sb["history"][j][energy])
            for i, j in time_pairs
        )
    result["peak_center_speeds"] = [
        max(
            s["history"][i]["center_max_speed"]
            for i in [pair[k] for pair in time_pairs]
        )
        for k, s in enumerate((sa, sb))
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
