"""Compare cavity runs at their exactly shared physical nodes (no interpolation).

Usage: python examples/pde/cavity_compare.py build/cavity_n33 build/cavity_2d build/cavity_n49
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=Path("build/cavity_2d/refinement.json"))
    args = ap.parse_args()
    if len(args.runs) < 2:
        ap.error("provide at least two runs")
    states = [dict(np.load(path / "state.npz")) for path in args.runs]
    summaries = [json.loads((path / "summary.json").read_text()) for path in args.runs]
    for key in ("time", "dt", "reynolds", "ramp_time"):
        if not all(
            np.isclose(state[key], states[0][key], rtol=0, atol=1e-12)
            for state in states
        ):
            raise ValueError(f"Runs must have identical {key}")
    common = states[0]["x"]
    for state in states[1:]:
        common = np.intersect1d(common, state["x"])
    if len(common) < 3:
        raise ValueError(
            "Fewer than three exactly shared nodes; choose compatible grids"
        )
    samples = []
    for state in states:
        indices = np.searchsorted(state["x"], common)
        samples.append(state["velocity"][indices[:, None], indices[None, :], :])
    pairs = []
    for i in range(len(states) - 1):
        pairs.append(
            dict(
                grids=[len(states[i]["x"]), len(states[i + 1]["x"])],
                velocity_shared_nodes_linf=float(
                    np.max(abs(samples[i + 1] - samples[i]))
                ),
                kinetic_energy_absolute_difference=abs(
                    summaries[i + 1]["final"]["kinetic_energy"]
                    - summaries[i]["final"]["kinetic_energy"]
                ),
            )
        )
    result = dict(
        time=float(states[0]["time"]),
        dt=float(states[0]["dt"]),
        shared_coordinates=common.tolist(),
        pairs=pairs,
        note="Sampled differences on shared nodes, not a full-domain error bound or an exact-solution error.",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
