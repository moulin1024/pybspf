"""Read flushed channel diagnostics without waiting for final field output."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(log, out):
    records = []
    for line in log.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and "t" in record and "acceleration_l2" in record:
            records.append(record)
    if not records:
        raise ValueError("No complete time-step diagnostics in the log yet")
    out.mkdir(parents=True, exist_ok=True)
    t = np.array([r["t"] for r in records])
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for ax, key, title in zip(
        axes.flat[:3],
        ("kinetic_energy", "acceleration_l2", "max_speed"),
        ("Kinetic energy", "Velocity change: ||du/dt|| L2", "Maximum speed (quadrature points)"),
    ):
        ax.plot(t, [r[key] for r in records], "o-", ms=3)
        ax.set(title=title, xlabel="Time")
        if key == "acceleration_l2":
            ax.set_yscale("log")
    ax = axes[1, 1]
    ax.semilogy(t, [max(abs(r["flux_out"] - r["flux_in"]), 1e-16) for r in records], "o-", ms=3)
    ax.set(title="Absolute inlet/outlet flux difference", xlabel="Time")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(f"Live diagnostics | BSPF + rational correction | last saved t={t[-1]:g}")
    fig.savefig(out / "progress.png", dpi=170)
    plt.close(fig)
    (out / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records[-1], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("build/immersed_flow/hybrid/channel.log"))
    parser.add_argument("--out", type=Path, default=Path("build/immersed_flow/hybrid/channel"))
    args = parser.parse_args()
    main(args.log, args.out)
