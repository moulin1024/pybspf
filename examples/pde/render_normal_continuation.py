"""Plot independent high-wave-number collar errors and extrapolation gain."""

import json
from pathlib import Path

import matplotlib
import numpy as np
from numpy.polynomial import chebyshev as ch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    out = Path("build/normal_continuation")
    rows = json.loads((out / "results.json").read_text())["results"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for i, domain in enumerate(("convex", "nonstar")):
        for j, metric in enumerate(("value", "gradient", "laplacian")):
            for degree in (6, 8, 10, 14):
                selected = sorted(
                    (
                        r
                        for r in rows
                        if r["domain"] == domain
                        and r["band"] == 12
                        and r["normal_degree"] == degree
                        and r["tangent_degree"] == 36
                        and r["location"] == "exterior_half"
                    ),
                    key=lambda r: r["width"],
                )
                axes[i, j].loglog(
                    [r["width"] for r in selected],
                    [r[metric] for r in selected],
                    "o-",
                    label=f"normal degree {degree}",
                )
            axes[i, j].set(
                title=f"{domain}: {metric}",
                xlabel="Interior collar width",
                ylabel="Relative error (independent samples)",
            )
            axes[i, j].grid(alpha=0.2)
            axes[i, j].legend(fontsize=8)
    fig.suptitle(
        "Local normal continuation | kmax=12 pi | exterior distance < 0.5 collar width"
    )
    fig.savefig(out / "errors.png", dpi=170)
    fig.savefig(out / "errors.pdf")
    gains = []
    for degree in (6, 8, 10, 14):
        z = ch.chebpts2(degree + 1)
        cardinal = ch.chebfit(z, np.eye(degree + 1), degree)
        for ratio in (0.5, 1.0):
            gains.append(
                dict(
                    degree=degree,
                    exterior_ratio=ratio,
                    value_infinity_gain=float(
                        np.sum(abs(ch.chebval(1 + 2 * ratio, cardinal)))
                    ),
                )
            )
    (out / "extrapolation_gain.json").write_text(json.dumps(gains, indent=2) + "\n")


if __name__ == "__main__":
    main()
