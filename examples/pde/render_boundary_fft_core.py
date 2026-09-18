"""Plot independent errors and iteration counts of the boundary-only prototype."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    out = Path("build/boundary_fft_core")
    rows = json.loads((out / "results.json").read_text())["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for name, label in (
        ("circle", "Circle (analytic)"),
        ("ellipse", "Ellipse (analytic)"),
        ("convex_bspline", "Cubic B-spline (C2)"),
    ):
        data = [row for row in rows if row["geometry"] == name]
        for ax, key, title in zip(
            axes,
            ("independent_boundary_relative", "interior_relative", "iterations"),
            (
                "Independent boundary error",
                "Interior error (away from boundary)",
                "GMRES iterations",
            ),
        ):
            ax.plot(
                [r["boundary_points"] for r in data],
                [r[key] for r in data],
                "o-",
                label=label,
            )
            ax.set_xscale("log", base=2)
            if key != "iterations":
                ax.set_yscale("log")
            ax.set_xticks([32, 64, 128, 256, 512], labels=[32, 64, 128, 256, 512])
            ax.set_title(title)
            ax.set_xlabel("M (boundary unknowns)")
            ax.grid(True, alpha=0.25)
    axes[0].legend()
    axes[2].set_ylim(0, 7)
    fig.suptitle(
        "Arclength FFT principal operator + geometry correction\n"
        "Laplace core only: no nonzero volume source or close-evaluation module"
    )
    fig.savefig(out / "convergence.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
