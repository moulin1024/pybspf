"""Render independently validated convergence, physical fields, and hole extension."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("build/immersed_poisson"))
    args = parser.parse_args()
    out = args.out
    records = json.loads((out / "results.json").read_text())
    styles = {
        "random4": ("64 random waves: |k| <= 4 pi", "o"),
        "random8": ("64 random waves: |k| <= 8 pi", "s"),
        "pole": ("Log singularity hidden inside hole", "^"),
    }
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.5), constrained_layout=True)
    for ax, key, title in zip(
        axes.ravel(),
        (
            "grid_relative_u",
            "grid_relative_gradient",
            "grid_relative_pde",
            "boundary_max_u",
            "collar_max_gradient",
            "collar_relative_pde",
        ),
        (
            "Fluid: relative solution L2",
            "Fluid: relative gradient L2",
            "Fluid: relative PDE residual L2",
            "Independent wall: max |u - g|",
            "Near wall: max gradient error",
            "Near wall: relative PDE residual L2",
        ),
    ):
        for kind, (label, marker) in styles.items():
            rows = [r for r in records if r["case"] == kind]
            ax.semilogy(
                [r["nodes"] for r in rows],
                [r[key] for r in rows],
                marker + "-",
                ms=4,
                label=label,
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("BSPF nodes per axis N (not validation resolution)")
        ax.grid(True, which="both", alpha=0.22)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "BSPF smooth-field immersed Poisson: fixed eccentric analytic hole\n"
        "Independent 257 x 257 fluid samples; shifted boundary and normal collars down to distance 1e-6",
        fontsize=12,
    )
    fig.savefig(out / "convergence.png", dpi=180)
    plt.close(fig)
    for kind in ("random8", "pole"):
        nodes = max(r["nodes"] for r in records if r["case"] == kind)
        data = np.load(out / f"{kind}_n{nodes}.npz")
        x, y = data["x"], data["y"]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), constrained_layout=True)
        extent = [x[0], x[-1], y[0], y[-1]]
        a = axes[0].imshow(data["exact"], origin="lower", extent=extent, cmap="RdBu_r")
        fig.colorbar(a, ax=axes[0], shrink=0.8)
        axes[0].set_title(f"{kind}: MMS in fluid, N={nodes}")
        error = np.abs(data["error"])
        a = axes[1].imshow(
            error,
            origin="lower",
            extent=extent,
            cmap="magma",
            norm=LogNorm(
                vmin=max(np.nanmax(error) * 1e-5, 1e-16),
                vmax=max(np.nanmax(error), 1e-15),
            ),
        )
        fig.colorbar(a, ax=axes[1], shrink=0.8)
        axes[1].set_title("Absolute solution error (fluid only)")
        a = axes[2].imshow(
            data["numerical"], origin="lower", extent=extent, cmap="RdBu_r"
        )
        fig.colorbar(a, ax=axes[2], shrink=0.8)
        axes[2].set_title("Jointly solved whole-box field")
        for i, ax in enumerate(axes):
            ax.add_patch(
                Ellipse(
                    data["center"],
                    *(2 * data["axes"]),
                    facecolor=".8" if i < 2 else "none",
                    edgecolor="black",
                    lw=1,
                )
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")
        axes[2].text(
            data["center"][0],
            data["center"][1],
            "Auxiliary\nfield",
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )
        fig.suptitle(
            "Physical problem: rectangle minus ellipse; values inside ellipse are NOT fluid values",
            fontsize=11,
        )
        fig.savefig(out / f"{kind}_fields.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
