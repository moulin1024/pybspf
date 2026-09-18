"""Summarize regularized continuation and its actual Poisson coupling."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    out = Path("build/regularized_normal")
    rows = json.loads((out / "results.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for i, domain in enumerate(("convex", "nonstar")):
        ns = [
            r
            for r in rows
            if r["domain"] == domain
            and r["stage"] == "normal_values"
            and r["input_noise"] > 0
        ]
        for metric in ("value", "normal_derivative_1", "normal_derivative_2"):
            axes[i, 0].semilogy(
                [r["retained"] for r in ns], [r[metric] for r in ns], "o-", label=metric
            )
        axes[i, 0].set(
            title=f"{domain}: input noise 1e-12",
            xlabel="Retained normal degree",
            ylabel="Relative error",
        )
        axes[i, 0].legend(fontsize=8)
        ps = [r for r in rows if r["domain"] == domain and r["stage"] == "poisson"]
        for metric in ("value", "gradient", "laplacian"):
            axes[i, 1].semilogy(
                range(len(ps)), [r[metric] for r in ps], "o-", label=metric
            )
        axes[i, 1].set_xticks(range(len(ps)), [f"{r['weight']:g}" for r in ps])
        axes[i, 1].set(
            title="Independent Poisson errors", xlabel="Collar constraint weight"
        )
        axes[i, 1].legend(fontsize=8)
        data = np.load(out / f"{domain}_weight1000.0.npz")
        im = axes[i, 2].scatter(
            *data["points"].T,
            c=np.log10(np.maximum(abs(data["value"] - data["exact"]), 1e-16)),
            s=3,
            cmap="magma",
        )
        axes[i, 2].set(
            title="Poisson log10 absolute error (weight 1000)", aspect="equal"
        )
        fig.colorbar(im, ax=axes[i, 2], shrink=0.8)
        for ax in axes[i, :2]:
            ax.grid(alpha=0.2)
    fig.suptitle(
        "Regularized normal continuation + shared BSPF representation | N=49 | kmax=12 pi"
    )
    fig.savefig(out / "summary.png", dpi=180)
    fig.savefig(out / "summary.pdf")


if __name__ == "__main__":
    main()
