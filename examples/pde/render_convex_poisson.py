"""Compare convex BSPF designs, refinement, and independent spatial errors."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from bspf_jax.embedded_poisson import benchmark_domains


def main():
    base = Path("build/convex_poisson")
    control = json.loads((base / "results.json").read_text())
    public = json.loads(
        Path("build/convex_poisson_unrestricted/results.json").read_text()
    )
    fine = json.loads(Path("build/convex_poisson_n65/results.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    selected = [r for r in control if r["band"] == 12]
    labels = ["L2 trace", "H3/2 trace", "H3/2 + H2 scale"]
    for metric in ("value", "gradient", "laplacian"):
        axes[0, 0].semilogy(
            range(len(selected)), [r[metric] for r in selected], "o-", label=metric
        )
    axes[0, 0].set_xticks(range(len(selected)), labels, rotation=15)
    axes[0, 0].set(
        title="Same space, N=49 | kmax=12 pi", ylabel="Independent relative error"
    )
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.2)
    for j, band in enumerate((4, 12), start=1):
        rs = [next(r for r in data if r["band"] == band) for data in (public, fine)]
        for metric in ("value", "gradient", "laplacian"):
            axes[0, j].semilogy([49, 65], [r[metric] for r in rs], "o-", label=metric)
        axes[0, j].set(
            title=f"Default unrestricted space | kmax={band} pi",
            xlabel="BSPF nodes per axis",
        )
        axes[0, j].legend(fontsize=8)
        axes[0, j].grid(alpha=0.2)
    cases = [
        (base / "l2_column_12pi.npz", "L2 trace, constrained N=49"),
        (
            Path("build/convex_poisson_unrestricted/h32_h2_12pi.npz"),
            "New solver, unrestricted N=49",
        ),
        (
            Path("build/convex_poisson_n65/h32_h2_12pi.npz"),
            "New solver, unrestricted N=65",
        ),
    ]
    domain = benchmark_domains()[0]
    curve = domain.curve(np.linspace(0, domain.period, 1001))
    for ax, (path, title) in zip(axes[1], cases):
        data = np.load(path)
        im = ax.scatter(
            *data["points"].T,
            c=np.maximum(abs(data["value"] - data["exact"]), 1e-14),
            norm=LogNorm(1e-12, 1e-5),
            s=3,
            cmap="magma",
        )
        ax.plot(*curve.T, "k-", lw=0.8)
        ax.set(title=title, aspect="equal")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Absolute error")
    fig.suptitle(
        "Convex B-spline Poisson | analytic BSPF derivatives + arclength H3/2 boundary norm"
    )
    fig.savefig(base / "solver_comparison.png", dpi=180)
    fig.savefig(base / "solver_comparison.pdf")


if __name__ == "__main__":
    main()
