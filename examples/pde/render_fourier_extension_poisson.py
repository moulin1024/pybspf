"""Plot the independently measured FE/Poisson convergence and CPU costs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("build/fourier_extension_poisson_final"))
    args = parser.parse_args()
    report = json.loads((args.out / "results.json").read_text())
    runs = report["runs"]
    modes = [r["modes"] for r in runs]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for name in runs[0]["cases"]:
        for ax, key in zip(axes[:2], ("relative_l2", "relative_h2")):
            ax.semilogy(modes, [r["cases"][name][key] for r in runs], "o-", label=name)
    for ax, title in zip(axes[:2], ("Solution relative L2 error", "Solution relative H2 error")):
        ax.set(title=title, xlabel="Fourier modes per axis", xticks=modes)
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    axes[2].semilogy(modes, [r["plan"]["setup_seconds"] for r in runs], "o-", label="Setup (s)")
    axes[2].semilogy(modes, [np.median([v["repeated_rhs_seconds"] for v in r["cases"].values()]) for r in runs], "o-", label="Repeated RHS (s)")
    axes[2].semilogy(modes, [np.median([v["offgrid_evaluation_seconds"] for v in r["cases"].values()]) for r in runs], "o-", label="Validation evaluation (s)")
    axes[2].set(title="Separate CPU costs", xlabel="Fourier modes per axis", ylabel="Seconds", xticks=modes)
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=8)
    fig.suptitle("FFT Fourier extension + harmonic boundary correction\nIndependent whole-domain validation; low-resolution failures retained")
    fig.savefig(args.out / "convergence.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
