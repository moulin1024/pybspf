"""Independent-error comparison and actual endpoint refinement pattern."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    out = Path("build/panel_poisson")
    result = json.loads((out / "results.json").read_text())
    rows = result["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for method, label, marker in (
        ("global_fft", "Global arclength FFT", "o"),
        ("panel_p", "12 panels, increase order", "s"),
        ("panel_uniform_h", "12 nodes/panel, uniform refinement", "^"),
        ("panel_adaptive", "12 nodes/panel, residual adaptation", "D"),
    ):
        data = [r for r in rows if r["method"] == method]
        if method == "panel_uniform_h":
            data = [
                next(r for r in rows if r["method"] == "panel_p" and r["order"] == 12)
            ] + data
        for ax, key in zip(axes[:2], ("boundary_relative", "boundary_max_scaled")):
            ax.loglog(
                [r["unknowns"] for r in data],
                [r[key] for r in data],
                marker + "-",
                ms=4,
                label=label,
            )
    axes[0].set_title("Independent boundary relative L2")
    axes[1].set_title("Max error / max |u|, including knot probes")
    for ax in axes[:2]:
        ax.set_xlabel("Number of boundary unknowns")
        ax.grid(True, which="both", alpha=0.2)
        ax.set_ylim(1e-15, 1e-2)
    axes[0].legend(fontsize=8)
    final = result["adaptation"][-1]
    breaks = np.array(final["breaks"])
    axes[2].stairs(np.diff(breaks), breaks, baseline=None, color="tab:red", lw=1.3)
    for knot in range(13):
        axes[2].axvline(knot, color="0.7", lw=0.6, ls="--")
    axes[2].set_yscale("log", base=2)
    axes[2].set_xlabel("Original spline parameter t")
    axes[2].set_ylabel("Panel width in t")
    axes[2].set_title(
        f"Adaptive mesh: {final['panels']} panels\nDashed lines: original spline knots"
    )
    axes[2].set_xlim(0, 12)
    fig.suptitle(
        "Exact cubic B-spline geometry: panel density + logarithmic product integration\n"
        "Poisson f=1 with known particular solution; independent nonzero boundary correction",
        fontsize=12,
    )
    fig.savefig(out / "convergence.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
