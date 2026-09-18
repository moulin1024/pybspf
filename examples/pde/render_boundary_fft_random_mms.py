"""Plot random-wave convergence with a linear M axis to expose spectral decay."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    out = Path("build/boundary_fft_random_mms")
    result = json.loads((out / "results.json").read_text())
    rows = result["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    fits = []
    for band in (8, 32, 64, 128):
        data = [r for r in rows if r["kmax_pi"] == band]
        counts = np.array([r["boundary_points"] for r in data])
        errors = np.array([r["boundary_relative"] for r in data])
        for ax, key in zip(axes, ("boundary_relative", "interior_relative")):
            ax.semilogy(
                counts,
                [r[key] for r in data],
                "o-",
                ms=3,
                label=rf"$k_{{\max}}={band}\pi$",
            )
        selected = (errors < 1e-3) & (errors > 1e-11)
        if selected.sum() >= 3:
            x, y = counts[selected], np.log10(errors[selected])
            slope, intercept = np.polyfit(x, y, 1)
            r2 = 1 - np.sum((y - (slope * x + intercept)) ** 2) / np.sum(
                (y - y.mean()) ** 2
            )
            fits.append(
                dict(
                    kmax_pi=band,
                    count_range=[int(x.min()), int(x.max())],
                    sample_count=len(x),
                    log10_slope=float(slope),
                    r_squared=float(r2),
                    points_per_decade=float(-1 / slope),
                )
            )
    for ax, title in zip(
        axes,
        (
            "Independent boundary error (4096 shifted points)",
            "Interior error (fixed grid, elliptic radius <= 0.92)",
        ),
    ):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("M: boundary unknowns (linear scale)")
        ax.set_ylabel("Relative discrete L2 error")
        ax.set_xticks([0, 256, 512, 768, 1024, 1536])
        ax.set_ylim(2e-16, 2)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle(
        "Analytic ellipse: 64 random waves with a k^(-5/3) shell variance\n"
        "Nonzero-source Poisson; independently converged, source-specific volume potential",
        fontsize=12,
    )
    fig.savefig(out / "convergence.png", dpi=180)
    plt.close(fig)
    (out / "decay_fits.json").write_text(json.dumps(fits, indent=2) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), constrained_layout=True)
    for count, color in ((512, "tab:orange"), (1024, "tab:blue")):
        data = np.load(out / f"fields_{count}.npz")
        coordinate = (np.arange(len(data["boundary"])) + 0.371) / len(data["boundary"])
        exact, numerical = (
            data["boundary_exact"][:, -1],
            data["boundary_predicted"][:, -1],
        )
        if count == 512:
            axes[0].plot(
                coordinate, exact, color="black", lw=1, label="Exact MMS trace"
            )
        axes[0].plot(
            coordinate, numerical, color=color, lw=0.7, alpha=0.85, label=f"M={count}"
        )
        axes[1].semilogy(
            coordinate,
            np.maximum(abs(numerical - exact), 1e-16),
            color=color,
            lw=0.7,
            label=f"M={count}",
        )
    axes[0].set_xlim(0, 0.25)
    axes[0].set_ylabel("u on the boundary")
    axes[0].set_title(
        "Highest band: k_max=128 pi; first quarter of boundary shown above"
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_ylabel("Absolute error")
    axes[1].set_xlabel("Normalized arclength s/L")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.savefig(out / "boundary_trace.png", dpi=180)
    plt.close(fig)
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
