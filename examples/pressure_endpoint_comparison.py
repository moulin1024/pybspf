"""Compare endpoint estimators while retaining the masked pressure equation.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python examples/pressure_endpoint_comparison.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pybspf import PressurePoisson2D
from pressure_projection2d_convergence import exact_fields


CONFIGS = {
    "Taylor degree 8 / 14 samples": {},
    "Chebyshev M12 / P16": dict(
        endpoint_method="chebyshev", chebyshev_modes=12, baseline_points=16
    ),
    "Chebyshev M14 / P18": dict(
        endpoint_method="chebyshev", chebyshev_modes=14, baseline_points=18
    ),
    "Chebyshev M16 / P20": dict(
        endpoint_method="chebyshev", chebyshev_modes=16, baseline_points=20
    ),
    "Chebyshev M14 / P18 + q11 d15": dict(
        endpoint_method="chebyshev",
        chebyshev_modes=14,
        baseline_points=18,
        q=11,
        degree=15,
    ),
    "Chebyshev M14 / P18 + 48 splines": dict(
        endpoint_method="chebyshev", chebyshev_modes=14, baseline_points=18, n_basis=48
    ),
}


def main():
    out = Path("build/pressure_endpoint_comparison")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in [40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 256]:
        grid = np.linspace(0, 1, n)
        x, y = np.meshgrid(grid, grid)
        w = np.ones(n) / (n - 1)
        w[[0, -1]] *= 0.5
        W = np.outer(w, w)
        for name, opts in CONFIGS.items():
            if n <= opts.get("n_basis", 32):
                continue
            solver = PressurePoisson2D(grid, grid, **opts)
            for case in ["oscillatory", "localized"]:
                p, g = exact_fields(case, x, y)
                v, result = solver.project(g)
                p = solver.remove_mean(p)
                rel = np.sqrt(
                    np.sum(W * (result.pressure - p) ** 2) / np.sum(W * p * p)
                )
                gradient_rel = np.sqrt(
                    np.sum(W[..., None] * (solver.gradient(p) - g) ** 2)
                    / np.sum(W[..., None] * g * g)
                )
                rows.append(
                    dict(
                        N=n,
                        configuration=name,
                        case=case,
                        pressure_relative_l2=float(rel),
                        gradient_relative_l2=float(gradient_rel),
                        divergence_linf=float(abs(solver.divergence(v)).max()),
                        schur_residual_l2=result.schur_residual_l2,
                        wall_gradient_fit_linf=result.wall_gradient_fit_linf,
                    )
                )
            print(n, name, [r["pressure_relative_l2"] for r in rows[-2:]], flush=True)
    (out / "measurements.json").write_text(
        json.dumps(
            dict(
                default_parameters=dict(
                    q=9,
                    n_basis=32,
                    degree=13,
                    baseline_points=14,
                    endpoint_regularization=1e-12,
                ),
                overrides=CONFIGS,
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), layout="constrained")
    colors = ["#64748b", "#059669", "#2563eb", "#d97706"]
    for ax, case in zip(axes, ["oscillatory", "localized"]):
        for (name, _), color in zip(CONFIGS.items(), colors):
            rec = [r for r in rows if r["case"] == case and r["configuration"] == name]
            ax.semilogy(
                [r["N"] for r in rec],
                [r["pressure_relative_l2"] for r in rec],
                "o-",
                label=name,
                color=color,
                ms=4,
            )
        ax.set(
            xlabel="Nodes per axis N",
            ylabel="Relative pressure error (weighted L2)",
            title=f"{case.capitalize()} analytic pressure",
        )
        ax.grid(alpha=0.2)
        ax.set_ylim(2e-15, 1e-3)
    axes[0].legend(fontsize=9)
    fig.suptitle(
        "Better endpoint jets substantially improve pressure accuracy\n"
        "Same q=9, degree=13, 32 splines and tensor solve in all plotted curves",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(out / "endpoint_comparison.png", dpi=180)
    fig.savefig(out / "endpoint_comparison.pdf")
    plt.close(fig)
    print(out.resolve())


if __name__ == "__main__":
    main()
