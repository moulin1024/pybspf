"""Grid convergence of the masked pressure solve with analytic gradient data.

Run from the checkout with:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
    python examples/pressure_projection2d_convergence.py

Requires matplotlib in addition to the package dependencies. The solver is
unchanged; every run uses the same BSPF parameters and solve tolerances.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pybspf import PressurePoisson2D


CASES = {
    "exponential": ("Smooth exponential", "#64748b"),
    "oscillatory": ("Oscillatory analytic", "#2563eb"),
    "localized": ("Localized analytic", "#d97706"),
}


def exact_fields(case, x, y):
    """Return pressure and its analytic (not discrete BSPF) gradient."""
    if case == "exponential":
        p = np.exp(x + 0.5 * y)
        return p, np.stack([p, 0.5 * p], axis=-1)
    if case == "oscillatory":
        a, b = 5.3 * np.pi, 3.7 * np.pi
        p = np.sin(a * x + 0.2) * np.cos(b * y - 0.1)
        return p, np.stack(
            [
                a * np.cos(a * x + 0.2) * np.cos(b * y - 0.1),
                -b * np.sin(a * x + 0.2) * np.sin(b * y - 0.1),
            ],
            axis=-1,
        )
    dx, dy = x - 0.37, y - 0.61
    p = 1 / (1 + 25 * (dx * dx + dy * dy))
    return p, np.stack([-50 * dx * p * p, -50 * dy * p * p], axis=-1)


def fit_models(rows):
    # Declare the fit window explicitly and use the same points for both models.
    subset = [r for r in rows if r["case"] == "oscillatory" and r["N"] >= 80]
    n = np.array([r["N"] for r in subset])
    errors = np.array([r["pressure_relative_l2"] for r in subset])
    result = {"case": "oscillatory", "N": n.tolist()}
    for name, x in [("algebraic", np.log(n - 1)), ("exponential", n)]:
        slope, intercept = np.polyfit(x, np.log(errors), 1)
        residual = np.log(errors) - (intercept + slope * x)
        result[name] = {
            "slope": float(slope),
            "log_prefactor": float(intercept),
            "log10_rmse": float(np.sqrt(np.mean(residual**2)) / np.log(10)),
        }
    return result


def plot_results(rows, fits, out):
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), layout="constrained")
    semilog, loglog, orders, diagnostics = axes.ravel()
    for case, (label, color) in CASES.items():
        r = [row for row in rows if row["case"] == case]
        n = np.array([row["N"] for row in r])
        intervals = n - 1
        err = np.array([row["pressure_relative_l2"] for row in r])
        semilog.semilogy(n, err, "o-", color=color, label=label, ms=4)
        loglog.loglog(intervals, err, "o-", color=color, label=label, ms=4)
        if case != "exponential":
            rate = np.log(err[:-1] / err[1:]) / np.log(intervals[1:] / intervals[:-1])
            orders.plot(
                np.sqrt(intervals[1:] * intervals[:-1]),
                rate,
                "o-",
                color=color,
                label=label,
                ms=4,
            )
        diagnostics.semilogy(
            n,
            [row["schur_relative_l2"] for row in r],
            "o-",
            color=color,
            label=label,
            ms=4,
        )
    nfit = np.linspace(min(fits["N"]), max(fits["N"]), 200)
    for model, style, label in [
        ("algebraic", "--", "Power-law fit"),
        ("exponential", ":", "Exponential fit"),
    ]:
        fit = fits[model]
        xfit = np.log(nfit - 1) if model == "algebraic" else nfit
        values = np.exp(fit["log_prefactor"] + fit["slope"] * xfit)
        semilog.semilogy(nfit, values, style, color="#0f172a", lw=1.5, label=label)
        loglog.loglog(nfit - 1, values, style, color="#0f172a", lw=1.5)
    semilog.set(
        title="Semilog: exponential convergence would be straight",
        xlabel="Nodes per axis N",
        ylabel="Relative pressure error (weighted L2)",
    )
    loglog.set(
        title="Log–log: power-law convergence is nearly straight",
        xlabel="Intervals per axis N − 1",
        ylabel="Relative pressure error (weighted L2)",
    )
    semilog.legend(loc="upper right", fontsize=8)
    for ax in [semilog, loglog]:
        ax.axhspan(1e-16, 1e-13, color="#94a3b8", alpha=0.10)
        ax.set_ylim(5e-16, 4e-3)
    loglog.text(
        0.04,
        0.09,
        "Smooth exponential already near floating-point accuracy",
        transform=loglog.transAxes,
        fontsize=8,
        color="#475569",
    )
    orders.set(
        title="Observed order between successive grids",
        xlabel="Geometric mean of interval counts",
        ylabel="p in error ∝ hᵖ",
    )
    orders.axhline(-fits["algebraic"]["slope"], color="#0f172a", ls="--", lw=1)
    orders.legend(fontsize=8)
    diagnostics.set(
        title="Algebraic solve residual stays small",
        xlabel="Nodes per axis N",
        ylabel="||S p − b||₂ / ||b||₂",
    )
    diagnostics.ticklabel_format(axis="x", style="plain")
    for ax in axes.ravel():
        ax.grid(True, which="major", alpha=0.20)
    power = -fits["algebraic"]["slope"]
    fig.suptitle(
        "2D BSPF pressure projection: high-order algebraic convergence\n"
        f"Fixed q=9, degree=13, 32 splines, 14 endpoint samples  |  "
        f"oscillatory tail ≈ h^{power:.2f}",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(out / "convergence.png", dpi=180)
    fig.savefig(out / "convergence.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("build/pressure_convergence2d")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ns = [34, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256]
    rows = []
    for n in ns:
        grid = np.linspace(0, 1, n)
        solver = PressurePoisson2D(grid, grid)
        x, y = np.meshgrid(grid, grid)
        w = np.ones(n) / (n - 1)
        w[[0, -1]] *= 0.5
        weights = np.outer(w, w)
        for case in CASES:
            exact, gradient = exact_fields(case, x, y)
            projected, result = solver.project(gradient)
            exact = solver.remove_mean(exact)
            error = result.pressure - exact
            rhs = solver.divergence(solver.mask[..., None] * gradient)
            row = {
                "case": case,
                "N": n,
                "h": 1 / (n - 1),
                "pressure_relative_l2": float(
                    np.sqrt(np.sum(weights * error**2) / np.sum(weights * exact**2))
                ),
                "pressure_relative_linf": float(abs(error).max() / abs(exact).max()),
                "projected_relative_l2": float(
                    np.sqrt(
                        np.sum(weights[..., None] * projected**2)
                        / np.sum(weights[..., None] * gradient**2)
                    )
                ),
                "schur_relative_l2": float(
                    result.schur_residual_l2 / np.linalg.norm(rhs)
                ),
                "schur_residual_linf": result.schur_residual_linf,
                "divergence_linf": float(abs(solver.divergence(projected)).max()),
                "wall_gradient_fit_linf": result.wall_gradient_fit_linf,
            }
            rows.append(row)
            print(
                f"N={n:3d} {case:12s} pressure L2={row['pressure_relative_l2']:.5e}",
                flush=True,
            )
    fits = fit_models(rows)
    data = {
        "domain": [0, 1, 0, 1],
        "config": {"q": 9, "n_basis": 32, "degree": 13, "baseline_points": 14},
        "solve_tolerances": {"rtol": 1e-10, "atol": 1e-9},
        "numpy_version": np.__version__,
        "method": "project analytic gradient; zero-mean exact pressure; trapezoidal L2",
        "formulas": {
            "exponential": "exp(x + 0.5*y)",
            "oscillatory": "sin(5.3*pi*x + 0.2)*cos(3.7*pi*y - 0.1)",
            "localized": "1/(1 + 25*((x-0.37)^2 + (y-0.61)^2))",
        },
        "rows": rows,
        "fits": fits,
    }
    (args.out / "convergence.json").write_text(json.dumps(data, indent=2) + "\n")
    plot_results(rows, fits, args.out)
    print(json.dumps(fits, indent=2))
    print(f"Saved figures and data to {args.out.resolve()}")


if __name__ == "__main__":
    main()
