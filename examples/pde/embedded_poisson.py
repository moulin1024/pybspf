"""Convex and non-star-shaped spline Poisson convergence benchmarks."""

import argparse
import json
import pickle
from pathlib import Path

import jax
import numpy as np

from bspf_jax.embedded_poisson import (
    benchmark_domains,
    sample_domain,
    solve_poisson,
    error_metrics,
    manufactured,
    background_line,
    basis_values,
)

jax.config.update("jax_enable_x64", True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("build/embedded_poisson"))
    parser.add_argument("--orders", type=int, nargs="+", default=[12, 16, 20])
    parser.add_argument("--check-order", type=int, default=24)
    parser.add_argument("--modes", type=int, nargs="+", default=[6, 10, 14, 18, 20])
    parser.add_argument(
        "--reuse-cache", action="store_true", help="Load trusted local sample pickles"
    )
    args = parser.parse_args()
    if args.check_order <= max(args.orders):
        raise ValueError("Validation quadrature must be finer than every assembly rule")
    args.out.mkdir(parents=True, exist_ok=True)
    results, plotted = {}, []
    for domain in benchmark_domains():
        samples = {}
        for order in sorted(set(args.orders + [args.check_order])):
            path = args.out / f"{domain.name}_q{order}.pkl"
            if args.reuse_cache and path.exists():
                with path.open("rb") as handle:
                    sample = pickle.load(handle)
                if sample.modes != max(args.modes):
                    raise ValueError("Cached mode count differs; regenerate samples")
            else:
                print(f"Sampling {domain.name}, quadrature {order}", flush=True)
                sample = sample_domain(domain, max(args.modes), order)
                with path.open("wb") as handle:
                    pickle.dump(sample, handle)
            samples[order] = sample
        check = samples[args.check_order]
        geometry = domain.geometry_checks()
        geometry["area_volume"] = float(check.weights.sum())
        geometry["area_relative_difference"] = abs(
            check.weights.sum() / geometry["area_boundary"] - 1
        )
        rows = []
        for order in args.orders:
            for modes in args.modes:
                coeff, info = solve_poisson(samples[order], modes)
                info.update(error_metrics(check, coeff))
                info.update(quadrature_order=order, validation_order=args.check_order)
                rows.append(info)
                print(domain.name, json.dumps(info), flush=True)
                if order == max(args.orders) and modes == max(args.modes):
                    plotted.append((domain, check, coeff))
                    np.savez(
                        args.out / f"{domain.name}_solution.npz",
                        controls=domain.controls,
                        points=check.points,
                        weights=check.weights,
                        coefficients=coeff,
                        value=check.basis[0] @ coeff,
                        exact=manufactured(check.points)[0],
                    )
        sensitivity = []
        for cutoff in [1e-10, 1e-12, 1e-14]:
            coeff, info = solve_poisson(
                samples[max(args.orders)], max(args.modes), mass_cutoff=cutoff
            )
            info.update(error_metrics(check, coeff))
            sensitivity.append(info)
        coarse, _ = solve_poisson(samples[max(args.orders)], max(args.modes))
        refined, _ = solve_poisson(check, max(args.modes))
        delta = check.basis[0] @ (coarse - refined)
        reference = manufactured(check.points)[0]
        quadrature_change = float(
            np.sqrt(
                np.sum(check.weights * delta**2) / np.sum(check.weights * reference**2)
            )
        )
        results[domain.name] = dict(
            geometry=geometry,
            convergence=rows,
            cutoff_sensitivity=sensitivity,
            final_quadrature_solution_relative_change=quadrature_change,
        )
        (args.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    render(args.out, results, plotted)


def render(out, results, plotted):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row, (domain, check, coefficients) in enumerate(plotted):
        curve = domain.curve(np.linspace(0, domain.period, 1601))
        grid = -1 + 2 * (np.arange(240) + 0.5) / 240
        x, y = np.meshgrid(grid, grid)
        inside = np.zeros_like(x, dtype=bool)
        for column, value in enumerate(grid):
            for lo, hi in domain.intersections(value):
                inside[:, column] |= (grid > lo) & (grid < hi)
        points = np.column_stack((x[inside], y[inside]))
        values = basis_values(background_line(), points, check.modes)[0] @ coefficients
        field = np.full_like(x, np.nan)
        error = field.copy()
        field[inside] = values
        error[inside] = np.log10(
            np.maximum(abs(values - manufactured(points)[0]), 1e-14)
        )
        ax = axes[row, 0]
        ax.plot(*curve.T, color="black", lw=1)
        ax.plot(*np.vstack((domain.controls, domain.controls[0])).T, ":", alpha=0.35)
        cloud = ax.imshow(field, origin="lower", extent=(-1, 1, -1, 1), cmap="viridis")
        fig.colorbar(cloud, ax=ax, label="u")
        ax.set(
            title=f"{domain.name}: BSPF solution",
            xlim=(-1, 1),
            ylim=(-1, 1),
            aspect="equal",
        )
        ax = axes[row, 1]
        cloud = ax.imshow(error, origin="lower", extent=(-1, 1, -1, 1), cmap="magma")
        ax.plot(*curve.T, color="black", lw=0.8)
        fig.colorbar(cloud, ax=ax, label="log10 absolute error")
        ax.set(
            title="Error on independent uniform samples",
            xlim=(-1, 1),
            ylim=(-1, 1),
            aspect="equal",
        )
        ax = axes[row, 2]
        rows = results[domain.name]["convergence"]
        for order in sorted({r["quadrature_order"] for r in rows}):
            selected = [r for r in rows if r["quadrature_order"] == order]
            ax.semilogy(
                [r["modes"] for r in selected],
                [r["relative_l2"] for r in selected],
                "o-",
                label=f"L2, Gauss {order}",
            )
            ax.semilogy(
                [r["modes"] for r in selected],
                [r["relative_h1_seminorm"] for r in selected],
                "s--",
                label=f"grad, Gauss {order}",
            )
        ax.set(
            title="Modal and quadrature convergence",
            xlabel="BSPF modes per direction",
            ylabel="relative error",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(out / "validation.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
