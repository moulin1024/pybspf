"""Solve convex-domain Poisson and return only a requested Cartesian grid."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from bspf_jax import ConvexPoissonGridPlan
from bspf_jax.embedded_poisson import benchmark_domains
from bspf_jax.random_wave_mms import RandomWaveMMS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=33)
    parser.add_argument("--grid", type=int, default=33)
    parser.add_argument("--band", type=float, default=4)
    parser.add_argument(
        "--out", type=Path, default=Path("build/convex_poisson_fixed_grid")
    )
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    domain = benchmark_domains()[0]
    x = y = np.linspace(-1.2, 1.2, args.grid)
    start = perf_counter()
    plan = ConvexPoissonGridPlan(domain, x, y, nodes=args.nodes)
    setup_seconds = perf_counter() - start
    mms = RandomWaveMMS.create(kmax=args.band * np.pi)

    def forcing(p):
        return mms.evaluate(p)[2]

    def boundary(p):
        return mms.evaluate(p)[0]

    result = plan.solve(forcing, boundary)
    # Verification is outside the solver API; its exact solution is never
    # supplied as interior data to the Poisson solve.
    xx, yy = np.meshgrid(result.x, result.y)
    points = np.column_stack((xx[result.inside], yy[result.inside]))
    exact = mms.evaluate(points)[0]
    error = result.values[result.inside] - exact
    report = dict(
        result.diagnostics,
        nodes=args.nodes,
        band=args.band,
        setup_seconds=setup_seconds,
        relative_sample_l2=float(np.linalg.norm(error) / np.linalg.norm(exact)),
        max_sample_error=float(np.max(abs(error))),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez(
        args.out / "grid_solution.npz",
        x=result.x,
        y=result.y,
        inside=result.inside,
        values=result.values,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
