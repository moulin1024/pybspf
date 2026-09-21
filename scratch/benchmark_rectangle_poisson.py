"""Synchronized rectangular inverse benchmark, excluding RHS assembly/transfers.

PYTHONPATH=jax/src python scratch/benchmark_rectangle_poisson.py
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from bspf_jax.rectangle_poisson import plan_rectangle_poisson, solve_rectangle_poisson

import jax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument(
        "--out", type=Path, default=Path("build/rectangle_poisson/results.json")
    )
    args = parser.parse_args()
    if args.repeats < 1 or min(args.sizes) < 4:
        parser.error("repeats must be positive and sizes at least 4")
    jax.config.update("jax_enable_x64", True)
    device = jax.devices(args.backend)[0]
    results = {
        "device": str(device),
        "device_kind": device.device_kind,
        "jax": jax.__version__,
        "dtype": "float64",
        "cases": [],
    }
    for nx in args.sizes:
        ny = 3 * nx // 4

        def axis(n, length):
            h = length / (n + 1)
            return (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)) / h**2

        kx, ky = axis(nx, 2), axis(ny, 3)
        start = perf_counter()
        plan = plan_rectangle_poisson(np.eye(nx), kx, np.eye(ny), ky, device=device)
        jax.block_until_ready(plan)
        setup = perf_counter() - start
        exact = np.random.default_rng(71).normal(size=(nx, ny))
        rhs_host = kx @ exact + exact @ ky.T
        rhs = jax.device_put(rhs_host, device)
        rhs.block_until_ready()
        start = perf_counter()
        solution = solve_rectangle_poisson(plan, rhs)
        solution.block_until_ready()
        first = perf_counter() - start
        times = []
        with jax.transfer_guard("disallow"):
            for _ in range(args.repeats):
                start = perf_counter()
                solution = solve_rectangle_poisson(plan, rhs)
                solution.block_until_ready()
                times.append(perf_counter() - start)
        result = np.asarray(solution)
        residual = np.linalg.norm(
            kx @ result + result @ ky.T - rhs_host
        ) / np.linalg.norm(rhs_host)
        error = np.linalg.norm(result - exact) / np.linalg.norm(exact)
        if residual > 1e-10 or error > 1e-9:
            raise RuntimeError(f"Accuracy failure: residual={residual}, error={error}")
        case = {
            "shape": [nx, ny],
            "setup_seconds": setup,
            "first_solve_seconds": first,
            "warm_median_seconds": float(np.median(times)),
            "warm_seconds": times,
            "factor_bytes": sum(a.nbytes for a in plan),
            "relative_residual": float(residual),
            "relative_error": float(error),
        }
        results["cases"].append(case)
        print(json.dumps(case), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
