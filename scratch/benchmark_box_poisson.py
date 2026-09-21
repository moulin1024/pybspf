"""3D continuous MMS and synchronized GPU timings for the shared box inverse.

Uses BSPF Galerkin and seven-point finite differences on [0,2]x[0,3]x[0,1.5].
Each resolution/discretization runs in an isolated process. No full 3D matrix.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from benchmark_rectangle_mms import axial, fd_axis, fd_evaluation
from scipy.special import roots_legendre

LENGTHS = (2.0, 3.0, 1.5)


def product(factors):
    return np.einsum("i,j,k->ijk", *factors)


def mms(points):
    exact = np.zeros(tuple(len(p) for p in points))
    forcing = np.zeros_like(exact)
    for term, amplitude in enumerate((1.0, 0.2)):
        values, seconds = zip(
            *(
                axial(p, length, axis, term)
                for axis, (p, length) in enumerate(zip(points, LENGTHS))
            )
        )
        exact += amplitude * product(values)
        for axis in range(3):
            factors = list(values)
            factors[axis] = seconds[axis]
            forcing -= amplitude * product(factors)
    return exact, forcing


def apply_axis(matrix, values, axis):
    return np.moveaxis(np.tensordot(matrix, values, axes=(1, axis)), 0, axis)


def worker(args, observer=None):
    import jax.numpy as jnp
    from bspf_jax import plan_box_poisson, solve_box_poisson

    import jax

    jax.config.update("jax_enable_x64", True)
    device = jax.devices("gpu")[0]

    def checkpoint(phase, **data):
        if observer is not None:
            observer(phase, device, **data)

    checkpoint("preparation")
    n = args.n
    start = perf_counter()
    grids = [np.linspace(0, length, n + 2) for length in LENGTHS]
    gp, gw = roots_legendre(257)
    check = [(gp + 1) * length / 2 for length in LENGTHS]
    masses, stiffnesses, evaluation = [], [], []
    if args.discretization == "fd":
        for length, points in zip(LENGTHS, check):
            m, k = fd_axis(n, length)
            masses.append(m)
            stiffnesses.append(k)
            evaluation.append(fd_evaluation(n, length, points))
        _, rhs = mms([p[1:-1] for p in grids])
    else:
        from bspf_jax import galerkin_1d, interpolate, plan_1d
        from bspf_jax.galerkin import _quadrature_rule

        weak, quad = [], []
        for grid, points in zip(grids, check):
            p = plan_1d(jnp.asarray(grid), degree=5, n_basis=16, boundary_points=7)
            w = galerkin_1d(p, constraints=((0, 0), (1, 0)), quadrature_order=8)
            masses.append(np.asarray(w.mass))
            stiffnesses.append(np.asarray(w.stiffness))
            evaluation.append(
                np.asarray(interpolate(p, w.extension, jnp.asarray(points)))
            )
            weak.append(w)
            quad.append(np.asarray(_quadrature_rule(p, 8)[0]))
        rhs = np.zeros((n, n, n))
        for term, amplitude in enumerate((1.0, 0.2)):
            loads, second_loads = [], []
            for axis, (points, length, w) in enumerate(zip(quad, LENGTHS, weak)):
                v, d2 = axial(points, length, axis, term)
                b, weight = np.asarray(w.values), np.asarray(w.quadrature_weights)
                loads.append(b.T @ (weight * v))
                second_loads.append(b.T @ (weight * d2))
            for axis in range(3):
                factors = list(loads)
                factors[axis] = second_loads[axis]
                rhs -= amplitude * product(factors)
    preparation = perf_counter() - start
    checkpoint("factor_setup")
    start = perf_counter()
    plan = plan_box_poisson(masses, stiffnesses, device=device)
    jax.block_until_ready(plan)
    setup = perf_counter() - start
    checkpoint("rhs_upload")
    start = perf_counter()
    load = jax.device_put(rhs, device)
    load.block_until_ready()
    upload = perf_counter() - start
    checkpoint("first_solve", plan=plan, load=load)
    start = perf_counter()
    solution = solve_box_poisson(plan, load)
    solution.block_until_ready()
    first = perf_counter() - start
    checkpoint("warm_solves")
    times = []
    with jax.transfer_guard("disallow"):
        for _ in range(args.repeats):
            start = perf_counter()
            solution = solve_box_poisson(plan, load)
            solution.block_until_ready()
            times.append(perf_counter() - start)
    checkpoint("validation")
    host = np.asarray(solution)
    if args.discretization == "fd":
        action = np.zeros_like(host)
        for axis, length in enumerate(LENGTHS):
            scale = ((n + 1) / length) ** 2
            action += 2 * scale * host
            lower, upper = [slice(None)] * 3, [slice(None)] * 3
            lower[axis], upper[axis] = slice(None, -1), slice(1, None)
            action[tuple(lower)] -= scale * host[tuple(upper)]
            action[tuple(upper)] -= scale * host[tuple(lower)]
    else:
        action = np.zeros_like(host)
        for stiffness_axis in range(3):
            term = host
            for axis in range(3):
                matrix = stiffnesses[axis] if axis == stiffness_axis else masses[axis]
                term = apply_axis(matrix, term, axis)
            action += term
    residual = np.linalg.norm(action - rhs) / np.linalg.norm(rhs)
    values = host
    for axis, matrix in enumerate(evaluation):
        values = apply_axis(matrix, values, axis)
    exact, _ = mms(check)
    weights = product([gw] * 3)
    l2 = np.sqrt(np.sum(weights * (values - exact) ** 2) / np.sum(weights * exact**2))
    if not np.isfinite(l2) or not residual <= 2e-10:
        raise RuntimeError(f"Accuracy failure: relative residual={residual}, L2={l2}")
    case = {
        "discretization": args.discretization,
        "shape": [n] * 3,
        "unknowns": n**3,
        "sampling_nodes": [n + 2] * 3,
        "error_quadrature": [257] * 3,
        "preparation_seconds": preparation,
        "setup_seconds": setup,
        "rhs_upload_seconds": upload,
        "first_solve_seconds": first,
        "warm_seconds": times,
        "warm_median_seconds": float(np.median(times)),
        "plan_bytes": sum(a.nbytes for a in jax.tree_util.tree_leaves(plan)),
        "relative_l2_error": float(l2),
        "relative_residual": float(residual),
        "max_error": float(np.max(np.abs(values - exact))),
        "device": device.device_kind,
        "jax": jax.__version__,
        "dtype": "float64",
    }
    Path(args.result).write_text(json.dumps(case, indent=2) + "\n")
    checkpoint("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("build/box_poisson"))
    parser.add_argument("--fd-sizes", type=int, nargs="+", default=[31, 63, 127, 255])
    parser.add_argument("--bspf-sizes", type=int, nargs="+", default=[15, 31, 63, 127])
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--discretization", choices=("fd", "bspf"))
    parser.add_argument("--n", type=int)
    parser.add_argument("--result")
    args = parser.parse_args()
    if args.repeats < 1 or min(args.fd_sizes + args.bspf_sizes) < 15:
        parser.error("Require positive repeats and sizes >= 15")
    if args.worker:
        worker(args)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[1]
    env.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    env["PYTHONPATH"] = str(root / "jax/src") + os.pathsep + env.get("PYTHONPATH", "")
    results = {"domain": LENGTHS, "cases": []}
    for disc, sizes in [("bspf", args.bspf_sizes), ("fd", args.fd_sizes)]:
        for n in sizes:
            key = f"{disc}_{n}"
            result = args.out / f"{key}.json"
            print(f"Running {key}", flush=True)
            with (args.out / f"{key}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--discretization",
                        disc,
                        "--n",
                        str(n),
                        "--repeats",
                        str(args.repeats),
                        "--result",
                        str(result),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            case = json.loads(result.read_text())
            results["cases"].append(case)
            (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(
                f"  {case['warm_median_seconds'] * 1000:.3f} ms; L2 {case['relative_l2_error']:.3e}; residual {case['relative_residual']:.3e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
