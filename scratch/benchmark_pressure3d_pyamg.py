"""7-point Dirichlet FD + classical PyAMG baseline, in isolated workers.

N counts boundary-inclusive points; (N-2)^3 interior unknowns. This is not
BSPF's masked/lifted operator. Discrete recovery and continuous PDE accuracy
are explicitly separate experiments.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def exact(x, y, z):
    import numpy as np

    return np.exp(x + 0.5 * y - 0.3 * z) + np.sin(3 * np.pi * x) * np.cos(
        2 * np.pi * y
    ) * np.cos(np.pi * z)


def continuous_rhs(n):
    """-Delta p=f on [0,1]^3, exact nonzero Dirichlet data eliminated."""
    import numpy as np

    grid = np.linspace(0, 1, n)
    x, y, z = grid[1:-1, None, None], grid[None, 1:-1, None], grid[None, None, 1:-1]
    rhs = -1.34 * np.exp(x + 0.5 * y - 0.3 * z) + 14 * np.pi**2 * np.sin(
        3 * np.pi * x
    ) * np.cos(2 * np.pi * y) * np.cos(np.pi * z)
    for axis in range(3):
        for side, boundary in [(0, 0.0), (-1, 1.0)]:
            coords = [x, y, z]
            coords[axis] = boundary
            sl = [slice(None)] * 3
            sl[axis] = side
            rhs[tuple(sl)] += np.squeeze(exact(*coords), axis=axis) * (n - 1) ** 2
    return rhs.ravel()


def worker(args):
    import gc
    import resource
    import numpy as np
    import scipy
    import pyamg

    def peak():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )

    def sparse_bytes(matrix):
        return sum(
            getattr(matrix, name).nbytes for name in ("data", "indices", "indptr")
        )

    n, m = args.size, args.size - 2
    start = time.perf_counter()
    A = pyamg.gallery.poisson((m, m, m), format="csr", dtype=np.float64)
    A.data *= (n - 1) ** 2
    matrix_seconds = time.perf_counter() - start
    print(f"{n}^3: matrix {matrix_seconds:.2f}s, nnz={A.nnz}", flush=True)
    start = time.perf_counter()
    ml = pyamg.ruge_stuben_solver(A)
    setup_seconds = time.perf_counter() - start
    print(f"{n}^3: AMG setup {setup_seconds:.2f}s, levels={len(ml.levels)}", flush=True)
    arrays = {}
    for level in ml.levels:
        for name in ("A", "P", "R"):
            if hasattr(level, name):
                mat = getattr(level, name)
                for key in ("data", "indices", "indptr"):
                    arr = getattr(mat, key)
                    arrays[arr.__array_interface__["data"][0]] = arr.nbytes
    row = dict(
        n=n,
        unknowns=m**3,
        nnz=A.nnz,
        method=f"Ruge-Stuben / default symmetric GS / V / accel={args.accel}",
        accelerator=args.accel,
        pyamg_version=pyamg.__version__,
        scipy_version=scipy.__version__,
        repeats=args.repeats,
        matrix_seconds=matrix_seconds,
        hierarchy_seconds=setup_seconds,
        matrix_bytes=sparse_bytes(A),
        hierarchy_csr_bytes=sum(arrays.values()),
        operator_complexity=float(ml.operator_complexity()),
        grid_complexity=float(ml.grid_complexity()),
        levels=[dict(n=level.A.shape[0], nnz=level.A.nnz) for level in ml.levels],
        setup_peak_rss_bytes=peak(),
        cases={},
    )
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{n}_pyamg.json"

    for kind in ("smooth_discrete", "random_discrete", "smooth_continuous"):
        if kind == "random_discrete":
            p = (
                np.random.default_rng(71)
                .standard_normal((n, n, n))[1:-1, 1:-1, 1:-1]
                .copy()
                .ravel()
            )
        else:
            g = np.linspace(0, 1, n)[1:-1]
            p = exact(g[:, None, None], g[None, :, None], g[None, None, :]).ravel()
        rhs = continuous_rhs(n) if kind == "smooth_continuous" else A @ p
        bn = float(np.linalg.norm(rhs))
        threshold = 1e-9 + 1e-10 * bn
        tol = threshold / bn
        timings, histories = [], []
        repeats = args.repeats + 1 if kind == "smooth_discrete" else 1
        for rep in range(repeats):
            residuals = []
            start = time.perf_counter()
            sol = ml.solve(
                rhs,
                x0=None,
                tol=tol,
                maxiter=100,
                cycle="V",
                accel=None if args.accel == "none" else args.accel,
                residuals=residuals,
            )
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
            histories.append([float(v / bn) for v in residuals])
            print(
                f"{n}^3 {kind} run {rep}: {elapsed:.3f}s, {len(residuals) - 1} iterations, relres={residuals[-1] / bn:.3e}",
                flush=True,
            )
        error = sol - p
        residual = A @ sol - rhs
        rn = float(np.linalg.norm(residual))
        row["cases"][kind] = dict(
            first_solve_seconds=timings[0],
            samples_seconds=timings[1:] if repeats > 1 else timings,
            median_seconds=float(np.median(timings[1:] if repeats > 1 else timings)),
            cycles=[len(h) - 1 for h in histories],
            residual_histories=histories,
            error_linf=float(np.max(abs(error))),
            error_relative_l2=float(np.linalg.norm(error) / np.linalg.norm(p)),
            residual_relative_l2=rn / bn,
            residual_linf=float(np.max(abs(residual))),
            converged=rn <= threshold,
        )
        row["peak_rss_bytes"] = peak()
        target.write_text(json.dumps(row, indent=2))
        del p, rhs, sol, residual, error
        gc.collect()
    print(f"Wrote {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int)
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--accel", choices=["none", "cg"], default="none")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out is None:
        args.out = Path(
            "build/pressure3d_pyamg" + ("_cg" if args.accel == "cg" else "")
        )
    if args.size:
        worker(args)
        return
    rows = []
    for n in args.sizes:
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--size",
                str(n),
                "--accel",
                args.accel,
                "--repeats",
                str(args.repeats),
                "--out",
                str(args.out),
            ],
            check=True,
            env=os.environ.copy(),
        )
        rows.append(json.loads((args.out / f"{n}_pyamg.json").read_text()))
        (args.out / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
