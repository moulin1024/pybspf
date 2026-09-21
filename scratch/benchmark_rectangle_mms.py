"""Rectangle MMS: tensor inverse, cuSOLVER, FD AMGX-PCG and unpreconditioned CG.

Each case/backend runs in a fresh process. Use --amgx-python for a separate
working PyAMGX environment and --amgx-runtime-path for its CUDA libraries.
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.special import roots_legendre

LENGTHS = (2.0, 3.0)


def axial(points, length, axis, term):
    """Value and analytic physical second derivative of one MMS factor."""
    t = np.asarray(points) / length
    p, dp, ddp = t * (1 - t), 1 - 2 * t, -2.0
    if term == 0:
        c = 0.7 if axis == 0 else -0.4
        q = np.exp(c * t)
        dq, ddq = c * q, c * c * q
    else:
        k = (5 if axis == 0 else 3) * np.pi
        q = np.sin(k * t) if axis == 0 else np.cos(k * t)
        dq = k * np.cos(k * t) if axis == 0 else -k * np.sin(k * t)
        ddq = -k * k * q
    return p * q, (ddp * q + 2 * dp * dq + p * ddq) / length**2


def manufactured(x, y):
    u = np.zeros((len(x), len(y)))
    f = np.zeros_like(u)
    for term, amplitude in enumerate((1.0, 0.2)):
        a, aa = axial(x, LENGTHS[0], 0, term)
        b, bb = axial(y, LENGTHS[1], 1, term)
        u += amplitude * np.outer(a, b)
        f -= amplitude * (np.outer(aa, b) + np.outer(a, bb))
    return u, f


def fd_axis(n, length):
    h = length / (n + 1)
    stiffness = (
        sparse.diags(
            (-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)), (-1, 0, 1), format="csr"
        )
        / h**2
    )
    return np.eye(n), stiffness.toarray()


def fd_evaluation(n, length, points):
    z = np.asarray(points) * (n + 1) / length
    left = np.minimum(z.astype(int), n)
    fraction = z - left
    values = np.zeros((len(points), n + 2))
    values[np.arange(len(points)), left] = 1 - fraction
    values[np.arange(len(points)), left + 1] = fraction
    return values[:, 1:-1]


def assemble(discretization, n, path):
    start = perf_counter()
    nodes = [np.linspace(0, length, n + 2) for length in LENGTHS]
    gauss, weights = roots_legendre(257)
    check = [(gauss + 1) * length / 2 for length in LENGTHS]
    arrays = {}
    if discretization == "fd":
        for label, length, points in zip(("x", "y"), LENGTHS, check):
            arrays["m" + label], arrays["k" + label] = fd_axis(n, length)
            arrays["e" + label] = fd_evaluation(n, length, points)
        arrays["load"] = manufactured(nodes[0][1:-1], nodes[1][1:-1])[1]
        arrays["nodal_x"] = arrays["nodal_y"] = np.eye(n)
    else:
        import jax.numpy as jnp
        from bspf_jax import galerkin_1d, interpolate, plan_1d
        from bspf_jax.galerkin import _quadrature_rule

        import jax

        jax.config.update("jax_enable_x64", True)
        plans, weak, quad = [], [], []
        for label, grid, points in zip(("x", "y"), nodes, check):
            p = plan_1d(jnp.asarray(grid), degree=5, n_basis=16, boundary_points=7)
            w = galerkin_1d(p, constraints=((0, 0), (1, 0)), quadrature_order=8)
            plans.append(p)
            weak.append(w)
            quad.append(np.asarray(_quadrature_rule(p, 8)[0]))
            arrays["m" + label] = np.asarray(w.mass)
            arrays["k" + label] = np.asarray(w.stiffness)
            arrays["e" + label] = np.asarray(
                interpolate(p, w.extension, jnp.asarray(points))
            )
            arrays["nodal_" + label] = np.asarray(w.extension)[1:-1]
        load = np.zeros((n, n))
        bx, by = (np.asarray(w.values) for w in weak)
        wx, wy = (np.asarray(w.quadrature_weights) for w in weak)
        for term, amplitude in enumerate((1.0, 0.2)):
            a, aa = axial(quad[0], LENGTHS[0], 0, term)
            b, bb = axial(quad[1], LENGTHS[1], 1, term)
            load -= amplitude * (
                np.outer(bx.T @ (wx * aa), by.T @ (wy * b))
                + np.outer(bx.T @ (wx * a), by.T @ (wy * bb))
            )
        arrays["load"] = load
    arrays["exact"] = manufactured(*check)[0]
    arrays["nodal_exact"] = manufactured(nodes[0][1:-1], nodes[1][1:-1])[0]
    arrays["check_weights"] = np.outer(weights, weights)
    arrays["assembly_seconds"] = np.array(perf_counter() - start)
    np.savez(path, **arrays)


def assembled_matrix(d, discretization):
    if discretization == "fd":
        return sparse.kron(
            sparse.csr_matrix(d["kx"]), sparse.eye(len(d["my"])), format="csr"
        ) + sparse.kron(
            sparse.eye(len(d["mx"])), sparse.csr_matrix(d["ky"]), format="csr"
        )
    return np.kron(d["kx"], d["my"]) + np.kron(d["mx"], d["ky"])


def diagnostics(d, solution):
    u = solution.reshape(d["load"].shape)
    residual = d["kx"] @ u @ d["my"].T + d["mx"] @ u @ d["ky"].T - d["load"]
    sampled = d["ex"] @ u @ d["ey"].T
    nodal = d["nodal_x"] @ u @ d["nodal_y"].T
    norm = lambda a: np.sqrt(np.sum(d["check_weights"] * a**2))
    return {
        "relative_residual": float(
            np.linalg.norm(residual) / np.linalg.norm(d["load"])
        ),
        "relative_l2_error": float(norm(sampled - d["exact"]) / norm(d["exact"])),
        "max_error": float(np.max(np.abs(sampled - d["exact"]))),
        "nodal_relative_error": float(
            np.linalg.norm(nodal - d["nodal_exact"]) / np.linalg.norm(d["nodal_exact"])
        ),
    }


def amgx_config(rtol):
    return {
        "config_version": 2,
        "determinism_flag": 1,
        "exception_handling": 1,
        "solver": {
            "solver": "PCG",
            "max_iters": 2000,
            "tolerance": rtol,
            "convergence": "RELATIVE_INI",
            "norm": "L2",
            "monitor_residual": 1,
            "store_res_history": 1,
            "print_solve_stats": 0,
            "preconditioner": {
                "solver": "AMG",
                "algorithm": "CLASSICAL",
                "max_iters": 1,
                "max_levels": 50,
                "cycle": "V",
                "presweeps": 1,
                "postsweeps": 1,
                "interpolator": "D2",
                "aggressive_levels": 0,
                "coarse_solver": "DENSE_LU_SOLVER",
                "smoother": {"solver": "BLOCK_JACOBI", "relaxation_factor": 0.8},
            },
        },
    }


def worker(args):
    d = dict(np.load(args.data))
    backend = args.worker
    result = {
        "backend": backend,
        "discretization": args.discretization,
        "n": args.n,
        "unknowns": args.n**2,
        "assembly_seconds": float(d["assembly_seconds"]),
        "rtol": args.rtol,
        "repeats": args.repeats,
    }
    resources = []
    if backend == "tensor":
        from bspf_jax import plan_rectangle_poisson, solve_rectangle_poisson

        import jax

        jax.config.update("jax_enable_x64", True)
        device = jax.devices("gpu")[0]
        start = perf_counter()
        plan = plan_rectangle_poisson(d["mx"], d["kx"], d["my"], d["ky"], device=device)
        rhs = jax.device_put(d["load"], device)
        jax.block_until_ready((plan, rhs))
        result["setup_seconds"] = perf_counter() - start
        result["factor_bytes"] = sum(a.nbytes for a in plan)
        result["device"] = device.device_kind
        result["library"] = "jax " + jax.__version__

        def solve():
            return solve_rectangle_poisson(plan, rhs)

        def sync(x=None):
            if x is not None:
                x.block_until_ready()

        download = np.asarray
    elif backend == "amgx_pcg":
        import pyamgx

        runtime = ctypes.CDLL("libcudart.so")

        def sync(x=None):
            if runtime.cudaDeviceSynchronize() != 0:
                raise RuntimeError("CUDA synchronization failed")

        pyamgx.initialize()
        start = perf_counter()
        config = amgx_config(args.rtol)
        cfg = pyamgx.Config().create_from_dict(config)
        resources.append(cfg)
        rsc = pyamgx.Resources().create_simple(cfg)
        resources.append(rsc)
        a = pyamgx.Matrix().create(rsc, mode="dDDI")
        b = pyamgx.Vector().create(rsc, mode="dDDI")
        x = pyamgx.Vector().create(rsc, mode="dDDI")
        solver = pyamgx.Solver().create(rsc, cfg, mode="dDDI")
        resources.extend((a, b, x, solver))
        matrix = assembled_matrix(d, "fd")
        a.upload_CSR(matrix)
        b.upload(d["load"].ravel())
        x.upload(np.zeros(args.n**2))
        solver.setup(a)
        sync()
        result["setup_seconds"] = perf_counter() - start
        result["amgx_config"] = config
        result["matrix_bytes"] = (
            matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        )
        result["library"] = "pyamgx API " + pyamgx.get_api_version()

        def solve():
            # Explicit device-side reset: do not assume the shortcut clears storage.
            x.set_zero()
            solver.solve(b, x, zero_initial_guess=True)
            return x

        def download(value):
            host = np.empty(args.n**2)
            value.download(host)
            return host
    else:
        import cupy as cp
        import cupyx.scipy.sparse as csp
        from cupyx.scipy.linalg import solve_triangular
        from cupyx.scipy.sparse.linalg import cg, spsolve

        cp.cuda.Device(0).use()

        def sync(x=None):
            cp.cuda.runtime.deviceSynchronize()

        start = perf_counter()
        matrix = assembled_matrix(d, args.discretization)
        rhs = cp.asarray(d["load"].ravel())
        if backend == "cusolver_dense":
            matrix = matrix.toarray() if sparse.issparse(matrix) else matrix
            a = cp.asarray(matrix)
            factor = cp.linalg.cholesky(a)
            result["matrix_bytes"] = a.nbytes
            result["factor_bytes"] = factor.nbytes

            def solve():
                y = solve_triangular(factor, rhs, lower=True)
                return solve_triangular(factor, y, lower=True, trans="T")
        else:
            a = csp.csr_matrix(matrix)
            result["matrix_bytes"] = a.data.nbytes + a.indices.nbytes + a.indptr.nbytes
            if backend == "cusolver_sparse_qr":
                result["solve_includes_factorization"] = True

                def solve():
                    return spsolve(a, rhs)
            else:

                def solve():
                    x, info = cg(a, rhs, rtol=args.rtol, atol=0, maxiter=10000)
                    if info:
                        raise RuntimeError(f"CG did not converge: {info}")
                    return x

        sync()
        result["setup_seconds"] = perf_counter() - start
        result["library"] = "cupy " + cp.__version__
        result["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
        download = cp.asnumpy
    try:
        start = perf_counter()
        solution = solve()
        sync(solution)
        result["first_solve_seconds"] = perf_counter() - start
        times = []
        for _ in range(args.repeats):
            start = perf_counter()
            solution = solve()
            sync(solution)
            times.append(perf_counter() - start)
        result["warm_seconds"] = times
        result["warm_median_seconds"] = float(np.median(times))
        if backend == "amgx_pcg":
            result["iterations"] = solver.iterations_number
            result["status"] = solver.status
            if solver.status != "success":
                raise RuntimeError("AMGX status: " + solver.status)
        if backend == "cg":
            # Separate untimed solve: callback would add Python overhead to timings.
            count = [0]

            def callback(x):
                count[0] += 1

            _, _info = cg(
                a, rhs, rtol=args.rtol, atol=0, maxiter=10000, callback=callback
            )
            sync()
            result["iterations"] = count[0]
        host = download(solution)
        result.update(diagnostics(d, host))
        result["algebraic_pass"] = result["relative_residual"] <= 2 * args.rtol
        if not result["algebraic_pass"]:
            raise RuntimeError(
                f"Independent residual failed: {result['relative_residual']}"
            )
        Path(args.result).write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if backend == "amgx_pcg":
            for resource in reversed(resources):
                resource.destroy()
            pyamgx.finalize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("build/rectangle_mms"))
    parser.add_argument(
        "--fd-sizes", nargs="+", type=int, default=[31, 63, 127, 255, 511]
    )
    parser.add_argument("--bspf-sizes", nargs="+", type=int, default=[15, 31, 63, 127])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--dense-max-unknowns", type=int, default=4096)
    parser.add_argument("--sparse-direct-max-size", type=int, default=255)
    parser.add_argument("--amgx-python", default=sys.executable)
    parser.add_argument("--amgx-runtime-path", default="")
    parser.add_argument(
        "--worker",
        choices=[
            "assemble",
            "tensor",
            "cusolver_dense",
            "cusolver_sparse_qr",
            "amgx_pcg",
            "cg",
        ],
    )
    parser.add_argument("--discretization", choices=["fd", "bspf"])
    parser.add_argument("--n", type=int)
    parser.add_argument("--data")
    parser.add_argument("--result")
    args = parser.parse_args()
    if args.repeats < 1 or not 0 < args.rtol < 1:
        parser.error("positive repeats and 0 < rtol < 1 required")
    if args.worker == "assemble":
        assemble(args.discretization, args.n, args.data)
        return
    if args.worker:
        worker(args)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root / "jax/src") + os.pathsep + env.get("PYTHONPATH", "")
    # Pip CUDA-12 wheels serve CuPy/JAX; PyAMGX can use its separate CUDA runtime.
    cuda12 = list((Path(sysconfig.get_paths()["purelib"]) / "nvidia").glob("*/lib"))
    base_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(map(str, cuda12)) + os.pathsep + base_ld
    results = {
        "description": "Continuous rectangle MMS; FP64; zero Dirichlet; 7-point endpoint fit; degree-5 BSPF",
        "cases": [],
        "skipped": [],
        "failures": [],
    }
    for disc, sizes in (("bspf", args.bspf_sizes), ("fd", args.fd_sizes)):
        for n in sizes:
            data = args.out / f"{disc}_{n}.npz"
            common = [
                str(Path(__file__).resolve()),
                "--discretization",
                disc,
                "--n",
                str(n),
                "--data",
                str(data),
                "--repeats",
                str(args.repeats),
                "--rtol",
                str(args.rtol),
            ]
            with (args.out / f"{disc}_{n}_assembly.log").open("w") as log:
                subprocess.run(
                    [sys.executable, *common, "--worker", "assemble"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            backends = ["tensor", "cusolver_dense"]
            if disc == "fd":
                backends += ["cusolver_sparse_qr", "amgx_pcg", "cg"]
            for backend in backends:
                key = f"{disc}_{n}_{backend}"
                reason = None
                if backend == "cusolver_dense" and n * n > args.dense_max_unknowns:
                    reason = "dense matrix size cap"
                if backend == "cusolver_sparse_qr" and n > args.sparse_direct_max_size:
                    reason = "sparse QR size cap"
                if reason:
                    results["skipped"].append({"case": key, "reason": reason})
                    continue
                child_env = env.copy()
                interpreter = sys.executable
                if backend == "amgx_pcg":
                    interpreter = args.amgx_python
                    child_env["LD_LIBRARY_PATH"] = (
                        args.amgx_runtime_path + os.pathsep + base_ld
                    )
                result = args.out / f"{key}.json"
                print(f"Running {key}", flush=True)
                with (args.out / f"{key}.log").open("w") as log:
                    run = subprocess.run(
                        [
                            interpreter,
                            *common,
                            "--worker",
                            backend,
                            "--result",
                            str(result),
                        ],
                        env=child_env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if run.returncode:
                    results["failures"].append(
                        {"case": key, "returncode": run.returncode}
                    )
                    print(f"FAILED {key}: see log", flush=True)
                else:
                    case = json.loads(result.read_text())
                    results["cases"].append(case)
                    print(
                        f"  {case['warm_median_seconds'] * 1e3:.3f} ms; L2 {case['relative_l2_error']:.3e}; residual {case['relative_residual']:.3e}",
                        flush=True,
                    )
                (args.out / "results.json").write_text(
                    json.dumps(results, indent=2) + "\n"
                )
    if results["failures"]:
        raise SystemExit("Some cases failed; inspect results.json and individual logs")


if __name__ == "__main__":
    main()
