"""Isolated-process CPU comparison of 3D direct cores; exactly zero refinement."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(args):
    import gc
    import resource
    from functools import partial
    import jax
    import jax.numpy as jnp
    import numpy as np
    import bspf_jax as b
    from bspf_jax.pressure3d import _tensor_solve3d
    from bspf_jax._compressed_transform import transform_storage

    jax.config.update("jax_enable_x64", True)
    n = args.size
    start = time.perf_counter()
    x = np.linspace(0, 1, n)
    plan = b.plan_pressure_poisson3d(x, x, x)
    jax.block_until_ready(plan)
    setup = time.perf_counter() - start
    storage = dict(stored_bytes=2 * (n - 2) ** 2 * 16, factor_ratio=1.0)
    compression = 0.0
    if args.backend == "compressed":
        start = time.perf_counter()
        plan = b.compress_pressure_plan3d(plan)
        jax.block_until_ready(plan)
        compression = time.perf_counter() - start
        storage = transform_storage(plan.lines[0].compressed)
    core = jax.jit(partial(_tensor_solve3d, batch_size=args.batch_size))
    action = jax.jit(b.pressure_action3d)
    schur = jax.jit(b.pressure_schur3d)

    @jax.jit
    def metrics(plan, got, expected, rhs, lifted):
        residual = (
            b.pressure_schur3d(plan, got) + lifted * b.pressure_lift3d(plan, got) - rhs
        )
        norm_b = jnp.linalg.norm(rhs)
        norm_r = jnp.linalg.norm(residual)
        return jnp.stack(
            [
                jnp.max(abs(got - expected)),
                jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected),
                jnp.max(abs(residual)),
                norm_r,
                norm_r / norm_b,
                norm_r <= 1e-9 + 1e-10 * norm_b,
            ]
        )

    row = dict(
        n=n,
        backend=args.backend,
        batch_size=args.batch_size,
        refinement_steps=0,
        dense_setup_seconds=setup,
        compression_setup_seconds=compression,
        transform_storage=storage,
        jax_version=jax.__version__,
        devices=[str(d) for d in jax.devices()],
        accuracy={},
    )
    for kind in ("smooth", "random"):
        if kind == "smooth":
            xx = jnp.asarray(x)[:, None, None]
            yy = jnp.asarray(x)[None, :, None]
            zz = jnp.asarray(x)[None, None, :]
            p = jnp.exp(xx + 0.5 * yy - 0.3 * zz) + jnp.sin(3 * jnp.pi * xx) * jnp.cos(
                2 * jnp.pi * yy
            ) * jnp.cos(jnp.pi * zz)
        else:
            p = jnp.asarray(np.random.default_rng(71).standard_normal((n, n, n)))
        rhs = jax.block_until_ready(action(plan, p))
        start = time.perf_counter()
        got = jax.block_until_ready(core(plan, rhs))
        first = time.perf_counter() - start
        if kind == "smooth":
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                got = jax.block_until_ready(core(plan, rhs))
                samples.append(time.perf_counter() - start)
            row["timing"] = dict(
                first_call_seconds=first,
                samples_seconds=samples,
                median_seconds=float(np.median(samples)),
                min_seconds=min(samples),
            )
            row["core_peak_rss_bytes"] = resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        vals = np.asarray(metrics(plan, got, p, rhs, True))
        row["accuracy"][kind] = dict(
            zip(
                [
                    "error_linf",
                    "error_relative_l2",
                    "residual_linf",
                    "residual_l2",
                    "residual_relative_l2",
                    "converged",
                ],
                [float(v) for v in vals],
            )
        )
        # Compatible original masked equation, not only the lifted equation.
        del rhs, got
        rhs = jax.block_until_ready(schur(plan, p))
        got = jax.block_until_ready(core(plan, rhs))
        vals = np.asarray(metrics(plan, got, p, rhs, False))
        row["accuracy"][kind]["unlifted_residual_relative_l2"] = float(vals[4])
        row["accuracy"][kind]["unlifted_converged"] = bool(vals[5])
        del rhs, got, p
        gc.collect()
    row["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{n}_{args.backend}.json"
    target.write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int)
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--backend", choices=["dense", "compressed"])
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("build/pressure3d_benchmark"))
    args = parser.parse_args()
    if args.size:
        worker(args)
        return
    records = []
    for n in args.sizes:
        for backend in ("dense", "compressed"):
            print(f"Running {n}^3 {backend}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--size",
                    str(n),
                    "--backend",
                    backend,
                    "--batch-size",
                    str(args.batch_size),
                    "--repeats",
                    str(args.repeats),
                    "--out",
                    str(args.out),
                ],
                check=True,
                env=os.environ.copy(),
            )
            records.append(json.loads((args.out / f"{n}_{backend}.json").read_text()))
            (args.out / "results.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
