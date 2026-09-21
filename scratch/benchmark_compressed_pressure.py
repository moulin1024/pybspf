"""CPU benchmark of actual factored JAX transforms; no refinement in any case."""

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from bspf_models.elliptic.pressure import plan_pressure_poisson2d
from bspf_models.elliptic.pressure import compress_pressure_plan
from bspf_models.elliptic.pressure import pressure_schur
from bspf_models.elliptic.pressure import pressure_gradient
from bspf_models.elliptic.pressure import solve_pressure_poisson2d
from bspf_models.elliptic.pressure import project_pressure2d
from bspf_models.elliptic.pressure import _tensor_solve
from bspf_models._numerics._compressed_transform import transform_storage

jax.config.update("jax_enable_x64", True)


def timing(fn, *args):
    start = time.perf_counter()
    result = jax.block_until_ready(fn(*args))
    first = time.perf_counter() - start
    samples = []
    for _ in range(9):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - start)
    return result, dict(
        first_call_seconds=first,
        median_seconds=float(np.median(samples)),
        min_seconds=min(samples),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument(
        "--out", type=Path, default=Path("build/layered_pressure_benchmark")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    core = jax.jit(_tensor_solve)
    stage = jax.jit(
        lambda plan, raw: project_pressure2d(
            plan, raw, completion=False, refinement_steps=0
        )
    )
    scalar = jax.jit(
        lambda plan, b, g: solve_pressure_poisson2d(
            plan, b, wall_gradient=g, refinement_steps=0
        )
    )
    for n in args.sizes:
        print(f"N={n}: building dense pressure plan", flush=True)
        start = time.perf_counter()
        grid = np.linspace(0, 1, n)
        dense = plan_pressure_poisson2d(
            grid,
            grid,
            endpoint_method="chebyshev",
            baseline_points=16,
            chebyshev_modes=12,
        )
        setup = time.perf_counter() - start
        print(f"N={n}: dense setup {setup:.2f}s; building compressed plans", flush=True)
        start = time.perf_counter()
        compressed = compress_pressure_plan(dense, layout="layered")
        grouped = compress_pressure_plan(dense, layout="grouped")
        compression_seconds = time.perf_counter() - start
        print(f"N={n}: compression setup {compression_seconds:.2f}s", flush=True)
        rng = np.random.default_rng(71)
        raw = jnp.asarray(rng.standard_normal((n, n, 2)))
        xx, yy = jnp.meshgrid(dense.x.x, dense.y.x, indexing="ij")
        p = jnp.exp(xx + 0.5 * yy) + jnp.sin(3 * jnp.pi * xx) * jnp.cos(2 * jnp.pi * yy)
        rhs = pressure_schur(dense, p)
        gradient = pressure_gradient(dense, p)
        plans = {
            "dense_complex": dense,
            "dct_hodlr_grouped": grouped,
            "dct_hodlr_layered": compressed,
        }
        row = dict(
            N=n,
            dense_setup_seconds=setup,
            compression_setup_seconds=compression_seconds,
            storage={
                "layered": transform_storage(compressed.x.compressed),
                "grouped": transform_storage(grouped.x.compressed),
            },
            backends={},
        )
        reference = None
        for name, plan in plans.items():
            print(f"N={n}: timing {name}", flush=True)
            _, ct = timing(core, plan, rhs)
            (velocity, diagnostic), st = timing(stage, plan, raw)
            result, _ = timing(scalar, plan, rhs, gradient)
            if reference is None:
                reference = velocity
            expected = p - jnp.sum(plan.weights * p) / jnp.sum(plan.weights)
            row["backends"][name] = dict(
                core=ct,
                projection=st,
                projected_velocity_relative_difference=float(
                    jnp.linalg.norm(velocity - reference) / jnp.linalg.norm(reference)
                ),
                projection_converged=bool(diagnostic.converged),
                projection_schur_linf=float(diagnostic.schur_residual_linf),
                smooth_converged=bool(result.converged),
                smooth_pressure_relative_error=float(
                    jnp.linalg.norm(result.pressure - expected)
                    / jnp.linalg.norm(expected)
                ),
                smooth_schur_linf=float(result.schur_residual_linf),
            )
            (args.out / f"partial_{n}.json").write_text(json.dumps(row, indent=2))
            print(
                f"N={n}: {name} core={ct['median_seconds']:.6f}s projection={st['median_seconds']:.6f}s",
                flush=True,
            )
        # Random scalar pressure checks completion, not just the projected field.
        p = jnp.asarray(rng.standard_normal((n, n)))
        rhs = pressure_schur(dense, p)
        gradient = pressure_gradient(dense, p)
        expected = p - jnp.sum(dense.weights * p) / jnp.sum(dense.weights)
        for name, plan in plans.items():
            result = jax.block_until_ready(scalar(plan, rhs, gradient))
            row["backends"][name]["random_pressure_relative_error"] = float(
                jnp.linalg.norm(result.pressure - expected) / jnp.linalg.norm(expected)
            )
            row["backends"][name]["random_converged"] = bool(result.converged)
        records.append(row)
        (args.out / "results.json").write_text(
            json.dumps(
                dict(
                    device=str(jax.devices()),
                    jax_version=jax.__version__,
                    refinement_steps=0,
                    results=records,
                ),
                indent=2,
            )
        )
        print(json.dumps(row), flush=True)
        jax.clear_caches()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name in ["dense_complex", "dct_hodlr_grouped", "dct_hodlr_layered"]:
        for ax, key in zip(axes, ["core", "projection"]):
            ax.plot(
                [r["N"] for r in records],
                [1000 * r["backends"][name][key]["median_seconds"] for r in records],
                "o-",
                label=name,
            )
            ax.set(
                xlabel="Grid nodes per axis",
                ylabel="Median milliseconds",
                title=key + "; zero refinement",
            )
            ax.grid(alpha=0.25)
            ax.legend()
    fig.savefig(args.out / "timings.png", dpi=170)


if __name__ == "__main__":
    main()
