"""Compare tensor two-level Poisson with a single-level iteration and dense SVD.

This is a research benchmark: unconverged cases are explicitly saved as failures,
not accepted by the fixed-grid solver API. Exact MMS data are used only for f/g
and independent error measurements.
"""

import argparse
from copy import copy
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import jax
import numpy as np

from bspf_models.elliptic.convex_poisson import ConvexPoissonPlan
from bspf_models.elliptic.convex_poisson import trace_transform
from bspf_models.elliptic.convex_poisson_tensor import TensorConvexPoissonPlan
from bspf_models.elliptic.embedded_poisson import benchmark_domains
from bspf_models.elliptic.random_wave_mms import RandomWaveMMS
from bspf_models.elliptic.smooth_extension import factors
from bspf_models.elliptic.smooth_extension import evaluate_factors
from embedded_poisson_approximation import interior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=33)
    parser.add_argument("--coarse", type=int, default=17)
    parser.add_argument(
        "--coarse-space", choices=["projected", "spectral"], default="projected"
    )
    parser.add_argument("--volume-order", type=int, default=12)
    parser.add_argument("--boundary-count", type=int, default=256)
    parser.add_argument("--regularization", type=float, default=1e-10)
    parser.add_argument("--coarse-rcond", type=float, default=1e-6)
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--data-tolerance", type=float)
    parser.add_argument("--bands", type=int, nargs="+", default=[4, 12])
    parser.add_argument("--reference", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=Path("build/convex_poisson_tensor/benchmark")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    domain = benchmark_domains()[0]
    settings = dict(
        nodes=args.nodes,
        volume_order=args.volume_order,
        boundary_count=args.boundary_count,
        coarse_nodes=args.coarse,
        coarse_space=args.coarse_space,
        coarse_rcond=args.coarse_rcond,
        regularization=args.regularization,
        maxiter=args.maxiter,
        tolerance=1e-11,
        data_tolerance=args.data_tolerance,
        cache_dir=args.out / "cache",
    )
    plan = TensorConvexPoissonPlan(domain, **settings)
    print("SETUP", json.dumps(plan.setup_diagnostics), flush=True)
    # Warm construction measures actual cache loading, separately from cold setup.
    warm = TensorConvexPoissonPlan(domain, **settings)
    warm_setup = warm.setup_diagnostics
    del warm
    single = copy(plan)
    single.coarse_left = np.empty((plan.augmented.shape[0], 0))
    single.coarse_right = np.empty((plan.ndofs, 0))
    single.coarse_singular = np.empty(0)
    single.setup_diagnostics = dict(
        plan.setup_diagnostics,
        coarse_rank=0,
        coarse_storage_bytes=0,
        shared_fine_setup=True,
    )
    cases = [("single_level", single), ("two_level", plan)]
    dense_seconds = None
    if args.reference:
        start = perf_counter()
        # Materialize A only for the reference, reusing EXACTLY the fine points
        # and basis to separate factorization cost from shared MPFR setup.
        x, xx = plan.bx[plan.index], plan.hx[plan.index]
        lap = -(
            np.einsum("pi,pj->pij", xx, plan.by) + np.einsum("pi,pj->pij", x, plan.hy)
        ).reshape(len(plan.points), -1)
        original = np.empty_like(lap)
        original[plan.order] = lap
        geometry = SimpleNamespace(
            domain=domain,
            nodes=args.nodes,
            half_width=1.2,
            volume_order=args.volume_order,
            line=plan.line,
            points=plan.points,
            weights=plan.weights,
            laplace=original,
        )
        ref = ConvexPoissonPlan(
            domain,
            nodes=args.nodes,
            volume_order=args.volume_order,
            boundary_count=args.boundary_count,
            geometry=geometry,
        )
        dense_seconds = perf_counter() - start
        del geometry, original, lap, x, xx
        cases.append(("dense_reference", ref))
    check = interior(domain, 127, 131, 0.613)
    check_basis = factors(plan.line, check)
    edge, _ = plan.arc.sample(2 * plan.boundary_count, offset=0.371)
    edge_basis = factors(plan.line, edge)
    rows = []
    for band in args.bands:
        mms = RandomWaveMMS.create(kmax=band * np.pi)
        exact, gradient, forcing = mms.evaluate(check)
        for name, solver in cases:
            start = perf_counter()
            kw = {} if name == "dense_reference" else dict(allow_unconverged=True)
            solution = solver.solve(
                lambda p: mms.evaluate(p)[2], lambda p: mms.evaluate(p)[0], **kw
            )
            seconds = perf_counter() - start
            u, g, lap = evaluate_factors(check_basis, solution.coefficients)
            boundary_error = (
                evaluate_factors(edge_basis, solution.coefficients)[0]
                - mms.evaluate(edge)[0]
            )
            row = dict(
                case=name,
                band=band,
                nodes=args.nodes,
                **solution.diagnostics,
                measured_solve_seconds=seconds,
                value=float(np.linalg.norm(u - exact) / np.linalg.norm(exact)),
                gradient=float(np.linalg.norm(g - gradient) / np.linalg.norm(gradient)),
                laplacian=float(
                    np.linalg.norm(lap + forcing) / np.linalg.norm(forcing)
                ),
                boundary_h32=float(
                    np.linalg.norm(trace_transform(boundary_error, plan.arc.length))
                ),
                boundary_linf=float(np.max(abs(boundary_error))),
            )
            rows.append(row)
            print("RESULT", json.dumps(row), flush=True)
            (args.out / "results.json").write_text(
                json.dumps(
                    dict(
                        rows=rows,
                        cold_tensor_setup=plan.setup_diagnostics,
                        warm_tensor_setup=warm_setup,
                        dense_additional_setup_seconds=dense_seconds,
                        reference_regularization=0.0,
                        shared_setup_note="Dense setup timing excludes shared line/volume-factor setup; no total speedup is implied.",
                    ),
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
