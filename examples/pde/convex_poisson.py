"""Accuracy benchmark for the convex BSPF H^(3/2)-trace Poisson solver."""

import argparse
import json
import pickle
from pathlib import Path

import jax
import numpy as np
import scipy.linalg as la

from bspf_jax.convex_poisson import ConvexPoissonPlan, evaluate_hessian_factors
from bspf_jax.embedded_poisson import benchmark_domains
from bspf_jax.random_wave_mms import RandomWaveMMS
from bspf_jax.smooth_extension import factors, evaluate_factors
from embedded_poisson_approximation import interior

jax.config.update("jax_enable_x64", True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=49)
    parser.add_argument("--cached-dirichlet", action="store_true")
    parser.add_argument("--boundary-count", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("build/convex_poisson"))
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    domain = benchmark_domains()[0]
    geom = None
    if args.cached_dirichlet:
        with Path(
            f"build/embedded_poisson_smooth_extension_n{args.nodes}/convex_geometry.pkl"
        ).open("rb") as f:
            geom = pickle.load(f)
    configurations = (
        [("h32_h2", 1.5, "h2", 1e-13)]
        if args.single
        else [
            ("l2_column", 0.0, "column", 1e-14),
            ("h32_column", 1.5, "column", 1e-14),
            ("h32_h2", 1.5, "h2", 1e-13),
        ]
    )
    rows = (
        json.loads((args.out / "results.json").read_text())
        if args.resume and (args.out / "results.json").exists()
        else []
    )
    check = interior(domain, 127, 131, 0.613)
    check_basis = None
    validation_cache = None
    for name, order, scaling, cutoff in configurations:
        if {
            r["band"]
            for r in rows
            if r["name"] == name
            and r["nodes"] == args.nodes
            and r["boundary_count"] == args.boundary_count
            and r["cached_dirichlet"] == args.cached_dirichlet
            and r["rcond"] == cutoff
        } == {4, 12}:
            continue
        print("assemble", name, flush=True)
        plan = ConvexPoissonPlan(
            domain,
            nodes=args.nodes,
            boundary_count=args.boundary_count,
            sobolev_order=order,
            coefficient_scaling=scaling,
            rcond=cutoff,
            geometry=geom,
        )
        if validation_cache is not None:
            plan.validation_cache = validation_cache
        if check_basis is None:
            check_basis = factors(plan.line, check)
        for band in (4, 12):
            mms = RandomWaveMMS.create(kmax=band * np.pi)

            def forcing(p):
                return mms.evaluate(p)[2]

            def boundary(p):
                return mms.evaluate(p)[0]

            solution = plan.solve(forcing, boundary)
            u, g, lap = evaluate_factors(check_basis, solution.coefficients)
            exact, eg, ef = mms.evaluate(check)
            cosine = np.cos(check @ mms.wavevectors.T + mms.phases) * mms.amplitudes
            eh = -np.einsum("nk,ki,kj->nij", cosine, mms.wavevectors, mms.wavevectors)
            h = evaluate_hessian_factors(check_basis, solution.coefficients)
            h2 = float(
                np.sqrt(
                    (
                        la.norm(u - exact) ** 2
                        + la.norm(g - eg) ** 2
                        + la.norm(h - eh) ** 2
                    )
                    / (la.norm(exact) ** 2 + la.norm(eg) ** 2 + la.norm(eh) ** 2)
                )
            )
            row = dict(
                name=name,
                band=band,
                nodes=args.nodes,
                cached_dirichlet=args.cached_dirichlet,
                value=float(la.norm(u - exact) / la.norm(exact)),
                gradient=float(la.norm(g - eg) / la.norm(eg)),
                laplacian=float(la.norm(lap + ef) / la.norm(ef)),
                h2=h2,
                **solution.diagnostics,
            )
            # Independent shifted boundary and volume quadrature, not training residual.
            row["validation"] = solution.validate(forcing, boundary, volume_order=20)
            rows.append(row)
            print(json.dumps(row), flush=True)
            np.savez(
                args.out / f"{name}_{band}pi.npz",
                coefficient=solution.coefficients,
                points=check,
                value=u,
                exact=exact,
            )
        (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        validation_cache = plan.validation_cache
        del plan, solution


if __name__ == "__main__":
    main()
