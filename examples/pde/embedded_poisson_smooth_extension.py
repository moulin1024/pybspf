"""Analytic-normal BSPF smooth extension on random-wave Poisson MMS."""

import argparse
import json
import pickle
from pathlib import Path

import jax
import numpy as np
import scipy.linalg as la

from bspf_models.elliptic.embedded_poisson import benchmark_domains
from bspf_models.elliptic.random_wave_mms import RandomWaveMMS
from bspf_models.elliptic.smooth_extension import assemble_geometry
from bspf_models.elliptic.smooth_extension import plan_smooth_extension
from bspf_models.elliptic.smooth_extension import basis_operators
from bspf_models.elliptic.smooth_extension import factors
from bspf_models.elliptic.smooth_extension import evaluate_factors
from bspf_models.elliptic.smooth_extension import replace_boundary_rule
from embedded_poisson_approximation import interior

jax.config.update("jax_enable_x64", True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=33)
    parser.add_argument("--volume-order", type=int, default=16)
    parser.add_argument("--boundary-order", type=int, default=4)
    parser.add_argument("--matching-orders", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--length", type=float, default=0.25)
    parser.add_argument("--half-width", type=float, default=1.2)
    parser.add_argument("--domains", nargs="+", default=["convex", "nonstar"])
    parser.add_argument(
        "--out", type=Path, default=Path("build/embedded_poisson_smooth_extension")
    )
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--geometry-cache", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = dict(
        nodes=args.nodes,
        volume_order=args.volume_order,
        boundary_order=args.boundary_order,
        half_width=args.half_width,
        extension_length=args.length,
        results=[],
    )
    cases = {f"kmax_{n}pi": RandomWaveMMS.create(kmax=n * np.pi) for n in [4, 12]}
    for domain in benchmark_domains():
        if domain.name not in args.domains:
            continue
        print("Assembling", domain.name, flush=True)
        cache = (args.geometry_cache or args.out) / f"{domain.name}_geometry.pkl"
        if args.reuse_cache and cache.exists():
            with cache.open("rb") as handle:
                geom = pickle.load(handle)
            if (geom.nodes, geom.volume_order, geom.half_width) != (
                args.nodes,
                args.volume_order,
                args.half_width,
            ):
                raise ValueError("Cached geometry configuration differs")
            if geom.boundary_order != args.boundary_order:
                geom = replace_boundary_rule(geom, args.boundary_order)
        else:
            geom = assemble_geometry(
                domain,
                nodes=args.nodes,
                volume_order=args.volume_order,
                boundary_order=args.boundary_order,
                half_width=args.half_width,
            )
            with cache.open("wb") as handle:
                pickle.dump(geom, handle)
        check = interior(domain, 101, 103, 0.61)
        check_factors = factors(geom.line, check)
        # Independent exact B-spline boundary, never reused collocation points.
        bp, bw, normals = domain.boundary_rule(20)
        traces = basis_operators(geom.line, bp, normals)
        bv, _, bn, bnn = traces
        for order in args.matching_orders:
            plan = plan_smooth_extension(
                geom, matching_order=order, extension_length=args.length
            )
            for name, mms in cases.items():
                u, xi, info = plan.solve(
                    mms.evaluate(geom.points)[2], mms.evaluate(geom.boundary)[0]
                )
                value, grad, laplace = evaluate_factors(check_factors, u)
                exact, eg, ef = mms.evaluate(check)
                boundary_error = bv @ u - mms.evaluate(bp)[0]
                trace_jumps = [
                    float(np.sqrt(np.sum(bw * (t @ (u - xi)) ** 2) / bw.sum()))
                    for t in (bv, bn, bnn)
                ]
                normal_error = bn @ u - np.sum(mms.evaluate(bp)[1] * normals, axis=1)
                info.update(
                    domain=domain.name,
                    case=name,
                    relative_l2=float(la.norm(value - exact) / la.norm(exact)),
                    relative_gradient=float(la.norm(grad - eg) / la.norm(eg)),
                    relative_pde_residual=float(la.norm(-laplace - ef) / la.norm(ef)),
                    boundary_rms=float(
                        np.sqrt(np.sum(bw * boundary_error**2) / bw.sum())
                    ),
                    normal_derivative_rms_error=float(
                        np.sqrt(np.sum(bw * normal_error**2) / bw.sum())
                    ),
                    independent_jet_jump_rms=trace_jumps,
                )
                results["results"].append(info)
                print(json.dumps(info), flush=True)
                np.savez(
                    args.out / f"{domain.name}_{name}_C{order}.npz",
                    solution=u,
                    extension=xi,
                    points=check,
                    value=value,
                    exact=exact,
                    boundary_points=bp,
                    boundary_error=boundary_error,
                    normal_derivative_error=normal_error,
                )
            (args.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
