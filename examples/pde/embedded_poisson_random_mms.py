"""Random-wave Poisson MMS on convex and non-star-shaped spline domains."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
import scipy.linalg as la

from bspf_models.elliptic.embedded_poisson import background_line
from bspf_models.elliptic.embedded_poisson import benchmark_domains
from bspf_models.elliptic.random_wave_mms import RandomWaveMMS
from bspf_models._numerics.trial_spaces import stream_evaluate_line
from pybspf.tensor import tensor_product
from embedded_poisson_approximation import interior

jax.config.update("jax_enable_x64", True)


def factors(line, points):
    result = []
    for axis in range(2):
        values, inverse = np.unique(points[:, axis], return_inverse=True)
        result.append([a[inverse] for a in stream_evaluate_line(line, values)])
    return result


def matrix(values, laplacian=False):
    (x, _, xx), (y, _, yy) = values
    if laplacian:
        return tensor_product(xx, y, paired=True) + tensor_product(x, yy, paired=True)
    return tensor_product(x, y, paired=True)


def fields(values, coefficient):
    (x, dx, xx), (y, dy, yy) = values
    c = coefficient.reshape(x.shape[1], y.shape[1])

    def apply(a, b):
        return np.sum((a @ c) * b, axis=1)

    return (
        apply(x, y),
        np.column_stack((apply(dx, y), apply(x, dy))),
        -(apply(xx, y) + apply(x, yy)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, nargs="+", default=[33, 41, 49])
    parser.add_argument(
        "--out", type=Path, default=Path("build/embedded_poisson_random_mms")
    )
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--sample-factor", type=int, default=2)
    parser.add_argument("--endpoint-points", type=int, default=16)
    parser.add_argument("--chebyshev-modes", type=int, default=12)
    parser.add_argument(
        "--endpoint-method", choices=["chebyshev", "taylor"], default="chebyshev"
    )
    parser.add_argument("--endpoint-regularization", type=float, default=1e-12)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = {
        f"kmax_{k}pi": RandomWaveMMS.create(seed=args.seed, kmax=k * np.pi)
        for k in [4, 8, 12]
    }
    for name, mms in cases.items():
        mms.save(args.out / f"{name}_spectrum.npz")
    results = dict(
        seed=args.seed,
        modes_per_mms=64,
        scalar_spectral_exponent=5 / 3,
        svd_cutoff=1e-14,
        sample_factor=args.sample_factor,
        endpoint=dict(
            method=args.endpoint_method,
            points=args.endpoint_points,
            modes=args.chebyshev_modes,
            regularization=args.endpoint_regularization,
            jet_order=9,
            spline_degree=13,
            spline_count=32,
        ),
        results=[],
    )
    for nodes in args.nodes:
        line = background_line(
            nodes,
            endpoint_method=args.endpoint_method,
            endpoint_points=args.endpoint_points,
            chebyshev_modes=args.chebyshev_modes,
            endpoint_regularization=args.endpoint_regularization,
        )
        for domain in benchmark_domains():
            count = args.sample_factor * nodes + 1
            train = interior(domain, count, count + 2, 0.37)
            check = interior(domain, 137, 139, 0.61)
            boundary, weights, _ = domain.boundary_rule(24)
            edge_check, check_weights, _ = domain.boundary_rule(40)
            print(
                f"Assembling {domain.name}, {nodes}x{nodes}, {len(train)} interior rows",
                flush=True,
            )
            lap = matrix(factors(line, train), laplacian=True)
            edge = matrix(factors(line, boundary))
            w = 2 / np.sqrt(count * (count + 2))
            operator = np.vstack((-w * lap, np.sqrt(weights[:, None]) * edge))
            del lap, edge
            rhs = np.column_stack(
                [
                    np.r_[
                        w * mms.evaluate(train)[2],
                        np.sqrt(weights) * mms.evaluate(boundary)[0],
                    ]
                    for mms in cases.values()
                ]
            )
            scale = la.norm(operator, axis=0)
            operator /= scale
            # One factorization, three distinct bandwidths; never fit interior u.
            coefficient, _, rank, singular = la.lstsq(
                operator, rhs, cond=1e-14, overwrite_a=False
            )
            training = operator @ coefficient - rhs
            coefficient /= scale[:, None]
            del operator
            test_factors, edge_factors = factors(line, check), factors(line, edge_check)
            for column, (name, mms) in enumerate(cases.items()):
                value, gradient, forcing = fields(test_factors, coefficient[:, column])
                ev = fields(edge_factors, coefficient[:, column])[0]
                exact, grad, force = mms.evaluate(check)
                eg = mms.evaluate(edge_check)[0]
                row = dict(
                    domain=domain.name,
                    nodes=nodes,
                    case=name,
                    training_points=len(train),
                    validation_points=len(check),
                    rank=int(rank),
                    candidate_dofs=nodes**2,
                    relative_l2=float(la.norm(value - exact) / la.norm(exact)),
                    relative_gradient=float(la.norm(gradient - grad) / la.norm(grad)),
                    relative_pde_residual=float(
                        la.norm(forcing - force) / la.norm(force)
                    ),
                    boundary_relative_l2=float(
                        np.sqrt(
                            np.sum(check_weights * (ev - eg) ** 2)
                            / np.sum(check_weights * eg**2)
                        )
                    ),
                    sampled_linf=float(np.max(abs(value - exact))),
                    training_relative_residual=float(
                        la.norm(training[:, column]) / la.norm(rhs[:, column])
                    ),
                    smallest_relative_singular=float(singular[-1] / singular[0]),
                )
                results["results"].append(row)
                print(json.dumps(row), flush=True)
                if nodes == max(args.nodes):
                    np.savez(
                        args.out / f"{domain.name}_{name}_solution.npz",
                        coefficient=coefficient[:, column],
                        points=check,
                        value=value,
                        exact=exact,
                        error=value - exact,
                        relative_pde_residual_field=(forcing - force)
                        / np.sqrt(np.mean(force**2)),
                        nodes=nodes,
                    )
            (args.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
