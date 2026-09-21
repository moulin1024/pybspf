"""Full-space BSPF oversampled strong-residual Poisson accuracy experiment.

This is a separate least-squares backend, not a change to the Nitsche method.
The linear solve receives only f and boundary g; interior exact values are
used exclusively afterwards for independent validation.
"""

import json
from pathlib import Path

import jax
import numpy as np
import scipy.linalg as la

from bspf_models.elliptic.embedded_poisson import background_line
from bspf_models.elliptic.embedded_poisson import benchmark_domains
from bspf_models.elliptic.embedded_poisson import manufactured
from bspf_models._numerics.trial_spaces import stream_evaluate_line
from pybspf.tensor import tensor_product
from embedded_poisson_approximation import interior

jax.config.update("jax_enable_x64", True)


def operators(line, points):
    factors = []
    for axis in range(2):
        unique, inverse = np.unique(points[:, axis], return_inverse=True)
        factors.append([v[inverse] for v in stream_evaluate_line(line, unique)])
    (x, dx, dxx), (y, dy, dyy) = factors

    def pair(a, b):
        return tensor_product(a, b, paired=True)

    return pair(x, y), pair(dx, y), pair(x, dy), pair(dxx, y) + pair(x, dyy)


def main():
    out = Path("build/embedded_poisson_diagnosis")
    out.mkdir(parents=True, exist_ok=True)
    line = background_line(33)
    results = {}
    for domain in benchmark_domains():
        check = interior(domain, 101, 103, 0.61)
        cb, cx, cy, cl = operators(line, check)
        ep, ew, _ = domain.boundary_rule(32)
        eb = operators(line, ep)[0]
        exact, grad, forcing = manufactured(check)
        rows = []
        for count, order in [(51, 16), (67, 24)]:
            points = interior(domain, count, count + 2, 0.37)
            lap = operators(line, points)[3]
            boundary, weights, _ = domain.boundary_rule(order)
            edge = operators(line, boundary)[0]
            f = manufactured(points)[2]
            g = manufactured(boundary)[0]
            # Dimensionless domain: volume residual and boundary-value residual.
            # Avoid normal equations; solve the rectangular system by SVD.
            volume_weight = 2 / np.sqrt(count * (count + 2))
            boundary_weight = np.sqrt(weights)
            matrix = np.vstack((-volume_weight * lap, boundary_weight[:, None] * edge))
            rhs = np.r_[volume_weight * f, boundary_weight * g]
            scale = la.norm(matrix, axis=0)
            normalized = matrix / scale
            for cutoff in [1e-12, 1e-14]:
                coefficient, _, rank, singular = la.lstsq(normalized, rhs, cond=cutoff)
                coefficient /= scale
                error = cb @ coefficient - exact
                ge = np.column_stack((cx @ coefficient, cy @ coefficient)) - grad
                be = eb @ coefficient - manufactured(ep)[0]
                residual = -cl @ coefficient - forcing
                row = dict(
                    training_grid=count,
                    boundary_order=order,
                    svd_cutoff=cutoff,
                    rank=int(rank),
                    raw_dofs=len(coefficient),
                    relative_l2=float(la.norm(error) / la.norm(exact)),
                    relative_gradient=float(la.norm(ge) / la.norm(grad)),
                    boundary_rms=float(np.sqrt(np.sum(ew * be**2) / ew.sum())),
                    boundary_linf=float(np.max(abs(be))),
                    pde_residual_rms=float(np.sqrt(np.mean(residual**2))),
                    sampled_linf=float(np.max(abs(error))),
                    training_relative_residual=float(
                        la.norm(matrix @ coefficient - rhs) / la.norm(rhs)
                    ),
                    minimum_relative_singular=float(singular[-1] / singular[0]),
                )
                rows.append(row)
                print(domain.name, json.dumps(row), flush=True)
                np.savez(
                    out / f"{domain.name}_full_space_n{count}_cut{cutoff}.npz",
                    coefficient=coefficient,
                    points=check,
                    value=cb @ coefficient,
                )
        results[domain.name] = rows
        (out / "full_space_poisson.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
