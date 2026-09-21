"""Separate BSPF approximation error from the curved Poisson discretization.

Fits known exact values on one interior grid and checks another grid and the
exact spline boundary. This is an approximation diagnostic, NOT a PDE solve.
"""

import json
from pathlib import Path

import jax
import numpy as np
import scipy.linalg as la

from bspf_models.elliptic.embedded_poisson import background_line
from bspf_models.elliptic.embedded_poisson import basis_values
from bspf_models.elliptic.embedded_poisson import benchmark_domains
from bspf_models.elliptic.embedded_poisson import manufactured

jax.config.update("jax_enable_x64", True)


def interior(domain, nx, ny, offset):
    x = -1 + 2 * (np.arange(nx) + offset) / nx
    y = -1 + 2 * (np.arange(ny) + offset) / ny
    points = []
    for xx in x:
        for lo, hi in domain.intersections(xx):
            yy = y[(y > lo) & (y < hi)]
            points.append(np.column_stack((np.full_like(yy, xx), yy)))
    return np.vstack(points)


def main():
    out = Path("build/embedded_poisson_diagnosis")
    out.mkdir(parents=True, exist_ok=True)
    line = background_line(33)
    results = {}
    for domain in benchmark_domains():
        train = interior(domain, 67, 69, 0.37)
        check = interior(domain, 89, 87, 0.61)
        boundary, _, _ = domain.boundary_rule(24)
        a = basis_values(line, train, 33)[0]
        b, dx, dy = basis_values(line, check, 33)
        edge = basis_values(line, boundary, 33)[0]
        target = manufactured(train)[0]
        exact, grad, _ = manufactured(check)
        rows = []
        for modes in [6, 10, 14, 20, 26, 33]:
            idx = (np.arange(modes)[:, None] * 33 + np.arange(modes)).ravel()
            coefficients, _, rank, singular = la.lstsq(a[:, idx], target, cond=1e-14)
            error = b[:, idx] @ coefficients - exact
            gradient = np.column_stack(
                (dx[:, idx] @ coefficients, dy[:, idx] @ coefficients)
            )
            edge_error = edge[:, idx] @ coefficients - manufactured(boundary)[0]
            row = dict(
                modes=modes,
                rank=int(rank),
                interior_relative_l2=float(la.norm(error) / la.norm(exact)),
                gradient_relative_l2=float(la.norm(gradient - grad) / la.norm(grad)),
                boundary_linf=float(np.max(abs(edge_error))),
                training_residual=float(
                    la.norm(a[:, idx] @ coefficients - target) / la.norm(target)
                ),
                smallest_relative_singular=float(singular[-1] / singular[0]),
            )
            rows.append(row)
            print(domain.name, json.dumps(row), flush=True)
        results[domain.name] = dict(
            training_points=len(train), check_points=len(check), rows=rows
        )
        (out / "approximation.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
