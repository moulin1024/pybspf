"""Same-space/same-quadrature direct residual control for the extension trial."""

import json
import pickle
from pathlib import Path

import jax
import numpy as np
import scipy.linalg as la

from bspf_jax.embedded_poisson import benchmark_domains
from bspf_jax.random_wave_mms import RandomWaveMMS
from bspf_jax.smooth_extension import replace_boundary_rule, factors, evaluate_factors
from embedded_poisson_approximation import interior

jax.config.update("jax_enable_x64", True)


def main():
    source = Path("build/embedded_poisson_smooth_extension_n49")
    out = Path("build/embedded_poisson_smooth_extension_dense")
    results = []
    for domain in benchmark_domains():
        with (source / f"{domain.name}_geometry.pkl").open("rb") as handle:
            geom = pickle.load(handle)
        geom = replace_boundary_rule(geom, 12)
        print("Control", domain.name, flush=True)
        operator = np.vstack(
            (
                np.sqrt(geom.weights[:, None]) * geom.laplace,
                np.sqrt(geom.boundary_weights[:, None]) * geom.traces[0],
            )
        )
        cases = [RandomWaveMMS.create(kmax=n * np.pi) for n in [4, 12]]
        rhs = np.column_stack(
            [
                np.r_[
                    np.sqrt(geom.weights) * m.evaluate(geom.points)[2],
                    np.sqrt(geom.boundary_weights) * m.evaluate(geom.boundary)[0],
                ]
                for m in cases
            ]
        )
        scale = la.norm(operator, axis=0)
        operator /= scale
        coefficients, _, rank, _ = la.lstsq(operator, rhs, cond=1e-14)
        coefficients /= scale[:, None]
        del operator
        points = interior(domain, 101, 103, 0.61)
        basis = factors(geom.line, points)
        for index, (band, mms) in enumerate(zip([4, 12], cases)):
            v, grad, lap = evaluate_factors(basis, coefficients[:, index])
            exact, eg, ef = mms.evaluate(points)
            row = dict(
                domain=domain.name,
                band=band,
                rank=int(rank),
                relative_l2=float(la.norm(v - exact) / la.norm(exact)),
                relative_gradient=float(la.norm(grad - eg) / la.norm(eg)),
                relative_pde_residual=float(la.norm(-lap - ef) / la.norm(ef)),
            )
            results.append(row)
            print(json.dumps(row), flush=True)
        (out / "same_space_control.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
