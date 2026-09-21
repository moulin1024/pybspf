"""Re-evaluate saved N=65 solutions on higher, common physical quadrature."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from bspf_models.elliptic.embedded_poisson import benchmark_domains
from bspf_models.elliptic.random_wave_mms import RandomWaveMMS
from compare_poisson_bases import ProfileMMS, TrialBasis, exact_jets, field_jets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("build/poisson_basis_comparison")
    )
    parser.add_argument("--order", type=int, default=24)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    points, weights, _ = benchmark_domains()[0].volume_rule(args.order)
    names = ["4pi", "12pi", "polynomial", "gaussian", "rational"]
    cases = [RandomWaveMMS.create(kmax=k * np.pi) for k in (4, 12)]
    cases += [ProfileMMS(name) for name in names[2:]]
    truth = [exact_jets(case, points) for case in cases]

    def norm(value):
        return np.sqrt(
            np.sum(weights.reshape((-1,) + (1,) * (value.ndim - 1)) * value**2)
        )

    rows = []
    for family in ("fourier", "bspline", "bspf"):
        print("BASIS", family, "Q", args.order, "points", len(points), flush=True)
        basis = TrialBasis(family, 65)
        factors = basis.factors(points)
        for task in ("pde", "h2_fit"):
            for name, exact in zip(names, truth):
                with np.load(
                    args.out / f"{family}_n65" / f"{task}_{name}.npz"
                ) as saved:
                    coefficients = saved["coefficient"]
                result = field_jets(factors, coefficients)
                error = result - exact
                row = dict(
                    family=family,
                    n=65,
                    task=task,
                    case=name,
                    value=float(norm(error[:, 0]) / norm(exact[:, 0])),
                    gradient=float(norm(error[:, 1:3]) / norm(exact[:, 1:3])),
                    h2=float(norm(error) / norm(exact)),
                    laplacian=float(
                        norm(error[:, 3] + error[:, 5])
                        / norm(exact[:, 3] + exact[:, 5])
                    ),
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
        (args.out / "refined_validation.json").write_text(
            json.dumps(
                dict(
                    order=args.order,
                    points=len(points),
                    cutoff=1e-13,
                    rows=rows,
                    complete=family == "bspf",
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
