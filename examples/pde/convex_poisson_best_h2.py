"""Independent H2 approximation diagnostic; exact jets NEVER enter PDE solver."""

import json
import pickle
from pathlib import Path

import jax
import numpy as np
import scipy.linalg as la

from bspf_models.elliptic.embedded_poisson import benchmark_domains
from pybspf.tensor import tensor_product
from bspf_models.elliptic.smooth_extension import factors
from bspf_models.elliptic.smooth_extension import evaluate_factors
from bspf_models.elliptic.random_wave_mms import RandomWaveMMS
from bspf_models.elliptic.convex_poisson import box_h2_root
from embedded_poisson_approximation import interior

jax.config.update("jax_enable_x64", True)


def jets(mms, p):
    u, g, _ = mms.evaluate(p)
    cosine = np.cos(p @ mms.wavevectors.T + mms.phases) * mms.amplitudes
    k = mms.wavevectors
    return np.column_stack(
        (
            u,
            g,
            -cosine @ (k[:, 0] ** 2),
            -np.sqrt(2) * cosine @ (k[:, 0] * k[:, 1]),
            -cosine @ (k[:, 1] ** 2),
        )
    )


def main():
    out = Path("build/convex_poisson")
    out.mkdir(parents=True, exist_ok=True)
    with Path("build/embedded_poisson_smooth_extension_n49/convex_geometry.pkl").open(
        "rb"
    ) as f:
        geom = pickle.load(f)
    domain = benchmark_domains()[0]
    train = interior(domain, 87, 89, 0.371)
    (x, dx, xx), (y, dy, yy) = factors(geom.line, train)
    cases = [RandomWaveMMS.create(kmax=n * np.pi) for n in (4, 12)]
    pairs = [(x, y), (dx, y), (x, dy), (xx, y), (np.sqrt(2) * dx, dy), (x, yy)]
    a = np.vstack([tensor_product(v, w, paired=True) for v, w in pairs]) / np.sqrt(
        len(train)
    )
    rhs = np.column_stack([jets(m, train).T.ravel() for m in cases]) / np.sqrt(
        len(train)
    )
    root = box_h2_root(geom.line)
    a = la.solve_triangular(root.T, a.T, lower=True).T
    print("H2 fit", a.shape, flush=True)
    coefficient, _, rank, _ = la.lstsq(a, rhs, cond=1e-13)
    coefficient = la.solve_triangular(root, coefficient)
    del a
    check = interior(domain, 127, 131, 0.613)
    basis = factors(geom.line, check)
    (x, dx, xx), (y, dy, yy) = basis
    rows = []
    for col, (band, mms) in enumerate(zip((4, 12), cases)):
        c = coefficient[:, col]
        u, g, lap = evaluate_factors(basis, c)
        eu, eg, ef = mms.evaluate(check)
        matrix = c.reshape(x.shape[1], y.shape[1])
        predicted = np.column_stack(
            [
                np.sum((v @ matrix) * w, axis=1)
                for v, w in [
                    (x, y),
                    (dx, y),
                    (x, dy),
                    (xx, y),
                    (np.sqrt(2) * dx, dy),
                    (x, yy),
                ]
            ]
        )
        truth = jets(mms, check)
        row = dict(
            band=band,
            rank=int(rank),
            value=float(la.norm(u - eu) / la.norm(eu)),
            gradient=float(la.norm(g - eg) / la.norm(eg)),
            laplacian=float(la.norm(lap + ef) / la.norm(ef)),
            h2=float(la.norm(predicted - truth) / la.norm(truth)),
            diagnostic_only=True,
        )
        rows.append(row)
        print(row, flush=True)
    (out / "best_h2.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
