"""Rejected 1D fixes: assess stability AND perturbation of a smooth BSPF field."""

import json
from pathlib import Path

import numpy as np
import scipy.linalg as la
from numpy.polynomial.legendre import legvander

from bspf_weak_advection_diffusion_1d import assemble, analytic_field

p = assemble(160)
a = p.generator(weak=False)
h = p.x[1] - p.x[0]
phi, px, pxx = analytic_field(p.x, "nonperiodic")
rows = {}


def record(name, matrix, extra=None):
    rows[name] = dict(
        max_real=float(la.eigvals(matrix).real.max()),
        smooth_rhs_change_linf=float(abs((matrix - a) @ phi[1:-1]).max()),
        **(extra or {}),
    )


record(
    "original",
    a,
    dict(
        original_truncation_linf=float(
            abs((a @ phi[1:-1]) - (-px + 0.002 * pxx)[1:-1]).max()
        )
    ),
)
values, vectors = la.eig(a)
inverse = la.solve(vectors, np.eye(len(a)))
unstable = values.real > 0
correction = (
    vectors[:, unstable] @ np.diag(values[unstable].real + 1) @ inverse[unstable]
).real
record("shift_unstable_eigenvalues", a - correction, dict(rank=int(sum(unstable))))
window, degree = 20, 9
q, _ = la.qr(legvander(np.linspace(-1, 1, window), degree), mode="full")
remainder = np.zeros((160, 160))
remainder[:window, :window] = q[:, degree + 1 :] @ q[:, degree + 1 :].T
record(
    "inflow_polynomial_highpass",
    a - remainder[1:-1, 1:-1] / h,
    dict(window=window, polynomial_degree=degree, rate=float(1 / h)),
)
broader = assemble(160, window=32)
record("larger_jet_window", broader.generator(weak=False), dict(window=32))
full = -p.strong_d1 + 0.002 * p.strong_d2
sat = []
for left in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]:
    for right in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]:
        m = full.copy()
        m[0, 0] -= left / h
        m[-1, -1] -= right * 0.002 / h**2
        sat.append(
            dict(left=left, right=right, max_real=float(la.eigvals(m).real.max()))
        )
rows["diagonal_sat_scan"] = dict(
    stable_count=sum(r["max_real"] < 0 for r in sat),
    trials=len(sat),
    best=min(sat, key=lambda r: r["max_real"]),
)
out = Path("build/bspf_advection_diffusion_1d")
out.mkdir(parents=True, exist_ok=True)
(out / "rejected_controls.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
