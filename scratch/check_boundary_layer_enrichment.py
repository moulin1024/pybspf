"""Independent scalar outflow-layer accuracy check, analytic continuous PDE."""

import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import scipy.linalg as la
from scipy.interpolate import BSpline
from scipy.special import roots_legendre
import jax
from bspf_jax.pressure import _make_line
from bspf_jax._weak_basis import mp_trial_values

jax.config.update("jax_enable_x64", True)
n = 48
nu = 0.002
x = np.linspace(0, 1, n)
original = _make_line(x, 9, 32, 13, 16, "chebyshev", 12, 1e-12)
host = SimpleNamespace(x=x, P=np.asarray(original.projector))
breaks = np.linspace(0, 1, 20)
knots = np.r_[np.repeat(0.0, 14), breaks[1:-1], np.repeat(1.0, 14)]
spline = BSpline(knots, np.eye(32), 13)
q, w = roots_legendre(32)
breaks = np.unique(np.r_[breaks, 1 - nu * np.array([0.5, 1, 2, 4, 8, 16, 32, 64])])
breaks = breaks[(breaks >= 0) & (breaks <= 1)]
t = np.concatenate(
    [(a + b) / 2 + (b - a) / 2 * q for a, b in zip(breaks[:-1], breaks[1:])]
)
weights = np.concatenate([(b - a) / 2 * w for a, b in zip(breaks[:-1], breaks[1:])])
b, g = mp_trial_values(host, spline, t)
bn, gn = mp_trial_values(host, spline, x)
records = []
curves = {}
for enriched in [False, True]:
    bb, gg, nn = b.copy(), g.copy(), bn.copy()
    if enriched:
        e = np.exp((t - 1) / nu)
        bb = np.column_stack((bb, e))
        gg = np.column_stack((gg, e / nu))
        nn = np.column_stack((nn, np.exp((x - 1) / nu)))
    z = la.null_space(nn[[0, -1]])
    bb = bb @ z
    gg = gg @ z
    a = bb.T @ (weights[:, None] * gg) + nu * gg.T @ (weights[:, None] * gg)
    c = la.solve(a, bb.T @ weights)
    result = bb @ c
    exact = t - np.expm1(t / nu) * np.exp(-1 / nu) / (1 - np.exp(-1 / nu))
    # exp(t/nu) is finite here (Pe=500), no discrete manufactured forcing.
    error = result - exact
    r = dict(
        enriched=enriched,
        linf=float(abs(error).max()),
        l2=float(np.sqrt(weights @ (error * error))),
    )
    records.append(r)
    curves["enriched" if enriched else "original"] = result
out = Path("build/kh_stream/layer_accuracy")
out.mkdir(parents=True, exist_ok=True)
(out / "results.json").write_text(json.dumps(records, indent=2))
np.savez_compressed(out / "curves.npz", x=t, exact=exact, **curves)
print(json.dumps(records, indent=2))
