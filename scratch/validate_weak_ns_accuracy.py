"""Continuous manufactured NS accuracy, including a nonzero pressure gradient.

The exact velocity is exp(-t) curl(psi), psi=f(x)f(y), with zero wall velocity.
Analytic forcing is integrated at quadrature points; no discrete manufactured
forcing is used. Pressure is cos(x+2y)*exp(-t).
"""

import argparse
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import Polynomial
from math import comb
from bspf_models.fluids.weak_navier_stokes import plan_weak_navier_stokes2d
from bspf_models.fluids.weak_navier_stokes import weak_ns_load
from bspf_models.fluids.weak_navier_stokes import weak_ns_rhs
from bspf_models.fluids.weak_navier_stokes import weak_ns_project
from bspf_models.fluids.weak_navier_stokes import weak_ns_divergence


def shape(x, k=2.3):
    p = Polynomial([1, 0, -2, 0, 1])
    c = 0.2 + 1j * k
    return [
        sum(comb(d, j) * p.deriv(j)(x) * c ** (d - j) for j in range(d + 1))
        .__mul__(np.exp(c * x))
        .real
        for d in range(4)
    ]


def fields(x, y):
    f = shape(np.asarray(x)[:, None])
    g = shape(np.asarray(y)[None, :], k=1.7)
    u = np.stack((f[0] * g[1], -f[1] * g[0]), axis=-1)
    ux = np.stack((f[1] * g[1], -f[2] * g[0]), axis=-1)
    uy = np.stack((f[0] * g[2], -f[1] * g[1]), axis=-1)
    lap = np.stack((f[2] * g[1] + f[0] * g[3], -f[3] * g[0] - f[1] * g[2]), axis=-1)
    s = -np.sin(np.asarray(x)[:, None] + 2 * np.asarray(y)[None, :])
    gradp = np.stack((s, 2 * s), axis=-1)
    adv = u[..., 0, None] * ux + u[..., 1, None] * uy
    return u, lap, gradp, adv


def main():
    jax.config.update("jax_enable_x64", True)
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[40, 64, 96, 128])
    p.add_argument("--T", type=float, default=0.1)
    p.add_argument("--dt", type=float, default=0.0005)
    p.add_argument("--out", type=Path, default=Path("build/kh_weak/accuracy"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for n in args.sizes:
        start = time.perf_counter()
        plan = plan_weak_navier_stokes2d(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
        exact = jnp.asarray(fields(plan.x.x, plan.y.x)[0])
        u, lap, gradp, adv = fields(plan.x.quadrature_x, plan.y.quadrature_x)
        linear = weak_ns_load(plan, -u - 0.002 * lap + gradp)
        nonlinear = weak_ns_load(plan, adv)
        pressure_load = weak_ns_load(plan, gradp)

        def from_load(load):
            a = jnp.einsum(
                "ij,jkc,lk->ilc", plan.x.inverse_mass, load, plan.y.inverse_mass
            )
            return weak_ns_project(plan, jnp.zeros_like(exact).at[1:-1, 1:-1].set(a))

        projected = weak_ns_project(plan, exact)
        residual = weak_ns_rhs(plan, projected, linear + nonlinear) + exact

        def step(state, i):
            t = i * args.dt

            def rhs(v, t):
                return weak_ns_rhs(
                    plan, v, jnp.exp(-t) * linear + jnp.exp(-2 * t) * nonlinear
                )

            a = rhs(state, t)
            b = rhs(state + args.dt / 2 * a, t + args.dt / 2)
            c = rhs(state + args.dt / 2 * b, t + args.dt / 2)
            d = rhs(state + args.dt * c, t + args.dt)
            return state + args.dt / 6 * (a + 2 * b + 2 * c + d), None

        final = jax.jit(
            lambda v: jax.lax.scan(step, v, jnp.arange(round(args.T / args.dt)))[0]
        )(projected)
        rec = dict(
            n=n,
            initial_projection_error=float(jnp.max(abs(projected - exact))),
            gradient_projection_linf=float(jnp.max(abs(from_load(pressure_load)))),
            rhs_consistency_linf=float(jnp.max(abs(residual))),
            pde_error_linf=float(jnp.max(abs(final - jnp.exp(-args.T) * exact))),
            divergence_linf=float(jnp.max(abs(weak_ns_divergence(plan, final)))),
            elapsed_s=time.perf_counter() - start,
        )
        records.append(rec)
        print(json.dumps(rec), flush=True)
        (args.out / "results.json").write_text(
            json.dumps(dict(T=args.T, dt=args.dt, records=records), indent=2)
        )


if __name__ == "__main__":
    main()
