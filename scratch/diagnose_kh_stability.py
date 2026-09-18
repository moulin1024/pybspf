"""BSPF KH stability diagnostics; production solver and old movie untouched."""

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eig, eigvalsh
import bspf_jax as b
from bspf_jax.navier_stokes import (
    plan_navier_stokes2d,
    ns_raw_rhs,
    ns_rhs,
    ns_rk4_step,
    kh_initial_velocity,
)


def make(nx, ny, closure="bspf"):
    p = b.plan_pressure_poisson2d(
        np.linspace(-3, 3, nx),
        np.linspace(-1, 1, ny),
        endpoint_method="chebyshev",
        chebyshev_modes=12,
        baseline_points=16,
    )
    plan = plan_navier_stokes2d(p, viscosity=0.002, closure=closure)
    initial, base, diag = kh_initial_velocity(plan, perturbation=0.03)
    assert bool(diag.converged)
    return plan, initial, base


def operators(plan):
    out = {}
    for name, d1, d2, coord, scale in [
        ("x", plan.dx, plan.dxx, plan.pressure.x.x, 3),
        ("y", plan.dy, plan.dyy, plan.pressure.y.x, 1),
    ]:
        d1, d2 = np.array(d1), np.array(d2)
        xi = np.array(coord)[1:-1]
        for label, A in [
            ("diffusion", 0.002 * d2[1:-1, 1:-1]),
            ("D1_squared_diffusion", 0.002 * (d1 @ d1)[1:-1, 1:-1]),
            ("advection_diffusion", (-d1 + 0.002 * d2)[1:-1, 1:-1]),
            (
                "advection_diffusion_sponge",
                (-d1 + 0.002 * d2)[1:-1, 1:-1] - np.diag(80 * (xi / scale) ** 16),
            ),
        ]:
            lam, v = eig(A)
            idx = np.argmax(lam.real)
            u = v[:, idx]
            edge = abs(xi) > scale * 0.8
            z = 0.002 * lam
            rk = 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
            out[name + "_" + label] = dict(
                max_real=float(lam[idx].real),
                imag=float(lam[idx].imag),
                positive_count=int(np.sum(lam.real > 1e-8)),
                edge_mode_energy_fraction=float(
                    np.sum(abs(u[edge]) ** 2) / np.sum(abs(u) ** 2)
                ),
                numerical_abscissa=float(
                    eigvalsh((A + A.T) / 2, subset_by_index=[len(A) - 1, len(A) - 1])[0]
                ),
                rk4_dt002_max_amplification=float(np.max(abs(rk))),
            )
    return out


def evolution(plan, initial, base, args):
    p = plan.pressure
    x, y = jnp.meshgrid(p.x.x, p.y.x, indexing="ij")
    sx, sy = 80 * (x / 3) ** 16, 80 * y**16
    if args.variant == "sponge":
        sigma = sx + sy
    elif args.variant == "sponge_x":
        sigma = sx
    elif args.variant == "sponge_y":
        sigma = sy
    else:
        sigma = jnp.zeros_like(x)
    if args.variant == "d1_squared":
        plan = plan._replace(dxx=plan.dx @ plan.dx, dyy=plan.dy @ plan.dy)
    force = -ns_raw_rhs(plan, base)
    center = (abs(x) < 2) & (abs(y) < 0.5)
    edge = (abs(x) > 2.4) | (abs(y) > 0.8)

    @jax.jit
    def diagnostic(u):
        w = u - base
        en = jnp.sum(w * w, axis=-1)
        maximum = jnp.argmax(en)
        return jnp.array(
            [
                jnp.max(jnp.sqrt(jnp.sum(u * u, axis=-1))),
                jnp.sqrt(jnp.sum(p.weights * en)),
                jnp.sqrt(
                    jnp.sum(p.weights * center * en) / jnp.sum(p.weights * center)
                ),
                jnp.sqrt(jnp.sum(p.weights * edge * en) / jnp.sum(p.weights * edge)),
                jnp.max(abs(b.ns_divergence(plan, u))),
                x.ravel()[maximum],
                y.ravel()[maximum],
                jnp.max(abs((1 - p.mask)[..., None] * w)),
            ]
        )

    steps = int(round(0.05 / args.dt))

    @jax.jit
    def advance(u):
        def body(c, _):
            u, ok, r = c
            u, d = ns_rk4_step(plan, u, args.dt, force, reference=base, sponge=sigma)
            return (u, ok & d.converged, jnp.maximum(r, d.schur_linf)), None

        return jax.lax.scan(
            body, (u, jnp.array(True), jnp.array(0.0)), None, length=steps
        )[0]

    records = []
    u = initial
    start = time.perf_counter()
    frames = []
    for i in range(round(args.T / 0.05) + 1):
        if i:
            u, valid, res = advance(u)
        else:
            valid, res = True, 0.0
        vals = np.array(diagnostic(u))
        row = dict(
            t=i * 0.05,
            valid=bool(valid),
            schur_linf=float(res),
            **dict(
                zip(
                    [
                        "max_speed",
                        "perturbation_l2",
                        "center_rms",
                        "edge_rms",
                        "div_linf",
                        "peak_x",
                        "peak_y",
                        "wall_linf",
                    ],
                    map(float, vals),
                )
            ),
        )
        records.append(row)
        if i % 10 == 0:
            print(args.variant, args.dt, row, flush=True)
            frames.append(np.asarray(u - base))
        if not np.all(np.isfinite(vals)) or vals[0] > 10 or vals[4] > 1e-6:
            break
    np.savez_compressed(
        args.out / f"{args.variant}_{args.nx}x{args.ny}_dt{args.dt}_state.npz",
        x=np.asarray(p.x.x),
        y=np.asarray(p.y.x),
        initial=np.asarray(initial),
        velocity=np.asarray(u),
        base=np.asarray(base),
        frames=np.array(frames),
    )
    return dict(records=records, elapsed_seconds=time.perf_counter() - start)


def spectrum(plan, base, args):
    from scipy.sparse.linalg import LinearOperator, eigs, ArpackNoConvergence

    x, y = jnp.meshgrid(plan.pressure.x.x, plan.pressure.y.x, indexing="ij")
    sigma = jnp.zeros_like(x)
    if args.variant in ("sponge", "sponge_x"):
        sigma = sigma + 80 * (x / 3) ** 16
    if args.variant in ("sponge", "sponge_y"):
        sigma = sigma + 80 * y**16
    if args.variant == "d1_squared":
        plan = plan._replace(dxx=plan.dx @ plan.dx, dyy=plan.dy @ plan.dy)
    if args.variant == "uniform":
        base = jnp.zeros_like(base).at[..., 0].set(1.0)
    shape = (base.shape[0] - 2, base.shape[1] - 2, 2)
    force = -ns_raw_rhs(plan, base)

    @jax.jit
    def apply(v):
        w = jnp.zeros_like(base).at[1:-1, 1:-1].set(v.reshape(shape))
        unprojected = w
        if plan.energy is not None:
            w, _ = b.ns_project_velocity(plan, w)
        z = jax.jvp(
            lambda u: ns_rhs(plan, u, force, reference=base, sponge=sigma)[0],
            (base,),
            (w,),
        )[1]
        if plan.energy is not None:
            # Restrict physical eigenmodes to div-free increments; move the
            # irrelevant constraint complement away from the leading spectrum.
            z -= 50 * (unprojected - w)
        return z[1:-1, 1:-1].ravel()

    def mv(v):
        return np.asarray(apply(jnp.asarray(v)))

    size = np.prod(shape)
    op = LinearOperator((size, size), matvec=mv, dtype=np.float64)
    start = time.perf_counter()
    try:
        values, vectors = eigs(
            op,
            k=6,
            which="LR",
            tol=1e-7,
            maxiter=1500,
            ncv=48,
            v0=np.random.default_rng(9).normal(size=size),
        )
        converged = True
    except ArpackNoConvergence as e:
        values, vectors = e.eigenvalues, e.eigenvectors
        converged = False
    out = []
    edge = (abs(np.asarray(x)[1:-1, 1:-1]) > 2.4) | (
        abs(np.asarray(y)[1:-1, 1:-1]) > 0.8
    )
    for val, v in zip(values, vectors.T):
        residual = mv(v.real) + 1j * mv(v.imag) - val * v
        vv = v.reshape(shape)
        en = np.sum(abs(vv) ** 2, axis=-1)
        out.append(
            dict(
                real=float(val.real),
                imag=float(val.imag),
                residual_relative=float(
                    np.linalg.norm(residual) / max(abs(val) * np.linalg.norm(v), 1e-30)
                ),
                edge_energy_fraction=float(en[edge].sum() / en.sum()),
            )
        )
    np.savez_compressed(
        args.out / f"{args.variant}_{args.nx}x{args.ny}_eigenmodes.npz",
        values=values,
        vectors=vectors,
        x=np.asarray(plan.pressure.x.x),
        y=np.asarray(plan.pressure.y.x),
    )
    return dict(
        converged=converged,
        modes=out,
        elapsed_seconds=time.perf_counter() - start,
        constraint_complement_shift=-50 if plan.energy is not None else None,
    )


def main():
    jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--closure", choices=["bspf", "sbp84"], default="bspf")
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--T", type=float, default=3)
    parser.add_argument(
        "--mode", choices=["operators", "evolve", "spectrum"], default="operators"
    )
    parser.add_argument(
        "--variant",
        choices=["none", "sponge", "sponge_x", "sponge_y", "d1_squared", "uniform"],
        default="none",
    )
    parser.add_argument("--out", type=Path, default=Path("build/kh_stability"))
    args = parser.parse_args()
    if args.variant == "uniform" and args.mode != "spectrum":
        parser.error("uniform control is implemented for spectrum mode only")
    if args.closure != "bspf":
        args.out = args.out / args.closure
    args.out.mkdir(parents=True, exist_ok=True)
    plan, initial, base = make(args.nx, args.ny, args.closure)
    result = dict(
        nx=args.nx, ny=args.ny, variant=args.variant, dt=args.dt, closure=args.closure
    )
    if args.mode == "operators":
        result["operators"] = operators(plan)
    elif args.mode == "spectrum":
        result.update(spectrum(plan, base, args))
    else:
        result.update(evolution(plan, initial, base, args))
    name = f"{args.mode}_{args.variant}_{args.nx}x{args.ny}_dt{args.dt}.json"
    (args.out / name).write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            result if args.mode != "evolve" else result["records"][-1], indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
