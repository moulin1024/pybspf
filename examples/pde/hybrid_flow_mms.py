"""Continuous unsteady NS MMS for BSPF + rational boundary correction.

Forcing comes from analytic bulk derivatives and an independently higher-order
homogeneous rational field, never from the tested Galerkin matrices. The time
forcing includes the rational velocity explicitly.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la

from bspf_jax.immersed_flow import ImmersedFlowPlan, channel_lift
from bspf_jax.rational_stokes import RationalStokesExtension


class ContinuousHybridMMS:
    def __init__(self, bounds, hole, *, peak=1, reference_degree=120, waves=True):
        self.bounds, self.hole, self.peak = bounds, hole, peak
        left, right, h = bounds

        def psi(z):
            x, y = z
            t = (x - left) / (right - left)
            wave = jnp.cos(jnp.pi * x / 3 + 0.4) * jnp.sin(
                jnp.pi * y / 2 + 0.2
            ) + 0.3 * jnp.sin(2 * jnp.pi * x / 3 - jnp.pi * y + 0.1)
            return (
                (25 * wave if waves else 5.0)
                * t
                * t
                * (1 - t) ** 3
                * (1 - (y / h) ** 2) ** 2
            )

        d = jax.grad(psi)
        dd = jax.jacfwd(d)
        ddd = jax.jacfwd(dd)

        @jax.jit
        def bulk(points):
            a = jax.vmap(psi)(points)
            b = jax.vmap(d)(points)
            c = jax.vmap(dd)(points)
            e = jax.vmap(ddd)(points)
            return jnp.stack(
                (
                    a,
                    b[:, 1],
                    -b[:, 0],
                    c[:, 0, 1],
                    c[:, 1, 1],
                    -c[:, 0, 0],
                    e[:, 1, 0, 0] + e[:, 1, 1, 1],
                    -e[:, 0, 0, 0] - e[:, 0, 1, 1],
                )
            )

        self._bulk = bulk
        self.reference = RationalStokesExtension(
            bounds,
            hole,
            degree=reference_degree,
            corner_poles=reference_degree // 3,
            laurent=80,
            samples=max(800, reference_degree * 8),
        )
        b = self.reference.hole_points
        self.base_coeff = self.reference.response(
            -np.concatenate(channel_lift(b, h, peak)[1:3])
        )
        self.delta_coeff = self.reference.response(-np.concatenate(self.bulk(b)[1:3]))

    def bulk(self, points):
        return np.asarray(self._bulk(np.asarray(points)))

    def parts(self, points):
        base = channel_lift(points, self.bounds[2], self.peak)
        bulk = self.bulk(points)
        rb = self.reference.evaluate(points, self.base_coeff)
        rd = self.reference.evaluate(points, self.delta_coeff)
        return (
            tuple(a + b for a, b in zip(base, rb)),
            tuple(a + b for a, b in zip(bulk[:6], rd)),
            bulk,
            rd,
        )

    @staticmethod
    def amplitude(t):
        return np.sin(1.3 * t), 1.3 * np.cos(1.3 * t)

    def prepare(self, p):
        base, delta, bulk, rdelta = self.parts(p.points)

        def convection(a, b):
            return np.column_stack(
                (a[1] * b[3] + a[2] * b[4], a[1] * b[5] - a[2] * b[3])
            )

        def uv(f):
            return np.column_stack(f[1:3])

        f0 = convection(base, base) + p.sigma[:, None] * (
            uv(base) - uv(channel_lift(p.points, p.bounds[2], p.peak))
        )
        fs = (
            convection(base, delta)
            + convection(delta, base)
            - p.nu * bulk[6:8].T
            + p.sigma[:, None] * uv(delta)
        )
        fss = convection(delta, delta)
        ft = uv(delta)
        loads = [p.force_load(f) for f in (f0, fs, fss, ft)]
        missing_time = p.force_load(uv(rdelta))

        def load(t, omit_rational_time=False):
            s, rate = self.amplitude(t)
            value = loads[0] + s * loads[1] + s * s * loads[2] + rate * loads[3]
            return value - rate * missing_time if omit_rational_time else value

        return base, delta, bulk, load, missing_time


def run(args):
    jax.config.update("jax_enable_x64", True)
    args.out.mkdir(parents=True, exist_ok=True)
    p = ImmersedFlowPlan(
        nx=args.nx,
        ny=args.ny,
        wall_method="rational",
        buffer_strength=args.sponge,
        quadrature_factor=args.quadrature_factor,
        rational_options=dict(
            degree=args.degree,
            corner_poles=args.degree // 3,
            laurent=64,
            samples=max(600, 8 * args.degree),
        ),
    )
    print(
        "PLAN", p.setup_seconds, p.dofs, p.energy_condition, p.rational.info, flush=True
    )
    exact = ContinuousHybridMMS(p.bounds, p.hole)
    base, delta, bulk, load, missing = exact.prepare(p)
    coefficient = la.cho_solve(p.mass_factor, p.force_load(np.column_stack(delta[1:3])))
    reconstructed = [o @ coefficient for o in p.operators_fluid]
    representation = np.sqrt(
        p.weights
        @ ((reconstructed[0] - delta[1]) ** 2 + (reconstructed[1] - delta[2]) ** 2)
    )
    at = 0.31
    s, rate = exact.amplitude(at)
    state = s * coefficient
    residual = (
        p.explicit(state) + load(at) - p.linear @ state - rate * (p.mass @ coefficient)
    )

    def dual(r):
        return float(np.sqrt(max(r @ la.cho_solve(p.mass_factor, r), 0)))

    result = dict(
        nx=p.nx,
        ny=p.ny,
        dofs=p.dofs,
        setup_seconds=p.setup_seconds,
        energy_condition=p.energy_condition,
        volume_modes_discarded=p.discarded_volume_modes,
        rational=p.rational.info,
        perturbation_velocity_representation_l2=float(representation),
        continuous_ns_residual_mass_dual=dual(residual),
        residual_if_rational_time_omitted=dual(residual - rate * missing),
    )
    print("CONTINUOUS", json.dumps(result), flush=True)
    # Steady forced Stokes: exact state is base+delta. This exercises the volume
    # space with a nonzero body force; the homogeneous case alone does not.
    reference_uv = np.column_stack((base[1] + delta[1], base[2] + delta[2]))
    stokes_force = -p.nu * bulk[6:8].T + p.sigma[:, None] * (
        reference_uv - np.column_stack(channel_lift(p.points, p.bounds[2], p.peak)[1:3])
    )
    steady = la.solve(
        p.linear, p.force_load(stokes_force) - p.linear_lift, assume_a="pos"
    )
    err = (
        np.column_stack(
            [
                o @ steady + lift
                for o, lift in zip(p.operators_fluid[:2], p.lift_fields[1:3])
            ]
        )
        - reference_uv
    )
    result["forced_stokes_velocity_l2"] = float(
        np.sqrt(p.weights @ np.sum(err * err, axis=1))
    )
    result["time_convergence"] = []
    for dt in (0.08, 0.04, 0.02):
        a = np.zeros(p.dofs)
        step = p.stepper(dt)
        start = perf_counter()
        for k in range(round(0.4 / dt)):
            a = step.step(a, k * dt, load)
        target = [b + exact.amplitude(0.4)[0] * d for b, d in zip(base, delta)]
        err = np.column_stack(
            [
                o @ a + lift - t
                for o, lift, t in zip(
                    p.operators_fluid[:2], p.lift_fields[1:3], target[1:3]
                )
            ]
        )
        record = dict(
            dt=dt,
            velocity_error_l2=float(np.sqrt(p.weights @ np.sum(err * err, axis=1))),
            seconds=perf_counter() - start,
        )
        result["time_convergence"].append(record)
        print("TIME", record, flush=True)
    x, y = np.linspace(-1, 5, 401), np.linspace(-1, 1, 161)
    xx, yy = np.meshgrid(x, y)
    pts = np.column_stack((xx.ravel(), yy.ravel()))
    physical = p.hole.level(pts) > 1
    physical[[0, 400, 160 * 401, 161 * 401 - 1]] = False
    refbase, refdelta, *_ = exact.parts(pts[physical])
    fields = p.grid(steady, x, y)
    ref = np.full((6, len(pts)), np.nan)
    for i, (b, d) in enumerate(zip(refbase, refdelta)):
        ref[i, physical] = b + d
    ref = ref.reshape(6, *xx.shape)
    erru = fields["u"] - ref[1]
    errv = fields["v"] - ref[2]
    errw = fields["vorticity"] - (ref[5] - ref[4])
    result["forced_stokes_velocity_relative_l2"] = float(
        np.sqrt(np.nansum(erru**2 + errv**2) / np.nansum(ref[1] ** 2 + ref[2] ** 2))
    )
    result["forced_stokes_vorticity_relative_l2"] = float(
        np.sqrt(np.nansum(errw**2) / np.nansum((ref[5] - ref[4]) ** 2))
    )
    boundary, _ = p.arc.sample(512, offset=0.371)
    f = p.evaluate(a, boundary)
    result["final_ns_wall_max"] = float(np.max(np.hypot(f[1], f[2])))
    np.savez(args.out / "fields.npz", x=x, y=y, reference=ref, **fields)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=73)
    parser.add_argument("--ny", type=int, default=33)
    parser.add_argument("--degree", type=int, default=96)
    parser.add_argument("--sponge", type=float, default=0)
    parser.add_argument("--quadrature-factor", type=float, default=2.5)
    parser.add_argument(
        "--out", type=Path, default=Path("build/immersed_flow/hybrid/mms")
    )
    run(parser.parse_args())
