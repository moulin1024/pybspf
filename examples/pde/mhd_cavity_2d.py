"""Release a central pair of magnetic islands in a BSPF incompressible cavity.

python examples/pde/mhd_cavity_2d.py \
  --background build/cavity_n49/state.npz
Run without --background for independently reproducible startup from rest.
"""

import argparse
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np

from bspf_models.plasma.mhd_cavity import plan_mhd_cavity
from bspf_models.plasma.mhd_cavity import island_pair
from bspf_models.plasma.mhd_cavity import mhd_step
from bspf_models.plasma.mhd_cavity import magnetic_fields
from bspf_models.plasma.mhd_cavity import mhd_velocity
from bspf_models.plasma.mhd_cavity import mhd_budget
from bspf_models.fluids.cavity import plan_cavity_stepper
from bspf_models.fluids.cavity import cavity_step
from bspf_models.fluids.cavity import cavity_ramp
from bspf_models.fluids.stream_navier_stokes import stream_ns_divergence


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=49)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--rm", type=float, default=200.0)
    ap.add_argument("--field", type=float, default=6.0)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--dt", type=float, default=0.0005)
    ap.add_argument("--sample", type=float, default=0.005)
    ap.add_argument("--background", type=Path)
    ap.add_argument("--out", type=Path, default=Path("build/mhd_cavity"))
    args = ap.parse_args()
    for name in ("T", "dt", "sample"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            ap.error(f"{name} must be finite and positive")
    jax.config.update("jax_enable_x64", True)
    started = time.perf_counter()
    background = None
    offset = 0.0
    if args.background:
        background = dict(np.load(args.background))
        if len(background["x"]) != args.n or not np.allclose(
            background["x"], np.linspace(0, 1, args.n)
        ):
            ap.error("background grid must match --n on [0,1]")
        if not np.isclose(float(background["reynolds"]), args.re) or not np.isclose(
            float(background["ramp_time"]), 1
        ):
            ap.error("background must have matching Re and ramp_time=1")
        offset = float(background["time"])
        if offset < 1:
            ap.error("background startup must have finished")
    args.out.mkdir(parents=True, exist_ok=True)
    p = plan_mhd_cavity(
        n=args.n, reynolds=args.re, magnetic_reynolds=args.rm, time_offset=offset
    )
    f, m = p.fluid.spatial, p.magnetic
    a = (
        jnp.zeros_like(f.denominator)
        if background is None
        else jnp.asarray(background["a"])
    )
    if a.shape != f.denominator.shape or not np.all(np.isfinite(a)):
        ap.error("invalid background modal array")
    b = island_pair(p, peak_field=args.field)
    control = a.copy()
    count = int(np.ceil(args.T / args.dt))
    dt = args.T / count
    stepper = plan_cavity_stepper(p.fluid, dt)
    print(
        f"MHD n={args.n} Re={args.re:g} Rm={args.rm:g} max B={args.field:g} dt={dt:g}; setup={time.perf_counter() - started:.1f}s",
        flush=True,
    )

    @jax.jit
    def advance(a, b, control, start, steps):
        def step(k, state):
            a, b, control = state
            t = (start + k) * dt
            anew, bnew = mhd_step(p, stepper, a, b, t)
            cnew = cavity_step(p.fluid, stepper, control, t + offset)
            return anew, bnew, cnew

        return jax.lax.fori_loop(0, steps, step, (a, b, control))

    @jax.jit
    def diagnostics(a, b, control, t):
        flux, magnetic, current = magnetic_fields(p, b, nodes=True)
        velocity, baseline = (
            mhd_velocity(p, a, t, nodes=True),
            mhd_velocity(p, control, t, nodes=True),
        )
        budget = mhd_budget(p, a, b, t)
        divb = (m.gn @ b) @ m.gn.T - m.gn @ (b @ m.gn.T)
        s, _ = cavity_ramp(t + offset, 1.0)
        lid = s * 16 * f.x.x**2 * (1 - f.x.x) ** 2
        wall = jnp.maximum(
            jnp.max(abs(velocity[jnp.array([0, -1])])),
            jnp.maximum(
                jnp.max(abs(velocity[:, 0])),
                jnp.maximum(
                    jnp.max(abs(velocity[:, -1, 0] - lid)),
                    jnp.max(abs(velocity[:, -1, 1])),
                ),
            ),
        )
        normal_b = jnp.maximum(
            jnp.max(abs(magnetic[[0, -1], :, 0])), jnp.max(abs(magnetic[:, [0, -1], 1]))
        )
        center = (abs(f.x.x[:, None] - 0.5) <= 0.25) & (
            abs(f.y.x[None, :] - 0.5) <= 0.25
        )
        speed, delta = (
            jnp.linalg.norm(velocity, axis=-1),
            jnp.linalg.norm(velocity - baseline, axis=-1),
        )
        strong_j = -m.h @ b @ m.b.T - m.b @ b @ m.h.T
        jq = magnetic_fields(p, b)[2]
        weights = m.weights[:, None] * m.weights[None, :]
        current_error = jnp.sqrt(
            jnp.sum(weights * (strong_j - jq) ** 2)
            / jnp.maximum(jnp.sum(weights * jq**2), 1e-30)
        )
        values = jnp.array(
            [
                jnp.max(abs(stream_ns_divergence(f, a, nodes=True))),
                jnp.max(abs(divb)),
                wall,
                normal_b,
                jnp.max(speed),
                jnp.max(jnp.where(center, speed, 0.0)),
                jnp.max(jnp.where(center, jnp.linalg.norm(baseline, axis=-1), 0.0)),
                jnp.max(jnp.where(center, delta, 0.0)),
                current_error,
            ]
        )
        return flux, velocity, magnetic, current, baseline, budget, values

    keys = [
        "kinetic_energy",
        "magnetic_energy",
        "lid_power",
        "viscous_dissipation",
        "ohmic_dissipation",
        "magnetic_to_kinetic_power",
        "spatial_energy_residual",
        "div_u_linf",
        "div_b_linf",
        "velocity_wall_error",
        "normal_b_wall_error",
        "max_speed",
        "center_max_speed",
        "control_center_max_speed",
        "center_max_velocity_change",
        "weak_vs_strong_current_relative_l2",
    ]
    history, fluxes, velocities, magnetics, currents, controls = [], [], [], [], [], []
    stride, done = max(1, round(args.sample / dt)), 0
    while True:
        t = done * dt
        flux, velocity, magnetic, current, baseline, budget, values = jax.device_get(
            diagnostics(a, b, control, t)
        )
        all_values = np.r_[budget, values]
        if not np.isfinite(all_values).all():
            raise RuntimeError(
                "Nonfinite MHD state: reduce dt and refine the current layer"
            )
        row = dict(zip(keys, map(float, all_values)))
        row["time"] = t
        history.append(row)
        for target, value in zip(
            (fluxes, velocities, magnetics, currents, controls),
            (flux, velocity, magnetic, current, baseline),
        ):
            target.append(value)
        if len(history) % 10 == 1 or done == count:
            print(
                f"t={t:.4f} K={budget[0]:.5f} M={budget[1]:.5f} center U={values[5]:.3f} delta={values[7]:.3f} energy residual={budget[-1]:.2e}",
                flush=True,
            )
        if done == count:
            break
        steps = min(stride, count - done)
        a, b, control = advance(a, b, control, done, steps)
        done += steps
    ts = np.array([r["time"] for r in history])
    total = np.array([r["kinetic_energy"] + r["magnetic_energy"] for r in history])
    power = np.array(
        [
            r["lid_power"] - r["viscous_dissipation"] - r["ohmic_dissipation"]
            for r in history
        ]
    )
    integral = np.r_[0, np.cumsum(np.diff(ts) * (power[1:] + power[:-1]) / 2)]
    balance = total - total[0] - integral
    np.savez_compressed(
        args.out / "evolution.npz",
        time=ts,
        x=np.asarray(f.x.x),
        flux=fluxes,
        velocity=velocities,
        magnetic=magnetics,
        current=currents,
        control_velocity=controls,
        final_a=np.asarray(a),
        final_b=np.asarray(b),
        energy_balance=balance,
    )
    peak_index = int(np.argmax([r["center_max_velocity_change"] for r in history]))
    summary = dict(
        n=args.n,
        reynolds=args.re,
        magnetic_reynolds=args.rm,
        peak_initial_field=args.field,
        dt=dt,
        final_time=float(ts[-1]),
        sample_interval=stride * dt,
        background=str(args.background),
        fluid_time_offset=offset,
        elapsed_seconds=time.perf_counter() - started,
        initial_condition="Two non-equilibrium same-sign Gaussian flux islands at (.37,.5),(.63,.5), widths (.10,.14), polynomial wall taper",
        magnetic_boundary="A=0 (Bn=0, Et=0) on all four conducting walls; tangential B free",
        control="Identical initial velocity and lid, advanced without magnetic field",
        peak=history[peak_index],
        max_sampled_energy_balance_relative=float(np.max(abs(balance)) / total[0]),
        energy_balance_note="Physical lid shear work and volume dissipation; trapezoidal integration at saved times includes sampling error",
        history=history,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Saved {args.out}; relative sampled energy balance={summary['max_sampled_energy_balance_relative']:.3e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
