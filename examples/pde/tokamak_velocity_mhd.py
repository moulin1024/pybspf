"""Advance velocity and magnetic induction with shared BSPF stream-NS kernels."""

import argparse
import json
import pickle
from pathlib import Path
import jax
import numpy as np
import scipy.linalg as la
from bspf_jax.tokamak_equilibrium import (
    plan_axisymmetric_bspf,
    fit_fixed_coils,
    solve_equilibrium,
)
from bspf_jax.tokamak_vacuum import EquilibriumEvaluator, assemble_plasma_vacuum
from bspf_jax.tokamak_velocity import plan_tokamak_velocity
from tokamak_confined import safety_profile, vertical_projection


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("build/tokamak_velocity"))
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--duration", type=float, default=100.0)
    ap.add_argument("--far-wall", action="store_true")
    args = ap.parse_args()
    if (
        not np.isfinite(args.dt)
        or args.dt <= 0
        or not np.isfinite(args.duration)
        or args.duration <= 0
    ):
        ap.error("dt and duration must be finite and positive")
    jax.config.update("jax_enable_x64", True)
    p = plan_axisymmetric_bspf(49)
    coils, offset, _ = fit_fixed_coils(quadrupole=-0.004, vertical=0.03, offset=-0.2)
    eq = solve_equilibrium(p, coils, offset=offset, max_iterations=700)
    f, profile, _ = safety_profile(EquilibriumEvaluator(p, eq, coils, offset))
    spatial = assemble_plasma_vacuum(
        p,
        eq,
        coils,
        offset,
        modes=14,
        angles=256,
        radial_quadrature=64,
        vacuum_layers=48,
        toroidal_f=f,
        wall_scale=None if args.far_wall else 1.2,
        vacuum_solver="tensor_pcg",
    )
    plan = plan_tokamak_velocity(spatial)
    h = vertical_projection(spatial)
    minor = np.ptp(spatial.boundary[:, 0]) / 2
    initial_q = h * (0.002 * minor / (h @ h))
    state = plan.initial_state(initial_q)
    initial = state.copy()
    zero = np.zeros_like(state)
    e0 = plan.energy(state)
    times = []
    states = []
    energies = []
    steps = int(np.ceil(args.duration / args.dt))
    dt = args.duration / steps
    stride = max(1, steps // 200)
    for k in range(steps + 1):
        if k % stride == 0 or k == steps:
            times.append(k * dt)
            states.append(state.copy())
            energies.append(plan.energy(state))
        if args.far_wall and abs(h @ plan.split(state)[0]) / minor >= 0.02:
            break
        if k < steps:
            state = plan.step(state, dt)
    times = np.asarray(times)
    states = np.asarray(states)
    q, v, b = np.split(states, 3, axis=1)
    # Independent reference uses only the old second-order displacement operator.
    eigen, rotation = spatial.modes()
    modal = initial_q @ rotation
    factor = np.cos(times[:, None] * np.sqrt(np.maximum(eigen, 0))[None, :])
    if (eigen < 0).any():
        factor[:, eigen < 0] = np.cosh(
            times[:, None] * np.sqrt(-eigen[eigen < 0])[None, :]
        )
    reference = (factor * modal[None, :]) @ rotation.T
    reference_error = la.norm(q - reference) / la.norm(reference)
    growth = {}
    if eigen[0] < 0:
        gamma = float(np.sqrt(-eigen[0]))
        mode = rotation[:, 0]
        # Isolate the growing component from q and v, without prescribing its evolution.
        amplitude = (q @ mode + (v @ mode) / gamma) / 2
        fitted = float(np.polyfit(times, np.log(abs(amplitude)), 1)[0])
        growth = dict(
            gamma_reference=gamma,
            gamma_time_fit=fitted,
            gamma_relative_difference=abs(fitted / gamma - 1),
        )
    induction_error = la.norm(b - q @ plan.induction.T) / max(la.norm(b), 1e-30)
    zero = plan.step(zero, dt)
    vacuum_flux = q @ (spatial.vacuum.extension @ spatial.trace).T
    target = q @ spatial.trace.T
    trace_error = la.norm(vacuum_flux[:, spatial.vacuum.inner] - target) / max(
        la.norm(target), 1e-30
    )
    physical_b = plan.magnetic_fields(b[-1])
    raw = spatial.basis.evaluate(spatial.quadrature_points)
    vr, vz = raw["xr"] @ spatial.transform, raw["xz"] @ spatial.transform
    rr = spatial.quadrature_points[:, 0, None]
    divergence = (raw["xrr"] + raw["xr"] / rr + raw["xzz"]) @ spatial.transform @ v[-1]
    summary = dict(
        model="Linear axisymmetric velocity-induction MHD; shared stream-NS RK4 stages",
        state_fields=["displacement", "velocity", "magnetic perturbation"],
        nonlinear_advection=False,
        pressure_treatment="Divergence-free stream basis; no Cartesian pressure projection",
        elliptic_treatment="Tensor Poisson preconditioned CG; fixed-geometry response cached",
        q_axis=profile["q_axis"],
        q95=profile["q"][-1],
        toroidal_f=f,
        dt=dt,
        duration=float(times[-1]),
        maximum_alfven_frequency=plan.maximum_frequency,
        operator_reference_relative_error=plan.stiffness_reference_error,
        trajectory_reference_relative_error=float(reference_error),
        frozen_flux_relative_error=float(induction_error),
        interface_trajectory_relative_error=float(trace_error),
        energy_relative_drift=float(
            np.max(abs(np.asarray(energies) - e0)) / max(abs(e0), 1e-30)
        ),
        divergence_linf=float(abs(divergence).max()),
        zero_control_linf=float(abs(zero).max()),
        max_centroid_fraction=float(abs(q @ h).max() / minor),
        **growth,
        **spatial.diagnostics,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "evolution.npz",
        time=times,
        displacement_coefficients=q,
        velocity_coefficients=v,
        magnetic_coefficients=b,
        centroid_z=q @ h,
        energy=energies,
        reference_displacement=reference,
        vacuum_flux=vacuum_flux,
        initial_state=initial,
        plasma_points=spatial.quadrature_points,
        final_magnetic_field=physical_b,
        final_poloidal_velocity=np.column_stack((vr @ v[-1], vz @ v[-1])),
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "model.pkl").write_bytes(pickle.dumps(plan))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
