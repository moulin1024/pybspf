"""Run the linear BSPF plasma / BSPF vacuum free-boundary model."""

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
from bspf_jax.tokamak_vacuum import assemble_plasma_vacuum


def evolve(model, out, dt=0.1):
    values, vectors = model.modes()
    if values[0] >= 0:
        raise RuntimeError("No growing mode in the resolved displacement space")
    gamma = np.sqrt(-values[0])
    mode = vectors[:, 0]
    displacement = model.displacement(mode, model.quadrature_points)
    mean = np.average(displacement[:, 1], weights=model.quadrature_weights)
    if abs(mean) < 1e-8:
        raise RuntimeError("Fastest mode is not a vertical displacement mode")
    minor = np.ptp(model.boundary[:, 0]) / 2
    q = mode * (1e-4 * minor / mean)
    v = gamma * q
    steps = int(np.ceil(np.log(200) / gamma / dt))
    dt = np.log(200) / gamma / steps
    stiffness = model.stiffness
    factor = la.cho_factor(np.eye(len(q)) + dt**2 / 4 * stiffness)
    # Implicit midpoint for BOTH displacement and velocity; not prescribed exp(t).
    times, qs, vs, energies = [], [], [], []
    zero_q, zero_v = np.zeros_like(q), np.zeros_like(v)
    stride = max(1, steps // 100)
    for k in range(steps + 1):
        if k % stride == 0 or k == steps:
            times.append(k * dt)
            qs.append(q.copy())
            vs.append(v.copy())
            energies.append((v @ v + q @ stiffness @ q) / 2)
        if k < steps:
            qnew = la.cho_solve(factor, q + dt * v - dt**2 / 4 * (stiffness @ q))
            v -= dt / 2 * (stiffness @ (q + qnew))
            q = qnew
            zn = la.cho_solve(
                factor, zero_q + dt * zero_v - dt**2 / 4 * (stiffness @ zero_q)
            )
            zero_v -= dt / 2 * (stiffness @ (zero_q + zn))
            zero_q = zn
    times, qs, vs = map(np.asarray, (times, qs, vs))
    centroids = (qs @ mode) * mean
    measured = np.polyfit(times, np.log(abs(centroids)), 1)[0]
    boundary_displacements = np.array(
        [model.displacement(a, model.boundary) for a in qs]
    )
    boundary = model.boundary[None, :, :] + boundary_displacements[:, :, :2]
    response = model.vacuum.extension @ model.trace
    final_flux = response @ q
    if hasattr(model.vacuum, "harmonic_residual"):
        harmonic = model.vacuum.harmonic_residual(model.trace @ q)
    else:
        harmonic = (model.vacuum.stiffness @ final_flux)[model.vacuum.free]
    final_xi = model.displacement(q, model.quadrature_points)
    final_mean = np.average(final_xi[:, 1], weights=model.quadrature_weights)
    rigid = np.zeros_like(final_xi)
    rigid[:, 1] = final_mean
    departure = np.sqrt(
        np.sum(model.quadrature_weights * np.sum((final_xi - rigid) ** 2, axis=1))
        / np.sum(model.quadrature_weights * np.sum(final_xi**2, axis=1))
    )
    eigen_res = la.norm(stiffness @ mode - values[0] * mode) / max(
        la.norm(stiffness @ mode), abs(values[0])
    )
    summary = dict(
        model="Linear ideal incompressible axisymmetric plasma-vacuum; vertical parity",
        exterior="Current-free magnetostatic vacuum; no fluid degrees of freedom",
        discretization=(
            "BSPF equilibrium, plasma displacement and mapped vacuum"
            if model.diagnostics.get("vacuum_method") == "mapped_bspf"
            else "Restricted BSPF plasma displacement + fitted P1 FEM vacuum"
        ),
        gamma=float(gamma),
        gamma_time_fit=float(measured),
        dt=float(dt),
        time_fit_relative_error=float(abs(measured / gamma - 1)),
        eigen_relative_residual=float(eigen_res),
        vacuum_interior_residual_linf=float(abs(harmonic).max()),
        vacuum_outer_flux_linf=float(abs(final_flux[model.vacuum.outer]).max()),
        interface_trace_error_linf=float(
            abs(final_flux[model.vacuum.inner] - model.trace @ q).max()
        ),
        interface_trace_relative_error=float(
            la.norm(final_flux[model.vacuum.inner] - model.trace @ q)
            / max(la.norm(model.trace @ q), 1e-30)
        ),
        final_mean_vertical_displacement=float(final_mean),
        max_final_displacement_over_minor_radius=float(
            np.linalg.norm(final_xi, axis=1).max() / minor
        ),
        departure_from_rigid_translation=float(departure),
        vacuum_energy=float(q @ model.vacuum_stiffness @ q / 2),
        plasma_potential_energy=float(q @ model.plasma_stiffness @ q / 2),
        zero_control_linf=float(max(abs(zero_q).max(), abs(zero_v).max())),
        exterior_velocity_dofs=0,
        minor_radius=float(minor),
        **model.diagnostics,
    )
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "evolution.npz",
        time=times,
        displacement_coefficients=qs,
        velocity_coefficients=vs,
        centroid_z=centroids,
        energy=energies,
        boundary=boundary,
        boundary0=model.boundary,
        vacuum_points=model.vacuum.points,
        vacuum_triangles=model.vacuum.triangles,
        vacuum_flux=qs @ response.T,
        eigenvalues=values,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=49)
    ap.add_argument("--modes", type=int, default=14)
    ap.add_argument("--angles", type=int, default=256)
    ap.add_argument("--radial-quadrature", type=int, default=64)
    ap.add_argument(
        "--vacuum-layers",
        type=int,
        default=48,
        help="BSPF display layers only; FEM backend radial mesh layers",
    )
    ap.add_argument("--vacuum-method", choices=("bspf", "fem"), default="bspf")
    ap.add_argument("--vacuum-radial-modes", type=int, default=None)
    ap.add_argument("--vacuum-angular-modes", type=int, default=None)
    ap.add_argument("--mass-cutoff", type=float, default=1e-10)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=Path("build/tokamak_vacuum"))
    args = ap.parse_args()
    if not np.isfinite(args.dt) or args.dt <= 0:
        ap.error("dt must be finite and positive")
    jax.config.update("jax_enable_x64", True)
    plan = plan_axisymmetric_bspf(args.n)
    coils, offset, _ = fit_fixed_coils(quadrupole=-0.004, vertical=0.03, offset=-0.2)
    eq = solve_equilibrium(plan, coils, offset=offset, max_iterations=700)
    model = assemble_plasma_vacuum(
        plan,
        eq,
        coils,
        offset,
        modes=args.modes,
        angles=args.angles,
        radial_quadrature=args.radial_quadrature,
        vacuum_layers=args.vacuum_layers,
        vacuum_method=args.vacuum_method,
        vacuum_radial_modes=args.vacuum_radial_modes,
        vacuum_angular_modes=args.vacuum_angular_modes,
        mass_cutoff=args.mass_cutoff,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "model.pkl").write_bytes(pickle.dumps(model))
    evolve(model, args.out, args.dt)


if __name__ == "__main__":
    main()
