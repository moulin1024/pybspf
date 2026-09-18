"""Engineering benchmark: passive ideal-wall stabilization of elongated plasma.

Not a reactor design: low-beta equilibrium, closed smooth flux surfaces, fixed
coils, a vacuum gap, and a shaped ideal conductor. No X-point or transport.
"""

import argparse
import json
import pickle
from pathlib import Path
import jax
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from bspf_jax.tokamak_equilibrium import (
    plan_axisymmetric_bspf,
    fit_fixed_coils,
    solve_equilibrium,
)
from bspf_jax.tokamak_vacuum import EquilibriumEvaluator, assemble_plasma_vacuum


def safety_profile(evaluator, target_axis_q=1.3):
    result = minimize(
        lambda x: -evaluator.evaluate(np.array([x]))[0][0],
        [2.2, 0.0],
        jac=lambda x: -np.array(evaluator.evaluate(np.array([x]))[1:]).ravel(),
        method="BFGS",
        tol=1e-10,
    )
    center = result.x
    psi, pr, pz, prr, prz, pzz = [
        a[0] for a in evaluator.evaluate(center[None, :], True)
    ]
    if np.hypot(pr, pz) > 1e-7 or prr >= 0 or pzz >= 0:
        raise RuntimeError("Magnetic axis location failed")
    f = target_axis_q * center[0] * np.sqrt(prr * pzz - prz**2)
    t = np.arange(256) * 2 * np.pi / 256
    levels = np.r_[0.01, np.linspace(0.05, 0.95, 19)]
    qs = []
    surfaces = []
    for level in levels:
        points, _, _ = evaluator.surface(
            t, [(1, 3), (-1.6, 1.6)], center=center, level=psi * (1 - level)
        )
        _, dr, dz = evaluator.evaluate(points)
        derivative = CubicSpline(
            np.r_[t, 2 * np.pi], np.vstack((points, points[0])), bc_type="periodic"
        )(t, 1)
        qs.append(
            f
            * np.mean(
                np.linalg.norm(derivative, axis=1) / (points[:, 0] * np.hypot(dr, dz))
            )
        )
        surfaces.append(points)
    return (
        f,
        dict(
            axis=center.tolist(),
            psi_axis=float(psi),
            q_axis=target_axis_q,
            normalized_poloidal_flux=levels.tolist(),
            q=qs,
        ),
        np.array(surfaces),
    )


def vertical_projection(model):
    f = model.basis.evaluate(model.quadrature_points)
    return (
        np.average(f["xz"], weights=model.quadrature_weights, axis=0) @ model.transform
    )


def trajectory(
    model, duration=100.0, dt=0.01, initial_fraction=0.002, stop_fraction=None
):
    h = vertical_projection(model)
    minor = np.ptp(model.boundary[:, 0]) / 2
    # Mass projection of uniform vertical displacement into all resolved fields.
    q = h * (initial_fraction * minor / (h @ h))
    v = np.zeros_like(q)
    initial = q.copy()
    k = model.stiffness
    if np.min(la.eigvalsh(k)) < -4 / dt**2:
        raise ValueError("Time step too large for midpoint solve")
    fac = la.cho_factor(np.eye(len(q)) + dt**2 / 4 * k)
    e0 = (v @ v + q @ k @ q) / 2
    times = []
    qs = []
    vs = []
    energy = []
    stride = max(1, int(round(0.5 / dt)))
    for step in range(int(duration / dt) + 1):
        if step % stride == 0:
            times.append(step * dt)
            qs.append(q.copy())
            vs.append(v.copy())
            energy.append((v @ v + q @ k @ q) / 2)
        if stop_fraction is not None and abs(h @ q) / minor >= stop_fraction:
            break
        qn = la.cho_solve(fac, q + dt * v - dt**2 / 4 * (k @ q))
        v -= dt / 2 * (k @ (q + qn))
        q = qn
    qs = np.asarray(qs)
    vs = np.asarray(vs)
    times = np.asarray(times)
    boundary_fields = model.basis.evaluate(model.boundary)
    boundary_ops = [
        boundary_fields[key] @ model.transform for key in ("xr", "xz", "xp")
    ]
    displacements = np.stack([qs @ op.T for op in boundary_ops], axis=-1)
    interface_flux = qs @ model.trace.T
    projected_interface = interface_flux @ model.vacuum.extension[model.vacuum.inner].T
    trace_relative = la.norm(projected_interface - interface_flux) / max(
        la.norm(interface_flux), 1e-30
    )
    # Distance from the displaced interface to the fixed wall, at recorded times.
    moved = model.boundary[None, :, :] + displacements[:, :, :2]
    from scipy.spatial import cKDTree

    gap = (
        cKDTree(model.vacuum.points[model.vacuum.outer])
        .query(moved.reshape(-1, 2))[0]
        .min()
    )
    values, vectors = model.modes()
    modal = (initial @ vectors)[None, :] * np.cos(
        times[:, None] * np.sqrt(np.maximum(values, 0))[None, :]
    )
    if values[0] >= -1e-8:
        reference = modal @ vectors.T
        time_error = float(la.norm(qs - reference) / la.norm(reference))
    else:
        time_error = None
    return dict(
        time=times,
        displacement_coefficients=qs,
        velocity_coefficients=vs,
        centroid_z=qs @ h,
        energy=np.array(energy),
        boundary=moved,
        boundary0=model.boundary,
        vacuum_points=model.vacuum.points,
        vacuum_triangles=model.vacuum.triangles,
        vacuum_flux=qs @ (model.vacuum.extension @ model.trace).T,
    ), dict(
        duration=float(times[-1]),
        initial_fraction=initial_fraction,
        dt=float(dt),
        max_centroid_fraction=float(abs(qs @ h).max() / minor),
        max_boundary_displacement_fraction=float(
            np.linalg.norm(displacements, axis=-1).max() / minor
        ),
        minimum_wall_clearance=float(gap),
        relative_energy_drift=float(
            np.max(abs(np.array(energy) - e0)) / max(abs(e0), 1e-30)
        ),
        relative_time_solution_error=time_error,
        interface_trajectory_relative_error=float(trace_relative),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("build/tokamak_confined"))
    ap.add_argument("--n", type=int, default=49)
    ap.add_argument("--modes", type=int, default=14)
    ap.add_argument("--vacuum-method", choices=("bspf", "fem"), default="bspf")
    args = ap.parse_args()
    jax.config.update("jax_enable_x64", True)
    args.out.mkdir(parents=True, exist_ok=True)
    p = plan_axisymmetric_bspf(args.n)
    coils, offset, _ = fit_fixed_coils(quadrupole=-0.004, vertical=0.03, offset=-0.2)
    eq = solve_equilibrium(p, coils, offset=offset, max_iterations=700)
    f, profile, surfaces = safety_profile(EquilibriumEvaluator(p, eq, coils, offset))
    common = dict(
        modes=args.modes,
        angles=256,
        radial_quadrature=64,
        vacuum_layers=48,
        vacuum_method=args.vacuum_method,
        toroidal_f=f,
    )
    model = assemble_plasma_vacuum(p, eq, coils, offset, wall_scale=1.2, **common)
    far = assemble_plasma_vacuum(p, eq, coils, offset, **common)
    eigen, vectors = model.modes()
    fe, _ = far.modes()
    if eigen[0] < -1e-8:
        raise RuntimeError("Chosen close-wall configuration is unstable")
    data, checks = trajectory(model)
    control, control_checks = trajectory(far, stop_fraction=0.02)
    gaps = []
    for scale in (1.1, 1.2, 1.3):
        m = (
            model
            if scale == 1.2
            else assemble_plasma_vacuum(
                p, eq, coils, offset, wall_scale=scale, **common
            )
        )
        e, v = m.modes()
        h = vertical_projection(m)
        overlap = abs(h @ v) ** 2
        index = int(np.argmax(overlap))
        gaps.append(
            dict(
                wall_scale=scale,
                min_omega_squared=float(e[0]),
                dominant_vertical_omega_squared=float(e[index]),
                dominant_vertical_omega=float(np.sqrt(max(e[index], 0))),
                resolved_growing_modes=int(np.sum(e < -1e-8)),
            )
        )
    field = model.evaluator.evaluate(model.quadrature_points)
    pressure = eq["alpha"] * field[0] ** 3 / 3
    r = model.quadrature_points[:, 0]
    b2 = (field[1] ** 2 + field[2] ** 2 + f**2) / r**2
    beta = (
        2
        * np.average(pressure, weights=model.quadrature_weights)
        / np.average(b2, weights=model.quadrature_weights)
    )
    summary = dict(
        case="Elongated tokamak: passive ideal-wall vertical stabilization",
        dimensional_status="Dimensionless engineering benchmark, not a reconstructed device or reactor design",
        wall_scale=1.2,
        toroidal_f=float(f),
        q_axis=profile["q_axis"],
        q95=profile["q"][-1],
        volume_average_beta=float(beta),
        elongation=float(np.ptp(model.boundary[:, 1]) / np.ptp(model.boundary[:, 0])),
        minor_radius=float(np.ptp(model.boundary[:, 0]) / 2),
        plasma_current=1.0,
        fixed_coils=np.asarray(coils).tolist(),
        minimum_omega_squared=float(eigen[0]),
        resolved_growing_modes=int(np.sum(eigen < -1e-8)),
        far_wall_growth_rate=float(np.sqrt(-fe[0])),
        wall_scan=gaps,
        trajectory=checks,
        far_wall_trajectory=control_checks,
        limitations="Ideal conductor; vertical parity n=0 only; no finite wall resistance, feedback, divertor, transport or fusion power",
        **{k: v for k, v in model.diagnostics.items() if k != "wall_scale"},
    )
    np.savez_compressed(args.out / "evolution.npz", **data)
    np.savez_compressed(args.out / "far_wall.npz", **control)
    np.savez_compressed(
        args.out / "equilibrium_surfaces.npz",
        surfaces=surfaces,
        normalized_flux=profile["normalized_poloidal_flux"],
    )
    (args.out / "q_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "model.pkl").write_bytes(pickle.dumps(model))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
