"""BSPF axisymmetric tokamak equilibrium and deformable linear vertical mode.

OPENBLAS_NUM_THREADS=1 python examples/pde/tokamak_axisymmetric.py
All quantities are dimensionless. No device-specific growth rate is claimed.
"""

import argparse
import json
from pathlib import Path
import time
import pickle
import jax
import numpy as np
import scipy.linalg as la
from bspf_models.plasma.tokamak_equilibrium import plan_axisymmetric_bspf
from bspf_models.plasma.tokamak_equilibrium import fit_fixed_coils
from bspf_models.plasma.tokamak_equilibrium import solve_equilibrium
from bspf_models.plasma.tokamak_equilibrium import external_field
from bspf_models.plasma.tokamak_linear import assemble_linear_tokamak
from bspf_models.plasma.tokamak_linear import growing_modes
from bspf_models.plasma.tokamak_linear import with_exterior


def run(
    model,
    out,
    *,
    dt=0.1,
    initial_displacement=1e-4,
    final_displacement=0.02,
    sensitivity=False,
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    p, eq = model.equilibrium_plan, model.equilibrium
    if np.asarray(p.radial.x).dtype != np.float64:
        raise ValueError("Enable jax_enable_x64 before building or loading the model")
    rr = np.asarray(p.radial.points)[:, None]
    zz = np.asarray(p.vertical.points)[None, :]
    w = np.asarray(p.radial.weights)[:, None] * np.asarray(p.vertical.weights)[None, :]
    core = eq["core"]
    i, j = np.where(core)
    minor_radius = float((rr[i, 0].max() - rr[i, 0].min()) / 2)
    elongation = (zz[0, j].max() - zz[0, j].min()) / (2 * minor_radius)
    gammas, vectors, residuals = growing_modes(model, shift=0.15)
    candidates = np.where(
        (gammas.real > 0) & (abs(gammas.imag) < 1e-8) & (residuals < 1e-7)
    )[0]
    if not len(candidates):
        raise RuntimeError("No verified real growing mode near the requested shift")
    index = candidates[0]
    gamma = float(gammas[index].real)
    mode = vectors[:, index].real
    flux, velocity, magnetic = model.fields(mode)
    mass = np.sum(w * rr * core)
    centroid = float(np.sum(w * rr * core * velocity[..., 1]) / mass / gamma)
    if abs(centroid) < 1e-10:
        raise RuntimeError("Selected mode has no vertical center-of-mass motion")
    state = mode * (initial_displacement * minor_radius / centroid)
    initial = state.copy()
    duration = np.log(final_displacement / initial_displacement) / gamma
    steps = int(np.ceil(duration / dt))
    dt = duration / steps
    a, m = model.matrices["A"], model.matrices["M"]
    left = la.lu_factor(m - dt / 2 * a)
    right = m + dt / 2 * a
    energy = model.matrices["energy"]
    times = []
    centroids = []
    energies = []
    states = []
    fluxes = []
    velocities = []
    magnetics = []
    zero = np.zeros_like(state)
    stride = max(1, steps // 100)
    mass_mode = float(mode @ m @ mode)
    for k in range(steps + 1):
        if k % stride == 0 or k == steps:
            fp, up, bp = model.fields(state, nodes=True)
            # State amplitude is measured from the time-advanced vector, not prescribed.
            amplitude = float(mode @ (m @ state) / mass_mode)
            times.append(k * dt)
            centroids.append(amplitude * centroid)
            energies.append(0.5 * float(state @ energy @ state))
            states.append(state.copy())
            fluxes.append(fp)
            velocities.append(up)
            magnetics.append(bp)
        if k < steps:
            state = la.lu_solve(left, right @ state)
            zero = la.lu_solve(left, right @ zero)
    measured = float(np.polyfit(times, np.log(np.abs(centroids)), 1)[0])
    rn = np.asarray(p.radial.x)
    zn = np.asarray(p.vertical.x)
    psi0 = (
        p.evaluate(eq["a"], nodes=True)[0]
        + external_field(rn[:, None], zn[None, :], model.coils, model.offset)[0]
    )
    aq = eq["a"]
    r, z = p.radial, p.vertical
    strong = (
        -(
            np.asarray(r.h) @ aq @ np.asarray(z.b).T
            - np.asarray(r.g) @ aq @ np.asarray(z.b).T / rr
            + np.asarray(r.b) @ aq @ np.asarray(z.h).T
        )
        / rr
    )
    current_error = float(
        np.sqrt(
            np.sum(w * (strong - eq["current"]) ** 2) / np.sum(w * eq["current"] ** 2)
        )
    )
    weak_error = float(
        la.norm(p.action(aq) - p.load(eq["current"])) / la.norm(p.load(eq["current"]))
    )
    _, ufinal, _ = model.fields(state)
    max_displacement = float(np.max(np.linalg.norm(ufinal[core], axis=-1)) / gamma)
    # Measure non-rigid internal structure using the full plasma displacement field.
    rigid = np.zeros_like(ufinal)
    rigid[..., 1] = centroids[-1] * gamma
    deformation = float(
        np.sqrt(
            np.sum(w * rr * core * np.sum((ufinal - rigid) ** 2, axis=-1))
            / np.sum(w * rr * core * np.sum(ufinal**2, axis=-1))
        )
    )
    sensitivities = {}
    if sensitivity:
        for name, kw in [
            ("half_halo_density", dict(halo_density=float(model.density.min()) / 2)),
            (
                "double_exterior_resistivity",
                dict(vacuum_resistivity=float(model.resistivity.max()) * 2),
            ),
        ]:
            altered = with_exterior(model, **kw)
            gv, _, rv = growing_modes(altered, shift=gamma)
            valid = (gv.real > 0) & (abs(gv.imag) < 1e-8) & (rv < 1e-7)
            if not valid.any():
                raise RuntimeError("No verified growing sensitivity mode")
            sensitivities[name] = float(gv[valid][0].real)
    np.savez_compressed(
        out / "evolution.npz",
        R=rn,
        Z=zn,
        psi0=psi0,
        time=times,
        delta_psi=fluxes,
        velocity=velocities,
        delta_B=magnetics,
        centroid_z=centroids,
        energy=energies,
        initial_state=initial,
        final_state=state,
        coils=model.coils,
    )
    summary = dict(
        n=len(rn),
        model="Axisymmetric full-vector incompressible linear MHD; vertical parity sector",
        exterior="Finite-density high-resistivity halo approximation; fixed poloidal flux and toroidal field at the outer boundary",
        rigid_motion=False,
        normalization="mu0=1; all quantities dimensionless",
        reynolds_note="Dynamic viscosity, not a fitted experimental transport coefficient",
        halo_density=float(model.density.min()),
        exterior_resistivity=float(model.resistivity.max()),
        toroidal_f=model.toroidal_f,
        elongation=float(elongation),
        minor_radius=float(minor_radius),
        equilibrium_iterations=int(eq["iterations"]),
        equilibrium_weak_residual=weak_error,
        equilibrium_strong_current_relative_l2=current_error,
        plasma_current=float(np.sum(w * eq["current"])),
        gamma_eigenvalue=gamma,
        gamma_time_fit=measured,
        growth_rate_relative_difference=abs(measured / gamma - 1),
        eigen_residual=float(residuals[index]),
        dt=dt,
        duration=duration,
        initial_centroid_z=float(centroids[0]),
        final_centroid_z=float(centroids[-1]),
        max_final_displacement_over_minor_radius=max_displacement / minor_radius,
        displacement_departure_from_rigid_translation=deformation,
        zero_control_linf=float(np.max(abs(zero))),
        sensitivities=sensitivities,
        coils=model.coils.tolist(),
        coil_flux_offset=model.offset,
        equilibrium_source_file="Self-consistent BSPF Grad-Shafranov solve; no prescribed moving contour",
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=33)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=Path("build/tokamak_axisymmetric"))
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument(
        "--control",
        action="store_true",
        help="Less elongated coil configuration; inspect modes near gamma=0.05 without assuming instability",
    )
    args = ap.parse_args()
    if not np.isfinite(args.dt) or args.dt <= 0:
        ap.error("dt must be finite and positive")
    jax.config.update("jax_enable_x64", True)
    start = time.perf_counter()
    p = plan_axisymmetric_bspf(n=args.n)
    coils, offset, _ = fit_fixed_coils(
        quadrupole=0.001 if args.control else -0.004,
        vertical=-0.018 if args.control else 0.03,
        offset=-0.2,
    )
    eq = solve_equilibrium(p, coils, offset=offset, max_iterations=700, tolerance=1e-10)
    model = assemble_linear_tokamak(p, eq, coils, offset)
    args.out.mkdir(parents=True, exist_ok=True)
    # Cache only local, self-generated objects for development verification.
    (args.out / "model.pkl").write_bytes(pickle.dumps(model))
    if args.control:
        values, _, residuals = growing_modes(model, shift=0.05, count=12)
        i, j = np.where(eq["core"])
        kappa = float(
            np.ptp(np.asarray(p.vertical.points)[j])
            / np.ptp(np.asarray(p.radial.points)[i])
        )
        result = dict(
            n=args.n,
            elongation=kappa,
            eigenvalues_real=values.real.tolist(),
            eigenvalues_imag=values.imag.tolist(),
            residuals=residuals.tolist(),
            near_shift_growing_modes=int(
                np.sum((values.real > 0) & (residuals < 1e-7))
            ),
            note="Spectrum near shift=0.05 in the vertical parity sector; not a proof of global stability",
        )
        (args.out / "spectrum.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return
    run(model, args.out, dt=args.dt, sensitivity=args.sensitivity)
    print(f"Elapsed {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
