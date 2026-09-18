"""Compute and render the nonperiodic, boundary-buffered JAX BSPF KH example.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src \
 MPLCONFIGDIR=/tmp/pybspf-mpl python scratch/run_kh_nonperiodic.py
"""

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

import bspf_jax as b
from bspf_jax.navier_stokes import (
    plan_navier_stokes2d,
    kh_initial_velocity,
    ns_raw_rhs,
    ns_rk4_step,
    ns_vorticity,
)


def render(out, fps=25):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    # The conda ffmpeg on this machine is killed before startup; prefer the
    # working Homebrew binary when present. Other platforms retain PATH lookup.
    homebrew_ffmpeg = Path("/opt/homebrew/bin/ffmpeg")
    if homebrew_ffmpeg.exists():
        matplotlib.rcParams["animation.ffmpeg_path"] = str(homebrew_ffmpeg)

    with np.load(out / "frames.npz") as data:
        x, y, times = data["x"], data["y"], data["times"]
        omega, v = data["vorticity"], data["transverse_velocity"]
    records = json.loads((out / "diagnostics.json").read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "figure.facecolor": "#f5f7fb",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(12.8, 9), dpi=100)
    gs = fig.add_gridspec(
        3,
        2,
        left=0.08,
        right=0.94,
        bottom=0.09,
        top=0.87,
        height_ratios=[1, 1, 0.65],
        hspace=0.5,
        wspace=0.3,
    )
    a, c = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])
    growth, constraint = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
    extent = [x[0], x[-1], y[0], y[-1]]
    im = a.imshow(
        omega[0].T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-10,
        vmax=10,
        interpolation="bilinear",
    )
    vm = c.imshow(
        v[0].T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-0.6,
        vmax=0.6,
        interpolation="bilinear",
    )
    for ax, title in [
        (a, "Vorticity: shear-layer roll-up"),
        (c, "Transverse velocity: KH perturbation growth"),
    ]:
        ax.set(xlabel="x", ylabel="y", title=title)
        ax.axvline(-2, color="#64748b", ls="--", lw=0.8)
        ax.axvline(2, color="#64748b", ls="--", lw=0.8)
    fig.colorbar(im, ax=a, fraction=0.027, pad=0.015, label="dv/dx − du/dy")
    fig.colorbar(vm, ax=c, fraction=0.027, pad=0.015, label="v")
    vrms = np.array([r["central_v_rms"] for r in records])
    div = np.array([r["divergence_linf"] for r in records])
    growth.semilogy(times, vrms, color="#cbd5e1")
    (line,) = growth.semilogy([], [], color="#2563eb", lw=2)
    (dot,) = growth.semilogy([], [], "o", color="#2563eb")
    growth.set(
        xlim=(0, times[-1]),
        xlabel="Physical time",
        ylabel="RMS(v)",
        title="Central region |x| < 2, |y| < 0.5",
    )
    constraint.semilogy(times, np.maximum(div, 1e-16), color="#cbd5e1")
    (dl,) = constraint.semilogy([], [], color="#059669", lw=2)
    constraint.set(
        xlim=(0, times[-1]),
        xlabel="Physical time",
        ylabel="max |div u|",
        title="Incompressibility residual",
    )
    for ax in (growth, constraint):
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Kelvin–Helmholtz instability · nonperiodic JAX BSPF",
        x=0.08,
        y=0.978,
        ha="left",
        fontsize=20,
        weight="bold",
    )
    fig.text(
        0.08,
        0.935,
        f"{x.size} × {y.size} uniform nodes  |  ν = 0.002  |  "
        "fixed shear boundaries + smooth sponge  |  maintained base flow",
        fontsize=10,
    )
    clock = fig.text(0.94, 0.895, "", ha="right", weight="bold", fontsize=12)
    status = fig.text(0.08, 0.028, "", fontsize=10)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={
            "title": "Nonperiodic JAX BSPF Kelvin–Helmholtz instability",
            "comment": "Computed NS states; fixed base forcing and explicit sponge; no periodic boundary conditions.",
        },
    )
    with writer.saving(fig, str(out / "kh_nonperiodic.mp4"), dpi=100):
        for i, t in enumerate(times):
            im.set_data(omega[i].T)
            vm.set_data(v[i].T)
            line.set_data(times[: i + 1], vrms[: i + 1])
            dot.set_data([t], [vrms[i]])
            dl.set_data(times[: i + 1], np.maximum(div[: i + 1], 1e-16))
            clock.set_text(f"t = {t:.2f}")
            status.set_text(
                f"max |div u| = {div[i]:.2e}   |   boundary error = {records[i]['boundary_linf']:.1e}"
                f"   |   central RMS(v) = {vrms[i]:.4f}"
            )
            writer.grab_frame()
            if i == len(times) - 1:
                fig.savefig(out / "kh_nonperiodic.png", dpi=150)
    plt.close(fig)


def main():
    jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=160)
    parser.add_argument("--ny", type=int, default=112)
    parser.add_argument("--T", type=float, default=6.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--frame-dt", type=float, default=0.02)
    parser.add_argument(
        "--out", type=Path, default=Path("examples/pde/results/kh_nonperiodic")
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        render(args.out)
        return
    steps = int(round(args.frame_dt / args.dt))
    frames = int(round(args.T / args.frame_dt))
    if (
        steps < 1
        or not np.isclose(steps * args.dt, args.frame_dt)
        or not np.isclose(frames * args.frame_dt, args.T)
    ):
        raise ValueError(
            "frame-dt must be an integer multiple of dt, and T of frame-dt"
        )
    start = time.perf_counter()
    p = b.plan_pressure_poisson2d(
        jnp.linspace(-3, 3, args.nx),
        jnp.linspace(-1, 1, args.ny),
        endpoint_method="chebyshev",
        chebyshev_modes=12,
        baseline_points=16,
    )
    plan = plan_navier_stokes2d(p, viscosity=0.002)
    initial, base, initial_diagnostic = kh_initial_velocity(plan, perturbation=0.03)
    if not bool(initial_diagnostic.converged):
        raise RuntimeError("Initial perturbation projection failed")
    x, y = jnp.meshgrid(p.x.x, p.y.x, indexing="ij")
    sponge = 80 * ((x / 3) ** 16 + y**16)
    force = -ns_raw_rhs(plan, base)
    center = (abs(x) < 2) & (abs(y) < 0.5)
    wall = 1 - p.mask

    @jax.jit
    def diagnostics(u):
        omega = ns_vorticity(plan, u)
        return omega, jnp.array(
            [
                jnp.sqrt(
                    jnp.sum(center * p.weights * u[..., 1] ** 2)
                    / jnp.sum(center * p.weights)
                ),
                jnp.max(abs(b.pressure_divergence(p, u))),
                jnp.max(abs(wall[..., None] * (u - base))),
                0.5 * jnp.sum(p.weights[..., None] * u * u),
                jnp.max(jnp.sqrt(jnp.sum(u * u, axis=-1))),
                args.dt
                * jnp.max(
                    abs(u[..., 0]) / (p.x.x[1] - p.x.x[0])
                    + abs(u[..., 1]) / (p.y.x[1] - p.y.x[0])
                ),
            ]
        )

    def advance(u, dt, steps):
        def step(carry, _):
            u, valid, residual = carry
            u, d = ns_rk4_step(plan, u, dt, force, reference=base, sponge=sponge)
            return (u, valid & d.converged, jnp.maximum(residual, d.schur_linf)), None

        return jax.lax.scan(
            step, (u, jnp.array(True), jnp.array(0.0)), None, length=steps
        )[0]

    advance = jax.jit(advance, static_argnums=2)
    velocity = initial
    records, omegas, transverse = [], [], []
    t1_velocity = None
    for i in range(frames + 1):
        if i:
            velocity, valid, residual = advance(velocity, args.dt, steps)
            if not bool(valid):
                np.savez_compressed(
                    args.out / "failure.npz",
                    velocity=np.asarray(velocity),
                    t=i * args.frame_dt,
                )
                raise RuntimeError(
                    f"Pressure projection failed at t={i * args.frame_dt}"
                )
        else:
            residual = 0.0
        omega, d = diagnostics(velocity)
        values = np.asarray(d)
        if not np.all(np.isfinite(values)) or values[4] > 5 or values[1] > 1e-7:
            raise RuntimeError(f"State check failed at t={i * args.frame_dt}: {values}")
        record = dict(
            t=i * args.frame_dt,
            central_v_rms=float(values[0]),
            divergence_linf=float(values[1]),
            boundary_linf=float(values[2]),
            energy=float(values[3]),
            max_speed=float(values[4]),
            cfl=float(values[5]),
            max_stage_schur_linf=float(residual),
        )
        records.append(record)
        omegas.append(np.asarray(omega, dtype=np.float32))
        transverse.append(np.asarray(velocity[..., 1], dtype=np.float32))
        if np.isclose(i * args.frame_dt, 1.0):
            t1_velocity = velocity
        if i % 20 == 0:
            print(
                json.dumps(dict(**record, elapsed_s=time.perf_counter() - start)),
                flush=True,
            )
    np.savez_compressed(
        args.out / "frames.npz",
        x=np.asarray(p.x.x),
        y=np.asarray(p.y.x),
        times=np.array([r["t"] for r in records]),
        vorticity=np.array(omegas),
        transverse_velocity=np.array(transverse),
    )
    (args.out / "diagnostics.json").write_text(json.dumps(records, indent=2) + "\n")
    # Same initial condition, half dt to t=1: assess temporal error independently.
    validation = {}
    if t1_velocity is not None:
        refined, valid, _ = advance(initial, args.dt / 2, int(round(2 / args.dt)))
        difference = refined - t1_velocity
        validation["half_dt_T1_valid"] = bool(valid)
        validation["half_dt_T1_relative_perturbation_l2"] = float(
            jnp.sqrt(
                jnp.sum(p.weights[..., None] * difference**2)
                / jnp.sum(p.weights[..., None] * (refined - base) ** 2)
            )
        )
        if not bool(valid) or validation["half_dt_T1_relative_perturbation_l2"] > 0.01:
            raise RuntimeError(f"Time refinement failed: {validation}")
    control, valid, _ = advance(base, args.dt, int(round(args.T / args.dt)))
    validation["unperturbed_control_valid"] = bool(valid)
    validation["unperturbed_control_linf"] = float(jnp.max(abs(control - base)))
    if not bool(valid) or validation["unperturbed_control_linf"] > 1e-8:
        raise RuntimeError(f"Base-flow control failed: {validation}")
    _, pressure_result = b.project_pressure2d(
        p, ns_raw_rhs(plan, velocity) + force - sponge[..., None] * (velocity - base)
    )
    if not bool(pressure_result.converged):
        raise RuntimeError("Final completed pressure solve failed")
    np.savez_compressed(
        args.out / "final_state.npz",
        x=np.asarray(p.x.x),
        y=np.asarray(p.y.x),
        velocity=np.asarray(velocity),
        initial=np.asarray(initial),
        base=np.asarray(base),
        pressure=np.asarray(pressure_result.pressure),
        sponge=np.asarray(sponge),
        force=np.asarray(force),
        t=args.T,
    )
    summary = dict(
        nx=args.nx,
        ny=args.ny,
        T=args.T,
        dt=args.dt,
        viscosity=0.002,
        thickness=0.12,
        perturbation=0.03,
        wavelength=1.5,
        frame_count=frames + 1,
        endpoint_method="chebyshev",
        chebyshev_modes=12,
        baseline_points=16,
        boundary="Fixed tanh shear velocity on all four boundaries; nonperiodic",
        forcing="Fixed -NS_raw(base), maintaining the discrete base shear",
        sponge="80*((x/3)^16+y^16), damping toward base",
        central_v_rms_growth=records[-1]["central_v_rms"] / records[0]["central_v_rms"],
        max_divergence=max(r["divergence_linf"] for r in records),
        max_boundary_error=max(r["boundary_linf"] for r in records),
        max_stage_schur=max(r["max_stage_schur_linf"] for r in records),
        max_cfl=max(r["cfl"] for r in records),
        validation=validation,
        wall_gradient_fit_linf=float(pressure_result.wall_gradient_fit_linf),
        elapsed_s=time.perf_counter() - start,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not args.no_video:
        render(args.out)
    print(args.out.resolve(), flush=True)


if __name__ == "__main__":
    main()
