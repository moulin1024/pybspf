"""Nonperiodic KH using same-space weak BSPF, direct projection, no sponge.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src \
 MPLCONFIGDIR=/tmp/pybspf-mpl python scratch/run_kh_weak.py
"""

import argparse
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from bspf_jax.weak_navier_stokes import (
    plan_weak_navier_stokes2d,
    weak_kh_initial_velocity,
    weak_ns_momentum_load,
    weak_ns_rk4_step,
    weak_ns_divergence,
    weak_ns_vorticity,
    weak_ns_pointwise_divergence,
)


def render(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    if Path("/opt/homebrew/bin/ffmpeg").exists():
        matplotlib.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"
    f = np.load(out / "frames.npz")
    records = json.loads((out / "diagnostics.json").read_text())
    t, x, y = f["times"], f["x"], f["y"]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [1, 1, 0.6]},
        layout="constrained",
    )
    fig.suptitle(
        "Nonperiodic KH | weak BSPF + direct mass-orthogonal projection", fontsize=15
    )
    extent = [x[0], x[-1], y[0], y[-1]]
    images = []
    for ax, values, label, vmax in zip(
        axes[:2],
        [f["vorticity"], f["transverse_velocity"]],
        ["Vorticity", "Transverse velocity"],
        [10, 0.8],
    ):
        images.append(
            ax.imshow(
                values[0].T,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
        )
        ax.set(xlabel="x", ylabel="y", title=label)
        fig.colorbar(images[-1], ax=ax, pad=0.01, shrink=0.85)
    div = np.maximum([r["divergence_linf"] for r in records], 1e-16)
    vrms = np.array([r["central_v_rms"] for r in records])
    ax = axes[2]
    ax.semilogy(t, vrms, color="lightgray")
    (line,) = ax.semilogy([], [], color="tab:blue", label="Central RMS(v)")
    ax.set(xlabel="Physical time", ylabel="Central RMS(v)", xlim=(0, t[-1]))
    ax.grid(alpha=0.2)
    title = axes[0].set_title("")
    writer = FFMpegWriter(
        fps=25,
        codec="libx264",
        extra_args=["-crf", "19", "-pix_fmt", "yuv420p"],
        metadata={
            "comment": "Computed weak BSPF NS. Fixed four-wall shear, maintained base, no sponge, zero refinement."
        },
    )
    with writer.saving(fig, str(out / "kh_weak.mp4"), dpi=110):
        for i, ti in enumerate(t):
            images[0].set_data(f["vorticity"][i].T)
            images[1].set_data(f["transverse_velocity"][i].T)
            line.set_data(t[: i + 1], vrms[: i + 1])
            title.set_text(
                f"{len(x)} x {len(y)} | t={ti:.2f} | no sponge | max |weak div u|={div[i]:.1e}"
            )
            writer.grab_frame()
        fig.savefig(out / "kh_weak.png", dpi=150)
    plt.close(fig)


def main():
    jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=160)
    parser.add_argument("--ny", type=int, default=112)
    parser.add_argument("--T", type=float, default=6)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--frame-dt", type=float, default=0.04)
    parser.add_argument("--out", type=Path, default=Path("build/kh_weak/160x112"))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        render(args.out)
        return
    steps, frames = round(args.frame_dt / args.dt), round(args.T / args.frame_dt)
    if (
        min(steps, frames) < 1
        or not np.isclose(steps * args.dt, args.frame_dt)
        or not np.isclose(frames * args.frame_dt, args.T)
    ):
        raise ValueError("Require integer time/frame counts")
    start = time.perf_counter()
    plan = plan_weak_navier_stokes2d(
        np.linspace(-3, 3, args.nx), np.linspace(-1, 1, args.ny)
    )
    initial, base = weak_kh_initial_velocity(plan)
    force = -weak_ns_momentum_load(plan, base)
    center = (abs(plan.x.x[:, None]) < 2) & (abs(plan.y.x[None, :]) < 0.5)
    setup = time.perf_counter() - start
    print(
        json.dumps(
            {"setup_s": setup, "quadrature": [len(plan.x.weights), len(plan.y.weights)]}
        ),
        flush=True,
    )

    @jax.jit
    def diagnostics(u):
        delta = u - base
        wall = jnp.maximum(
            jnp.max(abs(delta[jnp.array([0, -1])])),
            jnp.max(abs(delta[:, jnp.array([0, -1])])),
        )
        return jnp.array(
            [
                jnp.sqrt(jnp.sum(center * u[..., 1] ** 2) / jnp.sum(center)),
                jnp.max(abs(weak_ns_divergence(plan, u))),
                wall,
                jnp.max(jnp.linalg.norm(u, axis=-1)),
                jnp.max(abs(weak_ns_pointwise_divergence(plan, u))),
            ]
        )

    def advance(u, dt, steps):
        return jax.lax.fori_loop(
            0, steps, lambda _, v: weak_ns_rk4_step(plan, v, dt, force), u
        )

    advance = jax.jit(advance, static_argnums=2)
    records = []
    omegas = []
    transverse = []
    velocity = initial
    t1 = None
    for i in range(frames + 1):
        if i:
            velocity = advance(velocity, args.dt, steps)
        d = np.asarray(diagnostics(velocity))
        record = dict(
            t=i * args.frame_dt,
            central_v_rms=float(d[0]),
            divergence_linf=float(d[1]),
            boundary_linf=float(d[2]),
            max_speed=float(d[3]),
            pointwise_divergence_linf=float(d[4]),
        )
        if not np.all(np.isfinite(d)) or d[3] > 10 or d[1] > 1e-7:
            np.savez_compressed(
                args.out / "failure.npz", velocity=velocity, t=i * args.frame_dt
            )
            (args.out / "diagnostics.json").write_text(
                json.dumps(records + [record], indent=2)
            )
            raise RuntimeError(f"State failed: {record}")
        records.append(record)
        omegas.append(np.asarray(weak_ns_vorticity(plan, velocity), dtype=np.float32))
        transverse.append(np.asarray(velocity[..., 1], dtype=np.float32))
        if np.isclose(i * args.frame_dt, 1):
            t1 = np.asarray(velocity)
        if i % 10 == 0:
            print(
                json.dumps(dict(**record, elapsed_s=time.perf_counter() - start)),
                flush=True,
            )
    np.savez_compressed(
        args.out / "frames.npz",
        x=plan.x.x,
        y=plan.y.x,
        times=[r["t"] for r in records],
        vorticity=omegas,
        transverse_velocity=transverse,
    )
    np.savez_compressed(
        args.out / "final_state.npz",
        x=plan.x.x,
        y=plan.y.x,
        velocity=velocity,
        initial=initial,
        base=base,
        t=args.T,
    )
    (args.out / "diagnostics.json").write_text(json.dumps(records, indent=2))
    validation = {}
    if args.validate:
        if t1 is not None:
            refined = np.asarray(advance(initial, args.dt / 2, round(2 / args.dt)))
            validation["half_dt_T1_relative_perturbation_l2"] = float(
                np.linalg.norm(refined - t1)
                / np.linalg.norm(refined - np.asarray(base))
            )
        control = np.asarray(advance(base, args.dt, round(args.T / args.dt)))
        validation["base_control_linf"] = float(np.max(abs(control - np.asarray(base))))
    summary = dict(
        nx=args.nx,
        ny=args.ny,
        T=args.T,
        dt=args.dt,
        viscosity=0.002,
        sponge=False,
        refinement_steps=0,
        assembly_bits=113,
        quadrature=[len(plan.x.weights), len(plan.y.weights)],
        setup_s=setup,
        max_speed=max(r["max_speed"] for r in records),
        max_divergence=max(r["divergence_linf"] for r in records),
        max_pointwise_divergence=max(r["pointwise_divergence_linf"] for r in records),
        max_boundary_error=max(r["boundary_linf"] for r in records),
        central_growth=records[-1]["central_v_rms"] / records[0]["central_v_rms"],
        validation=validation,
        elapsed_s=time.perf_counter() - start,
        forcing="Fixed negative weak momentum load of base shear",
        boundary="Fixed (tanh(y/0.12),0) on all four walls",
        time_integrator="Unsplit RK4, explicit viscosity, direct projection at every stage",
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if not args.no_video:
        render(args.out)


if __name__ == "__main__":
    main()
