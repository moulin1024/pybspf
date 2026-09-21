"""Run and plot the BSPF regularized lid-driven incompressible NS cavity.

From the repository root:
  MPLCONFIGDIR=/tmp/bspf-mpl python examples/pde/cavity_2d.py
"""

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from bspf_models.fluids.cavity import plan_cavity
from bspf_models.fluids.cavity import plan_cavity_stepper
from bspf_models.fluids.cavity import cavity_step
from bspf_models.fluids.cavity import cavity_fields
from bspf_models.fluids.cavity import cavity_rhs
from bspf_models.fluids.cavity import cavity_ramp
from bspf_models.fluids.stream_navier_stokes import stream_ns_divergence
from bspf_models.fluids.stream_navier_stokes import stream_ns_velocity


def render(path, x, psi, velocity, omega, history, reynolds, t):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    speed = np.linalg.norm(velocity, axis=-1)
    ax = axes[0, 0]
    im = ax.pcolormesh(x, x, speed.T, shading="auto", cmap="viridis")
    if np.min(psi) < -1e-14:
        ax.contour(
            x,
            x,
            psi.T,
            levels=np.linspace(np.min(psi), 0, 20)[1:-1],
            colors="white",
            linewidths=0.65,
            linestyles="solid",
        )
    ax.set(
        title="Speed and streamfunction contours",
        xlabel="x",
        ylabel="y",
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="speed")
    ax = axes[0, 1]
    limit = np.max(abs(omega))
    im = ax.pcolormesh(
        x, x, omega.T, shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    ax.set(title="Vorticity", xlabel="x", ylabel="y", aspect="equal")
    fig.colorbar(im, ax=ax, label="dv/dx - du/dy")
    ax = axes[1, 0]
    # Interpolate onto the true center even for even node counts.
    u_mid = np.array([np.interp(0.5, x, velocity[:, j, 0]) for j in range(len(x))])
    v_mid = np.array([np.interp(0.5, x, velocity[i, :, 1]) for i in range(len(x))])
    ax.plot(x, u_mid, label="u(0.5, y)")
    ax.plot(x, v_mid, label="v(x, 0.5)")
    ax.axhline(0, color="gray", lw=0.7)
    ax.set(title="Centerline velocities", xlabel="y / x", ylabel="velocity")
    ax.legend()
    ax = axes[1, 1]
    ts = [r["time"] for r in history]
    ax.plot(ts, [r["kinetic_energy"] for r in history], color="tab:blue")
    ax.set(
        xlabel="time",
        ylabel="kinetic energy",
        title="Startup and approach to steady flow",
    )
    ax2 = ax.twinx()
    ax2.semilogy(
        ts[1:],
        [max(r["acceleration_linf"], 1e-16) for r in history[1:]],
        color="tab:orange",
    )
    ax2.set_ylabel("max |du/dt| (after startup: steady residual)")
    fig.suptitle(
        f"BSPF incompressible cavity | Re={reynolds:g} | t={t:g}\n"
        "Regularized lid: u(x,1)=16x²(1−x)²; all other walls stationary"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=41)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--T", type=float, default=30.0)
    ap.add_argument("--ramp-time", type=float, default=1.0)
    ap.add_argument("--steady-tol", type=float, default=1e-7)
    ap.add_argument("--out", type=Path, default=Path("build/cavity_2d"))
    args = ap.parse_args()
    if (
        not np.isfinite(args.T)
        or args.T <= 0
        or not np.isfinite(args.dt)
        or args.dt <= 0
    ):
        ap.error("T and dt must be finite and positive")
    if not np.isfinite(args.steady_tol) or args.steady_tol < 0:
        ap.error(
            "steady-tol must be finite and nonnegative (0 disables early stopping)"
        )
    jax.config.update("jax_enable_x64", True)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    c = plan_cavity(n=args.n, reynolds=args.re, ramp_time=args.ramp_time)
    count = int(np.ceil(args.T / args.dt))
    dt = args.T / count
    stepper = plan_cavity_stepper(c, dt)
    p = c.spatial
    a = jnp.zeros_like(p.denominator)
    print(
        f"BSPF {args.n}² nodes, {a.size} streamfunction modes; dt={dt:g}; setup {time.perf_counter() - started:.1f}s",
        flush=True,
    )

    @jax.jit
    def advance(a, start, steps):
        return jax.lax.fori_loop(
            0,
            steps,
            lambda k, state: cavity_step(c, stepper, state, (start + k) * dt),
            a,
        )

    @jax.jit
    def diagnostics(a, t):
        psi, vel, omega = cavity_fields(c, a, t)
        _, vq, _ = cavity_fields(c, a, t, nodes=False)
        energy = 0.5 * jnp.sum(
            p.x.weights[:, None] * p.y.weights[None, :] * jnp.sum(vq**2, axis=-1)
        )
        s, rate = cavity_ramp(t, c.ramp_time)
        from bspf_models.fluids.cavity import cavity_lift

        lid = 16 * p.x.x**2 * (1 - p.x.x) ** 2 * s
        boundary = jnp.maximum(
            jnp.max(abs(vel[[0, -1], :, :])),
            jnp.maximum(
                jnp.max(abs(vel[:, 0, :])),
                jnp.maximum(
                    jnp.max(abs(vel[:, -1, 1])), jnp.max(abs(vel[:, -1, 0] - lid))
                ),
            ),
        )
        acceleration = (
            stream_ns_velocity(p, cavity_rhs(c, a, t), nodes=True)
            + rate * cavity_lift(p.x.x, p.y.x)[1]
        )
        values = jnp.array(
            [
                energy,
                jnp.max(abs(stream_ns_divergence(p, a, nodes=True))),
                boundary,
                jnp.max(abs(acceleration)),
                jnp.max(jnp.linalg.norm(vel, axis=-1)),
            ]
        )
        return psi, vel, omega, values

    history = []
    stride = max(1, round(0.5 / dt))
    done = 0
    while True:
        t = done * dt
        psi, velocity, omega, values = jax.device_get(diagnostics(a, t))
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Nonfinite flow: reduce dt and check spatial resolution")
        row = dict(
            zip(
                (
                    "kinetic_energy",
                    "divergence_linf",
                    "boundary_linf",
                    "acceleration_linf",
                    "max_speed",
                ),
                map(float, values),
            )
        )
        row["time"] = t
        history.append(row)
        print(
            f"t={t:6.2f} E={values[0]:.6e} div={values[1]:.2e} wall={values[2]:.2e} residual={values[3]:.3e}",
            flush=True,
        )
        np.savez_compressed(
            args.out / "state.npz",
            a=np.asarray(a),
            x=np.asarray(p.x.x),
            psi=psi,
            velocity=velocity,
            vorticity=omega,
            time=t,
            dt=dt,
            reynolds=args.re,
            ramp_time=args.ramp_time,
        )
        steady = t >= args.ramp_time and values[3] < args.steady_tol
        if done >= count or steady:
            break
        steps = min(stride, count - done)
        a = advance(a, done, steps)
        done += steps
    summary = dict(
        n=args.n,
        reynolds=args.re,
        viscosity=1 / args.re,
        dt=dt,
        final_time=t,
        ramp_time=args.ramp_time,
        steady=bool(steady),
        steady_tolerance=args.steady_tol,
        elapsed_seconds=time.perf_counter() - started,
        boundary="u(x,1,t)=s(t)*16*x^2*(1-x)^2; all other velocity traces zero",
        method="BSPF clamped streamfunction Galerkin; IMEX midpoint/CN; no body force",
        final=history[-1],
        history=history,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    render(
        args.out / "cavity.png",
        np.asarray(p.x.x),
        psi,
        velocity,
        omega,
        history,
        args.re,
        t,
    )
    print(f"Saved {args.out}; steady={steady}", flush=True)


if __name__ == "__main__":
    main()
