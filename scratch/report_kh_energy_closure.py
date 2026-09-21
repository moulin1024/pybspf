"""Report SBP energy certificates and the no-sponge KH comparison."""

import json
from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bspf_models._numerics._energy_stable import sbp84_derivative
from bspf_models._numerics._energy_stable import sbp84_diffusion


def main():
    jax.config.update("jax_enable_x64", True)
    root = Path("build/kh_stability")
    out = root / "sbp84"
    old = json.loads((root / "evolve_none_160x112_dt0.002.json").read_text())
    runs = [
        json.loads((out / f"evolve_none_160x112_dt{dt}.json").read_text())
        for dt in (0.002, 0.001)
    ]
    states = [np.load(out / f"none_160x112_dt{dt}_state.npz") for dt in (0.002, 0.001)]
    scalar = []
    for name, x in (("x", states[0]["x"]), ("y", states[0]["y"])):
        d, h = map(np.asarray, sbp84_derivative(x))
        boundary = np.zeros(len(x))
        boundary[0], boundary[-1] = -1, 1
        diffusion = np.asarray(sbp84_diffusion(d, h))
        a = (-d + 0.002 * diffusion)[1:-1, 1:-1]
        root_h = np.sqrt(h[1:-1])
        scaled = root_h[:, None] * a / root_h[None, :]
        scalar.append(
            dict(
                axis=name,
                sbp_identity_max=float(
                    abs(h[:, None] * d + d.T * h - np.diag(boundary)).max()
                ),
                max_eigenvalue_real=float(np.linalg.eigvals(a).real.max()),
                weighted_energy_max_eigenvalue=float(
                    np.linalg.eigvalsh((scaled + scaled.T) / 2).max()
                ),
            )
        )
    convergence = []
    for n in (40, 80, 160):
        x = np.linspace(-1, 1, n)
        d, _ = sbp84_derivative(x)
        convergence.append(
            dict(
                n=n,
                derivative_exp_linf=float(
                    abs(np.asarray(d) @ np.exp(x) - np.exp(x)).max()
                ),
            )
        )
    s, fine = states
    relative = np.linalg.norm(s["velocity"] - fine["velocity"]) / np.linalg.norm(
        fine["velocity"] - fine["base"]
    )
    summary = dict(
        scalar_certificates=scalar,
        derivative_convergence=convergence,
        half_dt_relative_perturbation_difference=float(relative),
        runs=[
            dict(
                dt=r["dt"],
                last=r["records"][-1],
                max_divergence=max(v["div_linf"] for v in r["records"]),
                max_wall_error=max(v["wall_linf"] for v in r["records"]),
                all_pressure_checks_pass=all(v["valid"] for v in r["records"]),
            )
            for r in runs
        ],
    )
    fine_path = out / "none_240x168_dt0.002_state.npz"
    if fine_path.exists():
        from scipy.interpolate import RectBivariateSpline

        with np.load(fine_path) as grid_fine:
            center = (abs(s["x"][:, None]) < 2) & (abs(s["y"][None, :]) < 0.5)
            comparison = []
            for index, t in [(2, 1.0), (6, 3.0), (12, 6.0)]:
                reference = np.stack(
                    [
                        RectBivariateSpline(
                            grid_fine["x"],
                            grid_fine["y"],
                            grid_fine["frames"][index, ..., k],
                        )(s["x"], s["y"])
                        for k in range(2)
                    ],
                    axis=-1,
                )
                difference = s["frames"][index] - reference
                comparison.append(
                    dict(
                        t=t,
                        relative_full=float(
                            np.linalg.norm(difference) / np.linalg.norm(reference)
                        ),
                        relative_center=float(
                            np.linalg.norm(difference[center])
                            / np.linalg.norm(reference[center])
                        ),
                    )
                )
            summary["grid_comparison_160x112_vs_240x168"] = comparison
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for run, label, color, ls in [
        (old, "Original BSPF, no sponge", "#dc2626", "-"),
        (runs[0], "SBP(8,4), dt=0.002, no sponge", "#2563eb", "-"),
        (runs[1], "SBP(8,4), dt=0.001, no sponge", "#16a34a", "--"),
    ]:
        t = [r["t"] for r in run["records"]]
        axes[0, 0].semilogy(
            t, [r["max_speed"] for r in run["records"]], label=label, color=color, ls=ls
        )
        axes[0, 1].semilogy(
            t, [r["edge_rms"] for r in run["records"]], color=color, ls=ls
        )
    axes[0, 0].set(
        title="Removing the numerical blow-up", xlabel="Time", ylabel="Maximum speed"
    )
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(
        title="Boundary-region perturbation", xlabel="Time", ylabel="RMS(u - base)"
    )
    x, y = s["x"], s["y"]
    dx, _ = sbp84_derivative(x)
    dy, _ = sbp84_derivative(y)
    u = s["velocity"]
    omega = np.asarray(dx) @ u[..., 1] - u[..., 0] @ np.asarray(dy).T
    im = axes[1, 0].imshow(
        omega.T,
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        aspect="auto",
        cmap="RdBu_r",
        vmin=-10,
        vmax=10,
    )
    axes[1, 0].set(title="SBP(8,4) vorticity at t=6, no sponge", xlabel="x", ylabel="y")
    fig.colorbar(im, ax=axes[1, 0], label="dv/dx - du/dy", shrink=0.8)
    modes = np.load(out / "none_160x112_eigenmodes.npz")
    k = np.argmax(modes["values"].real)
    v = modes["vectors"][:, k].reshape(len(x) - 2, len(y) - 2, 2)
    amplitude = np.sqrt(np.sum(abs(v) ** 2, axis=-1))
    im = axes[1, 1].imshow(
        (amplitude / amplitude.max()).T,
        origin="lower",
        extent=[x[1], x[-2], y[1], y[-2]],
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    axes[1, 1].set(
        title=f"Leading KH mode: Re(lambda)={modes['values'][k].real:.4f}",
        xlabel="x",
        ylabel="y",
    )
    fig.colorbar(im, ax=axes[1, 1], label="Normalized mode amplitude", shrink=0.8)
    fig.suptitle(
        "Energy-compatible high-order closure | 160 x 112 | viscosity=0.002",
        fontsize=15,
    )
    for ax in axes[0]:
        ax.grid(alpha=0.2)
    fig.savefig(out / "comparison.png", dpi=160)
    fig.savefig(out / "comparison.pdf")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
