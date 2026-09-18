"""Plot fixed-space quadrature and unsteady NS checks for the hybrid method."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def render(root):
    low = json.loads((root / "mms/summary.json").read_text())
    high = json.loads((root / "mms_q4/summary.json").read_text())
    with np.load(root / "mms_q4/fields.npz") as data:
        x, y, ref = data["x"], data["y"], data["reference"]
        speed = np.hypot(ref[1], ref[2])
        velocity_error = np.hypot(data["u"] - ref[1], data["v"] - ref[2])
        vorticity_error = abs(data["vorticity"] - (ref[5] - ref[4]))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, field, title, logarithmic in zip(
        axes[0],
        (speed, velocity_error, vorticity_error),
        ("Forced Stokes MMS: reference speed", "Velocity absolute error", "Vorticity absolute error"),
        (False, True, True),
    ):
        norm = LogNorm(vmin=1e-13, vmax=max(1e-12, np.nanmax(field))) if logarithmic else None
        plot = ax.pcolormesh(x, y, field, shading="auto", norm=norm, cmap="magma")
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
        fig.colorbar(plot, ax=ax, shrink=0.65)
    ax = axes[1, 0]
    dt = np.array([r["dt"] for r in high["time_convergence"]])
    err = np.array([r["velocity_error_l2"] for r in high["time_convergence"]])
    ax.loglog(dt, err, "o-", label="Full nonlinear NS MMS")
    ax.loglog(dt, err[-1] * (dt / dt[-1]) ** 2, "--", label=r"$O(\Delta t^2)$")
    ax.set(title="Unsteady NS, final time 0.4", xlabel="Time step", ylabel="Velocity absolute L2 error")
    ax.legend()
    ax.grid(True, which="both", alpha=0.2)
    ax = axes[1, 1]
    ax.bar(
        ["Complete time term", "Rational time term omitted"],
        [high["continuous_ns_residual_mass_dual"], high["residual_if_rational_time_omitted"]],
        color=["#367ba8", "#cc603e"],
    )
    ax.set(yscale="log", title="Continuous NS residual: negative control", ylabel="Mass-dual norm")
    ax.tick_params(axis="x", labelsize=9)
    ax = axes[1, 2]
    for key, label in (
        ("forced_stokes_velocity_relative_l2", "Forced Stokes velocity"),
        ("forced_stokes_vorticity_relative_l2", "Forced Stokes vorticity"),
        ("continuous_ns_residual_mass_dual", "Continuous NS residual"),
    ):
        ax.semilogy([2.5, 4], [low[key], high[key]], "o-", label=label)
    ax.set(title="Quadrature only; unknowns unchanged", xlabel="Quadrature factor", ylabel="Relative L2 error / residual", xticks=[2.5, 4])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.suptitle("BSPF + rational boundary correction | fixed 73 x 33 space, 2059 unknowns", fontsize=15)
    path = root / "validation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("build/immersed_flow/hybrid"))
    render(parser.parse_args().root)
