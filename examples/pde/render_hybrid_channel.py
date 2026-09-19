"""Compare computed NS fields at fixed BSPF resolution, including rational lift."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


def interior_residual(data, summary):
    """Independent FD6 spatial / backward second-order temporal diagnostic.

    This is a sampled PDE defect, not an error against an exact NS solution.
    The mask excludes the sponge, outer walls and all hole-crossing stencils.
    """
    x, y, times = data["x"], data["y"], data["t"]
    u, v, omega, _ = data["fields"][-1]
    xx, yy = np.meshgrid(x, y)
    dx, dy = x[1] - x[0], y[1] - y[0]
    lap = np.zeros_like(omega[3:-3, 3:-3])
    wx, wy = lap.copy(), lap.copy()
    c2 = np.array([1 / 90, -3 / 20, 1.5, -49 / 18, 1.5, -3 / 20, 1 / 90])
    c1 = np.array([-1, 9, -45, 0, 45, -9, 1]) / 60
    for k in range(7):
        sx = omega[3:-3, k : k + len(x) - 6]
        sy = omega[k : k + len(y) - 6, 3:-3]
        lap += c2[k] * (sx / dx**2 + sy / dy**2)
        wx += c1[k] * sx / dx
        wy += c1[k] * sy / dy
    assert np.allclose(np.diff(times[-3:]), times[-1] - times[-2])
    wt = (3 * omega - 4 * data["fields"][-2, 2] + data["fields"][-3, 2]) / (
        2 * (times[-1] - times[-2])
    )
    residual = (
        wt[3:-3, 3:-3]
        + u[3:-3, 3:-3] * wx
        + v[3:-3, 3:-3] * wy
        - summary["viscosity"] * lap
    )
    xr, yr = xx[3:-3, 3:-3], yy[3:-3, 3:-3]
    mask = (
        (xr > -0.85)
        & (xr < 2.5)
        & (abs(yr) < 0.85)
        & (((xr - 0.19) / 0.41) ** 2 + ((yr + 0.13) / 0.33) ** 2 > 1)
    )
    assert np.all(np.isfinite(residual[mask]))
    return dict(
        sampled_ns_vorticity_residual_rms=float(np.sqrt(np.mean(residual[mask] ** 2))),
        sampled_ns_vorticity_residual_max=float(np.max(abs(residual[mask]))),
        samples=int(mask.sum()),
    )


def main():
    root = Path("build/immersed_flow")
    out = root / "hybrid/channel"
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout="constrained", width_ratios=(2, 1))
    records = []
    for row, (key, label) in enumerate(
        [("dense", "Original SVD"), ("factor", "Analytic wall factor"), ("hybrid/channel", "Rational boundary correction")]
    ):
        data = np.load(root / key / "fields.npz")
        summary = json.loads((root / key / "summary.json").read_text())
        assert (summary["nx"], summary["ny"], summary["final_time"]) == (73, 33, 20)
        x, y = data["x"], data["y"]
        xx, yy = np.meshgrid(x, y)
        omega = data["fields"][-1, 2] - 2 * yy
        hole = ((xx - data["center"][0]) / data["axes"][0]) ** 2 + ((yy - data["center"][1]) / data["axes"][1]) ** 2 <= 1
        ax = axes[row, 0]
        im = ax.pcolormesh(x, y, np.ma.array(omega, mask=hole), cmap="RdBu_r", vmin=-22, vmax=22, shading="auto")
        ax.add_patch(Ellipse(data["center"], *(2 * data["axes"]), facecolor=".65", edgecolor="k"))
        ax.set(xlim=(-1, 3), ylim=(-1, 1), aspect="equal", xlabel="x", ylabel="y", title=f"{label} | quadrature factor {summary['quadrature_factor']:g}")
        ax = axes[row, 1]
        for pos in (0.75, 1.5):
            k = np.argmin(abs(x - pos))
            ax.plot(omega[:, k], y, label=f"x={x[k]:.2f}")
        ax.set(xlim=(-8, 8), ylim=(-1, 1), xlabel="Vorticity perturbation", ylabel="y")
        ax.legend()
        ax.grid(alpha=0.2)
        records.append(dict(method=key, **interior_residual(data, summary)))
    fig.suptitle("Computed Navier-Stokes | Re=20, t=20, fixed BSPF 73 x 33; identical color scales")
    fig.colorbar(im, ax=axes[:, 0], label="Vorticity minus Poiseuille shear", shrink=0.8)
    fig.savefig(out / "comparison.png", dpi=170)
    plt.close(fig)
    (out / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
