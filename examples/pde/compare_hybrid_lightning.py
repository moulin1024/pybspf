"""Separate NS/Stokes model differences from a matched Stokes lift check."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from bspf_jax.immersed_flow import channel_lift
from bspf_jax.immersed_poisson import EllipticHole
from bspf_jax.rational_stokes import RationalStokesExtension


def metrics(u, v, w, ref, mask):
    du, dv, dw = u - ref[0], v - ref[1], w - ref[3]
    return dict(
        velocity_relative_l2=float(np.sqrt(np.sum((du**2 + dv**2)[mask]) / np.sum((ref[0]**2 + ref[1]**2)[mask]))),
        velocity_max=float(np.max(np.hypot(du, dv)[mask])),
        vorticity_relative_l2=float(np.sqrt(np.sum(dw[mask]**2) / np.sum(ref[3][mask]**2))),
        vorticity_max=float(np.max(abs(dw[mask]))),
    )


def main():
    root = Path("build/immersed_flow")
    out = root / "hybrid/channel/aaa_comparison"
    out.mkdir(parents=True, exist_ok=True)
    with np.load(root / "lightning/reference_120.npz") as data:
        x, y, ref = data["x"], data["y"], data["fields"]
    with np.load(root / "hybrid/channel/fields.npz") as data:
        assert np.array_equal(x, data["x"]) and np.array_equal(y, data["y"])
        u, v, w, _ = data["fields"][-1]
        assert data["t"][-1] == 20
    xx, yy = np.meshgrid(x, y)
    mask = np.isfinite(ref).all(axis=0)
    physical = mask & (xx <= 3)
    result = dict(
        comparison_type="NS Re=20 with sponge versus unforced Stokes without sponge: model difference, not numerical error",
        ns_vs_stokes_full_box=metrics(u, v, w, ref, mask),
        ns_vs_stokes_physical_region=metrics(u, v, w, ref, physical),
    )
    print("MODEL_DIFFERENCE", json.dumps(result), flush=True)
    # With no inertia or sponge, the hybrid solution is its analytic Stokes
    # lift. This checks the lift alone, not the NS volume/time discretization.
    hole = EllipticHole()
    ext = RationalStokesExtension((-1, 5, 1), hole)
    coeff = ext.response(-np.concatenate(channel_lift(ext.hole_points)[1:3]))
    pts = np.column_stack((xx[mask], yy[mask]))
    lift = [a + b for a, b in zip(channel_lift(pts), ext.evaluate(pts, coeff))]
    su, sv, sw = [np.full(xx.shape, np.nan) for _ in range(3)]
    su[mask], sv[mask], sw[mask] = lift[1], lift[2], lift[5] - lift[4]
    result["matched_stokes_lift_error"] = metrics(su, sv, sw, ref, mask)
    result["matched_stokes_scope"] = "Same no-inertia, no-sponge boundary-value problem; tests rational lift only"
    convergence = json.loads((root / "lightning/convergence.json").read_text())
    result["reference_96_to_120"] = {
        k: convergence[-1][k]
        for k in ("successive_velocity_max", "successive_vorticity_max")
    }
    np.savez_compressed(out / "matched_stokes.npz", x=x, y=y, u=su, v=sv, vorticity=sw)
    (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print("COMPLETE", json.dumps(result), flush=True)
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), layout="constrained")
    velocity_difference = np.where(mask, np.hypot(u-ref[0], v-ref[1]), np.nan)
    omega_difference = np.where(mask, w-ref[3], np.nan)
    speed_limit = max(np.nanmax(np.hypot(u, v)), np.nanmax(np.hypot(ref[0], ref[1])))
    omega_limit = max(np.nanmax(abs(w)), np.nanmax(abs(ref[3])))
    rows = (
        (np.hypot(ref[0], ref[1]), ref[3], "AAA-lightning: Stokes, sponge OFF"),
        (np.hypot(u, v), w, "Hybrid BSPF: NS Re=20, sponge ON, t=20"),
        (velocity_difference, omega_difference, "NS minus Stokes: MODEL DIFFERENCE"),
    )
    for row, (speed, omega, title) in enumerate(rows):
        for col, (field, cmap) in enumerate(((speed, "viridis"), (omega, "RdBu_r"))):
            ax = axes[row, col]
            limit = (speed_limit if col == 0 else omega_limit) if row < 2 else np.nanmax(abs(field))
            im = ax.pcolormesh(x, y, np.where(mask, field, np.nan), shading="auto", cmap=cmap, vmin=0 if col == 0 else -limit, vmax=limit)
            ax.add_patch(Ellipse(hole.center, *(2*np.array(hole.axes)), facecolor=".7", edgecolor="k"))
            ax.axvline(3, color="k", ls="--", lw=0.8)
            ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
            fig.colorbar(im, ax=ax, label=("Speed" if row < 2 else "Velocity difference magnitude") if col == 0 else "Vorticity")
    fig.suptitle("Same geometry and evaluation grid; different governing equations / sponge")
    fig.savefig(out / "ns_vs_stokes.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
