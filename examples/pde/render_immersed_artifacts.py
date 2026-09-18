"""Identical grids and color scales for the fixed-space NS wall experiments."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

root = Path("build/immersed_flow")
fig, axes = plt.subplots(3, 1, figsize=(11, 10), layout="constrained")
records = []
for ax, key, label in zip(
    axes,
    ["dense", "wall_rank_1e5", "factor"],
    [
        "Original wall SVD (rcond 1e-10)",
        "Relaxed wall SVD (rcond 1e-5)",
        "Analytic wall factor + BSPF",
    ],
):
    data = np.load(root / key / "fields.npz")
    x, y = data["x"], data["y"]
    xx, yy = np.meshgrid(x, y)
    u, v, w, psi = data["fields"][-1]
    hole = ((xx - 0.19) / 0.31) ** 2 + ((yy + 0.13) / 0.23) ** 2 <= 1
    im = ax.pcolormesh(
        x,
        y,
        np.ma.array(w - 2 * yy, mask=hole),
        cmap="RdBu_r",
        vmin=-22,
        vmax=22,
        shading="auto",
    )
    ax.add_patch(Ellipse((0.19, -0.13), 0.62, 0.46, facecolor=".65", edgecolor="k"))
    ax.set(
        xlim=(-1, 3), ylim=(-1, 1), xlabel="x", ylabel="y", title=label, aspect="equal"
    )
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    lap = np.zeros_like(w[3:-3, 3:-3])
    wx = lap.copy()
    wy = lap.copy()
    c2 = [1 / 90, -3 / 20, 1.5, -49 / 18, 1.5, -3 / 20, 1 / 90]
    c1 = np.array([-1, 9, -45, 0, 45, -9, 1]) / 60
    for k in range(7):
        sx = w[3:-3, k : k + len(x) - 6]
        sy = w[k : k + len(y) - 6, 3:-3]
        lap += c2[k] * (sx / dx**2 + sy / dy**2)
        wx += c1[k] * sx / dx
        wy += c1[k] * sy / dy
    xr, yr = xx[3:-3, 3:-3], yy[3:-3, 3:-3]
    mask = (
        (xr > -0.85)
        & (xr < 2.5)
        & (abs(yr) < 0.85)
        & (((xr - 0.19) / 0.41) ** 2 + ((yr + 0.13) / 0.33) ** 2 > 1)
    )
    residual = u[3:-3, 3:-3] * wx + v[3:-3, 3:-3] * wy - (0.46 * (2 / 3) / 20) * lap
    summary = json.loads((root / key / "summary.json").read_text())
    records.append(
        dict(
            method=key,
            steady_vorticity_residual_fd6_rms=float(
                np.sqrt(np.mean(residual[mask] ** 2))
            ),
            steady_vorticity_residual_fd6_max=float(np.max(abs(residual[mask]))),
        )
    )
fig.suptitle("Same BSPF 73 x 33, Re=20, t=20; no filtering or added viscosity")
fig.colorbar(im, ax=axes, label="Vorticity minus baseline shear", shrink=0.8)
fig.savefig(root / "artifact_diagnosis" / "ns_fixed_grid.png", dpi=160)
(root / "artifact_diagnosis" / "ns_curl_residual.json").write_text(
    json.dumps(records, indent=2)
)
print(json.dumps(records, indent=2))
