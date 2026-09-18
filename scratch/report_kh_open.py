"""Saved-data comparison of fixed and open KH boundaries, with honest color limits."""

from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from report_kh_stream import render_movie

root = Path("build/kh_stream")
fixed = np.load(root / "final96/frames.npz")
opened = np.load(root / "open96/frames.npz")
fig, axes = plt.subplots(3, 1, figsize=(11, 8), layout="constrained")
for ax, f, t, label in zip(
    axes,
    [fixed, opened, opened],
    [6, 6, 12],
    ["Fixed vertical velocity", "Open vertical faces", "Open vertical faces"],
):
    i = np.argmin(abs(f["times"] - t))
    om = f["vorticity"][i]
    im = ax.imshow(
        om.T,
        origin="lower",
        extent=[-3, 3, -1, 1],
        aspect="auto",
        cmap="RdBu_r",
        vmin=-12,
        vmax=12,
        interpolation="nearest",
    )
    ax.set(
        title=f"{label}, t={t}; actual max |vorticity|={abs(om).max():.2f}",
        xlabel="x",
        ylabel="y",
    )
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle(
    "96 x 80 BSPF | same viscosity and initial seed | colors clipped at +/-12\nHorizontal boundaries remain fixed; no sponge or filter"
)
fig.savefig(root / "open_boundary_comparison.png", dpi=150)
plt.close(fig)
coarse = np.load(root / "open64/frames.npz")
fine = np.load(root / "open64_dt002/frames.npz")
i = np.argmin(abs(fine["times"] - 6))
result = {
    "dt_comparison_64_t6": {
        "transverse_velocity_nodal_linf": float(
            abs(
                coarse["transverse_velocity"][-1] - fine["transverse_velocity"][i]
            ).max()
        ),
        "vorticity_nodal_linf": float(
            abs(coarse["vorticity"][-1] - fine["vorticity"][i]).max()
        ),
        "dt_coarse": 0.004,
        "dt_fine": 0.002,
    },
    "open96": json.loads((root / "open96/summary.json").read_text()),
}
(root / "open_validation.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
render_movie(root / "open96")
