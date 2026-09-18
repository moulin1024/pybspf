"""Same continuous PDE and Dirichlet data: BSPF vs FD error curves."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

root = Path("build/bspf3d_continuous")
rows = json.loads((root / "results.json").read_text())
fd = json.loads(Path("build/pressure3d_pyamg_cg/results.json").read_text())
ns = np.array([r["n"] for r in rows])
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.7), layout="constrained")
for backend, label, color, style in [
    ("dense", "BSPF dense", "#2463ac", "o-"),
    ("compressed", "BSPF DCT + blocks", "#db7432", "s--"),
]:
    e = np.array([r["results"][backend]["error_linf"] for r in rows])
    ax[0].loglog(ns, e, style, color=color, label=label, markerfacecolor="none", ms=7)
    ax[1].semilogy(ns, e, style, color=color, label=label, markerfacecolor="none", ms=7)
fn = np.array([r["n"] for r in fd])
fe = np.array([r["cases"]["smooth_continuous"]["error_linf"] for r in fd])
ax[0].loglog(fn, fe, "^-", color="#27825b", label="2nd-order FD + AMG-CG")
ax[0].loglog(
    fn,
    fe[0] * ((fn[0] - 1) / (fn - 1)) ** 2,
    ":",
    color="#555555",
    label="h² reference",
)
ax[0].set_title("Same PDE: maximum pressure error")
ax[0].set_xticks([40, 64, 96, 128, 192, 256], [40, 64, 96, 128, 192, 256])
ax[0].xaxis.set_minor_locator(NullLocator())
ax[1].set_title("BSPF detail: semilog scale")
ax[1].set_xticks([40, 64, 96, 128, 192, 256])
for a in ax:
    a.set_xlabel("Points per axis N")
    a.set_ylabel("Maximum error against continuous exact solution")
    a.grid(alpha=0.25)
    a.legend(fontsize=9)
fig.suptitle(
    "3D Poisson with exact Dirichlet data: analytic forcing / zero BSPF refinement\nBSPF uses a separate D1@D1 Dirichlet closure, not the masked pressure core",
    fontsize=12,
)
fig.savefig(root / "continuous_error.png", dpi=200)
fig.savefig(root / "continuous_error.pdf")
