"""Ablate advection/diffusion stencils in the stable continuous scalar problem."""

import json
from pathlib import Path
import jax
import numpy as np
from scipy.linalg import eig
from diagnose_kh_stability import make

jax.config.update("jax_enable_x64", True)
plan, _, _ = make(160, 112)
D = np.asarray(plan.dx)
D2 = np.asarray(plan.dxx)
x = np.asarray(plan.pressure.x.x)
n = len(x)
h = x[1] - x[0]
m = n - 2
nu = 0.002
F1 = (np.diag(np.ones(m - 1), 1) - np.diag(np.ones(m - 1), -1)) / (2 * h)
F2 = (np.diag(np.ones(m - 1), 1) + np.diag(np.ones(m - 1), -1) - 2 * np.eye(m)) / h**2
interior = np.s_[1:-1, 1:-1]
cases = {
    "BSPF_D1_BSPF_D2": -D[interior] + nu * D2[interior],
    "BSPF_D1_D1_squared": -D[interior] + nu * (D @ D)[interior],
    "FD_D1_FD_D2": -F1 + nu * F2,
    "BSPF_D1_FD_D2": -D[interior] + nu * F2,
    "FD_D1_BSPF_D2": -F1 + nu * D2[interior],
}
for rate in (10, 20, 40, 80):
    cases[f"BSPF_sponge_{rate}"] = (
        -D[interior] + nu * D2[interior] - np.diag(rate * (x[1:-1] / 3) ** 16)
    )
rows = {}
modes = {}
for name, A in cases.items():
    lam, v = eig(A)
    idx = np.argmax(lam.real)
    mode = v[:, idx]
    rows[name] = dict(
        max_real=float(lam[idx].real),
        imag=float(lam[idx].imag),
        positive_count=int(np.sum(lam.real > 1e-8)),
        peak_x=float(x[1:-1][np.argmax(abs(mode))]),
        left_boundary_fraction=float(
            np.sum(abs(mode[x[1:-1] < -2.4]) ** 2) / np.sum(abs(mode) ** 2)
        ),
    )
    modes[name + "_values"] = lam
    modes[name + "_mode"] = mode
root = Path("build/kh_stability")
(root / "line_ablation.json").write_text(json.dumps(rows, indent=2))
np.savez_compressed(root / "line_modes.npz", x=x, dx=D, dxx=D2, **modes)
print(json.dumps(rows, indent=2))

# Diagnostic row replacement, not a production NS discretization change.
row_ablation = []
for side in ("inflow", "outflow", "both"):
    for k in (1, 2, 4, 8, 16):
        hybrid = D[interior].copy()
        if side in ("inflow", "both"):
            hybrid[:k] = F1[:k]
        if side in ("outflow", "both"):
            hybrid[-k:] = F1[-k:]
        lam = np.linalg.eigvals(-hybrid + nu * D2[interior])
        row_ablation.append(
            dict(side=side, rows_per_side=k, max_real=float(lam.real.max()))
        )
(root / "boundary_row_ablation.json").write_text(json.dumps(row_ablation, indent=2))
