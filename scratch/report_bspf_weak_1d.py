"""Plot stability and continuous-PDE accuracy of the unchanged BSPF trial space."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("build/bspf_advection_diffusion_1d")
records = json.loads((root / "mp113/results.json").read_text())
ordinary = {
    r["n"]: r
    for r in json.loads((root / "results.json").read_text())
    if r["window"] == 16
}
n = np.array([r["n"] for r in records])
reference = next(r for r in records if r["n"] == 160)
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
ax = axes[0, 0]
for method, color, label in [
    ("strong", "#dc2626", "Original BSPF collocation"),
    ("weak", "#2563eb", "Weak BSPF, same trial space"),
]:
    ax.plot(
        n,
        [r["spectra"][method]["max_eigenvalue_real"] for r in records],
        "o-",
        color=color,
        label=label,
    )
ax.axhline(0, color="#64748b", ls="--", lw=1)
ax.set(
    xlabel="Uniform nodes N",
    ylabel="max Re(lambda)",
    title="Eliminating the unstable boundary modes",
)
ax.legend(fontsize=9)
ax = axes[0, 1]
for case, color, label in [
    ("nonperiodic", "#2563eb", "Tapered exp(x/3) cos(pi x)"),
    ("oscillatory", "#9333ea", "Two sine modes, k=1 and 7"),
    ("gaussian", "#059669", "Tapered Gaussian"),
]:
    ax.semilogy(
        n,
        [r["cases"][case]["weak"]["transient"][1]["nodal_linf"] for r in records],
        "o-",
        color=color,
        label=label,
    )
ax.set(
    xlabel="Uniform nodes N",
    ylabel="Max nodal PDE error at t=3",
    title="BSPF high-order accuracy is retained",
    ylim=(3e-16, 0.1),
)
ax.legend(fontsize=8)
ax = axes[1, 0]
s = reference["mass_semigroup"]
ax.semilogy(s["times"], s["strong"], color="#dc2626", label="Original collocation")
ax.semilogy(s["times"], s["weak"], color="#2563eb", label="Weak BSPF")
ax.set(
    xlabel="Time",
    ylabel="Induced M-norm of exp(t A)",
    title="Energy stability: no hidden transient amplification",
)
ax.legend(fontsize=9)
ax = axes[1, 1]
ax.semilogy(
    n,
    [
        ordinary[v]["cases"]["nonperiodic"]["weak"]["transient"][1]["nodal_linf"]
        for v in n
    ],
    "o--",
    color="#ea580c",
    label="Naive float64 basis assembly",
)
ax.semilogy(
    n,
    [r["cases"]["nonperiodic"]["weak"]["transient"][1]["nodal_linf"] for r in records],
    "o-",
    color="#2563eb",
    label="113-bit basis setup; float64 solve",
)
ax.set(
    xlabel="Uniform nodes N",
    ylabel="Max nodal PDE error at t=3",
    title="Separating cancellation error from instability",
    ylim=(3e-16, 0.001),
)
ax.legend(fontsize=8)
for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle(
    "1D advection-diffusion | same BSPF q=9, degree=13, Chebyshev 12/16 | no added dissipation",
    fontsize=13,
)
fig.savefig(root / "comparison.png", dpi=170)
fig.savefig(root / "comparison.pdf")
summary = dict(
    reference_n160=reference,
    convergence=[
        dict(
            n=r["n"],
            max_real=r["spectra"]["weak"]["max_eigenvalue_real"],
            nonperiodic_linf=r["cases"]["nonperiodic"]["weak"]["transient"][1][
                "nodal_linf"
            ],
            oscillatory_linf=r["cases"]["oscillatory"]["weak"]["transient"][1][
                "nodal_linf"
            ],
            gaussian_linf=r["cases"]["gaussian"]["weak"]["transient"][1]["nodal_linf"],
        )
        for r in records
    ],
    max_energy_identity_relative=max(r["energy_identity_relative"] for r in records),
    max_raw_integration_by_parts_defect=max(
        r["raw_integration_by_parts_defect"] for r in records
    ),
    max_mass_condition=max(r["mass_condition"] for r in records),
    all_cases_finite=all(
        np.isfinite(r["cases"][c]["weak"]["transient"][j]["continuous_l2"])
        for r in records
        for c in r["cases"]
        for j in range(3)
    ),
)
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != "reference_n160"}, indent=2))
