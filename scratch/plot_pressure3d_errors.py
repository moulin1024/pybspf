"""Error/convergence plots from recorded benchmarks; no solver reruns."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

out = Path("build/pressure3d_pyamg")
v = json.loads((out / "results.json").read_text())
cg = json.loads(Path("build/pressure3d_pyamg_cg/results.json").read_text())
bspf = json.loads(Path("build/pressure3d_benchmark/results.json").read_text())
ns = np.array([r["n"] for r in v])
colors = ["#2463ac", "#dc7935", "#27825b", "#8053a4"]
plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), constrained_layout=True)

ax = axes[0, 0]
orders = {}
for records, label, color, style in [
    (v, "FD + AMG V", colors[2], "o-"),
    (cg, "FD + AMG-CG", colors[3], "s--"),
]:
    e = np.array([r["cases"]["smooth_continuous"]["error_linf"] for r in records])
    ax.loglog(
        ns, e, style, color=color, label=label, markerfacecolor="none", markersize=8
    )
    orders[label] = (
        np.log(e[:-1] / e[1:]) / np.log((ns[1:] - 1) / (ns[:-1] - 1))
    ).tolist()
e0 = cg[0]["cases"]["smooth_continuous"]["error_linf"]
ax.loglog(
    ns,
    e0 * ((ns[0] - 1) / (ns - 1)) ** 2,
    ":",
    color="#333333",
    label="Second-order reference: h²",
    zorder=0,
)
ax.set_title("A  Continuous PDE error: FD only", loc="left")
ax.text(
    0.03,
    0.08,
    "Measured orders (AMG-CG): " + ", ".join(f"{p:.2f}" for p in orders["FD + AMG-CG"]),
    transform=ax.transAxes,
    fontsize=10,
)
ax.set_ylabel("Maximum pressure error")
ax.legend(fontsize=9)

for ax, kind, title in [
    (axes[0, 1], "smooth", "B  Discrete recovery: smooth pressure"),
    (axes[1, 0], "random", "C  Discrete recovery: random pressure"),
]:
    for backend, label, color, style in [
        ("dense", "BSPF dense", colors[0], "o-"),
        ("compressed", "BSPF compressed", colors[1], "s--"),
    ]:
        data = [r for r in bspf if r["backend"] == backend]
        ax.loglog(
            ns,
            [r["accuracy"][kind]["error_linf"] for r in data],
            style,
            color=color,
            label=label,
            markerfacecolor="none",
            markersize=7,
        )
    for records, label, color, style in [
        (v, "FD + AMG V", colors[2], "^-"),
        (cg, "FD + AMG-CG", colors[3], "D--"),
    ]:
        ax.loglog(
            ns,
            [r["cases"][kind + "_discrete"]["error_linf"] for r in records],
            style,
            color=color,
            label=label,
            markerfacecolor="none",
            markersize=6,
        )
    ax.set_title(title, loc="left")
    ax.set_ylabel("Maximum pressure recovery error")
    ax.legend(fontsize=9)

ax = axes[1, 1]
for i, n in enumerate(ns):
    for records, label, style in [(v, "V", "o-"), (cg, "CG", "s--")]:
        row = next(r for r in records if r["n"] == n)
        hist = row["cases"]["smooth_discrete"]["residual_histories"][-1]
        ax.semilogy(
            range(len(hist)),
            hist,
            style,
            color=colors[i],
            markersize=3,
            label=f"{n}³ / {label}",
        )
ax.axhline(1e-10, color="gray", ls=":", label="Relative target ~1e-10")
ax.set_title("D  AMG residual: smooth discrete RHS", loc="left")
ax.set_xlabel("V-cycle or preconditioned CG iteration")
ax.set_ylabel("Relative L2 residual (not pressure error)")
ax.legend(fontsize=8, ncol=2)
for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Points per axis N; h = 1/(N-1)")
for ax in axes.flat:
    ax.grid(which="major", alpha=0.25)
fig.suptitle(
    "3D pressure error and convergence / CPU double precision\nBSPF: zero refinement; FD: 7-point Dirichlet + PyAMG",
    fontsize=14,
)
fig.get_layout_engine().set(rect=(0, 0.065, 1, 0.84))
fig.text(
    0.5,
    0.015,
    "A: continuous Dirichlet problem.  B–C: each method’s own discrete system; boundary treatments differ.\nB–C do not measure PDE discretization convergence.  BSPF is direct and has no iteration history.",
    ha="center",
    fontsize=9,
    color="#444444",
)
fig.savefig(out / "error_convergence.png", dpi=200)
fig.savefig(out / "error_convergence.pdf")
(out / "observed_fd_orders.json").write_text(json.dumps(orders, indent=2))
print(json.dumps(orders, indent=2))
