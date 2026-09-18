"""Render the recorded KH instability diagnostics; no simulation reruns."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("build/kh_stability")


def read(name):
    return json.loads((root / name).read_text())


fig, ax = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
for variant, dt, label, color, style in [
    ("none", 0.002, "No sponge, dt=0.002", "#d65c35", "-"),
    ("none", 0.001, "No sponge, dt=0.001", "#385aac", "--"),
    ("sponge_x", 0.002, "x-only sponge, dt=0.002", "#27825b", "-"),
]:
    rows = read(f"evolve_{variant}_160x112_dt{dt}.json")["records"]
    ax[0, 0].semilogy(
        [r["t"] for r in rows],
        [r["edge_rms"] for r in rows],
        style,
        color=color,
        label=label,
        lw=2,
    )
ax[0, 0].set(
    title="Boundary-strip perturbation growth",
    xlabel="Time",
    ylabel="RMS of velocity perturbation",
)
ax[0, 0].legend(fontsize=9)
ax[0, 0].grid(alpha=0.25)
for variant, label, color, marker in [
    ("none", "KH, no sponge", "#d65c35", "o"),
    ("sponge", "KH, full sponge", "#27825b", "s"),
    ("uniform", "Uniform flow, no sponge", "#8053a4", "x"),
]:
    rows = read(f"spectrum_{variant}_160x112_dt0.002.json")["modes"]
    ax[0, 1].scatter(
        [r["real"] for r in rows],
        [r["imag"] for r in rows],
        label=label,
        color=color,
        marker=marker,
        s=48,
    )
ax[0, 1].axvline(0, color="gray", lw=1)
ax[0, 1].set(
    title="Six leading linearized NS eigenvalues",
    xlabel="Real part: growth rate",
    ylabel="Imaginary part: oscillation frequency",
    xlim=(-0.5, 9),
)
ax[0, 1].legend(fontsize=9, loc="upper left")
ax[0, 1].grid(alpha=0.25)
for a, variant, title in [
    (ax[1, 0], "none", "No sponge: boundary-localized mode"),
    (ax[1, 1], "sponge", "Full sponge: central shear-layer mode"),
]:
    d = np.load(root / f"{variant}_160x112_eigenmodes.npz")
    i = np.argmax(d["values"].real)
    v = d["vectors"][:, i].reshape(len(d["x"]) - 2, len(d["y"]) - 2, 2)
    amp = np.sqrt(np.sum(abs(v) ** 2, axis=-1))
    amp = amp / amp.max()
    im = a.imshow(
        np.log10(np.maximum(amp, 1e-5)).T,
        origin="lower",
        extent=[d["x"][1], d["x"][-2], d["y"][1], d["y"][-2]],
        aspect="auto",
        cmap="magma",
        vmin=-5,
        vmax=0,
    )
    lam = d["values"][i]
    a.set(title=f"{title}\nRe(λ) = {lam.real:.3f}", xlabel="x", ylabel="y")
    fig.colorbar(im, ax=a, label="log10(normalized mode amplitude)", shrink=0.85)
fig.suptitle(
    "Why the KH example needed a sponge: unstable boundary advection closure\n160 × 112 / ν=0.002 / original fixed-velocity boundaries",
    fontsize=14,
)
fig.savefig(root / "diagnosis.png", dpi=180)
fig.savefig(root / "diagnosis.pdf")

summary = {}
a = read("evolve_none_160x112_dt0.002.json")
h = read("evolve_none_160x112_dt0.001.json")
c = read("evolve_sponge_x_160x112_dt0.002.json")
summary["nonlinear_final"] = {
    k: v["records"][-1] for k, v in [("no_sponge", a), ("half_dt", h), ("x_sponge", c)]
}
for variant in ["none", "sponge", "uniform"]:
    d = np.load(root / f"{variant}_160x112_eigenmodes.npz")
    i = np.argmax(d["values"].real)
    x, y = np.meshgrid(d["x"][1:-1], d["y"][1:-1], indexing="ij")
    v = d["vectors"][:, i].reshape(x.shape + (2,))
    e = np.sum(abs(v) ** 2, axis=-1)
    summary[variant + "_leading_mode"] = {
        key: float(e[mask].sum() / e.sum())
        for key, mask in [
            ("x_boundary_fraction", abs(x) > 2.4),
            ("y_boundary_fraction", abs(y) > 0.8),
            (
                "inflow_fraction",
                (x < -2.4)
                if variant == "uniform"
                else (((x > 2.4) & (y < 0)) | ((x < -2.4) & (y > 0))),
            ),
            ("center_fraction", (abs(x) < 2) & (abs(y) < 0.5)),
        ]
    }
    lam = d["values"][i]
    summary[variant + "_leading_mode"].update(
        real=float(lam.real), imag=float(lam.imag)
    )
    if variant == "none":
        summary["rk4_effective_growth"] = {}
        for dt in [0.002, 0.001]:
            z = dt * lam
            R = 1 + z + z * z / 2 + z**3 / 6 + z**4 / 24
            summary["rk4_effective_growth"][str(dt)] = float(np.log(abs(R)) / dt)
# Half-step differences before the rapid nonlinear blow-up and at the stop time.
a = np.load(root / "none_160x112_dt0.002_state.npz")
h = np.load(root / "none_160x112_dt0.001_state.npz")
summary["half_dt_relative_perturbation_difference_T2"] = float(
    np.linalg.norm(a["frames"][4] - h["frames"][4]) / np.linalg.norm(h["frames"][4])
)
summary["half_dt_relative_perturbation_difference_stop"] = float(
    np.linalg.norm(a["velocity"] - h["velocity"])
    / np.linalg.norm(a["velocity"] - a["base"])
)
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
