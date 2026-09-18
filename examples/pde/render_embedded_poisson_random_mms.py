"""Visualize random-wave spectra and curved-domain Poisson convergence."""

import json
from pathlib import Path

import jax
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

from bspf_jax.embedded_poisson import background_line, benchmark_domains
from bspf_jax.random_wave_mms import RandomWaveMMS
from bspf_jax.stream_navier_stokes import stream_evaluate_line

jax.config.update("jax_enable_x64", True)


def load_mms(out, name):
    data = np.load(out / f"{name}_spectrum.npz")
    return RandomWaveMMS(**{key: data[key] for key in data.files})


def main():
    out = Path("build/embedded_poisson_random_mms")
    summary = json.loads((out / "summary.json").read_text())
    nodes = max(r["nodes"] for r in summary["results"])
    grid = -1 + 2 * (np.arange(200) + 0.5) / 200
    x, y = np.meshgrid(grid, grid, indexing="ij")
    b = stream_evaluate_line(background_line(nodes), grid)[0]
    mms = load_mms(out, "kmax_12pi")
    exact = mms.evaluate(np.column_stack((x.ravel(), y.ravel())))[0].reshape(x.shape)
    records = []
    for domain in benchmark_domains():
        coefficient = np.load(out / f"{domain.name}_kmax_12pi_solution.npz")[
            "coefficient"
        ].reshape(nodes, nodes)
        value = b @ coefficient @ b.T
        mask = np.zeros_like(x, dtype=bool)
        for i, xi in enumerate(grid):
            for lo, hi in domain.intersections(xi):
                mask[i] |= (grid > lo) & (grid < hi)
        records.append((domain, value, abs(value - exact), mask))
    maximum = max(np.max(abs(v[mask])) for _, v, _, mask in records)
    emax = max(np.max(error[mask]) for _, _, error, mask in records)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row, (domain, value, error, mask) in enumerate(records):
        curve = domain.curve(np.linspace(0, domain.period, 1601))
        for ax, field, norm, cmap, title in [
            (
                axes[row, 0],
                value,
                TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum),
                "RdBu_r",
                f"{domain.name}: numerical solution",
            ),
            (
                axes[row, 1],
                np.maximum(error, 1e-16),
                LogNorm(max(emax * 1e-6, 1e-16), emax),
                "magma",
                "Absolute error (shared scale)",
            ),
        ]:
            im = ax.imshow(
                np.where(mask, field, np.nan).T,
                extent=(-1, 1, -1, 1),
                origin="lower",
                norm=norm,
                cmap=cmap,
                interpolation="nearest",
            )
            ax.plot(*curve.T, color="#263447", lw=0.9)
            ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
            fig.colorbar(im, ax=ax, shrink=0.82)
        ax = axes[row, 2]
        for band in [4, 8, 12]:
            rows = sorted(
                [
                    r
                    for r in summary["results"]
                    if r["domain"] == domain.name and r["case"] == f"kmax_{band}pi"
                ],
                key=lambda r: r["nodes"],
            )
            ax.semilogy(
                [r["nodes"] for r in rows],
                [r["relative_l2"] for r in rows],
                "o-",
                label=rf"$k_{{max}}={band}\pi$",
            )
        ax.set(
            title="Independent relative solution error",
            xlabel="Full BSPF factors per direction",
            ylabel="relative L2",
            xticks=sorted({r["nodes"] for r in summary["results"]}),
        )
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(
        rf"Random-wave Poisson MMS: 64 waves, seed {summary['seed']}", fontsize=16
    )
    fig.supxlabel(
        rf"Field panels: $k_{{max}}=12\pi$, {nodes} x {nodes} candidate space. Spectrum is prescribed; no turbulent dynamics simulated.",
        fontsize=10,
    )
    fig.savefig(out / "random_wave_validation.png", dpi=180)
    fig.savefig(out / "random_wave_validation.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    k = mms.wavevectors / np.pi
    im = axes[0].scatter(
        *np.vstack((k, -k)).T, c=np.tile(mms.amplitudes, 2), cmap="viridis", s=28
    )
    axes[0].set(
        title="Continuous random wavevectors (both signs)",
        xlabel=r"$k_x/\pi$",
        ylabel=r"$k_y/\pi$",
        aspect="equal",
    )
    fig.colorbar(im, ax=axes[0], label="cosine amplitude")
    for band in [4, 8, 12]:
        m = load_mms(out, f"kmax_{band}pi")
        power = np.bincount(m.shell_indices, weights=m.amplitudes**2 / 2)
        center = np.sqrt(m.shell_edges[1:] * m.shell_edges[:-1])
        axes[1].loglog(
            center / np.pi,
            power / np.diff(m.shell_edges),
            "o-",
            label=rf"$k_{{max}}={band}\pi$",
        )
    ref = np.geomspace(1, 12, 100)
    axes[1].loglog(ref, 0.35 * ref ** (-5 / 3), "k--", lw=1, label=r"$k^{-5/3}$ guide")
    axes[1].set(
        title="Phase-ensemble scalar variance spectrum",
        xlabel=r"$k/\pi$",
        ylabel="shell variance / shell width",
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=9)
    fig.savefig(out / "random_wave_spectrum.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
