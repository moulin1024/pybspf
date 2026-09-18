"""Render the dense-interface C2 extension experiment and independent errors."""

import pickle
from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from bspf_jax.embedded_poisson import benchmark_domains
from bspf_jax.random_wave_mms import RandomWaveMMS
from bspf_jax.stream_navier_stokes import stream_evaluate_line

jax.config.update("jax_enable_x64", True)


def main():
    out = Path("build/embedded_poisson_smooth_extension_dense")
    source = Path("build/embedded_poisson_smooth_extension_n49")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    x = np.linspace(-1.2, 1.2, 181)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    exact = (
        RandomWaveMMS.create(kmax=12 * np.pi)
        .evaluate(np.column_stack((xx.ravel(), yy.ravel())))[0]
        .reshape(xx.shape)
    )
    for row, domain in enumerate(benchmark_domains()):
        with (source / f"{domain.name}_geometry.pkl").open("rb") as handle:
            geom = pickle.load(handle)
        data = np.load(out / f"{domain.name}_kmax_12pi_C2.npz")
        b = stream_evaluate_line(geom.line, x)[0]
        n = b.shape[1]
        u = b @ data["solution"].reshape(n, n) @ b.T
        xi = b @ data["extension"].reshape(n, n) @ b.T
        mask = np.zeros_like(xx, dtype=bool)
        for i, value in enumerate(x):
            hits = domain.intersections(value)
            for lower, upper in hits:
                mask[i] |= (x > lower) & (x < upper)
        curve = domain.curve(np.linspace(0, domain.period, 1001))
        t = np.linspace(0, domain.period, 24, endpoint=False)
        p, normal = domain.curve(t), domain.normal(t)
        for ax in axes[row, :2]:
            ax.plot(*curve.T, "k-", lw=1)
            ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect="equal")
        im = axes[row, 0].pcolormesh(
            xx,
            yy,
            np.where(mask, u, xi),
            shading="auto",
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
        )
        axes[row, 0].quiver(*p.T, *normal.T, scale=14, width=0.003)
        axes[row, 0].set_title(f"{domain.name}: u inside / extension outside")
        fig.colorbar(im, ax=axes[row, 0], shrink=0.8)
        im = axes[row, 1].pcolormesh(
            xx,
            yy,
            np.ma.masked_where(~mask, np.maximum(abs(u - exact), 1e-8)),
            shading="auto",
            cmap="magma",
            norm=LogNorm(1e-6, 0.3),
        )
        axes[row, 1].set_title("Physical absolute error (independent grid)")
        fig.colorbar(im, ax=axes[row, 1], shrink=0.8)
        from bspf_jax.smooth_extension import basis_operators

        bp, _, normals = domain.boundary_rule(20)
        _, _, dn, _ = basis_operators(geom.line, bp, normals)
        jump = dn @ (data["solution"] - data["extension"])
        s = np.arange(len(bp)) / len(bp)
        axes[row, 2].semilogy(
            s, np.maximum(abs(jump), 1e-14), label="normal matching error"
        )
        axes[row, 2].semilogy(
            s,
            np.maximum(abs(data["normal_derivative_error"]), 1e-14),
            alpha=0.8,
            label="physical normal derivative error",
        )
        axes[row, 2].set(
            xlabel="Boundary sample index / count",
            ylabel="Absolute error",
            title="Independent interface samples",
        )
        axes[row, 2].legend(fontsize=8)
        axes[row, 2].grid(alpha=0.2)
    fig.suptitle("BSPF unknown-field C2 extension | N=49 | random-wave MMS, kmax=12 pi")
    fig.savefig(out / "smooth_extension.png", dpi=180)
    fig.savefig(out / "smooth_extension.pdf")


if __name__ == "__main__":
    main()
