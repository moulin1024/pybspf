"""Render saved Solov'ev GS convergence, flux surfaces and field errors."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("build/solovev_bspf"))
    args = parser.parse_args()
    report = json.loads((args.out/"results.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    for case in report["profiles"]:
        rows = [row for row in report["runs"] if row["case"] == case]
        for ax, key, title in zip(axes, ("flux_relative_l2", "poloidal_field_relative_l2", "gs_relative_residual"),
                                  ("Flux error", "Poloidal magnetic-field error", "Grad-Shafranov residual")):
            ax.semilogy([r["nodes"] for r in rows], [r["errors"][key] for r in rows], "o-", label=case)
            ax.set(title=title, xlabel="BSPF nodes per axis", ylabel="Independent relative L2", xticks=report["settings"]["nodes"])
            ax.grid(alpha=0.2)
            ax.legend()
    fig.suptitle("Fixed-boundary Solov'ev equilibria: exact flux-surface geometry, BSPF fields")
    fig.savefig(args.out/"convergence.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for row, case in enumerate(report["profiles"]):
        n = max(r["nodes"] for r in report["runs"] if r["case"] == case)
        data = np.load(args.out/f"{case}_n{n}.npz")
        mask, r, z = data["mask"], data["raxis"], data["zaxis"]
        def grid(values):
            result = np.full(mask.shape, np.nan)
            result[mask] = values
            return np.ma.masked_invalid(result.T)
        axis_flux = report["profiles"][case]["axis_flux"]
        flux = grid(data["flux"])
        exact = grid(data["exact_flux"])
        im = axes[row, 0].pcolormesh(r, z, flux, shading="auto", cmap="viridis")
        levels = np.linspace(axis_flux/10, axis_flux*0.9, 7)
        axes[row, 0].contour(r, z, exact, levels=levels, colors="black", linewidths=0.7)
        axes[row, 0].contour(r, z, flux, levels=levels, colors="white", linewidths=0.7, linestyles="dashed")
        fig.colorbar(im, ax=axes[row, 0], label="Flux")
        flux_error = np.maximum(np.abs(data["flux"]-data["exact_flux"])/axis_flux, 1e-18)
        field_error = np.linalg.norm(data["poloidal_field"]-data["exact_poloidal_field"], axis=1)
        field_error /= np.sqrt(np.mean(np.sum(data["exact_poloidal_field"]**2, axis=1)))
        field_error = np.maximum(field_error, 1e-18)
        for col, values, title in ((1, flux_error, "Flux error / axis flux"),
                                   (2, field_error, "Poloidal B error / exact RMS")):
            upper = max(float(np.max(values)), 1e-17)
            im = axes[row, col].pcolormesh(r, z, grid(values), shading="auto", cmap="magma",
                                            norm=LogNorm(vmin=max(1e-18, upper*1e-4), vmax=upper))
            axes[row, col].set_title(title)
            fig.colorbar(im, ax=axes[row, col])
        axes[row, 0].set_title(f"{case}, N={n}\nExact contours: black; BSPF: white dashed")
        for ax in axes[row]:
            ax.set(xlabel="R", ylabel="Z", aspect="equal")
            edge = data["boundary_points"]
            ax.plot(np.r_[edge[:, 0], edge[0, 0]], np.r_[edge[:, 1], edge[0, 1]], color="tab:blue", lw=1)
            ax.plot(report["profiles"][case]["major_radius"], 0, "+", color="tab:red", ms=6)
    fig.savefig(args.out/"equilibria.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
