"""Render full-space BSPF fields and compare backends on one independent grid."""

import json
from pathlib import Path

import jax
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from bspf_models.elliptic.embedded_poisson import background_line
from bspf_models.elliptic.embedded_poisson import benchmark_domains
from bspf_models.elliptic.embedded_poisson import manufactured
from bspf_models._numerics.trial_spaces import stream_evaluate_line

jax.config.update("jax_enable_x64", True)


def main():
    out = Path("build/embedded_poisson_diagnosis")
    grid = -1 + 2 * (np.arange(180) + 0.5) / 180
    x, y = np.meshgrid(grid, grid, indexing="ij")
    b, d, h = stream_evaluate_line(background_line(33), grid)
    exact, grad, forcing = manufactured(np.column_stack((x.ravel(), y.ravel())))
    exact, forcing = exact.reshape(x.shape), forcing.reshape(x.shape)
    grad = grad.reshape((*x.shape, 2))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), constrained_layout=True)
    stats = {}
    labels = {"convex": "Convex B-spline domain", "nonstar": "Non-star-shaped C domain"}
    for row, domain in enumerate(benchmark_domains()):
        data = np.load(out / f"{domain.name}_full_space_n67_cut1e-14.npz")
        coefficients = data["coefficient"].reshape(33, 33)
        value = b @ coefficients @ b.T
        residual = -(h @ coefficients @ b.T + b @ coefficients @ h.T) - forcing
        error = value - exact
        mask = np.zeros_like(x, dtype=bool)
        for i, xi in enumerate(grid):
            for lo, hi in domain.intersections(xi):
                mask[i] |= (grid > lo) & (grid < hi)
        curve = domain.curve(np.linspace(0, domain.period, 1601))
        fields = [
            value,
            np.maximum(abs(error), 1e-16),
            np.maximum(abs(residual), 1e-16),
        ]
        norms = [Normalize(0.6, 3.5), LogNorm(1e-16, 1e-12), LogNorm(1e-16, 1e-9)]
        titles = [
            labels[domain.name],
            r"Absolute error $|u_h-u_*|$",
            r"PDE residual $|-\Delta u_h-f|$",
        ]
        for ax, field, norm, title, cmap in zip(
            axes[row], fields, norms, titles, ["viridis", "magma", "magma"]
        ):
            image = ax.imshow(
                np.where(mask, field, np.nan).T,
                origin="lower",
                extent=(-1, 1, -1, 1),
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )
            ax.plot(*curve.T, color="#263447", linewidth=1)
            ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
            fig.colorbar(image, ax=ax, shrink=0.85)
        stats[domain.name] = {}
        old = np.load(Path("build/embedded_poisson") / f"{domain.name}_solution.npz")[
            "coefficients"
        ].reshape(20, 20)
        for name, c, modes in [
            ("20-mode Nitsche", old, 20),
            ("Full-space least squares", coefficients, 33),
        ]:
            bb, dd = b[:, :modes], d[:, :modes]
            v = bb @ c @ bb.T
            gx, gy = dd @ c @ bb.T, bb @ c @ dd.T
            stats[domain.name][name] = dict(
                relative_l2=float(
                    np.linalg.norm((v - exact)[mask]) / np.linalg.norm(exact[mask])
                ),
                relative_gradient=float(
                    np.linalg.norm(
                        np.column_stack(
                            ((gx - grad[:, :, 0])[mask], (gy - grad[:, :, 1])[mask])
                        )
                    )
                    / np.linalg.norm(grad[mask])
                ),
            )
        current = stats[domain.name]["Full-space least squares"]
        current["pde_residual_rms"] = float(np.sqrt(np.mean(residual[mask] ** 2)))
        current["error_linf"] = float(np.max(abs(error[mask])))
        axes[row, 0].text(
            0.03,
            0.03,
            f"Relative L2: {current['relative_l2']:.2e}",
            transform=axes[row, 0].transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
        )
        np.savez(
            out / f"{domain.name}_plot_grid.npz",
            grid=grid,
            mask=mask,
            value=value,
            exact=exact,
            error=error,
            pde_residual=residual,
        )
    fig.suptitle("Full-space BSPF Poisson on curved domains", fontsize=17)
    fig.supxlabel(
        "Independent 180 x 180 cell-center samples; white = outside domain. Log plots floor at 1e-16.",
        fontsize=10,
    )
    fig.savefig(out / "full_space_fields.png", dpi=180)
    fig.savefig(out / "full_space_fields.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ["relative_l2", "relative_gradient"],
        ["Relative solution error", "Relative gradient error"],
    ):
        for i, method in enumerate(["20-mode Nitsche", "Full-space least squares"]):
            values = [stats[name][method][metric] for name in ["convex", "nonstar"]]
            positions = np.arange(2) + (i - 0.5) * 0.34
            ax.bar(
                positions,
                values,
                width=0.32,
                label=method,
                color=["#da8743", "#276f9f"][i],
            )
            for xx, yy in zip(positions, values):
                ax.text(xx, yy * 1.7, f"{yy:.2e}", ha="center", fontsize=9)
        ax.set(
            yscale="log",
            ylim=(1e-16, 1),
            xticks=[0, 1],
            xticklabels=["Convex", "Non-star-shaped"],
            title=title,
        )
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Comparison on the same independent grid", fontsize=15)
    fig.supxlabel(
        "Both approximation space and solver differ; this is not a mode-count-only comparison.",
        fontsize=9,
    )
    fig.savefig(out / "accuracy_comparison.png", dpi=180)
    plt.close(fig)
    (out / "plot_metrics.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
