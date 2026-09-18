"""Render the matched BSPF/Fourier/B-spline benchmark from saved results."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FAMILIES = ("bspf", "fourier", "bspline")
LABELS = {"bspf": "BSPF", "fourier": "Fourier", "bspline": "B-spline (p=13)"}
COLORS = {"bspf": "#006cb7", "fourier": "#dc6529", "bspline": "#28906e"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("build/poisson_basis_comparison")
    )
    args = parser.parse_args()
    rows = []
    for path in sorted(args.out.glob("*_n*/results.json")):
        data = json.loads(path.read_text())
        if not data.get("complete"):
            raise ValueError(f"Incomplete case: {path}")
        for row in data["rows"]:
            rows.append(
                dict(family=data["settings"]["family"], n=data["settings"]["n"], **row)
            )
    if not rows:
        raise ValueError("No results found")
    with (args.out / "all_results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def select(family, band, task, cutoff=1e-13):
        return sorted(
            [
                r
                for r in rows
                if r["family"] == family
                and (
                    r["band"] == band
                    if isinstance(band, int)
                    else r.get("case") == band
                )
                and r["task"] == task
                and r["cutoff"] == cutoff
            ],
            key=lambda r: r["n"],
        )

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), constrained_layout=True)
    for i, band in enumerate((4, 12)):
        for j, (task, key, title) in enumerate(
            (
                ("h2_fit", "h2", "Best-H2 diagnostic"),
                ("pde", "value", "PDE solution: relative L2 error"),
                ("pde", "h2", "PDE solution: relative H2 error"),
            )
        ):
            ax = axes[i, j]
            for family in FAMILIES:
                data = select(family, band, task)
                ax.semilogy(
                    [r["n"] for r in data],
                    [r[key] for r in data],
                    "o-",
                    color=COLORS[family],
                    label=LABELS[family],
                )
            ax.set_title(f"{title}\nkmax = {band} pi")
            ax.set_xlabel("N (basis functions per axis)")
            ax.set_xticks([17, 33, 65])
            ax.grid(True, which="both", alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle(
        "Same geometry, quadrature, H3/2 trace norm and independent validation points\n"
        "Box-H2 scaling; relative SVD cutoff = 1e-13; best-H2 is a regularized numerical fit"
    )
    fig.savefig(args.out / "convergence.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(13, 10.5), constrained_layout=True)
    for i, name in enumerate(("polynomial", "gaussian", "rational")):
        for j, (task, key, title) in enumerate(
            (
                ("h2_fit", "h2", "Best-H2 diagnostic"),
                ("pde", "value", "PDE: relative L2 error"),
                ("pde", "h2", "PDE: relative H2 error"),
            )
        ):
            ax = axes[i, j]
            for family in FAMILIES:
                data = select(family, name, task)
                ax.semilogy(
                    [r["n"] for r in data],
                    [r[key] for r in data],
                    "o-",
                    color=COLORS[family],
                    label=LABELS[family],
                )
            ax.set_title(f"{name.capitalize()}: {title}")
            ax.set_xlabel("N (basis functions per axis)")
            ax.set_xticks([17, 33, 65])
            ax.grid(True, which="both", alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle(
        "Non-Fourier-defined MMS; same geometry, quadrature and independent points\n"
        "Box-H2 scaling; SVD cutoff = 1e-13; polynomial is an exact-reproduction control"
    )
    fig.savefig(args.out / "non_fourier_convergence.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for ax, name in zip(axes, ("polynomial", "gaussian", "rational")):
        with np.load(args.out / "bspline_n65" / f"pde_{name}.npz") as data:
            points, exact = data["points"], data["exact"]
        im = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=exact,
            s=3,
            marker="s",
            linewidths=0,
            cmap="viridis",
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_title(name.capitalize())
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Exact solutions defined directly in physical space")
    fig.savefig(args.out / "non_fourier_mms.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for family in FAMILIES:
        data = select(family, 12, "pde")
        for ax, key, scale, title in zip(
            axes,
            ("setup_seconds", "repeated_rhs_seconds", "matrix_bytes"),
            (1, 1000, 1 / 1e9),
            ("PDE cold setup (s)", "PDE repeated RHS (ms)", "PDE matrix alone (GB)"),
        ):
            ax.semilogy(
                [r["n"] for r in data],
                [r[key] * scale for r in data],
                "o-",
                color=COLORS[family],
                label=LABELS[family],
            )
            ax.set_title(title)
            ax.set_xlabel("N")
            ax.set_xticks([17, 33, 65])
            ax.grid(True, which="both", alpha=0.2)
    axes[0].legend()
    fig.suptitle(
        "Common dense SVD implementation; serial runs; one BLAS thread\n"
        "RHS timing excludes source sampling and output evaluation; memory is not peak RSS"
    )
    fig.savefig(args.out / "cost.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for ax, task in zip(axes, ("pde", "h2_fit")):
        for family in FAMILIES:
            singular = np.load(args.out / f"{family}_n65" / f"{task}_singular.npy")
            ax.semilogy(
                np.arange(1, len(singular) + 1),
                singular / singular[0],
                label=LABELS[family],
                color=COLORS[family],
            )
        ax.axhline(1e-13, linestyle="--", color="gray", linewidth=1)
        ax.set_ylim(1e-18, 2)
        ax.set_title(f"N=65: {task}")
        ax.set_xlabel("Singular-value index")
        ax.set_ylabel("sigma / sigma_max")
        ax.grid(True, alpha=0.2)
    axes[0].legend()
    fig.savefig(args.out / "singular_values.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, family in zip(axes, FAMILIES):
        path = args.out / f"{family}_n65" / "pde_12pi.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            p, value, exact = data["points"], data["value"], data["exact"]
        error = np.log10(
            np.maximum(abs(value - exact) / np.sqrt(np.mean(exact**2)), 1e-16)
        )
        im = ax.scatter(
            p[:, 0],
            p[:, 1],
            c=error,
            s=3,
            marker="s",
            linewidths=0,
            cmap="magma",
            vmin=-13,
            vmax=-6,
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_title(LABELS[family])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im, ax=axes, label="log10(|error| / RMS(exact))", shrink=0.8)
    fig.suptitle("N=65, kmax=12 pi: PDE error at identical independent points")
    fig.savefig(args.out / "error_fields.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Matched approximation-space comparison",
        "",
        "Main cutoff: 1e-13.",
        "",
        "| MMS | N | Space | H2-fit H2 error | PDE L2 error | PDE H2 error | PDE setup (s) | Repeated RHS (ms) | H2-fit setup (s) |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for band in (4, 12, "polynomial", "gaussian", "rational"):
        for n in (17, 33, 65):
            for family in FAMILIES:
                pde = [r for r in select(family, band, "pde") if r["n"] == n]
                fit = [r for r in select(family, band, "h2_fit") if r["n"] == n]
                if pde and fit:
                    p, h = pde[0], fit[0]
                    lines.append(
                        f"| {band} | {n} | {LABELS[family]} | {h['h2']:.4e} | {p['value']:.4e} | "
                        f"{p['h2']:.4e} | {p['setup_seconds']:.3f} | {1000 * p['repeated_rhs_seconds']:.3f} | {h['setup_seconds']:.3f} |"
                    )
    lines += [
        "",
        "## N=65 threshold sensitivity (independent relative PDE H2 error)",
        "",
        "| MMS | Space | cutoff 1e-11 | cutoff 1e-13 | cutoff 1e-14 |",
        "|---|---|---:|---:|---:|",
    ]
    for case in (4, 12, "polynomial", "gaussian", "rational"):
        for family in FAMILIES:
            values = []
            for cutoff in (1e-11, 1e-13, 1e-14):
                data = [r for r in select(family, case, "pde", cutoff) if r["n"] == 65]
                values.append(f"{data[0]['h2']:.4e}" if data else "missing")
            lines.append(f"| {case} | {LABELS[family]} | " + " | ".join(values) + " |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print(args.out / "convergence.png")
    print(args.out / "summary.md")


if __name__ == "__main__":
    main()
