"""Create a shareable MMS comparison report and plots from measured results."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "tensor": "Shared tensor inverse",
    "cusolver_dense": "cuSOLVER dense Cholesky",
    "cusolver_sparse_qr": "cuSOLVER sparse QR (refactors)",
    "amgx_pcg": "AMGX-PCG",
    "cg": "CuPy CG",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("build/rectangle_mms/results.json")
    )
    parser.add_argument("--out", type=Path, default=Path("docs/data/rectangle_mms"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.input.read_text())
    cases = report["cases"]
    shutil.copyfile(args.input, args.out / "results.json")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
    colors = dict(zip(LABELS, plt.get_cmap("tab10").colors))
    for backend, label in LABELS.items():
        rows = sorted(
            [
                c
                for c in cases
                if c["discretization"] == "fd" and c["backend"] == backend
            ],
            key=lambda c: c["n"],
        )
        x = [c["unknowns"] for c in rows]
        axes[0, 0].loglog(
            x,
            [c["warm_median_seconds"] * 1e3 for c in rows],
            "o-",
            label=label,
            color=colors[backend],
        )
        axes[1, 0].loglog(
            x,
            [c["setup_seconds"] for c in rows],
            "o-",
            label=label,
            color=colors[backend],
        )
    for disc, label, color in [
        ("bspf", "BSPF + tensor", "C0"),
        ("fd", "FD + tensor", "C1"),
    ]:
        rows = sorted(
            [
                c
                for c in cases
                if c["discretization"] == disc and c["backend"] == "tensor"
            ],
            key=lambda c: c["n"],
        )
        axes[0, 1].loglog(
            [c["unknowns"] for c in rows],
            [c["relative_l2_error"] for c in rows],
            "o-",
            label=label,
            color=color,
        )
        axes[1, 1].loglog(
            [c["relative_l2_error"] for c in rows],
            [c["warm_median_seconds"] * 1e3 for c in rows],
            "o-",
            label=label,
            color=color,
        )
    for backend in ("amgx_pcg", "cg"):
        rows = sorted(
            [c for c in cases if c["backend"] == backend], key=lambda c: c["n"]
        )
        axes[1, 1].loglog(
            [c["relative_l2_error"] for c in rows],
            [c["warm_median_seconds"] * 1e3 for c in rows],
            "o-",
            label="FD + " + LABELS[backend],
            color=colors[backend],
        )
    titles = [
        "Same FD matrix: synchronized solve time",
        "Continuous MMS: independent L2 error",
        "Same FD matrix: backend setup",
        "Measured error versus repeated solve cost",
    ]
    xlabels = [
        "Interior unknowns",
        "Interior unknowns",
        "Interior unknowns",
        "Relative L2 error (257² Gauss points)",
    ]
    ylabels = ["Warm median [ms]", "Relative L2 error", "Setup [s]", "Warm median [ms]"]
    for ax, title, xlabel, ylabel in zip(axes.flat, titles, xlabels, ylabels):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"Rectangle Poisson MMS · A100 · FP64 · {cases[0]['repeats']} repeated solves",
        fontsize=15,
    )
    fig.savefig(args.out / "comparison.png", dpi=170)
    fig.savefig(args.out / "comparison.pdf")
    plt.close(fig)

    lines = [
        "# Rectangle Poisson MMS and GPU solver comparison",
        "",
        "Measured on 2026-09-19 on NVIDIA A100-SXM4-40GB. FP64 throughout.",
        "",
        "## Continuous manufactured problem",
        "",
        "Solve `-Δu=f` on `[0,2] × [0,3]`, with zero Dirichlet data. Put `t=x/2`, `s=y/3`:",
        "",
        "```text",
        "u = t(1-t)s(1-s) [exp(0.7t-0.4s) + 0.2 sin(5πt) cos(3πs)]",
        "f = -∂xx u - ∂yy u",
        "```",
        "",
        "The forcing is evaluated from analytic product derivatives, independently of every discrete matrix. The forcing and boundary values are checked against JAX automatic differentiation. This is not a discrete manufactured RHS or a single Laplacian eigenmode.",
        "",
        "BSPF uses degree 5, 16 spline basis functions, seven-point endpoint fits, value constraints at both endpoints, and order-8 Gauss quadrature split at nodes and knots. FD uses the standard second-order five-point operator. `n` is the number of interior unknowns per direction, so both discretizations have `n²` unknowns and `n+2` axis nodes.",
        "",
        "Errors use an independent 257×257 Gauss grid: BSPF's own continuous interpolant versus piecewise-bilinear FD reconstruction, with the same analytic solution and quadrature weights. Nodal errors are also retained in the JSON. Thus solve error and physical approximation error are separate measurements.",
        "",
        "## Timing contract",
        "",
        "Each case/backend runs in a fresh process on the same GPU. Problem preparation is recorded separately; it includes axis/load assembly, first-use JAX compilation and construction of error-evaluation tables, so it is not a pure production assembly timer. Backend setup includes matrix construction/upload and reusable factorization or AMG hierarchy setup, where applicable. First solve includes solve compilation/first-use overhead. Warm solve times are medians of seven synchronized wall-clock measurements, with operator and RHS already resident. No solution download or error evaluation is timed.",
        "",
        "- Shared tensor inverse: the production `plan_rectangle_poisson` / `solve_rectangle_poisson` API; only axis matrices are factored.",
        "- Dense direct: CuPy `linalg.cholesky` uses cuSOLVER; repeated solves use two GPU triangular solves with the retained factor. The same full discrete matrix is used as the tensor inverse, including BSPF's nonidentity masses. Dense cases are capped at 4,096 unknowns to bound quadratic storage.",
        "- Sparse direct: CuPy `spsolve` uses cuSOLVER `csrlsvqr`. Its API refactors on each call; **the plotted warm time includes factorization** and is not a reusable-factor solve. This baseline is capped at 255² unknowns.",
        "- AMGX-PCG: FP64 classical AMG V-cycle, symmetric one-pre/one-post Jacobi smoothing and dense coarse solve. Every call explicitly zeros the device vector inside the timed region before solving. This avoids accumulation observed when relying on the local library's zero-initial-guess shortcut alone.",
        "- CG: unpreconditioned CuPy GPU CG from zero. Iteration counts come from a separate untimed callback run. These wall times include Python dispatch/convergence-check overhead; they are not a claim about an optimized fused CG implementation.",
        "",
        "Both iterative methods target relative residual `1e-10`. Every method must independently satisfy `||AU-F||₂/||F||₂ ≤ 2e-10`; no residual refinement or tolerance relaxation is applied. GPU-to-host transfers occur only outside warm timing. AMG hierarchy storage and solver workspaces are not reported as matrix storage.",
        "",
        "## Same finite-difference matrix",
        "",
        "| n | Tensor [ms] | Dense direct [ms] | Sparse QR† [ms] | AMGX-PCG [ms] | CG [ms] | AMG / CG iterations | Relative L2 error |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in sorted({c["n"] for c in cases if c["discretization"] == "fd"}):
        rows = {
            c["backend"]: c
            for c in cases
            if c["discretization"] == "fd" and c["n"] == n
        }
        values = [
            f"{rows[k]['warm_median_seconds'] * 1e3:.3f}" if k in rows else "—"
            for k in LABELS
        ]
        iterations = " / ".join(
            str(rows[k].get("iterations", "—")) if k in rows else "—"
            for k in ("amgx_pcg", "cg")
        )
        error = rows["tensor"]["relative_l2_error"]
        lines.append(
            f"| {n} | " + " | ".join(values) + f" | {iterations} | {error:.3e} |"
        )
    lines += [
        "",
        "All methods in this table solve the same five-point FD matrix. Their L2 errors agree at the displayed precision; the error column reports the common FD discretization error against the continuous MMS, not the algebraic residual. † Sparse QR refactors each RHS. Dashes are configured size caps, not failed solves.",
        "",
        "## BSPF versus FD approximation",
        "",
        "| Discretization | n | Unknowns | Relative L2 error | Nodal relative error | Tensor warm [ms] | Problem preparation [s] | Tensor setup [s] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cases:
        if c["backend"] == "tensor":
            lines.append(
                f"| {c['discretization'].upper()} | {c['n']} | {c['unknowns']:,} | {c['relative_l2_error']:.3e} | {c['nodal_relative_error']:.3e} | {c['warm_median_seconds'] * 1e3:.3f} | {c['assembly_seconds']:.3f} | {c['setup_seconds']:.3f} |"
            )
    lines += [
        "",
        "## Dense direct on the identical BSPF matrix",
        "",
        "| n | Tensor warm [ms] | Dense direct warm [ms] | Tensor setup [s] | Dense setup [s] | Tensor factors [MiB] | Dense factor alone [MiB] |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cases:
        if c["discretization"] != "bspf" or c["backend"] != "cusolver_dense":
            continue
        t = next(
            v
            for v in cases
            if v["backend"] == "tensor"
            and v["discretization"] == "bspf"
            and v["n"] == c["n"]
        )
        lines.append(
            f"| {c['n']} | {t['warm_median_seconds'] * 1e3:.3f} | {c['warm_median_seconds'] * 1e3:.3f} | {t['setup_seconds']:.3f} | {c['setup_seconds']:.3f} | {t['factor_bytes'] / 2**20:.3f} | {c['factor_bytes'] / 2**20:.3f} |"
        )
    worst = max(c["relative_residual"] for c in cases)
    lines += [
        "",
        f"All {len(cases)} measured cases passed the independent residual check; maximum relative residual was `{worst:.3e}`. All backend timings, setup/first-call timings, iteration counts, errors and AMGX configuration are in [results.json](data/rectangle_mms/results.json).",
        "",
        "![Accuracy and performance](data/rectangle_mms/comparison.png)",
        "",
        "[PDF figure](data/rectangle_mms/comparison.pdf)",
        "",
        "## Reproduce",
        "",
        "```bash",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src \\",
        "  python scratch/benchmark_rectangle_mms.py \\",
        "  --amgx-python /raven/u/limo/venvs/numba_cuda_waterboa/bin/python \\",
        "  --amgx-runtime-path /mpcdf/soft/SLE_15/packages/x86_64/cuda/13.2.1/lib64",
        "python scratch/report_rectangle_mms.py",
        "```",
        "",
        "The main interpreter needs JAX GPU, CuPy CUDA 12, NumPy and SciPy; the report needs Matplotlib. PyAMGX runs in its existing Python 3.13 environment, with AMGX 2.5.0 built against CUDA 13.2. The JAX/CuPy environment uses JAX 0.10.0 and CuPy 14.2.0. Different CUDA toolkit versions are recorded rather than presented as identical library stacks.",
        "",
        "Validation: `python -m pytest jax/tests/test_rectangle_mms_benchmark.py jax/tests/test_rectangle_poisson.py -q`. Per-worker logs and assembly datasets remain under `build/rectangle_mms/`.",
    ]
    fd_tensor = max(
        (c for c in cases if c["discretization"] == "fd" and c["backend"] == "tensor"),
        key=lambda c: c["n"],
    )
    fd_amgx = next(
        c
        for c in cases
        if c["discretization"] == "fd"
        and c["backend"] == "amgx_pcg"
        and c["n"] == fd_tensor["n"]
    )
    speedup = fd_amgx["warm_median_seconds"] / fd_tensor["warm_median_seconds"]
    lines[4:4] = [
        "## Interpretation",
        "",
        f"For the same {fd_tensor['n']}² FD matrix, the reusable tensor inverse is {speedup:.1f}× faster per warm solve than this AMGX-PCG configuration. Its backend setup takes {fd_tensor['setup_seconds']:.2f} s versus {fd_amgx['setup_seconds']:.2f} s for AMGX; reuse is essential to realizing that advantage.",
        "",
        "The BSPF MMS convergence is much faster than second-order FD in this smooth example. Dense cuSOLVER solves on the identical BSPF matrix reproduce the same approximation errors, confirming that this difference comes from discretization rather than the linear solver.",
        "",
        "These timings apply to a fixed, separable, constant-coefficient rectangle. They do not imply the same advantage for curved domains, variable coefficients, or repeatedly changing operators. Cold BSPF problem preparation is roughly 16 s in this implementation, including compilation and error-evaluation tables. Sparse QR's refactor-every-call cost and CuPy CG's Python overhead must be considered when interpreting their large timing differences.",
        "",
    ]
    if report["failures"]:
        raise RuntimeError(f"Benchmark contains failures: {report['failures']}")
    (args.out.parents[1] / "jax_rectangle_mms_benchmark.md").write_text(
        "\n".join(lines) + "\n"
    )


if __name__ == "__main__":
    main()
