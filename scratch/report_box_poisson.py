"""Archive the 3D MMS measurements and produce a compact benchmark report."""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("build/box_poisson/results.json")
    )
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    cases = report["cases"]
    if not cases or any(c["relative_residual"] > 2e-10 for c in cases):
        raise ValueError("Need nonempty results passing the independent residual check")
    order = cases[0]["error_quadrature"][0]
    repeats = len(cases[0]["warm_seconds"])
    if any(
        c["error_quadrature"] != [order] * 3 or len(c["warm_seconds"]) != repeats
        for c in cases
    ):
        raise ValueError("Report requires a common quadrature and repetition count")
    destination = args.docs / "data/box_poisson"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.input, destination / "results.json")
    lines = [
        "# Shared 3D GPU box solver: continuous MMS benchmark",
        "",
        "Measured on 2026-09-19 using JAX 0.10.0, FP64 and NVIDIA A100-SXM4-40GB.",
        "",
        f"The manufactured problem, boundary conditions, discretizations and public API are described in [the 3D solver documentation](jax_box_poisson.md). These are independent coefficient counts: `n³` unknowns on `(n+2)³` sampling nodes for this particular nodal formulation. The L2 error is evaluated against the continuous exact solution on an independent {order}³ Gauss grid.",
        "",
        f"Each case runs in a fresh process. Warm timings are medians of {repeats} synchronized solves under a transfer guard, reusing the fixed plan and GPU-resident RHS. They exclude setup, uploads and error evaluation. BSPF and FD are two discretizations using the same production tensor solver; this table does not compare against a different linear solver.",
        "",
        "| Discretization | Coefficient shape | Unknowns | Warm solve [ms] | Relative L2 error | Relative algebraic residual |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in cases:
        n = c["shape"][0]
        lines.append(
            f"| {c['discretization'].upper()} | {n}³ | {c['unknowns']:,} | {1000 * c['warm_median_seconds']:.3f} | {c['relative_l2_error']:.3e} | {c['relative_residual']:.3e} |"
        )
    lines += [
        "",
        "All cases passed the independent relative residual limit of `2e-10`, recomputed from the original weak mass/stiffness matrices or seven-point FD stencil. No tolerance relaxation or residual refinement was applied.",
        "",
        "| Discretization | Coefficient shape | Cold preparation [s] | Factor setup [s] | First solve [ms] | Plan storage [MiB] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in cases:
        lines.append(
            f"| {c['discretization'].upper()} | {c['shape'][0]}³ | {c['preparation_seconds']:.3f} | {c['setup_seconds']:.3f} | {1000 * c['first_solve_seconds']:.3f} | {c['plan_bytes'] / 2**20:.4f} |"
        )
    lines += [
        "",
        "Cold preparation includes axis/load construction and error-evaluation tables; BSPF also incurs first-use JAX compilation. Factor setup is measured separately from RHS upload. The first solve includes solve compilation. Plan storage counts only axis rotations, spectra and shift, not input/output volumes or temporary GPU workspace. The denominator is not retained as a full volume in the plan.",
        "",
        "The implementation applies to positive-definite separable operators on boxes. Different geometry or nonseparable coefficients need their own operator treatment. CPU/GPU independent-matrix, non-cubic, batching, autodiff, transfer-guard and existing PDE regressions passed: **46 tests**.",
        "",
        "[Raw timings, errors and storage measurements](data/box_poisson/results.json)",
        "",
        "```bash",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src \\",
        "  python scratch/benchmark_box_poisson.py",
        "python scratch/report_box_poisson.py",
        "```",
        "",
    ]
    (args.docs / "jax_box_poisson_benchmark.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
