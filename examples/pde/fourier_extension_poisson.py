"""Reproduce Algorithm 1 and benchmark its Fourier-particular/MFS Poisson use.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1.
The dense baseline is the SAME extension problem, not a different PDE method.
"""

from importlib import import_module

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import scipy.linalg as la

from bspf_models.elliptic.embedded_poisson import benchmark_domains
from bspf_models.elliptic.fourier_poisson import FourierPoissonPlan
from compare_poisson_bases import ProfileMMS, RandomWaveMMS, exact_jets
from embedded_poisson_approximation import interior


class Exponential:
    def evaluate(self, points):
        u = np.exp(points[:, 0] + points[:, 1])
        return u, np.column_stack((u, u)), -2 * u

    def jets(self, points):
        u = self.evaluate(points)[0]
        return u[:, None] * np.array([1, 1, 1, 1, np.sqrt(2), 1])


def jets(case, points):
    return case.jets(points) if isinstance(case, Exponential) else exact_jets(case, points)


def median_time(call, repeats=5):
    call()
    times = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        times.append(perf_counter() - start)
    return float(np.median(times))


def dense_extension_comparison(plan, check, *, max_modes):
    p = plan.extension
    if p.modes > max_modes:
        return None
    # Use the same real orthonormal coordinates as the compressed SVD, so the
    # baseline is not disadvantaged by complex rather than real arithmetic.
    start = perf_counter()
    factors = []
    frequencies = np.arange(1, p.modes // 2 + 1) * np.pi / p.half_width
    for axis in (0, 1):
        phase = (p.points[:, axis, None] + p.half_width) * frequencies
        factor = np.ones((p.samples, p.modes))
        factor[:, 1::2] = np.sqrt(2) * np.cos(phase)
        factor[:, 2::2] = np.sqrt(2) * np.sin(phase)
        factors.append(factor)
    a = (factors[0][:, :, None] * factors[1][:, None, :]).reshape(p.samples, p.size) / p.normalization
    matrix_seconds = perf_counter() - start
    start = perf_counter()
    u, s, vh = la.svd(a, full_matrices=False)
    svd_seconds = perf_counter() - start
    keep = s > p.cutoff
    left, singular, right = u[:, keep].copy(), s[keep].copy(), vh[keep].T.copy()
    f = np.exp(p.points.sum(axis=1))

    def solve():
        return p._to_fourier(right @ ((left.T @ (f / p.normalization)) / singular))

    c = solve()
    fast = p.solve(f)
    truth = np.exp(check.sum(axis=1))
    direct_values, fast_values = p.evaluate(c, check), fast.evaluate(check)
    return dict(
        dense_assembly_seconds=matrix_seconds, dense_svd_seconds=svd_seconds,
        dense_total_setup_seconds=matrix_seconds + svd_seconds,
        dense_rhs_seconds=median_time(solve), fast_rhs_seconds=median_time(lambda: p.solve(f)),
        dense_rank=int(keep.sum()),
        dense_sample_relative_residual=float(la.norm(p.normalization * (a @ p._to_real_modes(c)) - f) / la.norm(f)),
        fast_sample_relative_residual=fast.diagnostics["sample_relative_residual"],
        dense_offgrid_relative_l2=float(la.norm(direct_values - truth) / la.norm(truth)),
        fast_offgrid_relative_l2=float(la.norm(fast_values - truth) / la.norm(truth)),
        fast_dense_offgrid_relative_difference=float(la.norm(fast_values - direct_values) / la.norm(truth)),
    )


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    domain = benchmark_domains()[0]
    check = interior(domain, 127, 131, 0.613)
    cases = {"exponential": Exponential(), **{name: ProfileMMS(name) for name in ("polynomial", "gaussian", "rational")},
             "wave4pi": RandomWaveMMS.create(kmax=4*np.pi),
             "wave12pi": RandomWaveMMS.create(kmax=12*np.pi)}
    report = dict(
        settings=vars(args) | {"out": str(args.out)},
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         machine=platform.machine(), blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
                         omp_threads=os.environ.get("OMP_NUM_THREADS")),
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                       [Path(__file__), Path(import_module("pybspf.fourier_extension").__file__), Path(import_module("bspf_models.elliptic.fourier_poisson").__file__)]},
        validation_points=len(check), runs=[],
    )
    for n in args.modes:
        plan = FourierPoissonPlan(domain, n, grid_size=args.oversampling*n,
                                  source_count=args.sources, boundary_count=2*args.sources,
                                  source_distance=args.source_distance, cutoff=args.cutoff,
                                  boundary_cutoff=args.boundary_cutoff, seed=args.seed)
        edge, _ = plan.arc.sample(4*args.sources, 0.371)
        entry = dict(modes=n, plan=plan.diagnostics, cases={})
        print("PLAN", n, json.dumps(plan.diagnostics), flush=True)
        for name, case in cases.items():
            # PDE solve only receives forcing and physical boundary values.
            f = case.evaluate(plan.extension.points)[2]
            g = case.evaluate(plan.boundary)[0]
            solution = plan.solve(f, g)
            rhs_seconds = median_time(lambda: plan.solve(f, g))
            start = perf_counter()
            value, grad, lap = solution.evaluate(check)
            h = solution.hessian(check)
            evaluation_seconds = perf_counter() - start
            predicted = np.column_stack((value, grad, h[:, 0, 0], np.sqrt(2)*h[:, 0, 1], h[:, 1, 1]))
            exact = jets(case, check)
            delta = predicted - exact
            edge_error = solution.derivative(edge) - case.evaluate(edge)[0]
            data = dict(
                relative_l2=float(la.norm(delta[:, 0]) / la.norm(exact[:, 0])),
                relative_gradient=float(la.norm(delta[:, 1:3]) / la.norm(exact[:, 1:3])),
                relative_h2=float(la.norm(delta) / la.norm(exact)),
                relative_laplacian=float(la.norm(lap - exact[:, 3] - exact[:, 5]) / la.norm(exact[:, 3] + exact[:, 5])),
                value_max=float(np.max(np.abs(delta[:, 0]))), boundary_max=float(np.max(np.abs(edge_error))),
                imaginary_max=float(np.max(np.abs(predicted.imag))),
                repeated_rhs_seconds=rhs_seconds, offgrid_evaluation_seconds=evaluation_seconds,
                diagnostics=solution.diagnostics,
            )
            entry["cases"][name] = data
            np.savez(args.out / f"{name}_n{n}.npz", particular=solution.particular,
                     forcing_coefficients=solution.forcing_extension.coefficients,
                     mean=solution.mean, harmonic=solution.harmonic, sources=plan.sources,
                     validation_points=check, boundary_validation_points=edge)
            print("POISSON", n, name, json.dumps(data), flush=True)
        entry["dense_extension_comparison"] = dense_extension_comparison(plan, check, max_modes=args.dense_max_modes)
        print("DENSE", n, json.dumps(entry["dense_extension_comparison"]), flush=True)
        report["runs"].append(entry)
        (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        write_summary(report, args.out)


def write_summary(report, out):
    lines = ["# Fourier extension + Poisson benchmark", "",
             "Independent whole-domain offset grid: %d points; separate offset boundary." % report["validation_points"],
             "Setup includes geometry, FE factors and boundary factors. RHS excludes input sampling and output evaluation.",
             "H2 errors are discrete validation norms, not continuous error certificates. Times are CPU single-process measurements.", "",
             "| N per axis | Case | Relative L2 | Relative H2 | Boundary max | Setup (s) | RHS (ms) |", "|---|---|---|---|---|---|---|"]
    for run in report["runs"]:
        for name, case in run["cases"].items():
            lines.append(f"| {run['modes']} | {name} | {case['relative_l2']:.3e} | {case['relative_h2']:.3e} | {case['boundary_max']:.3e} | {run['plan']['setup_seconds']:.3f} | {case['repeated_rhs_seconds']*1000:.3f} |")
    lines += ["", "## Same extension problem: direct TSVD versus Algorithm 1", "",
              "This comparison concerns function extension, not the full Poisson solver.",
              "Both use real modal coordinates; fast and direct cutoff operators differ (PA versus A).", "",
              "| N | Plunge rank / N² | Fast setup (s) | Dense setup (s) | Fast RHS (ms) | Dense RHS (ms) | Fast/direct off-grid difference |",
              "|---|---|---|---|---|---|---|"]
    for run in report["runs"]:
        d = run["dense_extension_comparison"]
        if d:
            e = run["plan"]["extension"]
            lines.append(f"| {run['modes']} | {e['plunge_rank']} / {run['modes']**2} | {e['setup_seconds']:.3f} | {d['dense_total_setup_seconds']:.3f} | {1000*d['fast_rhs_seconds']:.3f} | {1000*d['dense_rhs_seconds']:.3f} | {d['fast_dense_offgrid_relative_difference']:.3e} |")
    lines += ["", "Boundary-only MFS is an added Poisson construction, not part of the paper. No NUFFT is used.",
              "The range finder uses randomized probes; the residual estimate is not a deterministic certificate.",
              "At modest N the plunge rank can be most of N²; faster asymptotic complexity does not guarantee faster setup.", ""]
    (out / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", type=int, default=[17, 33, 49, 65])
    parser.add_argument("--oversampling", type=int, default=4)
    parser.add_argument("--sources", type=int, default=192)
    parser.add_argument("--source-distance", type=float, default=0.25)
    parser.add_argument("--cutoff", type=float, default=1e-12)
    parser.add_argument("--boundary-cutoff", type=float, default=1e-13)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dense-max-modes", type=int, default=49)
    parser.add_argument("--out", type=Path, default=Path("build/fourier_extension_poisson"))
    run(parser.parse_args())
