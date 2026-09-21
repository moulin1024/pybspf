#!/usr/bin/env python3
"""
Schrödinger FD4 – Method-of-Manufactured-Solutions example
===========================================================
Loads a multigrid mesh and MMS data, assembles the fourth-order
immersed-boundary Schrödinger operator, solves with AMG-preconditioned
BiCGSTAB, and reports error and timing statistics.

Run from the repo root:
    python src/pybspf/solvers/schrodinger_fd4/examples/solve_mms.py

Optional flags:
    --plot                 show solution / residual plots
    --save-plots           write PNG files next to the data file
    --tol TOL              BiCGSTAB relative tolerance  (default 1e-10)
    --maxiter N            max BiCGSTAB iterations      (default 1000)
    --mesh PATH            override default multigrid.nc path
    --data PATH            override default schrodinger_data.nc path
    --report-boundary-bands
    --report-operator-diagnostics
    --debug-checks
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when run directly from anywhere in the repo.
SRC = Path(__file__).resolve().parents[4]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import time

import numpy as np

from pybspf.solvers.schrodinger_fd4.assembly import (
    assemble_fd4_system_from_preprocessed,
    build_scipy_csr_matrix,
    preprocess_fd4_system,
)
from pybspf.solvers.schrodinger_fd4.boundary import (
    build_boundary_distance_query,
    evaluate_distance_to_boundary,
)
from pybspf.solvers.schrodinger_fd4.common import PINFO_GHOST
from pybspf.solvers.schrodinger_fd4.io import (
    load_fields,
    load_mesh,
    parse_args,
    point_coordinates,
    scatter_field,
)
from pybspf.solvers.schrodinger_fd4.linear import (
    BenchmarkStats,
    LinearSolveStats,
    build_amg_hierarchy,
    compute_operator_diagnostics,
    solve_with_amg_bicgstab,
    solve_with_existing_amg,
)

SOLVER_LABEL = "FD4+PyAMG-RSAMG-BiCGSTAB"


def run(args) -> int:
    # ------------------------------------------------------------------ load
    t0 = time.perf_counter()
    mesh = load_mesh(args.mesh.resolve())
    t_mesh = time.perf_counter()
    fields = load_fields(args.data.resolve())
    t_data = time.perf_counter()

    # ------------------------------------------------------------ preprocess
    pre = preprocess_fd4_system(mesh, fields, validate=args.debug_checks)
    t_pre = time.perf_counter()

    # --------------------------------------------------------------- assemble
    assembled = assemble_fd4_system_from_preprocessed(pre)
    A = build_scipy_csr_matrix(pre.pattern, assembled.csr_data)
    b = assembled.b
    active_indices = assembled.active_indices
    u_exact = assembled.u_exact
    meta = assembled.meta
    t_assembly = time.perf_counter()

    # ----------------------------------------------------------------- solve
    if args.benchmark_rhs <= 1:
        x, history, stats = solve_with_amg_bicgstab(A, b, tol=args.tol, maxiter=args.maxiter)
        benchmark_stats: BenchmarkStats | None = None
    else:
        ml, amg_stats = build_amg_hierarchy(A)
        rhs_list = [b] * args.benchmark_rhs
        solve_times: list[float] = []
        iterations: list[int] = []
        infos: list[int] = []
        histories: list[list[float]] = []
        x = np.zeros_like(b)
        for rhs_k in rhs_list:
            x, history_k, stats_k = solve_with_existing_amg(A, rhs_k, ml, tol=args.tol, maxiter=args.maxiter)
            solve_times.append(float(stats_k.solve_time_s))
            iterations.append(int(stats_k.iterations))
            infos.append(int(stats_k.info))
            histories.append(history_k)
        history = histories[0] if histories else []
        stats = LinearSolveStats(
            info=infos[0] if infos else 0,
            iterations=iterations[0] if iterations else 0,
            setup_time_s=float(amg_stats.setup_time_s),
            solve_time_s=solve_times[0] if solve_times else 0.0,
            total_time_s=float(amg_stats.setup_time_s) + (solve_times[0] if solve_times else 0.0),
            levels=int(amg_stats.levels),
            operator_complexity=float(amg_stats.operator_complexity),
            grid_complexity=float(amg_stats.grid_complexity),
        )
        benchmark_stats = BenchmarkStats(
            n_rhs=args.benchmark_rhs,
            amg_setup_time_s=float(amg_stats.setup_time_s),
            solve_time_first_s=solve_times[0] if solve_times else 0.0,
            solve_time_mean_s=float(np.mean(solve_times)) if solve_times else 0.0,
            solve_time_min_s=float(np.min(solve_times)) if solve_times else 0.0,
            solve_time_max_s=float(np.max(solve_times)) if solve_times else 0.0,
            iterations_mean=float(np.mean(iterations)) if iterations else 0.0,
            iterations_max=max(iterations) if iterations else 0,
            all_info_zero=int(all(info == 0 for info in infos)),
            amortized_total_per_rhs_s=(
                (t_pre - t_data)
                + (t_assembly - t_pre)
                + float(amg_stats.setup_time_s)
                + float(np.sum(solve_times))
            ) / float(args.benchmark_rhs)
            if solve_times
            else 0.0,
        )
    t_solve = time.perf_counter()

    # ------------------------------------------------------- error & timing
    u_num = np.asarray(u_exact, dtype=np.float64).copy()
    u_num[np.asarray(active_indices, dtype=np.int32) - 1] = x
    inner = np.asarray(mesh["inner_indices"], dtype=np.int32) - 1

    err = u_num - u_exact
    inner_rms = float(np.sqrt(np.mean(err[inner] ** 2)))
    inner_ref = float(np.sqrt(np.mean(u_exact[inner] ** 2)))
    active = np.asarray(mesh["pinfo"], dtype=np.int32) != PINFO_GHOST
    active_rms = float(np.sqrt(np.mean(err[active] ** 2)))
    active_ref = float(np.sqrt(np.mean(u_exact[active] ** 2)))

    print(f"{SOLVER_LABEL} error L2 rms (inner)      : {inner_rms:.6e}")
    print(f"{SOLVER_LABEL} relative L2 err (inner)   : {inner_rms / inner_ref:.6e}")
    print(f"{SOLVER_LABEL} error L2 rms (active)     : {active_rms:.6e}")
    print(f"{SOLVER_LABEL} relative L2 err (active)  : {active_rms / active_ref:.6e}")
    print(f"{SOLVER_LABEL} iterations                : {stats.iterations}")
    print(f"{SOLVER_LABEL} info                      : {stats.info}")
    print(f"{SOLVER_LABEL} setup time [s]            : {stats.setup_time_s:.6e}")
    print(f"{SOLVER_LABEL} solve time [s]            : {stats.solve_time_s:.6e}")
    print(f"{SOLVER_LABEL} total time [s]            : {stats.total_time_s:.6e}")
    print(f"{SOLVER_LABEL} mesh load time [s]        : {t_mesh - t0:.6e}")
    print(f"{SOLVER_LABEL} data load time [s]        : {t_data - t_mesh:.6e}")
    print(f"{SOLVER_LABEL} preprocessing time [s]   : {t_pre - t_data:.6e}")
    print(f"{SOLVER_LABEL} online assembly time [s] : {t_assembly - t_pre:.6e}")
    print(f"{SOLVER_LABEL} end-to-end time [s]       : {t_solve - t0:.6e}")
    print(f"{SOLVER_LABEL} AMG levels                : {stats.levels}")
    print(f"{SOLVER_LABEL} operator complexity       : {stats.operator_complexity:.6e}")
    print(f"{SOLVER_LABEL} grid complexity           : {stats.grid_complexity:.6e}")
    print(f"Unknown inner points                    : {meta.n_inner}")
    print(f"Total unknown points                    : {meta.n_unknowns}")
    print(f"Irregular x-closure rows                : {meta.n_irregular_x_rows}")
    print(f"Irregular y-closure rows                : {meta.n_irregular_y_rows}")
    print(f"Boundary setup time [s]                : {meta.boundary_setup_time_s:.6e}")
    print(f"Lookup setup time [s]                  : {meta.lookup_setup_time_s:.6e}")
    print(f"Template build time [s]                : {meta.template_build_time_s:.6e}")
    print(f"CSR pattern build time [s]            : {meta.csr_pattern_build_time_s:.6e}")
    print(f"Preprocess total time [s]              : {meta.preprocess_total_time_s:.6e}")
    print(f"Triplet concat time [s]                : {meta.triplet_concat_time_s:.6e}")
    print(f"CSR build time [s]                     : {meta.csr_build_time_s:.6e}")
    print(f"Online assembly time [s]               : {meta.online_assembly_time_s:.6e}")
    print(f"Boundary trace calls                   : {meta.boundary_trace_calls}")
    print(f"Boundary trace time [s]                : {meta.boundary_trace_time_s:.6e}")
    print(f"Boundary support points                 : {meta.n_boundary}")
    print(f"Ghost support points                    : {meta.n_ghost}")
    print(f"Matrix nnz                              : {meta.nnz}")

    # ------------------------------------------------- operator diagnostics
    if args.report_operator_diagnostics:
        diagnostics = compute_operator_diagnostics(
            A,
            np.asarray(pre.operator.potential, dtype=np.float64),
            np.asarray(pre.geometry.x_inner, dtype=np.float64),
            np.asarray(pre.geometry.y_inner, dtype=np.float64),
            history,
        )
        print(f"Potential/diag(K) ratio max             : {diagnostics['potential_diag_ratio_max']:.6e}")
        print(f"Potential/diag(K) ratio p95             : {diagnostics['potential_diag_ratio_p95']:.6e}")
        print(f"Potential/diag(K) ratio mean            : {diagnostics['potential_diag_ratio_mean']:.6e}")
        print(f"Potential Rayleigh abs max              : {diagnostics['potential_rayleigh_abs_max']:.6e}")
        print(f"Potential Rayleigh signed min           : {diagnostics['potential_rayleigh_signed_min']:.6e}")
        print(f"Potential Rayleigh signed max           : {diagnostics['potential_rayleigh_signed_max']:.6e}")
        for probe in ("ones", "x", "y", "rand"):
            key = f"potential_rayleigh_{probe}"
            if key in diagnostics:
                print(f"Potential Rayleigh probe ({probe:<4})         : {diagnostics[key]:.6e}")
        print(f"Krylov residual drop                    : {diagnostics['krylov_residual_drop']:.6e}")
        print(f"Krylov geometric contraction            : {diagnostics['krylov_geometric_contraction']:.6e}")

    # ---------------------------------------------------- benchmark summary
    if benchmark_stats is not None:
        print(f"Benchmark RHS count                     : {benchmark_stats.n_rhs}")
        print(f"Benchmark AMG setup time [s]           : {benchmark_stats.amg_setup_time_s:.6e}")
        print(f"Benchmark first solve time [s]         : {benchmark_stats.solve_time_first_s:.6e}")
        print(f"Benchmark mean solve time [s]          : {benchmark_stats.solve_time_mean_s:.6e}")
        print(f"Benchmark min solve time [s]           : {benchmark_stats.solve_time_min_s:.6e}")
        print(f"Benchmark max solve time [s]           : {benchmark_stats.solve_time_max_s:.6e}")
        print(f"Benchmark mean iterations              : {benchmark_stats.iterations_mean:.6f}")
        print(f"Benchmark max iterations               : {benchmark_stats.iterations_max}")
        print(f"Benchmark all converged                : {benchmark_stats.all_info_zero}")
        print(f"Benchmark amortized per RHS [s]        : {benchmark_stats.amortized_total_per_rhs_s:.6e}")

    # ------------------------------------------------- boundary-band errors
    if args.report_boundary_bands:
        x_all, y_all = point_coordinates(mesh)
        dquery = build_boundary_distance_query(mesh)
        dist_inner = evaluate_distance_to_boundary(dquery, x_all[inner], y_all[inner])
        dist_over_h = dist_inner / float(mesh["spacing"])
        err_inner = err[inner]
        sq_total = float(np.sum(err_inner**2))
        band_edges = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, np.inf]
        print("Boundary-Band Error Statistics (inner points)")
        for lo, hi in zip(band_edges[:-1], band_edges[1:]):
            if np.isinf(hi):
                mask = dist_over_h >= lo
                label = f"[{lo:.0f}h, inf)"
            else:
                mask = (dist_over_h >= lo) & (dist_over_h < hi)
                label = f"[{lo:.0f}h, {hi:.0f}h)"
            count = int(np.count_nonzero(mask))
            if count == 0:
                print(f"  {label:<14} count=0")
                continue
            band_err = err_inner[mask]
            band_sq = float(np.sum(band_err**2))
            band_rms = float(np.sqrt(np.mean(band_err**2)))
            band_max = float(np.max(np.abs(band_err)))
            frac = band_sq / sq_total if sq_total > 0.0 else 0.0
            print(
                f"  {label:<14} count={count:<8d} rms={band_rms:.6e} "
                f"max={band_max:.6e} energy_frac={frac:.6e}"
            )

    # --------------------------------------------------------------- plots
    if args.plot or args.save_plots:
        import matplotlib.pyplot as plt

        xmin = float(mesh["xmin"])
        ymin = float(mesh["ymin"])
        spacing = float(mesh["spacing"])
        xmax = xmin + (int(mesh["nx_f"]) - 1) * spacing
        ymax = ymin + (int(mesh["ny_f"]) - 1) * spacing
        extent = [xmin, xmax, ymin, ymax]

        u_num_grid = scatter_field(mesh, u_num)
        u_exact_grid = scatter_field(mesh, u_exact)
        err_grid = scatter_field(mesh, err)

        fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
        for ax, field, title in (
            (axes1[0], u_exact_grid, "Exact Solution"),
            (axes1[1], u_num_grid, "FD4+PyAMG-BiCGSTAB Solution"),
            (axes1[2], err_grid, "Error"),
        ):
            im = ax.imshow(field, origin="lower", extent=extent, aspect="equal", cmap="coolwarm")
            fig1.colorbar(im, ax=ax, shrink=0.85, format="%.2e")
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 4.5), constrained_layout=True)
        if history:
            ax2.semilogy(np.arange(1, len(history) + 1), history, label=SOLVER_LABEL)
        ax2.set_xlabel("BiCGSTAB iteration")
        ax2.set_ylabel(r"$\|r_k\|_2 / \|b\|_2$")
        ax2.set_title("Normalized Residual Convergence")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend()

        if args.save_plots:
            out_base = args.data.resolve().with_suffix("")
            fig1.savefig(out_base.with_name(out_base.name + "_fd4_amg_solution.png"), dpi=180)
            fig2.savefig(out_base.with_name(out_base.name + "_fd4_amg_residual.png"), dpi=180)
        if args.plot:
            plt.show()
        else:
            plt.close(fig1)
            plt.close(fig2)

    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
