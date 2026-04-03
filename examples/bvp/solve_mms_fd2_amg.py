#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyamg
from netCDF4 import Dataset
from scipy import sparse
from scipy.sparse import linalg as spla

PINFO_INNER = 1
PINFO_BOUNDARY = 2
PINFO_GHOST = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FD2 + AMG + PCG baseline for the circular MMS BVP.")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path(__file__).with_name("multigrid.nc"),
        help="Path to multigrid.nc.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).with_name("schrodinger_data.nc"),
        help="Path to helmholtz_data.nc or schrodinger_data.nc.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show diagnostic plots.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save diagnostic plots next to the data file.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1.0e-10,
        help="Relative tolerance for PCG.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=1000,
        help="Maximum PCG iterations.",
    )
    return parser.parse_args()


def load_mesh(mesh_path: Path) -> dict[str, np.ndarray | float | int]:
    with Dataset(mesh_path, "r") as nc:
        group = nc["mesh_lvl_001"]
        return {
            "spacing": float(group.getncattr("spacing_f")),
            "xmin": float(group.getncattr("xmin")),
            "ymin": float(group.getncattr("ymin")),
            "nx_f": int(group.getncattr("nx_f")),
            "ny_f": int(group.getncattr("ny_f")),
            "size_neighbor": (group.dimensions["size_neighbor"].size - 1) // 2,
            "cart_i": group["cart_i"][:].astype(np.int32),
            "cart_j": group["cart_j"][:].astype(np.int32),
            "pinfo": group["pinfo"][:].astype(np.int32),
            "index_neighbor": group["index_neighbor"][:].astype(np.int32),
            "inner_indices": group["inner_indices"][:].astype(np.int32),
            "boundary_indices": group["boundary_indices"][:].astype(np.int32),
            "ghost_indices": group["ghost_indices"][:].astype(np.int32),
        }


def load_fields(data_path: Path) -> dict[str, np.ndarray]:
    with Dataset(data_path, "r") as nc:
        fields = {
            "co": nc["co"][:].astype(np.float64),
            "lambda_inner": nc["lambda"][:].astype(np.float64),
            "xi_inner": nc["xi"][:].astype(np.float64),
            "rhs": nc["rhs"][:].astype(np.float64),
            "sol": nc["sol"][:].astype(np.float64),
            "guess": nc["guess"][:].astype(np.float64),
        }
        if "potential" in nc.variables:
            fields["potential"] = nc["potential"][:].astype(np.float64)
        if "transform_weight" in nc.variables:
            fields["transform_weight"] = nc["transform_weight"][:].astype(np.float64)
        return fields


def make_residual_callback(A: sparse.spmatrix, b: np.ndarray, history: list[float]):
    b_norm = float(np.linalg.norm(b))
    denom = b_norm if b_norm > 0.0 else 1.0

    def _callback(xk: np.ndarray) -> None:
        rk = b - A @ xk
        history.append(float(np.linalg.norm(rk) / denom))

    return _callback


def assemble_fd2_system(mesh: dict[str, np.ndarray | float | int], fields: dict[str, np.ndarray]) -> tuple[sparse.csr_matrix, np.ndarray, dict[str, int]]:
    spacing = float(mesh["spacing"])
    inv_h2 = 1.0 / (spacing * spacing)

    cart_i = np.asarray(mesh["cart_i"], dtype=np.int32)
    cart_j = np.asarray(mesh["cart_j"], dtype=np.int32)
    pinfo = np.asarray(mesh["pinfo"], dtype=np.int32)
    index_neighbor = np.asarray(mesh["index_neighbor"], dtype=np.int32)
    inner_indices = np.asarray(mesh["inner_indices"], dtype=np.int32)
    size_neighbor = int(mesh["size_neighbor"])

    co = np.asarray(fields["co"], dtype=np.float64)
    rhs = np.asarray(fields["rhs"], dtype=np.float64)
    sol = np.asarray(fields["sol"], dtype=np.float64)
    lambda_inner = np.asarray(fields["lambda_inner"], dtype=np.float64)
    xi_inner = np.asarray(fields["xi_inner"], dtype=np.float64)

    n_inner = inner_indices.size
    global_to_inner = {int(gidx): row for row, gidx in enumerate(inner_indices.tolist())}

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.empty(n_inner, dtype=np.float64)

    directions = [
        (size_neighbor - 1, size_neighbor),
        (size_neighbor, size_neighbor - 1),
        (size_neighbor, size_neighbor + 1),
        (size_neighbor + 1, size_neighbor),
    ]

    for row, gidx1 in enumerate(inner_indices.tolist()):
        gidx0 = gidx1 - 1
        xi = float(xi_inner[row])
        reaction = float(lambda_inner[row] / xi)
        rhs_row = float(rhs[gidx0] / xi)
        diag = reaction

        for jj, ii in directions:
            nidx1 = int(index_neighbor[gidx0, jj, ii])
            if nidx1 <= 0:
                raise ValueError(f"Inner point {gidx1} is missing a direct neighbor; mesh is inconsistent for FD2.")

            nidx0 = nidx1 - 1
            c_face = 0.5 * (co[gidx0] + co[nidx0])
            weight = c_face * inv_h2
            diag += weight

            if pinfo[nidx0] == PINFO_INNER:
                rows.append(row)
                cols.append(global_to_inner[nidx1])
                vals.append(-weight)
            else:
                rhs_row += weight * sol[nidx0]

        rows.append(row)
        cols.append(row)
        vals.append(diag)
        b[row] = rhs_row

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n_inner, n_inner))
    meta = {
        "n_inner": n_inner,
        "n_boundary": int(np.count_nonzero(pinfo == PINFO_BOUNDARY)),
        "n_ghost": int(np.count_nonzero(pinfo == PINFO_GHOST)),
        "nnz": int(A.nnz),
        "nx_f": int(mesh["nx_f"]),
        "ny_f": int(mesh["ny_f"]),
    }
    return A, b, meta


def scatter_field(mesh: dict[str, np.ndarray | float | int], values: np.ndarray) -> np.ndarray:
    field = np.full((int(mesh["ny_f"]), int(mesh["nx_f"])), np.nan, dtype=np.float64)
    ii = np.asarray(mesh["cart_i"], dtype=np.int64) - 1
    jj = np.asarray(mesh["cart_j"], dtype=np.int64) - 1
    field[jj, ii] = values
    return field


def solve_with_amg_pcg(A: sparse.csr_matrix, b: np.ndarray, *, tol: float, maxiter: int) -> tuple[np.ndarray, list[float], dict[str, float | int]]:
    start = time.perf_counter()
    ml = pyamg.ruge_stuben_solver(A)
    setup_time = time.perf_counter() - start

    history: list[float] = []
    start = time.perf_counter()
    x, info = spla.cg(
        A,
        b,
        M=ml.aspreconditioner(),
        rtol=tol,
        atol=0.0,
        maxiter=maxiter,
        callback=make_residual_callback(A, b, history),
    )
    solve_time = time.perf_counter() - start
    stats = {
        "info": int(info),
        "iterations": len(history),
        "setup_time_s": setup_time,
        "solve_time_s": solve_time,
        "total_time_s": setup_time + solve_time,
        "levels": len(ml.levels),
        "operator_complexity": float(ml.operator_complexity()),
        "grid_complexity": float(ml.grid_complexity()),
    }
    return x, history, stats


def main() -> int:
    args = parse_args()
    mesh = load_mesh(args.mesh.resolve())
    fields = load_fields(args.data.resolve())

    A, b, meta = assemble_fd2_system(mesh, fields)
    x, history, stats = solve_with_amg_pcg(A, b, tol=args.tol, maxiter=args.maxiter)

    u_num = np.asarray(fields["sol"], dtype=np.float64).copy()
    inner = np.asarray(mesh["inner_indices"], dtype=np.int32) - 1
    u_num[inner] = x
    u_exact = np.asarray(fields["sol"], dtype=np.float64)
    err = u_num - u_exact

    inner_rms = float(np.sqrt(np.mean(err[inner] ** 2)))
    inner_ref = float(np.sqrt(np.mean(u_exact[inner] ** 2)))
    active = np.asarray(mesh["pinfo"], dtype=np.int32) != PINFO_GHOST
    active_rms = float(np.sqrt(np.mean(err[active] ** 2)))
    active_ref = float(np.sqrt(np.mean(u_exact[active] ** 2)))

    print(f"FD2+PyAMG-PCG error L2 rms (inner)      : {inner_rms:.6e}")
    print(f"FD2+PyAMG-PCG relative L2 err (inner)   : {inner_rms / inner_ref:.6e}")
    print(f"FD2+PyAMG-PCG error L2 rms (active)     : {active_rms:.6e}")
    print(f"FD2+PyAMG-PCG relative L2 err (active)  : {active_rms / active_ref:.6e}")
    print(f"FD2+PyAMG-PCG iterations                : {stats['iterations']}")
    print(f"FD2+PyAMG-PCG info                      : {stats['info']}")
    print(f"FD2+PyAMG-PCG setup time [s]            : {stats['setup_time_s']:.6e}")
    print(f"FD2+PyAMG-PCG solve time [s]            : {stats['solve_time_s']:.6e}")
    print(f"FD2+PyAMG-PCG total time [s]            : {stats['total_time_s']:.6e}")
    print(f"FD2+PyAMG-PCG AMG levels                : {stats['levels']}")
    print(f"FD2+PyAMG-PCG operator complexity       : {stats['operator_complexity']:.6e}")
    print(f"FD2+PyAMG-PCG grid complexity           : {stats['grid_complexity']:.6e}")
    print(f"Unknown inner points                    : {meta['n_inner']}")
    print(f"Boundary support points                 : {meta['n_boundary']}")
    print(f"Ghost support points                    : {meta['n_ghost']}")
    print(f"Matrix nnz                              : {meta['nnz']}")

    if args.plot or args.save_plots:
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
            (axes1[1], u_num_grid, "FD2+PyAMG-PCG Solution"),
            (axes1[2], err_grid, "Error"),
        ):
            im = ax.imshow(field, origin="lower", extent=extent, aspect="equal", cmap="coolwarm")
            fig1.colorbar(im, ax=ax, shrink=0.85, format="%.2e")
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 4.5), constrained_layout=True)
        if history:
            ax2.semilogy(np.arange(1, len(history) + 1), history, label="FD2+PyAMG-PCG")
        ax2.set_xlabel("PCG iteration")
        ax2.set_ylabel(r"$\|r_k\|_2 / \|b\|_2$")
        ax2.set_title("Normalized Residual Convergence")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend()

        if args.save_plots:
            out_base = args.data.resolve().with_suffix("")
            fig1.savefig(out_base.with_name(out_base.name + "_fd2_amg_solution.png"), dpi=180)
            fig2.savefig(out_base.with_name(out_base.name + "_fd2_amg_residual.png"), dpi=180)
        if args.plot:
            plt.show()
        else:
            plt.close(fig1)
            plt.close(fig2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
