from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pyamg
from scipy import sparse
from scipy.sparse import linalg as spla


@dataclass(frozen=True)
class LinearSolveStats:
    info: int
    iterations: int
    setup_time_s: float = 0.0
    solve_time_s: float = 0.0
    total_time_s: float = 0.0
    levels: int = 0
    operator_complexity: float = 0.0
    grid_complexity: float = 0.0


@dataclass(frozen=True)
class AMGHierarchyStats:
    setup_time_s: float
    levels: int
    operator_complexity: float
    grid_complexity: float


@dataclass(frozen=True)
class BenchmarkStats:
    n_rhs: int
    amg_setup_time_s: float
    solve_time_first_s: float
    solve_time_mean_s: float
    solve_time_min_s: float
    solve_time_max_s: float
    iterations_mean: float
    iterations_max: int
    all_info_zero: int
    amortized_total_per_rhs_s: float


def build_amg_solver(A: sparse.csr_matrix, amg_type: str) -> object:
    if amg_type == "rs":
        return pyamg.ruge_stuben_solver(A)
    if amg_type == "sa":
        return pyamg.smoothed_aggregation_solver(A)
    raise ValueError(f"Unsupported AMG type: {amg_type}")


def make_residual_callback(A: sparse.spmatrix, b: np.ndarray, history: list[float]):
    b_norm = float(np.linalg.norm(b))
    denom = b_norm if b_norm > 0.0 else 1.0

    def _callback(xk: np.ndarray) -> None:
        rk = b - A @ xk
        history.append(float(np.linalg.norm(rk) / denom))

    return _callback


def compute_operator_diagnostics(
    A: sparse.csr_matrix,
    potential: np.ndarray,
    x_inner: np.ndarray,
    y_inner: np.ndarray,
    history: list[float],
) -> dict[str, float]:
    potential = np.asarray(potential, dtype=np.float64)
    x_inner = np.asarray(x_inner, dtype=np.float64)
    y_inner = np.asarray(y_inner, dtype=np.float64)

    a_diag = np.asarray(A.diagonal(), dtype=np.float64)
    k_diag = a_diag - potential
    diag_denom = np.maximum(np.abs(k_diag), 1.0e-14)
    diag_ratio = np.abs(potential) / diag_denom

    K = A.copy().tocsr()
    K.setdiag(k_diag)

    probes: list[tuple[str, np.ndarray]] = [("ones", np.ones_like(potential))]

    x_centered = x_inner - float(np.mean(x_inner))
    if np.linalg.norm(x_centered) > 0.0:
        probes.append(("x", x_centered))

    y_centered = y_inner - float(np.mean(y_inner))
    if np.linalg.norm(y_centered) > 0.0:
        probes.append(("y", y_centered))

    rng = np.random.default_rng(0)
    probes.append(("rand", rng.standard_normal(potential.size)))

    rayleigh_values: dict[str, float] = {}
    for name, vec in probes:
        denom = float(vec @ (K @ vec))
        numer = float(np.dot(potential * vec, vec))
        if abs(denom) <= 1.0e-14:
            continue
        rayleigh_values[name] = numer / denom

    contraction = math.nan
    residual_drop = math.nan
    if history:
        residual_drop = float(history[-1] / max(history[0], 1.0e-30))
        contraction = float(residual_drop ** (1.0 / max(len(history), 1)))

    out = {
        "potential_diag_ratio_max": float(np.max(diag_ratio)),
        "potential_diag_ratio_p95": float(np.percentile(diag_ratio, 95.0)),
        "potential_diag_ratio_mean": float(np.mean(diag_ratio)),
        "potential_rayleigh_abs_max": float(np.max(np.abs(np.fromiter(rayleigh_values.values(), dtype=np.float64)))) if rayleigh_values else math.nan,
        "potential_rayleigh_signed_min": float(np.min(np.fromiter(rayleigh_values.values(), dtype=np.float64))) if rayleigh_values else math.nan,
        "potential_rayleigh_signed_max": float(np.max(np.fromiter(rayleigh_values.values(), dtype=np.float64))) if rayleigh_values else math.nan,
        "krylov_geometric_contraction": contraction,
        "krylov_residual_drop": residual_drop,
    }
    for name, value in rayleigh_values.items():
        out[f"potential_rayleigh_{name}"] = float(value)
    return out


def solve_with_amg_bicgstab(
    A: sparse.csr_matrix,
    b: np.ndarray,
    *,
    tol: float,
    maxiter: int,
    amg_type: str,
) -> tuple[np.ndarray, list[float], LinearSolveStats]:
    start = time.perf_counter()
    ml = build_amg_solver(A, amg_type)
    setup_time = time.perf_counter() - start

    history: list[float] = []
    start = time.perf_counter()
    x, info = spla.bicgstab(
        A,
        b,
        M=ml.aspreconditioner(),
        rtol=tol,
        atol=0.0,
        maxiter=maxiter,
        callback=make_residual_callback(A, b, history),
    )
    solve_time = time.perf_counter() - start
    stats = LinearSolveStats(
        info=int(info),
        iterations=len(history),
        setup_time_s=setup_time,
        solve_time_s=solve_time,
        total_time_s=setup_time + solve_time,
        levels=len(ml.levels),
        operator_complexity=float(ml.operator_complexity()),
        grid_complexity=float(ml.grid_complexity()),
    )
    return x, history, stats


def build_amg_hierarchy(A: sparse.csr_matrix) -> tuple[object, AMGHierarchyStats]:
    start = time.perf_counter()
    ml = build_amg_solver(A, "rs")
    setup_time = time.perf_counter() - start
    stats = AMGHierarchyStats(
        setup_time_s=setup_time,
        levels=len(ml.levels),
        operator_complexity=float(ml.operator_complexity()),
        grid_complexity=float(ml.grid_complexity()),
    )
    return ml, stats


def build_amg_hierarchy_with_type(A: sparse.csr_matrix, amg_type: str) -> tuple[object, AMGHierarchyStats]:
    start = time.perf_counter()
    ml = build_amg_solver(A, amg_type)
    setup_time = time.perf_counter() - start
    stats = AMGHierarchyStats(
        setup_time_s=setup_time,
        levels=len(ml.levels),
        operator_complexity=float(ml.operator_complexity()),
        grid_complexity=float(ml.grid_complexity()),
    )
    return ml, stats


def solve_with_existing_amg(
    A: sparse.csr_matrix,
    b: np.ndarray,
    ml: object,
    *,
    tol: float,
    maxiter: int,
) -> tuple[np.ndarray, list[float], LinearSolveStats]:
    history: list[float] = []
    start = time.perf_counter()
    x, info = spla.bicgstab(
        A,
        b,
        M=ml.aspreconditioner(),
        rtol=tol,
        atol=0.0,
        maxiter=maxiter,
        callback=make_residual_callback(A, b, history),
    )
    solve_time = time.perf_counter() - start
    stats = LinearSolveStats(
        info=int(info),
        iterations=len(history),
        solve_time_s=solve_time,
    )
    return x, history, stats
