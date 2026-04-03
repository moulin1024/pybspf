from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .boundary import (
    build_axis_intersection_tables,
    build_boundary_models_from_samples,
    build_trace_splines,
    build_trace_splines_from_samples,
    evaluate_boundary_trace_value,
)
from .common import PINFO_BOUNDARY, PINFO_GHOST, PINFO_INNER, default_circular_params
from .io import point_coordinates


@dataclass(frozen=True)
class PreprocessMeta:
    n_inner: int
    n_boundary: int
    n_ghost: int
    n_unknowns: int
    nx_f: int
    ny_f: int
    n_irregular_x_rows: int
    n_irregular_y_rows: int
    boundary_setup_time_s: float
    lookup_setup_time_s: float
    template_build_time_s: float
    preprocess_total_time_s: float
    boundary_trace_calls: int
    boundary_trace_time_s: float
    triplet_concat_time_s: float = 0.0
    csr_build_time_s: float = 0.0
    online_assembly_time_s: float = 0.0
    nnz: int = 0


@dataclass(frozen=True)
class PreprocessedSystem:
    n_rows: int
    inner_indices: np.ndarray
    potential: np.ndarray
    x_inner: np.ndarray
    y_inner: np.ndarray
    u_exact: np.ndarray
    b_base: np.ndarray
    diag_rows: np.ndarray
    diag_cols: np.ndarray
    diag_vals: np.ndarray
    x_reg_rows: np.ndarray
    x_reg_cols: np.ndarray
    x_reg_vals: np.ndarray
    y_reg_rows: np.ndarray
    y_reg_cols: np.ndarray
    y_reg_vals: np.ndarray
    irr_rows: np.ndarray
    irr_cols: np.ndarray
    irr_vals: np.ndarray
    meta: PreprocessMeta


@dataclass(frozen=True)
class AssemblyResult:
    A: sparse.csr_matrix
    b: np.ndarray
    active_indices: np.ndarray
    u_exact: np.ndarray
    meta: PreprocessMeta


def build_cartesian_lookup(mesh: dict[str, np.ndarray | float | int]) -> tuple[np.ndarray, int, int]:
    cart_i = np.asarray(mesh["cart_i"], dtype=np.int32)
    cart_j = np.asarray(mesh["cart_j"], dtype=np.int32)
    imin = int(cart_i.min())
    imax = int(cart_i.max())
    jmin = int(cart_j.min())
    jmax = int(cart_j.max())
    lut = np.zeros((jmax - jmin + 1, imax - imin + 1), dtype=np.int32)
    lut[cart_j - jmin, cart_i - imin] = np.arange(1, cart_i.size + 1, dtype=np.int32)
    return lut, imin, jmin


def finite_difference_second_derivative_weights(offsets: list[float]) -> np.ndarray:
    pts = np.asarray(offsets, dtype=np.float64)
    n = pts.size
    if n < 3:
        raise ValueError("Need at least 3 stencil points for a second derivative.")
    vand = np.vstack([pts**k for k in range(n)])
    rhs = np.zeros(n, dtype=np.float64)
    rhs[2] = 2.0
    return np.linalg.solve(vand, rhs)


def build_axis_fd4_stencil(
    *,
    gidx0: int,
    row: int,
    axis: str,
    cart_i_all: np.ndarray,
    cart_j_all: np.ndarray,
    pinfo: np.ndarray,
    cart_lookup: np.ndarray,
    cart_i_min: int,
    cart_j_min: int,
    global_to_inner: np.ndarray,
    spacing: float,
    x_i: float,
    y_i: float,
    trace_splines: dict[str, object],
    rhomin: float,
    rhomax: float,
    boundary_models: dict[str, dict[str, np.ndarray | object]] | None = None,
    axis_tables: dict[str, object] | None = None,
    perf: dict[str, float | int] | None = None,
) -> tuple[list[tuple[int, float]], float, float, bool]:
    step = (1, 0) if axis == "x" else (0, 1)

    i0 = int(cart_i_all[gidx0])
    j0 = int(cart_j_all[gidx0])

    def collect_side(sign: int, max_steps: int = 8) -> tuple[np.ndarray, np.ndarray]:
        offsets = np.empty(max_steps, dtype=np.float64)
        cols = np.empty(max_steps, dtype=np.int32)
        n_found = 0
        di = sign * step[0]
        dj = sign * step[1]
        for k in range(1, max_steps + 1):
            ii = i0 + k * di - cart_i_min
            jj = j0 + k * dj - cart_j_min
            if ii < 0 or jj < 0 or jj >= cart_lookup.shape[0] or ii >= cart_lookup.shape[1]:
                break
            nidx1 = int(cart_lookup[jj, ii])
            if nidx1 <= 0 or pinfo[nidx1 - 1] != PINFO_INNER:
                break
            offsets[n_found] = sign * k * spacing
            cols[n_found] = int(global_to_inner[nidx1])
            n_found += 1
        return offsets[:n_found], cols[:n_found]

    neg_offsets, neg_cols = collect_side(-1)
    pos_offsets, pos_cols = collect_side(1)

    if neg_offsets.size >= 2 and pos_offsets.size >= 2:
        weights = {
            -2.0 * spacing: -1.0 / (12.0 * spacing * spacing),
            -1.0 * spacing: 16.0 / (12.0 * spacing * spacing),
            0.0: -30.0 / (12.0 * spacing * spacing),
            1.0 * spacing: 16.0 / (12.0 * spacing * spacing),
            2.0 * spacing: -1.0 / (12.0 * spacing * spacing),
        }
        entries: list[tuple[int, float]] = [(row, weights[0.0])]
        entries.append((int(neg_cols[1]), weights[-2.0 * spacing]))
        entries.append((int(neg_cols[0]), weights[-1.0 * spacing]))
        entries.append((int(pos_cols[0]), weights[1.0 * spacing]))
        entries.append((int(pos_cols[1]), weights[2.0 * spacing]))
        return entries, 0.0, 0.0, False

    start = time.perf_counter()
    neg_delta, neg_value = evaluate_boundary_trace_value(
        direction=(-step[0], -step[1]),
        cart_i0=i0,
        cart_j0=j0,
        x_i=x_i,
        y_i=y_i,
        trace_splines=trace_splines,
        rhomin=rhomin,
        rhomax=rhomax,
        boundary_models=boundary_models,
        axis_tables=axis_tables,
    )
    pos_delta, pos_value = evaluate_boundary_trace_value(
        direction=(step[0], step[1]),
        cart_i0=i0,
        cart_j0=j0,
        x_i=x_i,
        y_i=y_i,
        trace_splines=trace_splines,
        rhomin=rhomin,
        rhomax=rhomax,
        boundary_models=boundary_models,
        axis_tables=axis_tables,
    )
    if perf is not None:
        perf["boundary_trace_calls"] = int(perf.get("boundary_trace_calls", 0)) + 2
        perf["boundary_trace_time_s"] = float(perf.get("boundary_trace_time_s", 0.0)) + (time.perf_counter() - start)

    all_offsets = [0.0]
    all_kinds = [0]  # 0=center, 1=interior, 2=boundary
    all_cols = [row]
    all_vals = [0.0]

    if neg_offsets.size < 2:
        all_offsets.append(-neg_delta)
        all_kinds.append(2)
        all_cols.append(-1)
        all_vals.append(neg_value)
    if pos_offsets.size < 2:
        all_offsets.append(pos_delta)
        all_kinds.append(2)
        all_cols.append(-1)
        all_vals.append(pos_value)

    interior_offsets = np.concatenate((neg_offsets, pos_offsets))
    interior_cols = np.concatenate((neg_cols, pos_cols))
    if interior_offsets.size:
        order = np.lexsort((interior_offsets, np.abs(interior_offsets)))
        interior_offsets = interior_offsets[order]
        interior_cols = interior_cols[order]

    for off, col in zip(interior_offsets.tolist(), interior_cols.tolist()):
        all_offsets.append(off)
        all_kinds.append(1)
        all_cols.append(col)
        all_vals.append(0.0)

    offset_set = set(all_offsets)
    if -neg_delta not in offset_set:
        all_offsets.append(-neg_delta)
        all_kinds.append(2)
        all_cols.append(-1)
        all_vals.append(neg_value)
    if pos_delta not in offset_set:
        all_offsets.append(pos_delta)
        all_kinds.append(2)
        all_cols.append(-1)
        all_vals.append(pos_value)

    selected_idx_list: list[int] = []
    used_offsets: set[float] = set()
    for idx, offset in enumerate(all_offsets):
        if offset in used_offsets:
            continue
        selected_idx_list.append(idx)
        used_offsets.add(offset)
        if len(selected_idx_list) >= 6:
            break

    if len(selected_idx_list) < 6:
        raise SystemExit(f"Unable to build a 4th-order cut-cell stencil in {axis}-direction at row {row}.")

    selected_idx = np.asarray(selected_idx_list[:6], dtype=np.int32)
    selected_offsets = np.asarray([all_offsets[i] for i in selected_idx], dtype=np.float64)
    selected_kinds = np.asarray([all_kinds[i] for i in selected_idx], dtype=np.int8)
    selected_cols = np.asarray([all_cols[i] for i in selected_idx], dtype=np.int32)
    selected_vals = np.asarray([all_vals[i] for i in selected_idx], dtype=np.float64)

    order = np.argsort(selected_offsets)
    selected_offsets = selected_offsets[order]
    selected_kinds = selected_kinds[order]
    selected_cols = selected_cols[order]
    selected_vals = selected_vals[order]

    weights = finite_difference_second_derivative_weights(selected_offsets.tolist())

    interior_mask = selected_kinds == 1
    boundary_mask = selected_kinds == 2
    center_mask = selected_kinds == 0

    matrix_entries = [
        (int(col), float(weight))
        for col, weight in zip(selected_cols[interior_mask].tolist(), weights[interior_mask].tolist())
    ]
    rhs_shift = float(np.dot(weights[boundary_mask], selected_vals[boundary_mask]))
    center_weight = float(weights[center_mask][0])
    matrix_entries.append((row, center_weight))
    return matrix_entries, rhs_shift, center_weight, True


def preprocess_fd4_system(
    mesh: dict[str, np.ndarray | float | int],
    fields: dict[str, np.ndarray],
) -> PreprocessedSystem:
    t0 = time.perf_counter()
    if not np.allclose(fields["co"], 1.0) or not np.allclose(fields["xi_inner"], 1.0):
        raise SystemExit("shortley-weller-4th currently requires the transformed Schrodinger-form data with co=1 and xi=1.")

    params = default_circular_params()
    rhomin = params["rhomin"]
    rhomax = params["rhomax"]
    has_sample_geometry = all(
        key in mesh for key in (
            "inner_boundary_sample_x",
            "inner_boundary_sample_y",
            "outer_boundary_sample_x",
            "outer_boundary_sample_y",
        )
    ) and all(key in fields for key in ("inner_boundary_sample_value", "outer_boundary_sample_value"))
    if has_sample_geometry:
        trace_splines = build_trace_splines_from_samples(fields)
        boundary_models = build_boundary_models_from_samples(mesh)
        axis_tables = build_axis_intersection_tables(mesh, boundary_models)
    else:
        trace_splines = build_trace_splines(rhomin=rhomin, rhomax=rhomax, form="schrodinger")
        boundary_models = None
        axis_tables = None
    t_boundary_setup = time.perf_counter()

    pinfo = np.asarray(mesh["pinfo"], dtype=np.int32)
    inner_indices = np.asarray(mesh["inner_indices"], dtype=np.int32)
    global_to_inner = np.full(pinfo.size + 1, -1, dtype=np.int32)
    global_to_inner[inner_indices] = np.arange(inner_indices.size, dtype=np.int32)
    cart_i_all = np.asarray(mesh["cart_i"], dtype=np.int32)
    cart_j_all = np.asarray(mesh["cart_j"], dtype=np.int32)
    cart_lookup, cart_i_min, cart_j_min = build_cartesian_lookup(mesh)
    t_lookup = time.perf_counter()

    x_all, y_all = point_coordinates(mesh)
    rhs_all = np.asarray(fields["rhs"], dtype=np.float64)
    u_exact = np.asarray(fields["sol"], dtype=np.float64)
    lambda_inner = np.asarray(fields["lambda_inner"], dtype=np.float64)
    spacing = float(mesh["spacing"])

    n_rows = inner_indices.size
    n_irregular_x = 0
    n_irregular_y = 0
    perf: dict[str, float | int] = {"boundary_trace_calls": 0, "boundary_trace_time_s": 0.0}
    std_center = -30.0 / (12.0 * spacing * spacing)
    std_near = 16.0 / (12.0 * spacing * spacing)
    std_far = -1.0 / (12.0 * spacing * spacing)

    lut_pad = np.pad(cart_lookup, ((2, 2), (2, 2)), mode="constant")
    inner_gidx0 = inner_indices - 1
    b = rhs_all[inner_gidx0].astype(np.float64).copy()
    diag = lambda_inner.astype(np.float64).copy()
    ii0 = cart_i_all[inner_gidx0] - cart_i_min + 2
    jj0 = cart_j_all[inner_gidx0] - cart_j_min + 2

    x_gids = np.stack([lut_pad[jj0, ii0 - 2], lut_pad[jj0, ii0 - 1], lut_pad[jj0, ii0 + 1], lut_pad[jj0, ii0 + 2]], axis=1)
    y_gids = np.stack([lut_pad[jj0 - 2, ii0], lut_pad[jj0 - 1, ii0], lut_pad[jj0 + 1, ii0], lut_pad[jj0 + 2, ii0]], axis=1)

    x_valid = x_gids > 0
    y_valid = y_gids > 0
    x_pinfo = np.zeros_like(x_gids)
    y_pinfo = np.zeros_like(y_gids)
    x_pinfo[x_valid] = pinfo[x_gids[x_valid] - 1]
    y_pinfo[y_valid] = pinfo[y_gids[y_valid] - 1]
    x_regular = np.all(x_valid & (x_pinfo == PINFO_INNER), axis=1)
    y_regular = np.all(y_valid & (y_pinfo == PINFO_INNER), axis=1)
    x_cols_regular = global_to_inner[x_gids]
    y_cols_regular = global_to_inner[y_gids]

    regular_weights = np.asarray([std_far, std_near, std_near, std_far], dtype=np.float64)
    x_reg_rows = np.nonzero(x_regular)[0].astype(np.int32)
    y_reg_rows = np.nonzero(y_regular)[0].astype(np.int32)
    diag[x_reg_rows] -= std_center
    diag[y_reg_rows] -= std_center

    x_reg_triplet_rows = np.repeat(x_reg_rows, 4)
    x_reg_triplet_cols = x_cols_regular[x_reg_rows].reshape(-1).astype(np.int32, copy=False)
    x_reg_triplet_vals = -np.tile(regular_weights, x_reg_rows.size)
    y_reg_triplet_rows = np.repeat(y_reg_rows, 4)
    y_reg_triplet_cols = y_cols_regular[y_reg_rows].reshape(-1).astype(np.int32, copy=False)
    y_reg_triplet_vals = -np.tile(regular_weights, y_reg_rows.size)

    irr_rows: list[int] = []
    irr_cols: list[int] = []
    irr_vals: list[float] = []

    for row, gidx1 in enumerate(inner_indices.tolist()):
        gidx0 = gidx1 - 1
        x_i = float(x_all[gidx0])
        y_i = float(y_all[gidx0])
        if bool(x_regular[row]):
            x_rhs_shift = 0.0
            x_irregular = False
        else:
            x_entries, x_rhs_shift, _, x_irregular = build_axis_fd4_stencil(
                gidx0=gidx0,
                row=row,
                axis="x",
                cart_i_all=cart_i_all,
                cart_j_all=cart_j_all,
                pinfo=pinfo,
                cart_lookup=cart_lookup,
                cart_i_min=cart_i_min,
                cart_j_min=cart_j_min,
                global_to_inner=global_to_inner,
                spacing=spacing,
                x_i=x_i,
                y_i=y_i,
                trace_splines=trace_splines,
                rhomin=rhomin,
                rhomax=rhomax,
                boundary_models=boundary_models,
                axis_tables=axis_tables,
                perf=perf,
            )
            for col, weight in x_entries:
                if col == row:
                    diag[row] -= weight
                else:
                    irr_rows.append(row)
                    irr_cols.append(col)
                    irr_vals.append(-weight)

        if bool(y_regular[row]):
            y_rhs_shift = 0.0
            y_irregular = False
        else:
            y_entries, y_rhs_shift, _, y_irregular = build_axis_fd4_stencil(
                gidx0=gidx0,
                row=row,
                axis="y",
                cart_i_all=cart_i_all,
                cart_j_all=cart_j_all,
                pinfo=pinfo,
                cart_lookup=cart_lookup,
                cart_i_min=cart_i_min,
                cart_j_min=cart_j_min,
                global_to_inner=global_to_inner,
                spacing=spacing,
                x_i=x_i,
                y_i=y_i,
                trace_splines=trace_splines,
                rhomin=rhomin,
                rhomax=rhomax,
                boundary_models=boundary_models,
                axis_tables=axis_tables,
                perf=perf,
            )
            for col, weight in y_entries:
                if col == row:
                    diag[row] -= weight
                else:
                    irr_rows.append(row)
                    irr_cols.append(col)
                    irr_vals.append(-weight)

        b[row] += x_rhs_shift + y_rhs_shift
        n_irregular_x += int(x_irregular)
        n_irregular_y += int(y_irregular)

    t_loop = time.perf_counter()
    diag_rows = np.arange(n_rows, dtype=np.int32)
    diag_cols = diag_rows
    diag_vals = diag

    irr_rows_arr = np.asarray(irr_rows, dtype=np.int32) if irr_rows else np.empty(0, dtype=np.int32)
    irr_cols_arr = np.asarray(irr_cols, dtype=np.int32) if irr_cols else np.empty(0, dtype=np.int32)
    irr_vals_arr = np.asarray(irr_vals, dtype=np.float64) if irr_vals else np.empty(0, dtype=np.float64)

    t_pre = time.perf_counter()
    meta = PreprocessMeta(
        n_inner=int(np.count_nonzero(pinfo == PINFO_INNER)),
        n_boundary=int(np.count_nonzero(pinfo == PINFO_BOUNDARY)),
        n_ghost=int(np.count_nonzero(pinfo == PINFO_GHOST)),
        n_unknowns=int(inner_indices.size),
        nx_f=int(mesh["nx_f"]),
        ny_f=int(mesh["ny_f"]),
        n_irregular_x_rows=n_irregular_x,
        n_irregular_y_rows=n_irregular_y,
        boundary_setup_time_s=t_boundary_setup - t0,
        lookup_setup_time_s=t_lookup - t_boundary_setup,
        template_build_time_s=t_loop - t_lookup,
        preprocess_total_time_s=t_pre - t0,
        boundary_trace_calls=int(perf["boundary_trace_calls"]),
        boundary_trace_time_s=float(perf["boundary_trace_time_s"]),
    )
    return PreprocessedSystem(
        n_rows=n_rows,
        inner_indices=inner_indices,
        potential=lambda_inner,
        x_inner=x_all[inner_gidx0].astype(np.float64, copy=False),
        y_inner=y_all[inner_gidx0].astype(np.float64, copy=False),
        u_exact=u_exact,
        b_base=b,
        diag_rows=diag_rows,
        diag_cols=diag_cols,
        diag_vals=diag_vals,
        x_reg_rows=x_reg_triplet_rows,
        x_reg_cols=x_reg_triplet_cols,
        x_reg_vals=x_reg_triplet_vals,
        y_reg_rows=y_reg_triplet_rows,
        y_reg_cols=y_reg_triplet_cols,
        y_reg_vals=y_reg_triplet_vals,
        irr_rows=irr_rows_arr,
        irr_cols=irr_cols_arr,
        irr_vals=irr_vals_arr,
        meta=meta,
    )


def assemble_fd4_system_from_preprocessed(
    pre: PreprocessedSystem,
) -> AssemblyResult:
    t0 = time.perf_counter()
    n_rows = int(pre.n_rows)
    b = np.asarray(pre.b_base, dtype=np.float64).copy()

    all_rows = np.concatenate((
        np.asarray(pre.x_reg_rows, dtype=np.int32),
        np.asarray(pre.y_reg_rows, dtype=np.int32),
        np.asarray(pre.irr_rows, dtype=np.int32),
        np.asarray(pre.diag_rows, dtype=np.int32),
    ))
    all_cols = np.concatenate((
        np.asarray(pre.x_reg_cols, dtype=np.int32),
        np.asarray(pre.y_reg_cols, dtype=np.int32),
        np.asarray(pre.irr_cols, dtype=np.int32),
        np.asarray(pre.diag_cols, dtype=np.int32),
    ))
    all_vals = np.concatenate((
        np.asarray(pre.x_reg_vals, dtype=np.float64),
        np.asarray(pre.y_reg_vals, dtype=np.float64),
        np.asarray(pre.irr_vals, dtype=np.float64),
        np.asarray(pre.diag_vals, dtype=np.float64),
    ))
    t_triplet = time.perf_counter()
    A = sparse.coo_array((all_vals, (all_rows, all_cols)), shape=(n_rows, n_rows)).tocsr()
    t_csr = time.perf_counter()

    meta = PreprocessMeta(
        n_inner=pre.meta.n_inner,
        n_boundary=pre.meta.n_boundary,
        n_ghost=pre.meta.n_ghost,
        n_unknowns=pre.meta.n_unknowns,
        nx_f=pre.meta.nx_f,
        ny_f=pre.meta.ny_f,
        n_irregular_x_rows=pre.meta.n_irregular_x_rows,
        n_irregular_y_rows=pre.meta.n_irregular_y_rows,
        boundary_setup_time_s=pre.meta.boundary_setup_time_s,
        lookup_setup_time_s=pre.meta.lookup_setup_time_s,
        template_build_time_s=pre.meta.template_build_time_s,
        preprocess_total_time_s=pre.meta.preprocess_total_time_s,
        boundary_trace_calls=pre.meta.boundary_trace_calls,
        boundary_trace_time_s=pre.meta.boundary_trace_time_s,
        triplet_concat_time_s=t_triplet - t0,
        csr_build_time_s=t_csr - t_triplet,
        online_assembly_time_s=t_csr - t0,
        nnz=int(A.nnz),
    )
    return AssemblyResult(
        A=A,
        b=b,
        active_indices=np.asarray(pre.inner_indices, dtype=np.int32),
        u_exact=np.asarray(pre.u_exact, dtype=np.float64),
        meta=meta,
    )
