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
    evaluate_boundary_trace_values_batch,
)
from .common import PINFO_BOUNDARY, PINFO_GHOST, PINFO_INNER, default_circular_params
from .io import point_coordinates

MAX_TEMPLATE_STEPS = 8


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


def finite_difference_second_derivative_weights_batch(offsets: np.ndarray) -> np.ndarray:
    pts = np.asarray(offsets, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("Expected a 2D array of stencil offsets.")
    n_templates, n = pts.shape
    if n < 3:
        raise ValueError("Need at least 3 stencil points for a second derivative.")
    powers = np.arange(n, dtype=np.float64)
    vand = pts[:, np.newaxis, :] ** powers[np.newaxis, :, np.newaxis]
    rhs = np.zeros((n_templates, n), dtype=np.float64)
    rhs[:, 2] = 2.0
    return np.linalg.solve(vand, rhs[..., np.newaxis])[..., 0]


def collect_side_batch(
    *,
    lut_pad: np.ndarray,
    base_i: np.ndarray,
    base_j: np.ndarray,
    di: int,
    dj: int,
    pinfo: np.ndarray,
    global_to_inner: np.ndarray,
    spacing: float,
    sign: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = base_i.size
    steps = np.arange(1, MAX_TEMPLATE_STEPS + 1, dtype=np.int32)
    ii = base_i[:, None] + di * steps[None, :]
    jj = base_j[:, None] + dj * steps[None, :]
    gids = lut_pad[jj, ii]

    valid = gids > 0
    kinds = np.zeros_like(gids, dtype=np.int32)
    kinds[valid] = pinfo[gids[valid] - 1]
    prefix_valid = np.logical_and.accumulate(valid & (kinds == PINFO_INNER), axis=1)
    counts = np.sum(prefix_valid, axis=1, dtype=np.int32)

    cols = np.full((n, MAX_TEMPLATE_STEPS), -1, dtype=np.int32)
    cols[prefix_valid] = global_to_inner[gids[prefix_valid]]
    offsets = sign * spacing * np.broadcast_to(steps[None, :], (n, MAX_TEMPLATE_STEPS)).astype(np.float64)
    return offsets, cols, counts


def build_axis_fd4_templates_batch(
    *,
    axis: str,
    rows: np.ndarray,
    inner_indices: np.ndarray,
    cart_i_all: np.ndarray,
    cart_j_all: np.ndarray,
    pinfo: np.ndarray,
    lut_pad: np.ndarray,
    cart_i_min: int,
    cart_j_min: int,
    global_to_inner: np.ndarray,
    spacing: float,
    x_all: np.ndarray,
    y_all: np.ndarray,
    trace_splines: dict[str, object],
    rhomin: float,
    rhomax: float,
    boundary_models: dict[str, dict[str, np.ndarray | object]] | None,
    axis_tables: dict[str, object] | None,
    perf: dict[str, float | int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int32)
    if rows.size == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty((0, 6), dtype=np.float64),
            np.empty((0, 6), dtype=np.int8),
            np.empty((0, 6), dtype=np.int32),
            np.empty((0, 6), dtype=np.float64),
        )

    gidx0 = inner_indices[rows] - 1
    x_i = np.asarray(x_all[gidx0], dtype=np.float64)
    y_i = np.asarray(y_all[gidx0], dtype=np.float64)
    i0 = np.asarray(cart_i_all[gidx0], dtype=np.int32)
    j0 = np.asarray(cart_j_all[gidx0], dtype=np.int32)
    base_i = i0 - cart_i_min + MAX_TEMPLATE_STEPS
    base_j = j0 - cart_j_min + MAX_TEMPLATE_STEPS

    if axis == "x":
        neg_offsets_batch, neg_cols_batch, neg_counts = collect_side_batch(
            lut_pad=lut_pad,
            base_i=base_i,
            base_j=base_j,
            di=-1,
            dj=0,
            pinfo=pinfo,
            global_to_inner=global_to_inner,
            spacing=spacing,
            sign=-1.0,
        )
        pos_offsets_batch, pos_cols_batch, pos_counts = collect_side_batch(
            lut_pad=lut_pad,
            base_i=base_i,
            base_j=base_j,
            di=1,
            dj=0,
            pinfo=pinfo,
            global_to_inner=global_to_inner,
            spacing=spacing,
            sign=1.0,
        )
    else:
        neg_offsets_batch, neg_cols_batch, neg_counts = collect_side_batch(
            lut_pad=lut_pad,
            base_i=base_i,
            base_j=base_j,
            di=0,
            dj=-1,
            pinfo=pinfo,
            global_to_inner=global_to_inner,
            spacing=spacing,
            sign=-1.0,
        )
        pos_offsets_batch, pos_cols_batch, pos_counts = collect_side_batch(
            lut_pad=lut_pad,
            base_i=base_i,
            base_j=base_j,
            di=0,
            dj=1,
            pinfo=pinfo,
            global_to_inner=global_to_inner,
            spacing=spacing,
            sign=1.0,
        )

    start = time.perf_counter()
    neg_delta, neg_value = evaluate_boundary_trace_values_batch(
        axis=axis,
        direction_sign=-1,
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
    pos_delta, pos_value = evaluate_boundary_trace_values_batch(
        axis=axis,
        direction_sign=1,
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
        perf["boundary_trace_calls"] = int(perf.get("boundary_trace_calls", 0)) + 2 * rows.size
        perf["boundary_trace_time_s"] = float(perf.get("boundary_trace_time_s", 0.0)) + (time.perf_counter() - start)

    n_rows = rows.size
    n_interior_slots = 2 * MAX_TEMPLATE_STEPS
    n_candidate_slots = 1 + 2 + n_interior_slots + 2
    candidate_offsets = np.zeros((n_rows, n_candidate_slots), dtype=np.float64)
    candidate_kinds = np.full((n_rows, n_candidate_slots), -1, dtype=np.int8)
    candidate_cols = np.full((n_rows, n_candidate_slots), -1, dtype=np.int32)
    candidate_vals = np.zeros((n_rows, n_candidate_slots), dtype=np.float64)
    candidate_valid = np.zeros((n_rows, n_candidate_slots), dtype=bool)

    # Center point is always present.
    candidate_kinds[:, 0] = 0
    candidate_cols[:, 0] = rows
    candidate_valid[:, 0] = True

    neg_boundary_offset = -neg_delta
    pos_boundary_offset = pos_delta
    neg_need_early = neg_counts < 2
    pos_need_early = pos_counts < 2

    # Early boundary points preserve the original ordering when a side has too
    # few interior points to anchor the stencil near the cut boundary.
    candidate_offsets[:, 1] = neg_boundary_offset
    candidate_kinds[:, 1] = 2
    candidate_vals[:, 1] = neg_value
    candidate_valid[:, 1] = neg_need_early

    candidate_offsets[:, 2] = pos_boundary_offset
    candidate_kinds[:, 2] = 2
    candidate_vals[:, 2] = pos_value
    candidate_valid[:, 2] = pos_need_early

    # Interior candidates are already ordered by increasing |offset|, with the
    # negative side preceding the positive side at equal distance.
    interior_offsets = np.empty((n_rows, n_interior_slots), dtype=np.float64)
    interior_cols = np.empty((n_rows, n_interior_slots), dtype=np.int32)
    interior_valid = np.empty((n_rows, n_interior_slots), dtype=bool)
    step_ids = np.arange(MAX_TEMPLATE_STEPS, dtype=np.int32)
    interior_offsets[:, 0::2] = neg_offsets_batch
    interior_offsets[:, 1::2] = pos_offsets_batch
    interior_cols[:, 0::2] = neg_cols_batch
    interior_cols[:, 1::2] = pos_cols_batch
    interior_valid[:, 0::2] = step_ids[None, :] < neg_counts[:, None]
    interior_valid[:, 1::2] = step_ids[None, :] < pos_counts[:, None]

    interior_slice = slice(3, 3 + n_interior_slots)
    candidate_offsets[:, interior_slice] = interior_offsets
    candidate_kinds[:, interior_slice] = 1
    candidate_cols[:, interior_slice] = interior_cols
    candidate_valid[:, interior_slice] = interior_valid

    # Late boundary points are always appended; duplicate suppression is handled
    # in bulk below, which reproduces the prior first-occurrence semantics.
    candidate_offsets[:, -2] = neg_boundary_offset
    candidate_kinds[:, -2] = 2
    candidate_vals[:, -2] = neg_value
    candidate_valid[:, -2] = True

    candidate_offsets[:, -1] = pos_boundary_offset
    candidate_kinds[:, -1] = 2
    candidate_vals[:, -1] = pos_value
    candidate_valid[:, -1] = True

    prev_mask = np.tril(np.ones((n_candidate_slots, n_candidate_slots), dtype=bool), k=-1)
    equal_offsets = candidate_offsets[:, :, None] == candidate_offsets[:, None, :]
    valid_pairs = candidate_valid[:, :, None] & candidate_valid[:, None, :]
    duplicate = np.any(equal_offsets & valid_pairs & prev_mask[None, :, :], axis=2)
    first_occurrence = candidate_valid & ~duplicate

    n_unique = np.sum(first_occurrence, axis=1, dtype=np.int32)
    if np.any(n_unique < 6):
        bad_idx = int(np.nonzero(n_unique < 6)[0][0])
        raise SystemExit(f"Unable to build a 4th-order cut-cell stencil in {axis}-direction at row {int(rows[bad_idx])}.")

    candidate_pos = np.broadcast_to(np.arange(n_candidate_slots, dtype=np.int32), (n_rows, n_candidate_slots))
    select_score = np.where(first_occurrence, candidate_pos, n_candidate_slots + candidate_pos)
    selected_pos = np.argsort(select_score, axis=1)[:, :6]

    selected_offsets = np.take_along_axis(candidate_offsets, selected_pos, axis=1)
    selected_kinds = np.take_along_axis(candidate_kinds, selected_pos, axis=1)
    selected_cols = np.take_along_axis(candidate_cols, selected_pos, axis=1)
    selected_vals = np.take_along_axis(candidate_vals, selected_pos, axis=1)

    order = np.argsort(selected_offsets, axis=1)
    templ_rows = rows.astype(np.int32, copy=False)
    templ_offsets = np.take_along_axis(selected_offsets, order, axis=1)
    templ_kinds = np.take_along_axis(selected_kinds, order, axis=1)
    templ_cols = np.take_along_axis(selected_cols, order, axis=1)
    templ_vals = np.take_along_axis(selected_vals, order, axis=1)
    return templ_rows, templ_offsets, templ_kinds, templ_cols, templ_vals


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

    lut_pad = np.pad(cart_lookup, ((MAX_TEMPLATE_STEPS, MAX_TEMPLATE_STEPS), (MAX_TEMPLATE_STEPS, MAX_TEMPLATE_STEPS)), mode="constant")
    inner_gidx0 = inner_indices - 1
    b = rhs_all[inner_gidx0].astype(np.float64).copy()
    diag = lambda_inner.astype(np.float64).copy()
    ii0 = cart_i_all[inner_gidx0] - cart_i_min + MAX_TEMPLATE_STEPS
    jj0 = cart_j_all[inner_gidx0] - cart_j_min + MAX_TEMPLATE_STEPS

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

    x_irregular_rows = np.nonzero(~x_regular)[0].astype(np.int32)
    y_irregular_rows = np.nonzero(~y_regular)[0].astype(np.int32)
    n_irregular_x = int(x_irregular_rows.size)
    n_irregular_y = int(y_irregular_rows.size)

    x_templ_rows, x_templ_offsets, x_templ_kinds, x_templ_cols, x_templ_vals = build_axis_fd4_templates_batch(
        axis="x",
        rows=x_irregular_rows,
        inner_indices=inner_indices,
        cart_i_all=cart_i_all,
        cart_j_all=cart_j_all,
        pinfo=pinfo,
        lut_pad=lut_pad,
        cart_i_min=cart_i_min,
        cart_j_min=cart_j_min,
        global_to_inner=global_to_inner,
        spacing=spacing,
        x_all=x_all,
        y_all=y_all,
        trace_splines=trace_splines,
        rhomin=rhomin,
        rhomax=rhomax,
        boundary_models=boundary_models,
        axis_tables=axis_tables,
        perf=perf,
    )

    y_templ_rows, y_templ_offsets, y_templ_kinds, y_templ_cols, y_templ_vals = build_axis_fd4_templates_batch(
        axis="y",
        rows=y_irregular_rows,
        inner_indices=inner_indices,
        cart_i_all=cart_i_all,
        cart_j_all=cart_j_all,
        pinfo=pinfo,
        lut_pad=lut_pad,
        cart_i_min=cart_i_min,
        cart_j_min=cart_j_min,
        global_to_inner=global_to_inner,
        spacing=spacing,
        x_all=x_all,
        y_all=y_all,
        trace_splines=trace_splines,
        rhomin=rhomin,
        rhomax=rhomax,
        boundary_models=boundary_models,
        axis_tables=axis_tables,
        perf=perf,
    )
    templ_rows_arr = np.concatenate((x_templ_rows, y_templ_rows))
    offsets_batch = np.concatenate((x_templ_offsets, y_templ_offsets), axis=0)
    kinds_batch = np.concatenate((x_templ_kinds, y_templ_kinds), axis=0)
    cols_batch = np.concatenate((x_templ_cols, y_templ_cols), axis=0)
    vals_batch = np.concatenate((x_templ_vals, y_templ_vals), axis=0)

    weights_batch = finite_difference_second_derivative_weights_batch(offsets_batch)

    boundary_contrib = np.where(kinds_batch == 2, vals_batch, 0.0)
    center_mask = kinds_batch == 0
    interior_mask = kinds_batch == 1
    rhs_shifts = np.sum(weights_batch * boundary_contrib, axis=1)
    center_weights = np.sum(weights_batch * center_mask.astype(np.float64), axis=1)

    np.add.at(b, templ_rows_arr, rhs_shifts)
    np.add.at(diag, templ_rows_arr, -center_weights)

    interior_counts = np.sum(interior_mask, axis=1, dtype=np.int32)
    irr_rows_arr = np.repeat(templ_rows_arr, interior_counts).astype(np.int32, copy=False)
    irr_cols_arr = cols_batch[interior_mask].astype(np.int32, copy=False)
    irr_vals_arr = (-weights_batch[interior_mask]).astype(np.float64, copy=False)

    t_loop = time.perf_counter()
    diag_rows = np.arange(n_rows, dtype=np.int32)
    diag_cols = diag_rows
    diag_vals = diag

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
