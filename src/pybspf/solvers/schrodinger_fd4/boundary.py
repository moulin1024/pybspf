from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial import cKDTree

from .common import circular_theta

BOUNDARY_SPLINE_DEGREE = 4


@dataclass(frozen=True)
class BoundaryDistanceQuery:
    mode: str
    tree: cKDTree | None = None
    rhomin: float | None = None
    rhomax: float | None = None


def evaluate_mms_solution(radius: np.ndarray, theta_vals: np.ndarray, *, rhomin: float, rhomax: float) -> np.ndarray:
    rhon = (radius - rhomin) / (rhomax - rhomin)
    return np.cos(1.5 * math.pi * rhon) * np.sin(4.0 * theta_vals) + 1.3


def evaluate_mms_co(radius: np.ndarray, theta_vals: np.ndarray, *, rhomin: float, rhomax: float) -> np.ndarray:
    rhon = (radius - rhomin) / (rhomax - rhomin)
    return 1.1 + np.cos(0.5 * math.pi * rhon) * np.cos(3.0 * theta_vals)


def evaluate_exact_solution_form(
    radius: np.ndarray,
    theta_vals: np.ndarray,
    *,
    form: str,
    rhomin: float,
    rhomax: float,
) -> np.ndarray:
    u = evaluate_mms_solution(radius, theta_vals, rhomin=rhomin, rhomax=rhomax)
    if form == "helmholtz":
        return u
    co = evaluate_mms_co(radius, theta_vals, rhomin=rhomin, rhomax=rhomax)
    return np.sqrt(co) * u


def build_trace_splines(*, rhomin: float, rhomax: float, form: str = "helmholtz", n_samples: int = 512) -> dict[str, object]:
    theta = np.linspace(0.0, 2.0 * math.pi, n_samples + 1, dtype=np.float64)
    inner_values = evaluate_exact_solution_form(
        np.full_like(theta, rhomin),
        theta,
        form=form,
        rhomin=rhomin,
        rhomax=rhomax,
    )
    outer_values = evaluate_exact_solution_form(
        np.full_like(theta, rhomax),
        theta,
        form=form,
        rhomin=rhomin,
        rhomax=rhomax,
    )
    return {
        "inner": make_interp_spline(theta, inner_values, k=BOUNDARY_SPLINE_DEGREE, bc_type="periodic"),
        "outer": make_interp_spline(theta, outer_values, k=BOUNDARY_SPLINE_DEGREE, bc_type="periodic"),
    }


def build_trace_splines_from_samples(fields: dict[str, np.ndarray]) -> dict[str, object]:
    traces: dict[str, object] = {}
    for side in ("inner", "outer"):
        key = f"{side}_boundary_sample_value"
        values = np.asarray(fields[key], dtype=np.float64)
        t = np.linspace(0.0, 2.0 * math.pi, values.size, dtype=np.float64)
        traces[side] = make_interp_spline(t, values, k=BOUNDARY_SPLINE_DEGREE, bc_type="periodic")
    return traces


def build_boundary_models_from_samples(
    mesh: dict[str, np.ndarray | float | int],
    *,
    n_dense: int = 4096,
) -> dict[str, dict[str, np.ndarray | object]]:
    models: dict[str, dict[str, np.ndarray | object]] = {}
    for side in ("inner", "outer"):
        x = np.asarray(mesh[f"{side}_boundary_sample_x"], dtype=np.float64)
        y = np.asarray(mesh[f"{side}_boundary_sample_y"], dtype=np.float64)
        t = np.linspace(0.0, 2.0 * math.pi, x.size, dtype=np.float64)
        spline_x = make_interp_spline(t, x, k=BOUNDARY_SPLINE_DEGREE, bc_type="periodic")
        spline_y = make_interp_spline(t, y, k=BOUNDARY_SPLINE_DEGREE, bc_type="periodic")
        dspline_x = spline_x.derivative()
        dspline_y = spline_y.derivative()
        dense_t = np.linspace(0.0, 2.0 * math.pi, n_dense + 1, dtype=np.float64)
        dense_x = np.asarray(spline_x(dense_t), dtype=np.float64)
        dense_y = np.asarray(spline_y(dense_t), dtype=np.float64)
        models[side] = {
            "sample_t": t,
            "spline_x": spline_x,
            "spline_y": spline_y,
            "dspline_x": dspline_x,
            "dspline_y": dspline_y,
            "dense_t": dense_t,
            "dense_x": dense_x,
            "dense_y": dense_y,
            "seg_t1": dense_t[:-1],
            "seg_t2": dense_t[1:],
            "seg_x1": dense_x[:-1],
            "seg_x2": dense_x[1:],
            "seg_y1": dense_y[:-1],
            "seg_y2": dense_y[1:],
        }
    return models


def _deduplicate_parametric_roots(
    params: np.ndarray,
    coords: np.ndarray,
    *,
    period: float = 2.0 * math.pi,
    tol: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray]:
    if params.size == 0:
        return params, coords
    wrapped = np.mod(params, period)
    order = np.argsort(wrapped)
    wrapped = wrapped[order]
    coords = coords[order]

    keep_params = [float(wrapped[0])]
    keep_coords = [float(coords[0])]
    for param, coord in zip(wrapped[1:], coords[1:]):
        if abs(float(param) - keep_params[-1]) > tol:
            keep_params.append(float(param))
            keep_coords.append(float(coord))
    if len(keep_params) > 1 and abs(keep_params[-1] - keep_params[0]) < tol:
        keep_params.pop()
        keep_coords.pop()
    return np.asarray(keep_params, dtype=np.float64), np.asarray(keep_coords, dtype=np.float64)


def _spline_axis_roots(
    *,
    model: dict[str, np.ndarray | object],
    axis: str,
    value: float,
    root_tol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    spline = model["spline_y"] if axis == "x" else model["spline_x"]
    dspline = model["dspline_y"] if axis == "x" else model["dspline_x"]
    other_spline = model["spline_x"] if axis == "x" else model["spline_y"]
    dense_t = np.asarray(model["dense_t"], dtype=np.float64)
    dense_coords = np.asarray(model["dense_y" if axis == "x" else "dense_x"], dtype=np.float64)
    dense_vals = dense_coords - value

    roots: list[float] = []

    exact = np.nonzero(np.abs(dense_vals) <= root_tol)[0]
    for idx in exact.tolist():
        roots.append(float(dense_t[idx]))

    left = dense_vals[:-1]
    right = dense_vals[1:]
    brackets = np.nonzero(left * right < 0.0)[0]
    if brackets.size:
        a = dense_t[brackets].astype(np.float64, copy=True)
        b = dense_t[brackets + 1].astype(np.float64, copy=True)
        fa = left[brackets].astype(np.float64, copy=True)
        fb = right[brackets].astype(np.float64, copy=True)

        denom = fb - fa
        secant = np.where(
            np.abs(denom) > 1.0e-30,
            (a * fb - b * fa) / denom,
            0.5 * (a + b),
        )
        t = np.clip(secant, a, b)
        active = np.ones_like(t, dtype=bool)

        for _ in range(12):
            if not np.any(active):
                break
            idx = np.nonzero(active)[0]
            t_act = t[idx]
            f = np.asarray(spline(t_act), dtype=np.float64) - value
            df = np.asarray(dspline(t_act), dtype=np.float64)

            left_act = a[idx]
            right_act = b[idx]
            fa_act = fa[idx]
            fb_act = fb[idx]

            left_keeps_root = fa_act * f <= 0.0
            b[idx] = np.where(left_keeps_root, t_act, right_act)
            fb[idx] = np.where(left_keeps_root, f, fb_act)
            a[idx] = np.where(left_keeps_root, left_act, t_act)
            fa[idx] = np.where(left_keeps_root, fa_act, f)

            converged = (np.abs(f) <= root_tol) | ((b[idx] - a[idx]) <= 1.0e-14)
            midpoint = 0.5 * (a[idx] + b[idx])
            newton = t_act - np.where(np.abs(df) > 1.0e-14, f / df, 0.0)
            use_newton = (np.abs(df) > 1.0e-14) & (newton > a[idx]) & (newton < b[idx])
            t[idx] = np.where(converged, t_act, np.where(use_newton, newton, midpoint))
            active[idx[converged]] = False

        roots.extend(t.tolist())

    if not roots:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    root_params = np.asarray(roots, dtype=np.float64)
    other_coords = np.asarray(other_spline(root_params), dtype=np.float64)
    return _deduplicate_parametric_roots(root_params, other_coords)


def _spline_axis_roots_for_values_batch(
    *,
    model: dict[str, np.ndarray | object],
    axis: str,
    values: np.ndarray,
    root_tol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        empty_i = np.empty(0, dtype=np.int32)
        empty_f = np.empty(0, dtype=np.float64)
        return empty_i, empty_f, empty_f

    spline = model["spline_y"] if axis == "x" else model["spline_x"]
    dspline = model["dspline_y"] if axis == "x" else model["dspline_x"]
    other_spline = model["spline_x"] if axis == "x" else model["spline_y"]
    dense_t = np.asarray(model["dense_t"], dtype=np.float64)
    dense_coords = np.asarray(model["dense_y" if axis == "x" else "dense_x"], dtype=np.float64)

    line_idx_list: list[np.ndarray] = []
    param_list: list[np.ndarray] = []

    exact_mask = np.abs(dense_coords[None, :] - values[:, None]) <= root_tol
    exact_lines, exact_cols = np.nonzero(exact_mask)
    if exact_lines.size:
        line_idx_list.append(exact_lines.astype(np.int32, copy=False))
        param_list.append(dense_t[exact_cols].astype(np.float64, copy=False))

    left = dense_coords[:-1]
    right = dense_coords[1:]
    left_vals = left[None, :] - values[:, None]
    right_vals = right[None, :] - values[:, None]
    bracket_mask = left_vals * right_vals < 0.0
    bracket_lines, bracket_cols = np.nonzero(bracket_mask)
    if bracket_lines.size:
        a = dense_t[bracket_cols].astype(np.float64, copy=True)
        b = dense_t[bracket_cols + 1].astype(np.float64, copy=True)
        fa = left_vals[bracket_lines, bracket_cols].astype(np.float64, copy=True)
        fb = right_vals[bracket_lines, bracket_cols].astype(np.float64, copy=True)
        target = values[bracket_lines].astype(np.float64, copy=False)

        denom = fb - fa
        secant = np.where(
            np.abs(denom) > 1.0e-30,
            (a * fb - b * fa) / denom,
            0.5 * (a + b),
        )
        t = np.clip(secant, a, b)
        active = np.ones_like(t, dtype=bool)

        for _ in range(12):
            if not np.any(active):
                break
            idx = np.nonzero(active)[0]
            t_act = t[idx]
            f = np.asarray(spline(t_act), dtype=np.float64) - target[idx]
            df = np.asarray(dspline(t_act), dtype=np.float64)

            left_act = a[idx]
            right_act = b[idx]
            fa_act = fa[idx]
            fb_act = fb[idx]

            left_keeps_root = fa_act * f <= 0.0
            b[idx] = np.where(left_keeps_root, t_act, right_act)
            fb[idx] = np.where(left_keeps_root, f, fb_act)
            a[idx] = np.where(left_keeps_root, left_act, t_act)
            fa[idx] = np.where(left_keeps_root, fa_act, f)

            converged = (np.abs(f) <= root_tol) | ((b[idx] - a[idx]) <= 1.0e-14)
            midpoint = 0.5 * (a[idx] + b[idx])
            newton = t_act - np.where(np.abs(df) > 1.0e-14, f / df, 0.0)
            use_newton = (np.abs(df) > 1.0e-14) & (newton > a[idx]) & (newton < b[idx])
            t[idx] = np.where(converged, t_act, np.where(use_newton, newton, midpoint))
            active[idx[converged]] = False

        line_idx_list.append(bracket_lines.astype(np.int32, copy=False))
        param_list.append(t.astype(np.float64, copy=False))

    if not line_idx_list:
        empty_i = np.empty(0, dtype=np.int32)
        empty_f = np.empty(0, dtype=np.float64)
        return empty_i, empty_f, empty_f

    line_idx = np.concatenate(line_idx_list)
    params = np.concatenate(param_list)
    order = np.lexsort((params, line_idx))
    line_idx = line_idx[order]
    params = params[order]
    coords = np.asarray(other_spline(params), dtype=np.float64)

    keep_line: list[int] = []
    keep_param: list[float] = []
    keep_coord: list[float] = []
    last_line = -1
    last_param = 0.0
    for line, param, coord in zip(line_idx.tolist(), params.tolist(), coords.tolist()):
        if line != last_line:
            keep_line.append(int(line))
            keep_param.append(float(param % (2.0 * math.pi)))
            keep_coord.append(float(coord))
            last_line = int(line)
            last_param = float(param % (2.0 * math.pi))
            continue
        current = float(param % (2.0 * math.pi))
        if abs(current - last_param) > 1.0e-10:
            keep_line.append(int(line))
            keep_param.append(current)
            keep_coord.append(float(coord))
            last_param = current

    return (
        np.asarray(keep_line, dtype=np.int32),
        np.asarray(keep_param, dtype=np.float64),
        np.asarray(keep_coord, dtype=np.float64),
    )


def build_boundary_distance_query(
    mesh: dict[str, np.ndarray | float | int],
    *,
    n_dense: int = 8192,
) -> BoundaryDistanceQuery:
    has_sample_geometry = all(
        key in mesh for key in (
            "inner_boundary_sample_x",
            "inner_boundary_sample_y",
            "outer_boundary_sample_x",
            "outer_boundary_sample_y",
        )
    )
    if has_sample_geometry:
        models = build_boundary_models_from_samples(mesh, n_dense=n_dense)
        points = []
        for side in ("inner", "outer"):
            dense_x = np.asarray(models[side]["dense_x"], dtype=np.float64)
            dense_y = np.asarray(models[side]["dense_y"], dtype=np.float64)
            points.append(np.column_stack([dense_x[:-1], dense_y[:-1]]))
        cloud = np.vstack(points)
        return BoundaryDistanceQuery(mode="sampled", tree=cKDTree(cloud))

    params = default_circular_params()
    return BoundaryDistanceQuery(mode="circular", rhomin=params["rhomin"], rhomax=params["rhomax"])


def evaluate_distance_to_boundary(
    query: BoundaryDistanceQuery,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if query.mode == "sampled":
        assert query.tree is not None
        dist, _ = query.tree.query(np.column_stack([x, y]), workers=-1)
        return np.asarray(dist, dtype=np.float64)

    assert query.rhomin is not None and query.rhomax is not None
    r = np.hypot(x, y)
    return np.minimum(np.abs(r - query.rhomin), np.abs(r - query.rhomax))


def build_axis_intersection_tables(
    mesh: dict[str, np.ndarray | float | int],
    boundary_models: dict[str, dict[str, np.ndarray | object]],
) -> dict[str, object]:
    spacing = float(mesh["spacing"])
    xmin = float(mesh["xmin"])
    ymin = float(mesh["ymin"])
    cart_i = np.asarray(mesh["cart_i"], dtype=np.int32)
    cart_j = np.asarray(mesh["cart_j"], dtype=np.int32)
    i_min = int(cart_i.min())
    i_max = int(cart_i.max())
    j_min = int(cart_j.min())
    j_max = int(cart_j.max())
    tol = 1.0e-14

    horiz: list[dict[str, np.ndarray] | None] = [None] * (j_max - j_min + 1)
    vert: list[dict[str, np.ndarray] | None] = [None] * (i_max - i_min + 1)
    y_lines = ymin + (np.arange(j_min, j_max + 1, dtype=np.float64) - 1.0) * spacing
    x_lines = xmin + (np.arange(i_min, i_max + 1, dtype=np.float64) - 1.0) * spacing

    horiz_coords: list[list[float]] = [[] for _ in range(y_lines.size)]
    horiz_params: list[list[float]] = [[] for _ in range(y_lines.size)]
    horiz_names: list[list[object]] = [[] for _ in range(y_lines.size)]
    vert_coords: list[list[float]] = [[] for _ in range(x_lines.size)]
    vert_params: list[list[float]] = [[] for _ in range(x_lines.size)]
    vert_names: list[list[object]] = [[] for _ in range(x_lines.size)]

    for side_name, model in boundary_models.items():
        line_idx, root_params, coords = _spline_axis_roots_for_values_batch(
            model=model,
            axis="x",
            values=y_lines,
            root_tol=tol,
        )
        for idx, param, coord in zip(line_idx.tolist(), root_params.tolist(), coords.tolist()):
            horiz_coords[idx].append(float(coord))
            horiz_params[idx].append(float(param))
            horiz_names[idx].append(side_name)

        line_idx, root_params, coords = _spline_axis_roots_for_values_batch(
            model=model,
            axis="y",
            values=x_lines,
            root_tol=tol,
        )
        for idx, param, coord in zip(line_idx.tolist(), root_params.tolist(), coords.tolist()):
            vert_coords[idx].append(float(coord))
            vert_params[idx].append(float(param))
            vert_names[idx].append(side_name)

    for j in range(j_min, j_max + 1):
        idx = j - j_min
        if horiz_coords[idx]:
            x_all = np.asarray(horiz_coords[idx], dtype=np.float64)
            p_all = np.asarray(horiz_params[idx], dtype=np.float64)
            n_all = np.asarray(horiz_names[idx], dtype=object)
            order = np.argsort(x_all)
            horiz[idx] = {"coord": x_all[order], "param": p_all[order], "name": n_all[order]}

    for i in range(i_min, i_max + 1):
        idx = i - i_min
        if vert_coords[idx]:
            y_all = np.asarray(vert_coords[idx], dtype=np.float64)
            p_all = np.asarray(vert_params[idx], dtype=np.float64)
            n_all = np.asarray(vert_names[idx], dtype=object)
            order = np.argsort(y_all)
            vert[idx] = {"coord": y_all[order], "param": p_all[order], "name": n_all[order]}

    return {"horiz": horiz, "vert": vert, "cart_i_min": i_min, "cart_j_min": j_min}


def axis_boundary_intersection(
    *,
    x_i: float,
    y_i: float,
    direction: tuple[int, int],
    rhomin: float,
    rhomax: float,
) -> tuple[float, float, float, str]:
    di, dj = direction
    candidates: list[tuple[float, float, float, str]] = []

    if di != 0:
        for radius, name in ((rhomin, "inner"), (rhomax, "outer")):
            rad2 = radius * radius - y_i * y_i
            if rad2 < 0.0:
                continue
            x_abs = math.sqrt(max(rad2, 0.0))
            for xb in (-x_abs, x_abs):
                delta = di * (xb - x_i)
                if delta > 1.0e-12:
                    candidates.append((delta, xb, y_i, name))
    else:
        for radius, name in ((rhomin, "inner"), (rhomax, "outer")):
            rad2 = radius * radius - x_i * x_i
            if rad2 < 0.0:
                continue
            y_abs = math.sqrt(max(rad2, 0.0))
            for yb in (-y_abs, y_abs):
                delta = dj * (yb - y_i)
                if delta > 1.0e-12:
                    candidates.append((delta, x_i, yb, name))

    if not candidates:
        raise ValueError("No forward axis-aligned boundary intersection found.")
    delta, xb, yb, name = min(candidates, key=lambda item: item[0])
    return xb, yb, delta, name


def spline_axis_boundary_intersection(
    *,
    x_i: float,
    y_i: float,
    direction: tuple[int, int],
    boundary_models: dict[str, dict[str, np.ndarray | object]],
) -> tuple[float, float, float, str, float]:
    di, dj = direction
    candidates: list[tuple[float, float, float, str, float]] = []

    for name, model in boundary_models.items():
        if di != 0:
            root_params, xb = _spline_axis_roots(model=model, axis="x", value=y_i)
            if root_params.size == 0:
                continue
            delta = di * (xb - x_i)
            pos = delta > 1.0e-12
            if not np.any(pos):
                continue
            idx = int(np.argmin(delta[pos]))
            delta_pos = delta[pos]
            xb_pos = xb[pos]
            candidates.append((
                float(delta_pos[idx]),
                float(xb_pos[idx]),
                y_i,
                name,
                float(root_params[pos][idx]),
            ))
        else:
            root_params, yb = _spline_axis_roots(model=model, axis="y", value=x_i)
            if root_params.size == 0:
                continue
            delta = dj * (yb - y_i)
            pos = delta > 1.0e-12
            if not np.any(pos):
                continue
            idx = int(np.argmin(delta[pos]))
            delta_pos = delta[pos]
            yb_pos = yb[pos]
            candidates.append((
                float(delta_pos[idx]),
                x_i,
                float(yb_pos[idx]),
                name,
                float(root_params[pos][idx]),
            ))

    if not candidates:
        raise ValueError("No forward spline boundary intersection found.")
    delta, xb, yb, name, tb = min(candidates, key=lambda item: item[0])
    return xb, yb, delta, name, tb


def evaluate_boundary_trace_value(
    *,
    direction: tuple[int, int],
    cart_i0: int,
    cart_j0: int,
    x_i: float,
    y_i: float,
    trace_splines: dict[str, object],
    rhomin: float,
    rhomax: float,
    boundary_models: dict[str, dict[str, np.ndarray | object]] | None = None,
    axis_tables: dict[str, object] | None = None,
) -> tuple[float, float]:
    if boundary_models is None:
        xb, yb, delta, boundary_name = axis_boundary_intersection(
            x_i=x_i,
            y_i=y_i,
            direction=direction,
            rhomin=rhomin,
            rhomax=rhomax,
        )
        param = float(circular_theta(np.asarray([xb]), np.asarray([yb]))[0])
    else:
        di, dj = direction
        if axis_tables is not None:
            if di != 0:
                line = axis_tables["horiz"][cart_j0 - int(axis_tables["cart_j_min"])]
                if line is None:
                    raise ValueError("No horizontal intersection table for requested row.")
                coords = np.asarray(line["coord"], dtype=np.float64)
                names = np.asarray(line["name"], dtype=object)
                params = np.asarray(line["param"], dtype=np.float64)
                deltas = di * (coords - x_i)
            else:
                line = axis_tables["vert"][cart_i0 - int(axis_tables["cart_i_min"])]
                if line is None:
                    raise ValueError("No vertical intersection table for requested column.")
                coords = np.asarray(line["coord"], dtype=np.float64)
                names = np.asarray(line["name"], dtype=object)
                params = np.asarray(line["param"], dtype=np.float64)
                deltas = dj * (coords - y_i)
            mask = deltas > 1.0e-12
            if not np.any(mask):
                raise ValueError("No forward precomputed boundary intersection found.")
            idx = int(np.argmin(deltas[mask]))
            delta = float(deltas[mask][idx])
            boundary_name = str(names[mask][idx])
            param = float(params[mask][idx])
        else:
            _, _, delta, boundary_name, param = spline_axis_boundary_intersection(
                x_i=x_i,
                y_i=y_i,
                direction=direction,
                boundary_models=boundary_models,
            )
    value = float(trace_splines[boundary_name](np.asarray([param]))[0])
    return delta, value


def evaluate_boundary_trace_values_batch(
    *,
    axis: str,
    direction_sign: int,
    cart_i0: np.ndarray,
    cart_j0: np.ndarray,
    x_i: np.ndarray,
    y_i: np.ndarray,
    trace_splines: dict[str, object],
    rhomin: float,
    rhomax: float,
    boundary_models: dict[str, dict[str, np.ndarray | object]] | None = None,
    axis_tables: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cart_i0 = np.asarray(cart_i0, dtype=np.int32)
    cart_j0 = np.asarray(cart_j0, dtype=np.int32)
    x_i = np.asarray(x_i, dtype=np.float64)
    y_i = np.asarray(y_i, dtype=np.float64)
    n = x_i.size
    delta = np.empty(n, dtype=np.float64)
    value = np.empty(n, dtype=np.float64)
    tol = 1.0e-12

    if boundary_models is None or axis_tables is None:
        if axis == "x":
            rad_inner = rhomin * rhomin - y_i * y_i
            rad_outer = rhomax * rhomax - y_i * y_i
            cand_coords = np.full((n, 4), np.nan, dtype=np.float64)
            cand_names = np.empty((n, 4), dtype=np.int8)
            inner_ok = rad_inner >= 0.0
            outer_ok = rad_outer >= 0.0
            inner_abs = np.sqrt(np.maximum(rad_inner, 0.0))
            outer_abs = np.sqrt(np.maximum(rad_outer, 0.0))
            cand_coords[:, 0] = -inner_abs
            cand_coords[:, 1] = inner_abs
            cand_coords[:, 2] = -outer_abs
            cand_coords[:, 3] = outer_abs
            cand_names[:, 0:2] = 0
            cand_names[:, 2:4] = 1
            valid = np.column_stack([inner_ok, inner_ok, outer_ok, outer_ok])
            cand_delta = direction_sign * (cand_coords - x_i[:, None])
            cand_delta = np.where(valid & (cand_delta > tol), cand_delta, np.inf)
            idx = np.argmin(cand_delta, axis=1)
            delta = cand_delta[np.arange(n), idx]
            xb = cand_coords[np.arange(n), idx]
            yb = y_i
            names = cand_names[np.arange(n), idx]
            params = circular_theta(np.asarray(xb, dtype=np.float64), np.asarray(yb, dtype=np.float64))
        else:
            rad_inner = rhomin * rhomin - x_i * x_i
            rad_outer = rhomax * rhomax - x_i * x_i
            cand_coords = np.full((n, 4), np.nan, dtype=np.float64)
            cand_names = np.empty((n, 4), dtype=np.int8)
            inner_ok = rad_inner >= 0.0
            outer_ok = rad_outer >= 0.0
            inner_abs = np.sqrt(np.maximum(rad_inner, 0.0))
            outer_abs = np.sqrt(np.maximum(rad_outer, 0.0))
            cand_coords[:, 0] = -inner_abs
            cand_coords[:, 1] = inner_abs
            cand_coords[:, 2] = -outer_abs
            cand_coords[:, 3] = outer_abs
            cand_names[:, 0:2] = 0
            cand_names[:, 2:4] = 1
            valid = np.column_stack([inner_ok, inner_ok, outer_ok, outer_ok])
            cand_delta = direction_sign * (cand_coords - y_i[:, None])
            cand_delta = np.where(valid & (cand_delta > tol), cand_delta, np.inf)
            idx = np.argmin(cand_delta, axis=1)
            delta = cand_delta[np.arange(n), idx]
            xb = x_i
            yb = cand_coords[np.arange(n), idx]
            names = cand_names[np.arange(n), idx]
            params = circular_theta(np.asarray(xb, dtype=np.float64), np.asarray(yb, dtype=np.float64))

        if not np.all(np.isfinite(delta)):
            raise ValueError("No forward analytic boundary intersection found in batch evaluation.")
        inner_mask = names == 0
        outer_mask = ~inner_mask
        value[inner_mask] = np.asarray(trace_splines["inner"](params[inner_mask]), dtype=np.float64)
        value[outer_mask] = np.asarray(trace_splines["outer"](params[outer_mask]), dtype=np.float64)
        return delta, value

    if axis == "x":
        line_ids = cart_j0
        coord0 = x_i
        lines = axis_tables["horiz"]
        line_min = int(axis_tables["cart_j_min"])
    else:
        line_ids = cart_i0
        coord0 = y_i
        lines = axis_tables["vert"]
        line_min = int(axis_tables["cart_i_min"])

    name_codes = np.empty(n, dtype=np.int8)
    params = np.empty(n, dtype=np.float64)

    unique_lines = np.unique(line_ids)
    for line_id in unique_lines.tolist():
        line = lines[int(line_id - line_min)]
        if line is None:
            raise ValueError("No precomputed boundary intersection table for requested line.")
        coords = np.asarray(line["coord"], dtype=np.float64)
        line_params = np.asarray(line["param"], dtype=np.float64)
        line_names = np.asarray(line["name"], dtype=object)
        mask = line_ids == line_id
        q = coord0[mask]
        if direction_sign > 0:
            idx = np.searchsorted(coords, q + tol, side="left")
            if np.any(idx >= coords.size):
                raise ValueError("No forward precomputed boundary intersection found.")
            chosen = coords[idx]
            delta[mask] = chosen - q
        else:
            idx = np.searchsorted(coords, q - tol, side="right") - 1
            if np.any(idx < 0):
                raise ValueError("No forward precomputed boundary intersection found.")
            chosen = coords[idx]
            delta[mask] = q - chosen
        params[mask] = line_params[idx]
        picked_names = line_names[idx]
        name_codes[mask] = np.where(picked_names == "inner", 0, 1)

    inner_mask = name_codes == 0
    outer_mask = ~inner_mask
    value[inner_mask] = np.asarray(trace_splines["inner"](params[inner_mask]), dtype=np.float64)
    value[outer_mask] = np.asarray(trace_splines["outer"](params[outer_mask]), dtype=np.float64)
    return delta, value
