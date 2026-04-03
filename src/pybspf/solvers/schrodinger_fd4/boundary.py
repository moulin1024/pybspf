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
        dense_t = np.linspace(0.0, 2.0 * math.pi, n_dense + 1, dtype=np.float64)
        models[side] = {
            "spline_x": spline_x,
            "spline_y": spline_y,
            "dense_t": dense_t,
            "dense_x": spline_x(dense_t),
            "dense_y": spline_y(dense_t),
            "seg_t1": dense_t[:-1],
            "seg_t2": dense_t[1:],
            "seg_x1": spline_x(dense_t[:-1]),
            "seg_x2": spline_x(dense_t[1:]),
            "seg_y1": spline_y(dense_t[:-1]),
            "seg_y2": spline_y(dense_t[1:]),
        }
    return models


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

    for j in range(j_min, j_max + 1):
        y = ymin + (j - 1.0) * spacing
        xs: list[np.ndarray] = []
        params: list[np.ndarray] = []
        names: list[np.ndarray] = []
        for side_name, model in boundary_models.items():
            x1 = np.asarray(model["seg_x1"], dtype=np.float64)
            x2 = np.asarray(model["seg_x2"], dtype=np.float64)
            y1 = np.asarray(model["seg_y1"], dtype=np.float64)
            y2 = np.asarray(model["seg_y2"], dtype=np.float64)
            t1 = np.asarray(model["seg_t1"], dtype=np.float64)
            t2 = np.asarray(model["seg_t2"], dtype=np.float64)
            dy = y2 - y1
            mask = ((y1 - y) * (y2 - y) <= 0.0) & (np.abs(dy) >= tol)
            if not np.any(mask):
                continue
            alpha = (y - y1[mask]) / dy[mask]
            valid = (alpha >= -1.0e-12) & (alpha <= 1.0 + 1.0e-12)
            if not np.any(valid):
                continue
            alpha = alpha[valid]
            xs.append(x1[mask][valid] + alpha * (x2[mask][valid] - x1[mask][valid]))
            params.append(t1[mask][valid] + alpha * (t2[mask][valid] - t1[mask][valid]))
            names.append(np.full(alpha.size, side_name, dtype=object))
        if xs:
            x_all = np.concatenate(xs)
            p_all = np.concatenate(params)
            n_all = np.concatenate(names)
            order = np.argsort(x_all)
            horiz[j - j_min] = {"coord": x_all[order], "param": p_all[order], "name": n_all[order]}

    for i in range(i_min, i_max + 1):
        x = xmin + (i - 1.0) * spacing
        ys: list[np.ndarray] = []
        params: list[np.ndarray] = []
        names: list[np.ndarray] = []
        for side_name, model in boundary_models.items():
            x1 = np.asarray(model["seg_x1"], dtype=np.float64)
            x2 = np.asarray(model["seg_x2"], dtype=np.float64)
            y1 = np.asarray(model["seg_y1"], dtype=np.float64)
            y2 = np.asarray(model["seg_y2"], dtype=np.float64)
            t1 = np.asarray(model["seg_t1"], dtype=np.float64)
            t2 = np.asarray(model["seg_t2"], dtype=np.float64)
            dx = x2 - x1
            mask = ((x1 - x) * (x2 - x) <= 0.0) & (np.abs(dx) >= tol)
            if not np.any(mask):
                continue
            alpha = (x - x1[mask]) / dx[mask]
            valid = (alpha >= -1.0e-12) & (alpha <= 1.0 + 1.0e-12)
            if not np.any(valid):
                continue
            alpha = alpha[valid]
            ys.append(y1[mask][valid] + alpha * (y2[mask][valid] - y1[mask][valid]))
            params.append(t1[mask][valid] + alpha * (t2[mask][valid] - t1[mask][valid]))
            names.append(np.full(alpha.size, side_name, dtype=object))
        if ys:
            y_all = np.concatenate(ys)
            p_all = np.concatenate(params)
            n_all = np.concatenate(names)
            order = np.argsort(y_all)
            vert[i - i_min] = {"coord": y_all[order], "param": p_all[order], "name": n_all[order]}

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
    tol = 1.0e-14

    for name, model in boundary_models.items():
        x1 = np.asarray(model["seg_x1"], dtype=np.float64)
        x2 = np.asarray(model["seg_x2"], dtype=np.float64)
        y1 = np.asarray(model["seg_y1"], dtype=np.float64)
        y2 = np.asarray(model["seg_y2"], dtype=np.float64)
        t1 = np.asarray(model["seg_t1"], dtype=np.float64)
        t2 = np.asarray(model["seg_t2"], dtype=np.float64)

        if di != 0:
            dy = y2 - y1
            mask = ((y1 - y_i) * (y2 - y_i) <= 0.0) & (np.abs(dy) >= tol)
            if not np.any(mask):
                continue
            alpha = (y_i - y1[mask]) / dy[mask]
            valid = (alpha >= -1.0e-12) & (alpha <= 1.0 + 1.0e-12)
            if not np.any(valid):
                continue
            alpha = alpha[valid]
            xb = x1[mask][valid] + alpha * (x2[mask][valid] - x1[mask][valid])
            delta = di * (xb - x_i)
            pos = delta > 1.0e-12
            if not np.any(pos):
                continue
            idx = int(np.argmin(delta[pos]))
            delta_pos = delta[pos]
            xb_pos = xb[pos]
            alpha_pos = alpha[pos]
            t1_pos = t1[mask][valid][pos]
            t2_pos = t2[mask][valid][pos]
            candidates.append((
                float(delta_pos[idx]),
                float(xb_pos[idx]),
                y_i,
                name,
                float(t1_pos[idx] + alpha_pos[idx] * (t2_pos[idx] - t1_pos[idx])),
            ))
        else:
            dx = x2 - x1
            mask = ((x1 - x_i) * (x2 - x_i) <= 0.0) & (np.abs(dx) >= tol)
            if not np.any(mask):
                continue
            alpha = (x_i - x1[mask]) / dx[mask]
            valid = (alpha >= -1.0e-12) & (alpha <= 1.0 + 1.0e-12)
            if not np.any(valid):
                continue
            alpha = alpha[valid]
            yb = y1[mask][valid] + alpha * (y2[mask][valid] - y1[mask][valid])
            delta = dj * (yb - y_i)
            pos = delta > 1.0e-12
            if not np.any(pos):
                continue
            idx = int(np.argmin(delta[pos]))
            delta_pos = delta[pos]
            yb_pos = yb[pos]
            alpha_pos = alpha[pos]
            t1_pos = t1[mask][valid][pos]
            t2_pos = t2[mask][valid][pos]
            candidates.append((
                float(delta_pos[idx]),
                x_i,
                float(yb_pos[idx]),
                name,
                float(t1_pos[idx] + alpha_pos[idx] * (t2_pos[idx] - t1_pos[idx])),
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
