#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pyamg
from netCDF4 import Dataset
from scipy.interpolate import make_interp_spline
from scipy import sparse
from scipy.sparse import linalg as spla

PINFO_INNER = 1
PINFO_BOUNDARY = 2
PINFO_GHOST = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fourth-order immersed-boundary Schrödinger solver with AMG + BiCGSTAB.")
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
        help="Path to schrodinger_data.nc.",
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
        help="Relative tolerance for BiCGSTAB.",
    )
    parser.add_argument("--maxiter", type=int, default=1000, help="Maximum BiCGSTAB iterations.")
    return parser.parse_args()


def load_mesh(mesh_path: Path) -> dict[str, np.ndarray | float | int]:
    with Dataset(mesh_path, "r") as nc:
        group = nc["mesh_lvl_001"]
        mesh = {
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
        for name in (
            "inner_boundary_sample_x",
            "inner_boundary_sample_y",
            "outer_boundary_sample_x",
            "outer_boundary_sample_y",
        ):
            if name in group.variables:
                mesh[name] = group[name][:].astype(np.float64)
        if "geometry_type" in group.ncattrs():
            mesh["geometry_type"] = str(group.getncattr("geometry_type"))
        return mesh


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
        for name in ("inner_boundary_sample_value", "outer_boundary_sample_value"):
            if name in nc.variables:
                fields[name] = nc[name][:].astype(np.float64)
        return fields


def circular_theta(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.mod(np.arctan2(y, x), 2.0 * math.pi)


def point_coordinates(mesh: dict[str, np.ndarray | float | int]) -> tuple[np.ndarray, np.ndarray]:
    spacing = float(mesh["spacing"])
    xmin = float(mesh["xmin"])
    ymin = float(mesh["ymin"])
    cart_i = np.asarray(mesh["cart_i"], dtype=np.float64)
    cart_j = np.asarray(mesh["cart_j"], dtype=np.float64)
    x = xmin + (cart_i - 1.0) * spacing
    y = ymin + (cart_j - 1.0) * spacing
    return x, y


def default_circular_params() -> dict[str, float]:
    return {
        "rhomin": 0.2,
        "rhomax": 0.4,
    }


def evaluate_mms_solution(radius: np.ndarray, theta_vals: np.ndarray, *, rhomin: float, rhomax: float) -> np.ndarray:
    rhon = (radius - rhomin) / (rhomax - rhomin)
    return np.cos(1.5 * math.pi * rhon) * np.sin(4.0 * theta_vals) + 1.3


def evaluate_mms_rhs(radius: np.ndarray, theta_vals: np.ndarray, *, rhomin: float, rhomax: float) -> np.ndarray:
    delta_r = rhomax - rhomin
    r = radius
    t = theta_vals
    rmin = rhomin
    return (
        r * np.sin(t) * (1.3 + np.cos((3.0 * math.pi * (r - rmin)) / (2.0 * delta_r)) * np.sin(4.0 * t))
        - ((2.0 + np.cos(t)) * (
            -12.0
            * np.cos((math.pi * (r - rmin)) / (2.0 * delta_r))
            * np.cos((3.0 * math.pi * (r - rmin)) / (2.0 * delta_r))
            * np.cos(4.0 * t)
            * np.sin(3.0 * t)
            - 16.0
            * np.cos((3.0 * math.pi * (r - rmin)) / (2.0 * delta_r))
            * (1.1 + np.cos((math.pi * (r - rmin)) / (2.0 * delta_r)) * np.cos(3.0 * t))
            * np.sin(4.0 * t)
        )) / np.power(r, 1.5)
        - ((2.0 + np.cos(t)) * (
            -(9.0 * math.pi**2 * r * np.cos((3.0 * math.pi * (r - rmin)) / (2.0 * delta_r))
              * (1.1 + np.cos((math.pi * (r - rmin)) / (2.0 * delta_r)) * np.cos(3.0 * t))
              * np.sin(4.0 * t)) / (4.0 * delta_r**2)
            - (3.0 * math.pi
               * (1.1 + np.cos((math.pi * (r - rmin)) / (2.0 * delta_r)) * np.cos(3.0 * t))
               * np.sin((3.0 * math.pi * (r - rmin)) / (2.0 * delta_r))
               * np.sin(4.0 * t)) / (2.0 * delta_r)
            + (3.0 * math.pi**2 * r * np.cos(3.0 * t)
               * np.sin((math.pi * (r - rmin)) / (2.0 * delta_r))
               * np.sin((3.0 * math.pi * (r - rmin)) / (2.0 * delta_r))
               * np.sin(4.0 * t)) / (4.0 * delta_r**2)
        )) / np.sqrt(r)
    )


def evaluate_mms_co(radius: np.ndarray, theta_vals: np.ndarray, *, rhomin: float, rhomax: float) -> np.ndarray:
    rhon = (radius - rhomin) / (rhomax - rhomin)
    return 1.1 + np.cos(0.5 * math.pi * rhon) * np.cos(3.0 * theta_vals)


def build_trace_splines(*, rhomin: float, rhomax: float, form: str = "helmholtz", n_samples: int = 512):
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
        "inner": make_interp_spline(theta, inner_values, k=3, bc_type="periodic"),
        "outer": make_interp_spline(theta, outer_values, k=3, bc_type="periodic"),
    }


def build_trace_splines_from_samples(fields: dict[str, np.ndarray]) -> dict[str, object]:
    traces: dict[str, object] = {}
    for side in ("inner", "outer"):
        key = f"{side}_boundary_sample_value"
        values = np.asarray(fields[key], dtype=np.float64)
        t = np.linspace(0.0, 2.0 * math.pi, values.size, dtype=np.float64)
        traces[side] = make_interp_spline(t, values, k=3, bc_type="periodic")
    return traces


def build_boundary_models_from_samples(mesh: dict[str, np.ndarray | float | int], *, n_dense: int = 4096) -> dict[str, dict[str, np.ndarray | object]]:
    models: dict[str, dict[str, np.ndarray | object]] = {}
    for side in ("inner", "outer"):
        x = np.asarray(mesh[f"{side}_boundary_sample_x"], dtype=np.float64)
        y = np.asarray(mesh[f"{side}_boundary_sample_y"], dtype=np.float64)
        t = np.linspace(0.0, 2.0 * math.pi, x.size, dtype=np.float64)
        spline_x = make_interp_spline(t, x, k=3, bc_type="periodic")
        spline_y = make_interp_spline(t, y, k=3, bc_type="periodic")
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
            x_sel1 = x1[mask][valid]
            x_sel2 = x2[mask][valid]
            t_sel1 = t1[mask][valid]
            t_sel2 = t2[mask][valid]
            xs.append(x_sel1 + alpha * (x_sel2 - x_sel1))
            params.append(t_sel1 + alpha * (t_sel2 - t_sel1))
            names.append(np.full(alpha.size, side_name, dtype=object))
        if xs:
            x_all = np.concatenate(xs)
            p_all = np.concatenate(params)
            n_all = np.concatenate(names)
            order = np.argsort(x_all)
            horiz[j - j_min] = {
                "coord": x_all[order],
                "param": p_all[order],
                "name": n_all[order],
            }

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
            y_sel1 = y1[mask][valid]
            y_sel2 = y2[mask][valid]
            t_sel1 = t1[mask][valid]
            t_sel2 = t2[mask][valid]
            ys.append(y_sel1 + alpha * (y_sel2 - y_sel1))
            params.append(t_sel1 + alpha * (t_sel2 - t_sel1))
            names.append(np.full(alpha.size, side_name, dtype=object))
        if ys:
            y_all = np.concatenate(ys)
            p_all = np.concatenate(params)
            n_all = np.concatenate(names)
            order = np.argsort(y_all)
            vert[i - i_min] = {
                "coord": y_all[order],
                "param": p_all[order],
                "name": n_all[order],
            }

    return {
        "horiz": horiz,
        "vert": vert,
        "cart_i_min": i_min,
        "cart_j_min": j_min,
    }


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


def detect_data_form(fields: dict[str, np.ndarray]) -> str:
    co = np.asarray(fields["co"], dtype=np.float64)
    if "transform_weight" in fields and np.allclose(co, 1.0):
        return "schrodinger"
    return "helmholtz"


def build_linear_support_values(
    mesh: dict[str, np.ndarray | float | int],
    fields: dict[str, np.ndarray],
) -> np.ndarray:
    params = default_circular_params()
    rhomin = params["rhomin"]
    rhomax = params["rhomax"]
    h = float(mesh["spacing"])
    form = detect_data_form(fields)

    pinfo = np.asarray(mesh["pinfo"], dtype=np.int32)
    x_all, y_all = point_coordinates(mesh)
    rho_all = np.hypot(x_all, y_all)
    theta_all = circular_theta(x_all, y_all)

    support = np.asarray(fields["sol"], dtype=np.float64).copy()
    non_inner = np.nonzero(pinfo != PINFO_INNER)[0]
    if non_inner.size == 0:
        return support

    rho_s = rho_all[non_inner]
    theta_s = theta_all[non_inner]
    use_inner_bnd = np.abs(rho_s - rhomin) <= np.abs(rho_s - rhomax)

    rb = np.where(use_inner_bnd, rhomin, rhomax)
    ri = np.where(use_inner_bnd, rhomin + h, rhomax - h)
    g_b = evaluate_exact_solution_form(rb, theta_s, form=form, rhomin=rhomin, rhomax=rhomax)
    g_i = evaluate_exact_solution_form(ri, theta_s, form=form, rhomin=rhomin, rhomax=rhomax)

    denom = ri - rb
    support[non_inner] = g_b + (rho_s - rb) / denom * (g_i - g_b)
    return support


def circle_segment_intersection(
    x_i: float,
    y_i: float,
    x_nb: float,
    y_nb: float,
    *,
    radii: tuple[float, float],
) -> tuple[float, float, float, str]:
    dx = x_nb - x_i
    dy = y_nb - y_i
    a = dx * dx + dy * dy
    b = 2.0 * (x_i * dx + y_i * dy)
    candidates: list[tuple[float, float, str]] = []
    for radius, name in ((radii[0], "inner"), (radii[1], "outer")):
        c = x_i * x_i + y_i * y_i - radius * radius
        roots = np.roots([a, b, c])
        roots = np.real(roots[np.isreal(roots)])
        mask = (roots >= -1.0e-12) & (roots <= 1.0 + 1.0e-12)
        if np.any(mask):
            t = float(np.min(roots[mask]))
            candidates.append((t, radius, name))
    if not candidates:
        raise ValueError("Boundary segment does not intersect either circle within the stencil edge.")
    t, radius, name = min(candidates, key=lambda item: item[0])
    xb = x_i + t * dx
    yb = y_i + t * dy
    delta = math.hypot(xb - x_i, yb - y_i)
    return xb, yb, delta, name


def quadratic_ghost_coefficients(delta: float, h: float) -> tuple[float, float, float]:
    a_opp = (h - delta) / (delta + h)
    b_center = 2.0 * (delta - h) / delta
    c_bnd = 2.0 * h * h / (delta * (delta + h))
    return a_opp, b_center, c_bnd


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
            x_sel1 = x1[mask][valid]
            x_sel2 = x2[mask][valid]
            t_sel1 = t1[mask][valid]
            t_sel2 = t2[mask][valid]
            alpha = alpha[valid]
            xb = x_sel1 + alpha * (x_sel2 - x_sel1)
            delta = di * (xb - x_i)
            pos = delta > 1.0e-12
            if not np.any(pos):
                continue
            idx = int(np.argmin(delta[pos]))
            delta_pos = delta[pos]
            xb_pos = xb[pos]
            alpha_pos = alpha[pos]
            t1_pos = t_sel1[pos]
            t2_pos = t_sel2[pos]
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
            y_sel1 = y1[mask][valid]
            y_sel2 = y2[mask][valid]
            t_sel1 = t1[mask][valid]
            t_sel2 = t2[mask][valid]
            alpha = alpha[valid]
            yb = y_sel1 + alpha * (y_sel2 - y_sel1)
            delta = dj * (yb - y_i)
            pos = delta > 1.0e-12
            if not np.any(pos):
                continue
            idx = int(np.argmin(delta[pos]))
            delta_pos = delta[pos]
            yb_pos = yb[pos]
            alpha_pos = alpha[pos]
            t1_pos = t_sel1[pos]
            t2_pos = t_sel2[pos]
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


def side_geometry(
    *,
    gidx0: int,
    x_i: float,
    y_i: float,
    di: int,
    dj: int,
    spacing: float,
    pinfo: np.ndarray,
    index_neighbor: np.ndarray,
    size_neighbor: int,
    x_all: np.ndarray,
    y_all: np.ndarray,
    co_all: np.ndarray,
    global_to_inner: dict[int, int],
    trace_splines: dict[str, object],
    rhomin: float,
    rhomax: float,
) -> dict[str, float | int | str]:
    jj = size_neighbor + dj
    ii = size_neighbor + di
    nidx1 = int(index_neighbor[gidx0, jj, ii])
    if nidx1 > 0 and pinfo[nidx1 - 1] == PINFO_INNER:
        return {
            "kind": "interior",
            "dist": spacing,
            "coef": 0.5 * (co_all[gidx0] + co_all[nidx1 - 1]),
            "col": global_to_inner[nidx1],
        }

    x_nb = x_i + di * spacing
    y_nb = y_i + dj * spacing
    try:
        xb, yb, delta, boundary_name = circle_segment_intersection(
            x_i,
            y_i,
            x_nb,
            y_nb,
            radii=(rhomin, rhomax),
        )
        theta_b = float(circular_theta(np.asarray([xb]), np.asarray([yb]))[0])
    except ValueError:
        rho_nb = math.hypot(x_nb, y_nb)
        boundary_name = "inner" if abs(rho_nb - rhomin) < abs(rho_nb - rhomax) else "outer"
        theta_b = float(circular_theta(np.asarray([x_nb]), np.asarray([y_nb]))[0])
        delta = spacing

    boundary_radius = rhomin if boundary_name == "inner" else rhomax
    g_b = float(trace_splines[boundary_name](np.asarray([theta_b]))[0])
    c_boundary = float(
        evaluate_mms_co(
            np.asarray([boundary_radius]),
            np.asarray([theta_b]),
            rhomin=rhomin,
            rhomax=rhomax,
        )[0]
    )
    return {
        "kind": "boundary",
        "dist": max(float(delta), 1.0e-12),
        "coef": 0.5 * (co_all[gidx0] + c_boundary),
        "value": g_b,
    }


def side_geometry_shortley_weller(
    *,
    gidx0: int,
    x_i: float,
    y_i: float,
    di: int,
    dj: int,
    spacing: float,
    pinfo: np.ndarray,
    index_neighbor: np.ndarray,
    size_neighbor: int,
    x_all: np.ndarray,
    y_all: np.ndarray,
    co_all: np.ndarray,
    global_to_inner: dict[int, int],
    trace_splines: dict[str, object],
    rhomin: float,
    rhomax: float,
) -> dict[str, float | int | str]:
    jj = size_neighbor + dj
    ii = size_neighbor + di
    nidx1 = int(index_neighbor[gidx0, jj, ii])
    if nidx1 > 0 and pinfo[nidx1 - 1] == PINFO_INNER:
        return {
            "kind": "interior",
            "dist": spacing,
            "coef": 0.5 * (co_all[gidx0] + co_all[nidx1 - 1]),
            "col": global_to_inner[nidx1],
        }

    xb, yb, delta, boundary_name = axis_boundary_intersection(
        x_i=x_i,
        y_i=y_i,
        direction=(di, dj),
        rhomin=rhomin,
        rhomax=rhomax,
    )
    theta_b = float(circular_theta(np.asarray([xb]), np.asarray([yb]))[0])
    boundary_radius = rhomin if boundary_name == "inner" else rhomax
    g_b = float(trace_splines[boundary_name](np.asarray([theta_b]))[0])
    c_boundary = float(
        evaluate_mms_co(
            np.asarray([boundary_radius]),
            np.asarray([theta_b]),
            rhomin=rhomin,
            rhomax=rhomax,
        )[0]
    )
    return {
        "kind": "boundary",
        "dist": max(float(delta), 1.0e-12),
        "coef": 0.5 * (co_all[gidx0] + c_boundary),
        "value": g_b,
    }


def make_residual_callback(A: sparse.spmatrix, b: np.ndarray, history: list[float]):
    b_norm = float(np.linalg.norm(b))
    denom = b_norm if b_norm > 0.0 else 1.0

    def _callback(xk: np.ndarray) -> None:
        rk = b - A @ xk
        history.append(float(np.linalg.norm(rk) / denom))

    return _callback


def append_shortley_weller_row(
    *,
    row: int,
    gidx0: int,
    rhs_row: float,
    reaction: float,
    x_i: float,
    y_i: float,
    spacing: float,
    pinfo: np.ndarray,
    index_neighbor: np.ndarray,
    size_neighbor: int,
    x_all: np.ndarray,
    y_all: np.ndarray,
    co_all: np.ndarray,
    global_to_inner: dict[int, int],
    trace_splines: dict[str, object],
    rhomin: float,
    rhomax: float,
    rows: list[int],
    cols: list[int],
    vals: list[float],
) -> float:
    diag = reaction

    left = side_geometry_shortley_weller(
        gidx0=gidx0, x_i=x_i, y_i=y_i, di=-1, dj=0, spacing=spacing,
        pinfo=pinfo, index_neighbor=index_neighbor, size_neighbor=size_neighbor,
        x_all=x_all, y_all=y_all, co_all=co_all, global_to_inner=global_to_inner,
        trace_splines=trace_splines, rhomin=rhomin, rhomax=rhomax,
    )
    right = side_geometry_shortley_weller(
        gidx0=gidx0, x_i=x_i, y_i=y_i, di=1, dj=0, spacing=spacing,
        pinfo=pinfo, index_neighbor=index_neighbor, size_neighbor=size_neighbor,
        x_all=x_all, y_all=y_all, co_all=co_all, global_to_inner=global_to_inner,
        trace_splines=trace_splines, rhomin=rhomin, rhomax=rhomax,
    )
    bottom = side_geometry_shortley_weller(
        gidx0=gidx0, x_i=x_i, y_i=y_i, di=0, dj=-1, spacing=spacing,
        pinfo=pinfo, index_neighbor=index_neighbor, size_neighbor=size_neighbor,
        x_all=x_all, y_all=y_all, co_all=co_all, global_to_inner=global_to_inner,
        trace_splines=trace_splines, rhomin=rhomin, rhomax=rhomax,
    )
    top = side_geometry_shortley_weller(
        gidx0=gidx0, x_i=x_i, y_i=y_i, di=0, dj=1, spacing=spacing,
        pinfo=pinfo, index_neighbor=index_neighbor, size_neighbor=size_neighbor,
        x_all=x_all, y_all=y_all, co_all=co_all, global_to_inner=global_to_inner,
        trace_splines=trace_splines, rhomin=rhomin, rhomax=rhomax,
    )

    pref_x = 2.0 / (float(left["dist"]) + float(right["dist"]))
    diag += pref_x * (float(left["coef"]) / float(left["dist"]) + float(right["coef"]) / float(right["dist"]))
    if left["kind"] == "interior":
        rows.append(row)
        cols.append(int(left["col"]))
        vals.append(-pref_x * float(left["coef"]) / float(left["dist"]))
    else:
        rhs_row += pref_x * float(left["coef"]) / float(left["dist"]) * float(left["value"])
    if right["kind"] == "interior":
        rows.append(row)
        cols.append(int(right["col"]))
        vals.append(-pref_x * float(right["coef"]) / float(right["dist"]))
    else:
        rhs_row += pref_x * float(right["coef"]) / float(right["dist"]) * float(right["value"])

    pref_y = 2.0 / (float(bottom["dist"]) + float(top["dist"]))
    diag += pref_y * (float(bottom["coef"]) / float(bottom["dist"]) + float(top["coef"]) / float(top["dist"]))
    if bottom["kind"] == "interior":
        rows.append(row)
        cols.append(int(bottom["col"]))
        vals.append(-pref_y * float(bottom["coef"]) / float(bottom["dist"]))
    else:
        rhs_row += pref_y * float(bottom["coef"]) / float(bottom["dist"]) * float(bottom["value"])
    if top["kind"] == "interior":
        rows.append(row)
        cols.append(int(top["col"]))
        vals.append(-pref_y * float(top["coef"]) / float(top["dist"]))
    else:
        rhs_row += pref_y * float(top["coef"]) / float(top["dist"]) * float(top["value"])

    rows.append(row)
    cols.append(row)
    vals.append(diag)
    return rhs_row


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
    vand = np.vstack([pts ** k for k in range(n)])
    rhs = np.zeros(n, dtype=np.float64)
    rhs[2] = 2.0
    return np.linalg.solve(vand, rhs)


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
    if axis == "x":
        step = (1, 0)
    else:
        step = (0, 1)

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

    center_offset = np.array([0.0], dtype=np.float64)
    center_kind = np.array([0], dtype=np.int8)  # 0=center, 1=interior, 2=boundary
    center_cols = np.array([row], dtype=np.int32)
    center_vals = np.array([0.0], dtype=np.float64)

    boundary_offsets_list: list[float] = []
    boundary_values_list: list[float] = []
    if neg_offsets.size < 2:
        boundary_offsets_list.append(-neg_delta)
        boundary_values_list.append(neg_value)
    if pos_offsets.size < 2:
        boundary_offsets_list.append(pos_delta)
        boundary_values_list.append(pos_value)

    boundary_offsets = np.asarray(boundary_offsets_list, dtype=np.float64)
    boundary_values = np.asarray(boundary_values_list, dtype=np.float64)
    boundary_count = boundary_offsets.size
    boundary_kind = np.full(boundary_count, 2, dtype=np.int8)
    boundary_cols = np.full(boundary_count, -1, dtype=np.int32)

    interior_offsets = np.concatenate((neg_offsets, pos_offsets))
    interior_cols = np.concatenate((neg_cols, pos_cols))
    if interior_offsets.size:
        order = np.lexsort((interior_offsets, np.abs(interior_offsets)))
        interior_offsets = interior_offsets[order]
        interior_cols = interior_cols[order]
    interior_kind = np.full(interior_offsets.size, 1, dtype=np.int8)
    interior_vals = np.zeros(interior_offsets.size, dtype=np.float64)

    all_offsets = np.concatenate((center_offset, boundary_offsets, interior_offsets))
    all_kinds = np.concatenate((center_kind, boundary_kind, interior_kind))
    all_cols = np.concatenate((center_cols, boundary_cols, interior_cols))
    all_vals = np.concatenate((center_vals, boundary_values, interior_vals))

    extra_offsets = []
    extra_kinds = []
    extra_cols = []
    extra_vals = []
    if -neg_delta not in set(all_offsets.tolist()):
        extra_offsets.append(-neg_delta)
        extra_kinds.append(2)
        extra_cols.append(-1)
        extra_vals.append(neg_value)
    if pos_delta not in set(all_offsets.tolist()):
        extra_offsets.append(pos_delta)
        extra_kinds.append(2)
        extra_cols.append(-1)
        extra_vals.append(pos_value)
    if extra_offsets:
        all_offsets = np.concatenate((all_offsets, np.asarray(extra_offsets, dtype=np.float64)))
        all_kinds = np.concatenate((all_kinds, np.asarray(extra_kinds, dtype=np.int8)))
        all_cols = np.concatenate((all_cols, np.asarray(extra_cols, dtype=np.int32)))
        all_vals = np.concatenate((all_vals, np.asarray(extra_vals, dtype=np.float64)))

    selected_idx_list: list[int] = []
    used_offsets: set[float] = set()
    for idx, offset in enumerate(all_offsets.tolist()):
        if offset in used_offsets:
            continue
        selected_idx_list.append(idx)
        used_offsets.add(offset)
        if len(selected_idx_list) >= 6:
            break

    n_selected = len(selected_idx_list)
    if n_selected < 6:
        raise SystemExit(f"Unable to build a 4th-order cut-cell stencil in {axis}-direction at row {row}.")

    selected_idx = np.asarray(selected_idx_list[:6], dtype=np.int32)
    selected_offsets = all_offsets[selected_idx[:n_selected]]
    selected_kinds = all_kinds[selected_idx[:n_selected]]
    selected_cols = all_cols[selected_idx[:n_selected]]
    selected_vals = all_vals[selected_idx[:n_selected]]

    order = np.argsort(selected_offsets)
    selected_offsets = selected_offsets[order]
    selected_kinds = selected_kinds[order]
    selected_cols = selected_cols[order]
    selected_vals = selected_vals[order]

    weights = finite_difference_second_derivative_weights(selected_offsets.tolist())

    interior_mask = selected_kinds == 1
    boundary_mask = selected_kinds == 2
    center_mask = selected_kinds == 0

    matrix_entries: list[tuple[int, float]] = [
        (int(col), float(weight))
        for col, weight in zip(selected_cols[interior_mask].tolist(), weights[interior_mask].tolist())
    ]
    rhs_shift = float(np.dot(weights[boundary_mask], selected_vals[boundary_mask]))
    center_weight = float(weights[center_mask][0])
    matrix_entries.append((row, center_weight))
    return matrix_entries, rhs_shift, center_weight, True

def assemble_fd4_system_shortley_weller_hybrid(
    mesh: dict[str, np.ndarray | float | int],
    fields: dict[str, np.ndarray],
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
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
    ) and all(
        key in fields for key in ("inner_boundary_sample_value", "outer_boundary_sample_value")
    )
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

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.empty(inner_indices.size, dtype=np.float64)
    n_irregular_x = 0
    n_irregular_y = 0
    perf: dict[str, float | int] = {
        "boundary_trace_calls": 0,
        "boundary_trace_time_s": 0.0,
    }
    std_center = -30.0 / (12.0 * spacing * spacing)
    std_near = 16.0 / (12.0 * spacing * spacing)
    std_far = -1.0 / (12.0 * spacing * spacing)

    lut_pad = np.pad(cart_lookup, ((2, 2), (2, 2)), mode="constant")
    inner_gidx0 = inner_indices - 1
    ii0 = cart_i_all[inner_gidx0] - cart_i_min + 2
    jj0 = cart_j_all[inner_gidx0] - cart_j_min + 2

    x_gids = np.stack(
        [lut_pad[jj0, ii0 - 2], lut_pad[jj0, ii0 - 1], lut_pad[jj0, ii0 + 1], lut_pad[jj0, ii0 + 2]],
        axis=1,
    )
    y_gids = np.stack(
        [lut_pad[jj0 - 2, ii0], lut_pad[jj0 - 1, ii0], lut_pad[jj0 + 1, ii0], lut_pad[jj0 + 2, ii0]],
        axis=1,
    )

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

    for row, gidx1 in enumerate(inner_indices.tolist()):
        gidx0 = gidx1 - 1
        reaction = float(lambda_inner[row])
        rhs_row = float(rhs_all[gidx0])

        x_i = float(x_all[gidx0])
        y_i = float(y_all[gidx0])
        i0 = int(cart_i_all[gidx0])
        j0 = int(cart_j_all[gidx0])
        diag = reaction

        if bool(x_regular[row]):
            x_entries = [
                (row, std_center),
                (int(x_cols_regular[row, 0]), std_far),
                (int(x_cols_regular[row, 1]), std_near),
                (int(x_cols_regular[row, 2]), std_near),
                (int(x_cols_regular[row, 3]), std_far),
            ]
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

        if bool(y_regular[row]):
            y_entries = [
                (row, std_center),
                (int(y_cols_regular[row, 0]), std_far),
                (int(y_cols_regular[row, 1]), std_near),
                (int(y_cols_regular[row, 2]), std_near),
                (int(y_cols_regular[row, 3]), std_far),
            ]
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

        for col, weight in x_entries:
            if col == row:
                diag -= weight
            else:
                rows.append(row)
                cols.append(col)
                vals.append(-weight)
        for col, weight in y_entries:
            if col == row:
                diag -= weight
            else:
                rows.append(row)
                cols.append(col)
                vals.append(-weight)

        rhs_row += x_rhs_shift + y_rhs_shift
        rows.append(row)
        cols.append(row)
        vals.append(diag)
        b[row] = rhs_row
        n_irregular_x += int(x_irregular)
        n_irregular_y += int(y_irregular)

    t_loop = time.perf_counter()
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(inner_indices.size, inner_indices.size))
    t_csr = time.perf_counter()
    meta = {
        "n_inner": int(np.count_nonzero(pinfo == PINFO_INNER)),
        "n_boundary": int(np.count_nonzero(pinfo == PINFO_BOUNDARY)),
        "n_ghost": int(np.count_nonzero(pinfo == PINFO_GHOST)),
        "n_unknowns": int(inner_indices.size),
        "nnz": int(A.nnz),
        "nx_f": int(mesh["nx_f"]),
        "ny_f": int(mesh["ny_f"]),
        "n_irregular_x_rows": n_irregular_x,
        "n_irregular_y_rows": n_irregular_y,
        "boundary_setup_time_s": t_boundary_setup - t0,
        "lookup_setup_time_s": t_lookup - t_boundary_setup,
        "assembly_loop_time_s": t_loop - t_lookup,
        "csr_build_time_s": t_csr - t_loop,
        "assembly_total_time_s": t_csr - t0,
        "boundary_trace_calls": int(perf["boundary_trace_calls"]),
        "boundary_trace_time_s": float(perf["boundary_trace_time_s"]),
    }
    return A, b, inner_indices, u_exact, meta


def scatter_field(mesh: dict[str, np.ndarray | float | int], values: np.ndarray) -> np.ndarray:
    field = np.full((int(mesh["ny_f"]), int(mesh["nx_f"])), np.nan, dtype=np.float64)
    ii = np.asarray(mesh["cart_i"], dtype=np.int64) - 1
    jj = np.asarray(mesh["cart_j"], dtype=np.int64) - 1
    field[jj, ii] = values
    return field


def solve_with_amg_bicgstab(A: sparse.csr_matrix, b: np.ndarray, *, tol: float, maxiter: int) -> tuple[np.ndarray, list[float], dict[str, float | int]]:
    start = time.perf_counter()
    ml = pyamg.ruge_stuben_solver(A)
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
    t0 = time.perf_counter()
    mesh = load_mesh(args.mesh.resolve())
    t_mesh = time.perf_counter()
    fields = load_fields(args.data.resolve())
    t_data = time.perf_counter()

    solver_label = "FD4+PyAMG-BiCGSTAB"
    A, b, active_indices, u_exact, meta = assemble_fd4_system_shortley_weller_hybrid(mesh, fields)
    t_assembly = time.perf_counter()
    x, history, stats = solve_with_amg_bicgstab(A, b, tol=args.tol, maxiter=args.maxiter)
    t_solve = time.perf_counter()
    u_num = np.asarray(u_exact, dtype=np.float64).copy()
    u_num[np.asarray(active_indices, dtype=np.int32) - 1] = x
    inner = np.asarray(mesh["inner_indices"], dtype=np.int32) - 1

    err = u_num - u_exact
    inner_rms = float(np.sqrt(np.mean(err[inner] ** 2)))
    inner_ref = float(np.sqrt(np.mean(u_exact[inner] ** 2)))
    active = np.asarray(mesh["pinfo"], dtype=np.int32) != PINFO_GHOST
    active_rms = float(np.sqrt(np.mean(err[active] ** 2)))
    active_ref = float(np.sqrt(np.mean(u_exact[active] ** 2)))

    print(f"{solver_label} error L2 rms (inner)      : {inner_rms:.6e}")
    print(f"{solver_label} relative L2 err (inner)   : {inner_rms / inner_ref:.6e}")
    print(f"{solver_label} error L2 rms (active)     : {active_rms:.6e}")
    print(f"{solver_label} relative L2 err (active)  : {active_rms / active_ref:.6e}")
    print(f"{solver_label} iterations                : {stats['iterations']}")
    print(f"{solver_label} info                      : {stats['info']}")
    print(f"{solver_label} setup time [s]            : {stats['setup_time_s']:.6e}")
    print(f"{solver_label} solve time [s]            : {stats['solve_time_s']:.6e}")
    print(f"{solver_label} total time [s]            : {stats['total_time_s']:.6e}")
    print(f"{solver_label} mesh load time [s]        : {t_mesh - t0:.6e}")
    print(f"{solver_label} data load time [s]        : {t_data - t_mesh:.6e}")
    print(f"{solver_label} assembly wall time [s]    : {t_assembly - t_data:.6e}")
    print(f"{solver_label} end-to-end time [s]       : {t_solve - t0:.6e}")
    print(f"{solver_label} AMG levels                : {stats['levels']}")
    print(f"{solver_label} operator complexity       : {stats['operator_complexity']:.6e}")
    print(f"{solver_label} grid complexity           : {stats['grid_complexity']:.6e}")
    print(f"Unknown inner points                    : {meta['n_inner']}")
    if "n_unknowns" in meta:
        print(f"Total unknown points                    : {meta['n_unknowns']}")
    if "n_irregular_x_rows" in meta:
        print(f"Irregular x-closure rows                : {meta['n_irregular_x_rows']}")
        print(f"Irregular y-closure rows                : {meta['n_irregular_y_rows']}")
        print(f"Boundary setup time [s]                : {meta['boundary_setup_time_s']:.6e}")
        print(f"Lookup setup time [s]                  : {meta['lookup_setup_time_s']:.6e}")
        print(f"Assembly loop time [s]                 : {meta['assembly_loop_time_s']:.6e}")
        print(f"CSR build time [s]                     : {meta['csr_build_time_s']:.6e}")
        print(f"Boundary trace calls                   : {meta['boundary_trace_calls']}")
        print(f"Boundary trace time [s]                : {meta['boundary_trace_time_s']:.6e}")
    print(f"Boundary support points                 : {meta['n_boundary']}")
    print(f"Ghost support points                    : {meta['n_ghost']}")
    print(f"Matrix nnz                              : {meta['nnz']}")

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
            ax2.semilogy(np.arange(1, len(history) + 1), history, label=solver_label)
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
