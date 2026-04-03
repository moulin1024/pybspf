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
from scipy.interpolate import make_interp_spline
from scipy import sparse
from scipy.sparse import linalg as spla

PINFO_INNER = 1
PINFO_BOUNDARY = 2
PINFO_GHOST = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finite-difference + AMG baseline for the circular MMS BVP.")
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
    parser.add_argument(
        "--boundary-treatment",
        choices=("linear-extrapolation", "trace-only-bspline", "ghost-bspline", "shortley-weller", "shortley-weller-4th"),
        default="linear-extrapolation",
        help="How Dirichlet boundary data is incorporated.",
    )
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
        }
    return models


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

    for name, model in boundary_models.items():
        dense_x = np.asarray(model["dense_x"], dtype=np.float64)
        dense_y = np.asarray(model["dense_y"], dtype=np.float64)
        dense_t = np.asarray(model["dense_t"], dtype=np.float64)
        for idx in range(dense_t.size - 1):
            x1 = dense_x[idx]
            y1 = dense_y[idx]
            x2 = dense_x[idx + 1]
            y2 = dense_y[idx + 1]
            t1 = dense_t[idx]
            t2 = dense_t[idx + 1]

            if di != 0:
                if (y1 - y_i) * (y2 - y_i) > 0.0 or abs(y2 - y1) < 1.0e-14:
                    continue
                alpha = (y_i - y1) / (y2 - y1)
                if alpha < -1.0e-12 or alpha > 1.0 + 1.0e-12:
                    continue
                xb = x1 + alpha * (x2 - x1)
                delta = di * (xb - x_i)
                if delta > 1.0e-12:
                    tb = t1 + alpha * (t2 - t1)
                    candidates.append((delta, xb, y_i, name, tb))
            else:
                if (x1 - x_i) * (x2 - x_i) > 0.0 or abs(x2 - x1) < 1.0e-14:
                    continue
                alpha = (x_i - x1) / (x2 - x1)
                if alpha < -1.0e-12 or alpha > 1.0 + 1.0e-12:
                    continue
                yb = y1 + alpha * (y2 - y1)
                delta = dj * (yb - y_i)
                if delta > 1.0e-12:
                    tb = t1 + alpha * (t2 - t1)
                    candidates.append((delta, x_i, yb, name, tb))

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


def build_cartesian_lookup(mesh: dict[str, np.ndarray | float | int]) -> dict[tuple[int, int], int]:
    cart_i = np.asarray(mesh["cart_i"], dtype=np.int32)
    cart_j = np.asarray(mesh["cart_j"], dtype=np.int32)
    return {(int(i), int(j)): idx + 1 for idx, (i, j) in enumerate(zip(cart_i.tolist(), cart_j.tolist()))}


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
    x_i: float,
    y_i: float,
    trace_splines: dict[str, object],
    rhomin: float,
    rhomax: float,
    boundary_models: dict[str, dict[str, np.ndarray | object]] | None = None,
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
    cart_lookup: dict[tuple[int, int], int],
    global_to_inner: dict[int, int],
    spacing: float,
    x_i: float,
    y_i: float,
    trace_splines: dict[str, object],
    rhomin: float,
    rhomax: float,
    boundary_models: dict[str, dict[str, np.ndarray | object]] | None = None,
) -> tuple[list[tuple[int, float]], float, float, bool]:
    if axis == "x":
        step = (1, 0)
    else:
        step = (0, 1)

    i0 = int(cart_i_all[gidx0])
    j0 = int(cart_j_all[gidx0])

    def collect_side(sign: int, max_steps: int = 8) -> list[tuple[float, int]]:
        entries: list[tuple[float, int]] = []
        di = sign * step[0]
        dj = sign * step[1]
        for k in range(1, max_steps + 1):
            key = (i0 + k * di, j0 + k * dj)
            nidx1 = cart_lookup.get(key, 0)
            if nidx1 <= 0 or pinfo[nidx1 - 1] != PINFO_INNER:
                break
            entries.append((sign * k * spacing, global_to_inner[nidx1]))
        return entries

    neg_interior = collect_side(-1)
    pos_interior = collect_side(1)

    if len(neg_interior) >= 2 and len(pos_interior) >= 2:
        weights = {
            -2.0 * spacing: -1.0 / (12.0 * spacing * spacing),
            -1.0 * spacing: 16.0 / (12.0 * spacing * spacing),
            0.0: -30.0 / (12.0 * spacing * spacing),
            1.0 * spacing: 16.0 / (12.0 * spacing * spacing),
            2.0 * spacing: -1.0 / (12.0 * spacing * spacing),
        }
        entries: list[tuple[int, float]] = [(row, weights[0.0])]
        entries.append((neg_interior[1][1], weights[-2.0 * spacing]))
        entries.append((neg_interior[0][1], weights[-1.0 * spacing]))
        entries.append((pos_interior[0][1], weights[1.0 * spacing]))
        entries.append((pos_interior[1][1], weights[2.0 * spacing]))
        return entries, 0.0, 0.0, False

    neg_delta, neg_value = evaluate_boundary_trace_value(
        direction=(-step[0], -step[1]),
        x_i=x_i,
        y_i=y_i,
        trace_splines=trace_splines,
        rhomin=rhomin,
        rhomax=rhomax,
        boundary_models=boundary_models,
    )
    pos_delta, pos_value = evaluate_boundary_trace_value(
        direction=(step[0], step[1]),
        x_i=x_i,
        y_i=y_i,
        trace_splines=trace_splines,
        rhomin=rhomin,
        rhomax=rhomax,
        boundary_models=boundary_models,
    )

    selected: list[tuple[float, str, int | None, float | None]] = [(0.0, "center", row, None)]
    if len(neg_interior) < 2:
        selected.append((-neg_delta, "boundary", None, neg_value))
    if len(pos_interior) < 2:
        selected.append((pos_delta, "boundary", None, pos_value))

    pool: list[tuple[float, str, int | None, float | None]] = []
    for offset, col in neg_interior:
        pool.append((offset, "interior", col, None))
    for offset, col in pos_interior:
        pool.append((offset, "interior", col, None))
    pool.sort(key=lambda item: (abs(item[0]), item[0]))

    used_offsets = {item[0] for item in selected}
    for item in pool:
        if len(selected) >= 6:
            break
        if item[0] in used_offsets:
            continue
        selected.append(item)
        used_offsets.add(item[0])

    if len(selected) < 6 and -neg_delta not in used_offsets:
        selected.append((-neg_delta, "boundary", None, neg_value))
        used_offsets.add(-neg_delta)
    if len(selected) < 6 and pos_delta not in used_offsets:
        selected.append((pos_delta, "boundary", None, pos_value))
        used_offsets.add(pos_delta)

    if len(selected) < 6:
        raise SystemExit(f"Unable to build a 4th-order cut-cell stencil in {axis}-direction at row {row}.")

    selected.sort(key=lambda item: item[0])
    offsets = [item[0] for item in selected]
    weights = finite_difference_second_derivative_weights(offsets)

    matrix_entries: list[tuple[int, float]] = []
    rhs_shift = 0.0
    center_weight = 0.0
    for weight, (_, kind, col, value) in zip(weights.tolist(), selected):
        if kind == "center":
            center_weight = weight
        elif kind == "interior":
            matrix_entries.append((int(col), weight))
        else:
            rhs_shift += weight * float(value)
    matrix_entries.append((row, center_weight))
    return matrix_entries, rhs_shift, center_weight, True


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
    support_values = build_linear_support_values(mesh, fields)
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
                rhs_row += weight * support_values[nidx0]

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


def assemble_fd2_system_trace_only_bspline(
    mesh: dict[str, np.ndarray | float | int],
    fields: dict[str, np.ndarray],
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, dict[str, int]]:
    params = default_circular_params()
    rhomin = params["rhomin"]
    rhomax = params["rhomax"]
    trace_splines = build_trace_splines(rhomin=rhomin, rhomax=rhomax, form="helmholtz")

    spacing = float(mesh["spacing"])
    inv_h2 = 1.0 / (spacing * spacing)

    pinfo = np.asarray(mesh["pinfo"], dtype=np.int32)
    index_neighbor = np.asarray(mesh["index_neighbor"], dtype=np.int32)
    size_neighbor = int(mesh["size_neighbor"])
    inner_indices = np.asarray(mesh["inner_indices"], dtype=np.int32)
    global_to_inner = {int(gidx): row for row, gidx in enumerate(inner_indices.tolist())}

    x_all, y_all = point_coordinates(mesh)
    rho_all = np.hypot(x_all, y_all)
    theta_all = circular_theta(x_all, y_all)
    co_all = np.asarray(fields["co"], dtype=np.float64)
    lambda_inner = y_all[inner_indices - 1]
    xi_inner = np.sqrt(rho_all[inner_indices - 1]) * (2.0 + np.cos(theta_all[inner_indices - 1]))
    rhs_all = evaluate_mms_rhs(rho_all, theta_all, rhomin=rhomin, rhomax=rhomax)
    sol_exact = evaluate_mms_solution(rho_all, theta_all, rhomin=rhomin, rhomax=rhomax)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.empty(inner_indices.size, dtype=np.float64)

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
        rhs_row = float(rhs_all[gidx0] / xi)
        diag = reaction

        for jj, ii in directions:
            nidx1 = int(index_neighbor[gidx0, jj, ii])
            if nidx1 > 0 and pinfo[nidx1 - 1] == PINFO_INNER:
                c_face = 0.5 * (co_all[gidx0] + co_all[nidx1 - 1])
                weight = c_face * inv_h2
                diag += weight
                rows.append(row)
                cols.append(global_to_inner[nidx1])
                vals.append(-weight)
                continue

            if nidx1 > 0:
                x_nb = float(x_all[nidx1 - 1])
                y_nb = float(y_all[nidx1 - 1])
                theta_b = float(theta_all[nidx1 - 1])
                rho_nb = float(rho_all[nidx1 - 1])
            else:
                di = ii - size_neighbor
                dj = jj - size_neighbor
                x_nb = float(x_all[gidx0] + di * spacing)
                y_nb = float(y_all[gidx0] + dj * spacing)
                theta_b = float(circular_theta(np.asarray([x_nb]), np.asarray([y_nb]))[0])
                rho_nb = math.hypot(x_nb, y_nb)

            boundary_name = "inner" if abs(rho_nb - rhomin) < abs(rho_nb - rhomax) else "outer"
            boundary_radius = rhomin if boundary_name == "inner" else rhomax
            g_b = float(trace_splines[boundary_name](np.asarray([theta_b]))[0])
            c_boundary = float(evaluate_mms_co(np.asarray([boundary_radius]), np.asarray([theta_b]), rhomin=rhomin, rhomax=rhomax)[0])
            c_face = 0.5 * (co_all[gidx0] + c_boundary)
            weight = c_face * inv_h2
            diag += weight
            rhs_row += weight * g_b

        rows.append(row)
        cols.append(row)
        vals.append(diag)
        b[row] = rhs_row

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(inner_indices.size, inner_indices.size))
    meta = {
        "n_inner": int(np.count_nonzero(pinfo == PINFO_INNER)),
        "n_boundary": int(np.count_nonzero(pinfo == PINFO_BOUNDARY)),
        "n_ghost": int(np.count_nonzero(pinfo == PINFO_GHOST)),
        "n_unknowns": int(inner_indices.size),
        "nnz": int(A.nnz),
        "nx_f": int(mesh["nx_f"]),
        "ny_f": int(mesh["ny_f"]),
    }
    return A, b, inner_indices, sol_exact, meta


def assemble_fd2_system_ghost_bspline(
    mesh: dict[str, np.ndarray | float | int],
    fields: dict[str, np.ndarray],
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    params = default_circular_params()
    rhomin = params["rhomin"]
    rhomax = params["rhomax"]
    trace_splines = build_trace_splines(rhomin=rhomin, rhomax=rhomax, form="helmholtz")

    spacing = float(mesh["spacing"])
    inv_h2 = 1.0 / (spacing * spacing)

    pinfo = np.asarray(mesh["pinfo"], dtype=np.int32)
    index_neighbor = np.asarray(mesh["index_neighbor"], dtype=np.int32)
    size_neighbor = int(mesh["size_neighbor"])
    inner_indices = np.asarray(mesh["inner_indices"], dtype=np.int32)
    global_to_inner = {int(gidx): row for row, gidx in enumerate(inner_indices.tolist())}

    x_all, y_all = point_coordinates(mesh)
    rho_all = np.hypot(x_all, y_all)
    theta_all = circular_theta(x_all, y_all)
    co_all = np.asarray(fields["co"], dtype=np.float64)
    rhs_all = np.asarray(fields["rhs"], dtype=np.float64)
    u_exact = np.asarray(fields["sol"], dtype=np.float64)
    lambda_inner = np.asarray(fields["lambda_inner"], dtype=np.float64)
    xi_inner = np.asarray(fields["xi_inner"], dtype=np.float64)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.empty(inner_indices.size, dtype=np.float64)

    for row, gidx1 in enumerate(inner_indices.tolist()):
        gidx0 = gidx1 - 1
        xi = float(xi_inner[row])
        reaction = float(lambda_inner[row] / xi)
        rhs_row = float(rhs_all[gidx0] / xi)
        diag = reaction

        x_i = float(x_all[gidx0])
        y_i = float(y_all[gidx0])

        left = side_geometry(
            gidx0=gidx0,
            x_i=x_i,
            y_i=y_i,
            di=-1,
            dj=0,
            spacing=spacing,
            pinfo=pinfo,
            index_neighbor=index_neighbor,
            size_neighbor=size_neighbor,
            x_all=x_all,
            y_all=y_all,
            co_all=co_all,
            global_to_inner=global_to_inner,
            trace_splines=trace_splines,
            rhomin=rhomin,
            rhomax=rhomax,
        )
        right = side_geometry(
            gidx0=gidx0,
            x_i=x_i,
            y_i=y_i,
            di=1,
            dj=0,
            spacing=spacing,
            pinfo=pinfo,
            index_neighbor=index_neighbor,
            size_neighbor=size_neighbor,
            x_all=x_all,
            y_all=y_all,
            co_all=co_all,
            global_to_inner=global_to_inner,
            trace_splines=trace_splines,
            rhomin=rhomin,
            rhomax=rhomax,
        )
        bottom = side_geometry(
            gidx0=gidx0,
            x_i=x_i,
            y_i=y_i,
            di=0,
            dj=-1,
            spacing=spacing,
            pinfo=pinfo,
            index_neighbor=index_neighbor,
            size_neighbor=size_neighbor,
            x_all=x_all,
            y_all=y_all,
            co_all=co_all,
            global_to_inner=global_to_inner,
            trace_splines=trace_splines,
            rhomin=rhomin,
            rhomax=rhomax,
        )
        top = side_geometry(
            gidx0=gidx0,
            x_i=x_i,
            y_i=y_i,
            di=0,
            dj=1,
            spacing=spacing,
            pinfo=pinfo,
            index_neighbor=index_neighbor,
            size_neighbor=size_neighbor,
            x_all=x_all,
            y_all=y_all,
            co_all=co_all,
            global_to_inner=global_to_inner,
            trace_splines=trace_splines,
            rhomin=rhomin,
            rhomax=rhomax,
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
        b[row] = rhs_row

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(inner_indices.size, inner_indices.size))
    meta = {
        "n_inner": int(np.count_nonzero(pinfo == PINFO_INNER)),
        "n_boundary": int(np.count_nonzero(pinfo == PINFO_BOUNDARY)),
        "n_ghost": int(np.count_nonzero(pinfo == PINFO_GHOST)),
        "n_unknowns": int(inner_indices.size),
        "nnz": int(A.nnz),
        "nx_f": int(mesh["nx_f"]),
        "ny_f": int(mesh["ny_f"]),
    }
    return A, b, inner_indices, u_exact, meta


def assemble_fd2_system_shortley_weller(
    mesh: dict[str, np.ndarray | float | int],
    fields: dict[str, np.ndarray],
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    params = default_circular_params()
    rhomin = params["rhomin"]
    rhomax = params["rhomax"]
    trace_splines = build_trace_splines(rhomin=rhomin, rhomax=rhomax, form="helmholtz")

    pinfo = np.asarray(mesh["pinfo"], dtype=np.int32)
    index_neighbor = np.asarray(mesh["index_neighbor"], dtype=np.int32)
    size_neighbor = int(mesh["size_neighbor"])
    inner_indices = np.asarray(mesh["inner_indices"], dtype=np.int32)
    global_to_inner = {int(gidx): row for row, gidx in enumerate(inner_indices.tolist())}

    x_all, y_all = point_coordinates(mesh)
    co_all = np.asarray(fields["co"], dtype=np.float64)
    rhs_all = np.asarray(fields["rhs"], dtype=np.float64)
    u_exact = np.asarray(fields["sol"], dtype=np.float64)
    lambda_inner = np.asarray(fields["lambda_inner"], dtype=np.float64)
    xi_inner = np.asarray(fields["xi_inner"], dtype=np.float64)
    spacing = float(mesh["spacing"])

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.empty(inner_indices.size, dtype=np.float64)

    for row, gidx1 in enumerate(inner_indices.tolist()):
        gidx0 = gidx1 - 1
        xi = float(xi_inner[row])
        reaction = float(lambda_inner[row] / xi)
        rhs_row = float(rhs_all[gidx0] / xi)

        x_i = float(x_all[gidx0])
        y_i = float(y_all[gidx0])
        b[row] = append_shortley_weller_row(
            row=row,
            gidx0=gidx0,
            rhs_row=rhs_row,
            reaction=reaction,
            x_i=x_i,
            y_i=y_i,
            spacing=spacing,
            pinfo=pinfo,
            index_neighbor=index_neighbor,
            size_neighbor=size_neighbor,
            x_all=x_all,
            y_all=y_all,
            co_all=co_all,
            global_to_inner=global_to_inner,
            trace_splines=trace_splines,
            rhomin=rhomin,
            rhomax=rhomax,
            rows=rows,
            cols=cols,
            vals=vals,
        )

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(inner_indices.size, inner_indices.size))
    meta = {
        "n_inner": int(np.count_nonzero(pinfo == PINFO_INNER)),
        "n_boundary": int(np.count_nonzero(pinfo == PINFO_BOUNDARY)),
        "n_ghost": int(np.count_nonzero(pinfo == PINFO_GHOST)),
        "n_unknowns": int(inner_indices.size),
        "nnz": int(A.nnz),
        "nx_f": int(mesh["nx_f"]),
        "ny_f": int(mesh["ny_f"]),
    }
    return A, b, inner_indices, u_exact, meta


def assemble_fd4_system_shortley_weller_hybrid(
    mesh: dict[str, np.ndarray | float | int],
    fields: dict[str, np.ndarray],
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
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
    else:
        trace_splines = build_trace_splines(rhomin=rhomin, rhomax=rhomax, form="schrodinger")
        boundary_models = None

    pinfo = np.asarray(mesh["pinfo"], dtype=np.int32)
    inner_indices = np.asarray(mesh["inner_indices"], dtype=np.int32)
    global_to_inner = {int(gidx): row for row, gidx in enumerate(inner_indices.tolist())}
    cart_i_all = np.asarray(mesh["cart_i"], dtype=np.int32)
    cart_j_all = np.asarray(mesh["cart_j"], dtype=np.int32)
    cart_lookup = build_cartesian_lookup(mesh)

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

    for row, gidx1 in enumerate(inner_indices.tolist()):
        gidx0 = gidx1 - 1
        reaction = float(lambda_inner[row])
        rhs_row = float(rhs_all[gidx0])

        x_i = float(x_all[gidx0])
        y_i = float(y_all[gidx0])
        diag = reaction

        x_entries, x_rhs_shift, _, x_irregular = build_axis_fd4_stencil(
            gidx0=gidx0,
            row=row,
            axis="x",
            cart_i_all=cart_i_all,
            cart_j_all=cart_j_all,
            pinfo=pinfo,
            cart_lookup=cart_lookup,
            global_to_inner=global_to_inner,
            spacing=spacing,
            x_i=x_i,
            y_i=y_i,
            trace_splines=trace_splines,
            rhomin=rhomin,
            rhomax=rhomax,
            boundary_models=boundary_models,
        )
        y_entries, y_rhs_shift, _, y_irregular = build_axis_fd4_stencil(
            gidx0=gidx0,
            row=row,
            axis="y",
            cart_i_all=cart_i_all,
            cart_j_all=cart_j_all,
            pinfo=pinfo,
            cart_lookup=cart_lookup,
            global_to_inner=global_to_inner,
            spacing=spacing,
            x_i=x_i,
            y_i=y_i,
            trace_splines=trace_splines,
            rhomin=rhomin,
            rhomax=rhomax,
            boundary_models=boundary_models,
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

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(inner_indices.size, inner_indices.size))
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
    }
    return A, b, inner_indices, u_exact, meta


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
    mesh = load_mesh(args.mesh.resolve())
    fields = load_fields(args.data.resolve())

    solver_label = "FD2+PyAMG-PCG"

    if args.boundary_treatment == "linear-extrapolation":
        A, b, meta = assemble_fd2_system(mesh, fields)
        x, history, stats = solve_with_amg_pcg(A, b, tol=args.tol, maxiter=args.maxiter)
        u_num = np.asarray(fields["sol"], dtype=np.float64).copy()
        inner = np.asarray(mesh["inner_indices"], dtype=np.int32) - 1
        u_num[inner] = x
        u_exact = np.asarray(fields["sol"], dtype=np.float64)
    else:
        if args.boundary_treatment == "shortley-weller-4th":
            solver_label = "FD4+PyAMG-BiCGSTAB"
            A, b, active_indices, u_exact, meta = assemble_fd4_system_shortley_weller_hybrid(mesh, fields)
            x, history, stats = solve_with_amg_bicgstab(A, b, tol=args.tol, maxiter=args.maxiter)
        else:
            if args.data.name != "helmholtz_data.nc":
                raise SystemExit("B-spline boundary treatments are currently implemented for the circular Helmholtz MMS only.")
            if args.boundary_treatment == "trace-only-bspline":
                A, b, active_indices, u_exact, meta = assemble_fd2_system_trace_only_bspline(mesh, fields)
            elif args.boundary_treatment == "ghost-bspline":
                A, b, active_indices, u_exact, meta = assemble_fd2_system_ghost_bspline(mesh, fields)
            else:
                A, b, active_indices, u_exact, meta = assemble_fd2_system_shortley_weller(mesh, fields)
            x, history, stats = solve_with_amg_pcg(A, b, tol=args.tol, maxiter=args.maxiter)
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
    print(f"{solver_label} AMG levels                : {stats['levels']}")
    print(f"{solver_label} operator complexity       : {stats['operator_complexity']:.6e}")
    print(f"{solver_label} grid complexity           : {stats['grid_complexity']:.6e}")
    print(f"Boundary treatment                      : {args.boundary_treatment}")
    print(f"Unknown inner points                    : {meta['n_inner']}")
    if "n_unknowns" in meta:
        print(f"Total unknown points                    : {meta['n_unknowns']}")
    if "n_irregular_x_rows" in meta:
        print(f"Irregular x-closure rows                : {meta['n_irregular_x_rows']}")
        print(f"Irregular y-closure rows                : {meta['n_irregular_y_rows']}")
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
