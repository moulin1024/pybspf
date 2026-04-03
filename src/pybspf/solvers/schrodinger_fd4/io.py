from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fourth-order immersed-boundary Schrödinger solver with AMG + BiCGSTAB.")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path(__file__).resolve().parents[4] / "examples" / "bvp" / "multigrid.nc",
        help="Path to multigrid.nc.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[4] / "examples" / "bvp" / "schrodinger_data.nc",
        help="Path to schrodinger_data.nc.",
    )
    parser.add_argument("--plot", action="store_true", help="Show diagnostic plots.")
    parser.add_argument("--save-plots", action="store_true", help="Save diagnostic plots next to the data file.")
    parser.add_argument("--tol", type=float, default=1.0e-10, help="Relative tolerance for BiCGSTAB.")
    parser.add_argument("--maxiter", type=int, default=1000, help="Maximum BiCGSTAB iterations.")
    parser.add_argument(
        "--amg-type",
        choices=("rs", "sa"),
        default="sa",
        help="AMG hierarchy type: classical Ruge-Stuben (rs) or smoothed aggregation (sa).",
    )
    parser.add_argument(
        "--benchmark-rhs",
        type=int,
        default=1,
        help="Number of in-memory RHS solves to benchmark with the same preprocessed operator.",
    )
    parser.add_argument(
        "--report-boundary-bands",
        action="store_true",
        help="Print error statistics in distance-to-boundary bands.",
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


def point_coordinates(mesh: dict[str, np.ndarray | float | int]) -> tuple[np.ndarray, np.ndarray]:
    spacing = float(mesh["spacing"])
    xmin = float(mesh["xmin"])
    ymin = float(mesh["ymin"])
    cart_i = np.asarray(mesh["cart_i"], dtype=np.float64)
    cart_j = np.asarray(mesh["cart_j"], dtype=np.float64)
    x = xmin + (cart_i - 1.0) * spacing
    y = ymin + (cart_j - 1.0) * spacing
    return x, y


def scatter_field(mesh: dict[str, np.ndarray | float | int], values: np.ndarray) -> np.ndarray:
    field = np.full((int(mesh["ny_f"]), int(mesh["nx_f"])), np.nan, dtype=np.float64)
    ii = np.asarray(mesh["cart_i"], dtype=np.int64) - 1
    jj = np.asarray(mesh["cart_j"], dtype=np.int64) - 1
    field[jj, ii] = values
    return field
