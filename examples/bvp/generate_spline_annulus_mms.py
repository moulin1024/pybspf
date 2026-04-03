#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from netCDF4 import Dataset
from scipy.interpolate import make_interp_spline

PINFO_INNER = 1
PINFO_BOUNDARY = 2
PINFO_GHOST = 3

DISTRICT_CORE = 813
DISTRICT_CLOSED = 814
DISTRICT_WALL = 817

TWO_PI = 2.0 * math.pi
BOUNDARY_SPLINE_DEGREE = 4


def fortran_nint(value: float) -> int:
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return -int(math.floor(-value + 0.5))


@dataclass
class SplineAnnulusParams:
    spacing_f: float = 1.0e-3
    size_neighbor: int = 2
    size_ghost_layer: int = 2
    n_boundary_samples: int = 128
    outer_a: float = 0.42
    outer_b: float = 0.34
    outer_eps_c: float = 0.06
    outer_eps_s: float = 0.03
    inner_a: float = 0.18
    inner_b: float = 0.14
    inner_eps_c: float = 0.05
    inner_eps_s: float = -0.025


@dataclass
class SplineBoundary:
    name: str
    sample_t: np.ndarray
    sample_x: np.ndarray
    sample_y: np.ndarray
    spline_x: object
    spline_y: object
    dense_t: np.ndarray
    dense_x: np.ndarray
    dense_y: np.ndarray
    path: MplPath


@dataclass
class MeshLevel:
    lvl: int
    lvst: int
    spacing_f: float
    spacing_c: float
    size_neighbor: int
    size_ghost_layer: int
    xmin: float
    ymin: float
    nx_f: int
    ny_f: int
    cart_i: np.ndarray
    cart_j: np.ndarray
    index_neighbor: np.ndarray
    inner_indices: np.ndarray
    boundary_indices: np.ndarray
    ghost_indices: np.ndarray
    pinfo: np.ndarray
    district: np.ndarray

    @property
    def n_points(self) -> int:
        return int(self.cart_i.size)

    @property
    def n_points_inner(self) -> int:
        return int(self.inner_indices.size)

    @property
    def n_points_boundary(self) -> int:
        return int(self.boundary_indices.size)

    @property
    def n_points_ghost(self) -> int:
        return int(self.ghost_indices.size)


def periodic_samples(n_samples: int) -> np.ndarray:
    return np.linspace(0.0, TWO_PI, n_samples + 1, dtype=np.float64)


def deformed_ellipse(theta: np.ndarray, *, a: float, b: float, eps_c: float, eps_s: float) -> tuple[np.ndarray, np.ndarray]:
    modulation = 1.0 + eps_c * np.cos(2.0 * theta) + eps_s * np.sin(3.0 * theta)
    x = a * modulation * np.cos(theta)
    y = b * modulation * np.sin(theta)
    return x, y


def build_boundary(theta: np.ndarray, x: np.ndarray, y: np.ndarray, *, name: str, n_dense: int = 4096) -> SplineBoundary:
    spline_x = make_interp_spline(theta, x, k=BOUNDARY_SPLINE_DEGREE, bc_type="periodic")
    spline_y = make_interp_spline(theta, y, k=BOUNDARY_SPLINE_DEGREE, bc_type="periodic")
    dense_t = np.linspace(0.0, TWO_PI, n_dense + 1, dtype=np.float64)
    dense_x = spline_x(dense_t)
    dense_y = spline_y(dense_t)
    path = MplPath(np.column_stack([dense_x, dense_y]), closed=True)
    return SplineBoundary(
        name=name,
        sample_t=theta,
        sample_x=x,
        sample_y=y,
        spline_x=spline_x,
        spline_y=spline_y,
        dense_t=dense_t,
        dense_x=dense_x,
        dense_y=dense_y,
        path=path,
    )


def build_boundaries(params: SplineAnnulusParams) -> dict[str, SplineBoundary]:
    theta = periodic_samples(params.n_boundary_samples)
    outer_x, outer_y = deformed_ellipse(
        theta,
        a=params.outer_a,
        b=params.outer_b,
        eps_c=params.outer_eps_c,
        eps_s=params.outer_eps_s,
    )
    inner_x, inner_y = deformed_ellipse(
        theta,
        a=params.inner_a,
        b=params.inner_b,
        eps_c=params.inner_eps_c,
        eps_s=params.inner_eps_s,
    )
    return {
        "inner": build_boundary(theta, inner_x, inner_y, name="inner"),
        "outer": build_boundary(theta, outer_x, outer_y, name="outer"),
    }


def compute_bbox(boundaries: dict[str, SplineBoundary], spacing: float) -> tuple[float, float, int, int]:
    outer = boundaries["outer"]
    margin = 4.0 * spacing
    xmin = float(np.min(outer.dense_x) - margin)
    xmax = float(np.max(outer.dense_x) + margin)
    ymin = float(np.min(outer.dense_y) - margin)
    ymax = float(np.max(outer.dense_y) + margin)
    nx_f = fortran_nint((xmax - xmin) / spacing)
    ny_f = fortran_nint((ymax - ymin) / spacing)
    return xmin, ymin, nx_f, ny_f


def point_coordinates(cart_i: np.ndarray, cart_j: np.ndarray, xmin: float, ymin: float, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    x = xmin + (cart_i.astype(np.float64) - 1.0) * spacing
    y = ymin + (cart_j.astype(np.float64) - 1.0) * spacing
    return x, y


def inside_annulus(points: np.ndarray, boundaries: dict[str, SplineBoundary]) -> np.ndarray:
    inside_outer = boundaries["outer"].path.contains_points(points, radius=1.0e-12)
    inside_inner = boundaries["inner"].path.contains_points(points, radius=1.0e-12)
    return inside_outer & ~inside_inner


def build_cartesian_points(
    params: SplineAnnulusParams,
    boundaries: dict[str, SplineBoundary],
    xmin: float,
    ymin: float,
    nx_f: int,
    ny_f: int,
    lvst: int,
) -> tuple[np.ndarray, np.ndarray]:
    spacing_c = params.spacing_f * lvst
    i = np.arange(1, nx_f + 1, lvst, dtype=np.int32)
    j = np.arange(1, ny_f + 1, lvst, dtype=np.int32)
    ii, jj = np.meshgrid(i, j, indexing="xy")

    x = xmin + (ii.astype(np.float64) - 1.0) * params.spacing_f
    y = ymin + (jj.astype(np.float64) - 1.0) * params.spacing_f

    shifts = np.array([-0.5, 0.0, 0.5], dtype=np.float64) * spacing_c
    x_s = np.broadcast_to(x[..., None, None] + shifts[None, None, :, None], x.shape + (3, 3))
    y_s = np.broadcast_to(y[..., None, None] + shifts[None, None, None, :], y.shape + (3, 3))
    pts = np.stack([x_s, y_s], axis=-1).reshape(-1, 2)
    include = inside_annulus(pts, boundaries).reshape(x.shape + (3, 3)).any(axis=(2, 3))
    return ii[include].astype(np.int32), jj[include].astype(np.int32)


def build_connectivity(cart_i: np.ndarray, cart_j: np.ndarray, lvst: int, size_neighbor: int) -> np.ndarray:
    n_points = cart_i.size
    sis = 2 * size_neighbor + 1

    imin = int(cart_i.min())
    imax = int(cart_i.max())
    jmin = int(cart_j.min())
    jmax = int(cart_j.max())
    ni = (imax - imin) // lvst + 1
    nj = (jmax - jmin) // lvst + 1

    lut = np.zeros((nj, ni), dtype=np.int32)
    lut[(cart_j - jmin) // lvst, (cart_i - imin) // lvst] = np.arange(1, n_points + 1, dtype=np.int32)

    ishifts = np.arange(-size_neighbor, size_neighbor + 1, dtype=np.int32)
    jshifts = np.arange(-size_neighbor, size_neighbor + 1, dtype=np.int32)
    gi = cart_i[:, None, None] + lvst * ishifts[None, None, :]
    gj = cart_j[:, None, None] + lvst * jshifts[None, :, None]
    gi_full = np.broadcast_to(gi, (n_points, sis, sis))
    gj_full = np.broadcast_to(gj, (n_points, sis, sis))

    valid = (
        (gi_full >= imin) & (gi_full <= imax) &
        (gj_full >= jmin) & (gj_full <= jmax) &
        ((gi_full - imin) % lvst == 0) &
        ((gj_full - jmin) % lvst == 0)
    )

    index_neighbor = np.zeros((n_points, sis, sis), dtype=np.int32)
    index_neighbor[valid] = lut[
        ((gj_full[valid] - jmin) // lvst),
        ((gi_full[valid] - imin) // lvst),
    ]
    return index_neighbor


def build_boundary_and_inner(index_neighbor: np.ndarray, size_neighbor: int) -> tuple[np.ndarray, np.ndarray]:
    direct = [
        index_neighbor[:, -1 + size_neighbor, 0 + size_neighbor],
        index_neighbor[:, 0 + size_neighbor, -1 + size_neighbor],
        index_neighbor[:, 0 + size_neighbor, 1 + size_neighbor],
        index_neighbor[:, 1 + size_neighbor, 0 + size_neighbor],
    ]
    is_boundary = np.logical_or.reduce([arr == 0 for arr in direct])
    boundary_indices = np.nonzero(is_boundary)[0].astype(np.int32) + 1
    inner_indices = np.nonzero(~is_boundary)[0].astype(np.int32) + 1
    return inner_indices, boundary_indices


def build_ghost_layer(
    cart_i: np.ndarray,
    cart_j: np.ndarray,
    inner_indices: np.ndarray,
    lvst: int,
    size_ghost_layer: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if size_ghost_layer <= 0:
        return cart_i.copy(), cart_j.copy(), np.empty(0, dtype=np.int32)

    inner_i = cart_i[inner_indices - 1]
    inner_j = cart_j[inner_indices - 1]
    shifts = np.arange(-size_ghost_layer, size_ghost_layer + 1, dtype=np.int32)
    di, dj = np.meshgrid(shifts, shifts, indexing="xy")
    offsets = np.stack([lvst * di.ravel(), lvst * dj.ravel()], axis=1)
    candidates = np.stack(
        [inner_i[:, None] + offsets[None, :, 0], inner_j[:, None] + offsets[None, :, 1]],
        axis=2,
    ).reshape(-1, 2)
    candidates = np.unique(candidates, axis=0)
    existing = np.unique(np.stack([cart_i, cart_j], axis=1), axis=0)
    existing_view = existing.view([("", existing.dtype)] * 2)
    candidate_view = candidates.view([("", candidates.dtype)] * 2)
    ghosts = candidates[(~np.isin(candidate_view, existing_view)).ravel()]

    if ghosts.size == 0:
        return cart_i.copy(), cart_j.copy(), np.empty(0, dtype=np.int32)

    order = np.lexsort((ghosts[:, 0], ghosts[:, 1]))
    ghosts = ghosts[order]
    ghost_i = ghosts[:, 0].astype(np.int32)
    ghost_j = ghosts[:, 1].astype(np.int32)
    cart_i_all = np.concatenate([cart_i, ghost_i])
    cart_j_all = np.concatenate([cart_j, ghost_j])
    ghost_indices = np.arange(cart_i.size + 1, cart_i_all.size + 1, dtype=np.int32)
    return cart_i_all, cart_j_all, ghost_indices


def build_pinfo(n_points: int, inner_indices: np.ndarray, boundary_indices: np.ndarray, ghost_indices: np.ndarray) -> np.ndarray:
    pinfo = np.zeros(n_points, dtype=np.int32)
    pinfo[inner_indices - 1] = PINFO_INNER
    pinfo[boundary_indices - 1] = PINFO_BOUNDARY
    pinfo[ghost_indices - 1] = PINFO_GHOST
    return pinfo


def build_district(
    cart_i: np.ndarray,
    cart_j: np.ndarray,
    pinfo: np.ndarray,
    boundaries: dict[str, SplineBoundary],
    xmin: float,
    ymin: float,
    spacing: float,
) -> np.ndarray:
    x, y = point_coordinates(cart_i, cart_j, xmin, ymin, spacing)
    pts = np.column_stack([x, y])
    district = np.full(cart_i.size, DISTRICT_CLOSED, dtype=np.int32)
    inner_hole = boundaries["inner"].path.contains_points(pts, radius=1.0e-12)
    outer_inside = boundaries["outer"].path.contains_points(pts, radius=1.0e-12)
    district[(pinfo == PINFO_GHOST) & inner_hole] = DISTRICT_CORE
    district[(pinfo == PINFO_GHOST) & (~outer_inside)] = DISTRICT_WALL
    return district


def apply_patch(
    boundary_indices: np.ndarray,
    ghost_indices: np.ndarray,
    index_neighbor: np.ndarray,
    pinfo: np.ndarray,
    size_neighbor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if boundary_indices.size == 0:
        return boundary_indices, ghost_indices, pinfo

    local = index_neighbor[
        boundary_indices - 1,
        size_neighbor - 1:size_neighbor + 2,
        size_neighbor - 1:size_neighbor + 2,
    ]
    neighbor_pinfo = np.zeros_like(local, dtype=np.int32)
    mask = local > 0
    neighbor_pinfo[mask] = pinfo[local[mask] - 1]

    has_inner = np.any(neighbor_pinfo == PINFO_INNER, axis=(1, 2))
    patched = boundary_indices[~has_inner]
    if patched.size == 0:
        return boundary_indices, ghost_indices, pinfo

    pinfo_new = pinfo.copy()
    pinfo_new[patched - 1] = PINFO_GHOST
    boundary_new = boundary_indices[has_inner]
    ghost_new = np.unique(np.concatenate([ghost_indices, patched])).astype(np.int32)
    return boundary_new, ghost_new, pinfo_new


def build_finest_mesh(params: SplineAnnulusParams, boundaries: dict[str, SplineBoundary]) -> MeshLevel:
    lvl = 1
    lvst = 1
    xmin, ymin, nx_f, ny_f = compute_bbox(boundaries, params.spacing_f)
    spacing_c = params.spacing_f

    cart_i, cart_j = build_cartesian_points(params, boundaries, xmin, ymin, nx_f, ny_f, lvst)
    index_neighbor = build_connectivity(cart_i, cart_j, lvst, params.size_neighbor)
    inner_indices, boundary_indices = build_boundary_and_inner(index_neighbor, params.size_neighbor)
    cart_i, cart_j, ghost_indices = build_ghost_layer(cart_i, cart_j, inner_indices, lvst, params.size_ghost_layer)
    index_neighbor = build_connectivity(cart_i, cart_j, lvst, params.size_neighbor)
    pinfo = build_pinfo(cart_i.size, inner_indices, boundary_indices, ghost_indices)
    boundary_indices, ghost_indices, pinfo = apply_patch(boundary_indices, ghost_indices, index_neighbor, pinfo, params.size_neighbor)
    district = build_district(cart_i, cart_j, pinfo, boundaries, xmin, ymin, params.spacing_f)

    return MeshLevel(
        lvl=lvl,
        lvst=lvst,
        spacing_f=params.spacing_f,
        spacing_c=spacing_c,
        size_neighbor=params.size_neighbor,
        size_ghost_layer=params.size_ghost_layer,
        xmin=xmin,
        ymin=ymin,
        nx_f=nx_f,
        ny_f=ny_f,
        cart_i=cart_i,
        cart_j=cart_j,
        index_neighbor=index_neighbor,
        inner_indices=inner_indices,
        boundary_indices=boundary_indices,
        ghost_indices=ghost_indices,
        pinfo=pinfo,
        district=district,
    )


def manufactured_solution(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    psi = 1.15 + 0.18 * np.sin(8.0 * x + 0.3) * np.cos(7.0 * y - 0.2) + 0.07 * np.cos(5.0 * x - 4.0 * y)
    lap = (
        -(8.0**2 + 7.0**2) * 0.18 * np.sin(8.0 * x + 0.3) * np.cos(7.0 * y - 0.2)
        -(5.0**2 + 4.0**2) * 0.07 * np.cos(5.0 * x - 4.0 * y)
    )
    return psi, lap


def manufactured_potential(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 1.4 + 0.25 * np.cos(3.0 * x + 2.0 * y) + 0.15 * np.sin(4.0 * x - y)


def compute_schrodinger_fields(mesh: MeshLevel) -> dict[str, np.ndarray]:
    x, y = point_coordinates(mesh.cart_i, mesh.cart_j, mesh.xmin, mesh.ymin, mesh.spacing_f)
    psi, lap_psi = manufactured_solution(x, y)
    potential = manufactured_potential(x, y)
    rhs = psi.copy()
    inner = mesh.inner_indices - 1
    rhs[inner] = -lap_psi[inner] + potential[inner] * psi[inner]
    return {
        "co": np.ones(mesh.n_points, dtype=np.float64),
        "lambda": potential[inner].astype(np.float64),
        "xi": np.ones(mesh.n_points_inner, dtype=np.float64),
        "rhs": rhs.astype(np.float64),
        "sol": psi.astype(np.float64),
        "guess": np.zeros(mesh.n_points, dtype=np.float64),
        "potential": potential.astype(np.float64),
    }


def sample_boundary_values(boundary: SplineBoundary) -> np.ndarray:
    psi, _ = manufactured_solution(boundary.sample_x, boundary.sample_y)
    return psi.astype(np.float64)


def create_dimension(group: Dataset, name: str, size: int) -> None:
    group.createDimension(name, None if size == 0 else size)


def write_mesh_group(group: Dataset, mesh: MeshLevel, boundaries: dict[str, SplineBoundary]) -> None:
    group.setncattr("lvl", np.int32(mesh.lvl))
    group.setncattr("lvst", np.int32(mesh.lvst))
    group.setncattr("spacing_f", np.float64(mesh.spacing_f))
    group.setncattr("spacing_c", np.float64(mesh.spacing_c))
    group.setncattr("xmin", np.float64(mesh.xmin))
    group.setncattr("ymin", np.float64(mesh.ymin))
    group.setncattr("nx_f", np.int32(mesh.nx_f))
    group.setncattr("ny_f", np.int32(mesh.ny_f))
    group.setncattr("size_ghost_layer", np.int32(mesh.size_ghost_layer))
    group.setncattr("geometry_type", "spline_annulus")

    sis = 2 * mesh.size_neighbor + 1
    create_dimension(group, "n_points", mesh.n_points)
    create_dimension(group, "n_points_inner", mesh.n_points_inner)
    create_dimension(group, "n_points_boundary", mesh.n_points_boundary)
    create_dimension(group, "n_points_ghost", mesh.n_points_ghost)
    create_dimension(group, "size_neighbor", sis)
    create_dimension(group, "n_boundary_samples_inner", boundaries["inner"].sample_x.size)
    create_dimension(group, "n_boundary_samples_outer", boundaries["outer"].sample_x.size)

    group.createVariable("cart_i", "i4", ("n_points",))[:] = mesh.cart_i
    group.createVariable("cart_j", "i4", ("n_points",))[:] = mesh.cart_j
    group.createVariable("pinfo", "i4", ("n_points",))[:] = mesh.pinfo
    group.createVariable("district", "i4", ("n_points",))[:] = mesh.district
    group.createVariable("inner_indices", "i4", ("n_points_inner",))[:] = mesh.inner_indices
    group.createVariable("boundary_indices", "i4", ("n_points_boundary",))[:] = mesh.boundary_indices
    group.createVariable("ghost_indices", "i4", ("n_points_ghost",))[:] = mesh.ghost_indices
    group.createVariable("index_neighbor", "i4", ("n_points", "size_neighbor", "size_neighbor"))[:] = mesh.index_neighbor
    group.createVariable("inner_boundary_sample_x", "f8", ("n_boundary_samples_inner",))[:] = boundaries["inner"].sample_x
    group.createVariable("inner_boundary_sample_y", "f8", ("n_boundary_samples_inner",))[:] = boundaries["inner"].sample_y
    group.createVariable("outer_boundary_sample_x", "f8", ("n_boundary_samples_outer",))[:] = boundaries["outer"].sample_x
    group.createVariable("outer_boundary_sample_y", "f8", ("n_boundary_samples_outer",))[:] = boundaries["outer"].sample_y


def write_schrodinger(
    path: Path,
    mesh: MeshLevel,
    fields: dict[str, np.ndarray],
    boundaries: dict[str, SplineBoundary],
) -> None:
    inner_bval = sample_boundary_values(boundaries["inner"])
    outer_bval = sample_boundary_values(boundaries["outer"])
    with Dataset(path, "w", format="NETCDF4") as nc:
        nc.setncattr("transform", "schrodinger_direct_mms")
        create_dimension(nc, "n_points", mesh.n_points)
        create_dimension(nc, "n_points_inner", mesh.n_points_inner)
        create_dimension(nc, "n_points_boundary", mesh.n_points_boundary)
        create_dimension(nc, "n_points_ghost", mesh.n_points_ghost)
        create_dimension(nc, "n_boundary_samples_inner", inner_bval.size)
        create_dimension(nc, "n_boundary_samples_outer", outer_bval.size)

        nc.createVariable("co", "f8", ("n_points",))[:] = fields["co"]
        nc.createVariable("lambda", "f8", ("n_points_inner",))[:] = fields["lambda"]
        nc.createVariable("xi", "f8", ("n_points_inner",))[:] = fields["xi"]
        nc.createVariable("rhs", "f8", ("n_points",))[:] = fields["rhs"]
        nc.createVariable("sol", "f8", ("n_points",))[:] = fields["sol"]
        nc.createVariable("guess", "f8", ("n_points",))[:] = fields["guess"]
        nc.createVariable("potential", "f8", ("n_points",))[:] = fields["potential"]
        nc.createVariable("inner_boundary_sample_value", "f8", ("n_boundary_samples_inner",))[:] = inner_bval
        nc.createVariable("outer_boundary_sample_value", "f8", ("n_boundary_samples_outer",))[:] = outer_bval


def scatter_field(mesh: MeshLevel, values: np.ndarray) -> np.ndarray:
    field = np.full((mesh.ny_f, mesh.nx_f), np.nan, dtype=np.float64)
    ii = mesh.cart_i.astype(np.int64) - 1
    jj = mesh.cart_j.astype(np.int64) - 1
    field[jj, ii] = values
    return field


def plot_geometry(mesh: MeshLevel, boundaries: dict[str, SplineBoundary], output_dir: Path) -> None:
    x, y = point_coordinates(mesh.cart_i, mesh.cart_j, mesh.xmin, mesh.ymin, mesh.spacing_f)
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.scatter(x[mesh.pinfo == PINFO_INNER], y[mesh.pinfo == PINFO_INNER], s=1.0, c="#2c7fb8", label="inner")
    ax.scatter(x[mesh.pinfo == PINFO_BOUNDARY], y[mesh.pinfo == PINFO_BOUNDARY], s=5.0, c="#d95f0e", label="boundary")
    ax.scatter(x[mesh.pinfo == PINFO_GHOST], y[mesh.pinfo == PINFO_GHOST], s=2.0, c="#999999", alpha=0.4, label="ghost")
    for name, boundary in boundaries.items():
        ax.plot(boundary.dense_x, boundary.dense_y, lw=2.0, label=f"{name} spline")
        ax.scatter(boundary.sample_x[:-1], boundary.sample_y[:-1], s=12.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Spline-Defined Annulus and Cartesian Mesh")
    ax.legend(loc="best")
    fig.savefig(output_dir / "spline_annulus_geometry.png", dpi=180)
    plt.close(fig)


def plot_fields(mesh: MeshLevel, fields: dict[str, np.ndarray], output_dir: Path) -> None:
    extent = [
        mesh.xmin,
        mesh.xmin + (mesh.nx_f - 1) * mesh.spacing_f,
        mesh.ymin,
        mesh.ymin + (mesh.ny_f - 1) * mesh.spacing_f,
    ]
    specs = [
        ("sol", "Schrodinger Solution"),
        ("rhs", "Schrodinger Source"),
        ("potential", "Potential"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, (key, title) in zip(axes, specs):
        field = scatter_field(mesh, np.asarray(fields[key], dtype=np.float64))
        im = ax.imshow(field, origin="lower", extent=extent, aspect="equal", cmap="coolwarm")
        fig.colorbar(im, ax=ax, shrink=0.85, format="%.2e")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.savefig(output_dir / "spline_annulus_fields.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a spline-annulus Schrodinger MMS dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Directory where outputs are written.")
    parser.add_argument("--spacing", type=float, default=1.0e-3, help="Cartesian spacing for the generated mesh.")
    parser.add_argument("--boundary-samples", type=int, default=128, help="Number of sample points per closed boundary.")
    parser.add_argument("--plot", action="store_true", help="Save geometry and field plots.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params = SplineAnnulusParams(spacing_f=float(args.spacing), n_boundary_samples=int(args.boundary_samples))
    boundaries = build_boundaries(params)
    mesh = build_finest_mesh(params, boundaries)
    fields = compute_schrodinger_fields(mesh)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = output_dir / "multigrid.nc"
    data_path = output_dir / "schrodinger_data.nc"

    with Dataset(mesh_path, "w", format="NETCDF4") as nc:
        nc.setncattr("nlvls", np.int32(1))
        write_mesh_group(nc.createGroup("mesh_lvl_001"), mesh, boundaries)
    write_schrodinger(data_path, mesh, fields, boundaries)

    print(f"Wrote {mesh_path}")
    print(f"Wrote {data_path}")
    print(
        f"Level 1: n_points={mesh.n_points}, "
        f"n_inner={mesh.n_points_inner}, "
        f"n_boundary={mesh.n_points_boundary}, "
        f"n_ghost={mesh.n_points_ghost}"
    )

    if args.plot:
        plot_geometry(mesh, boundaries, output_dir)
        plot_fields(mesh, fields, output_dir)
        print(f"Wrote {output_dir / 'spline_annulus_geometry.png'}")
        print(f"Wrote {output_dir / 'spline_annulus_fields.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
