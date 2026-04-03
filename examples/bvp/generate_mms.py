#!/usr/bin/env python3
"""Generate a minimal dataset containing only the finest mesh.

This script is intentionally standalone and hardcodes the circular MMS mesh
parameters so it can be used in lightweight AMGX workflows without depending on
the precomputed dataset.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from netCDF4 import Dataset
except ModuleNotFoundError as exc:
    missing_module = exc.name or "dependency"
    raise SystemExit(
        "generate_finest_mesh.py requires Python packages "
        f"`numpy`, `netCDF4`, and `matplotlib` (missing: {missing_module})."
    ) from exc


BOXFAC = 0.4 / 0.3685
INV_SQRT2 = math.sqrt(2.0) / 2.0

PINFO_INNER = 1
PINFO_BOUNDARY = 2
PINFO_GHOST = 3

DISTRICT_CORE = 813
DISTRICT_CLOSED = 814
DISTRICT_WALL = 817
DISTRICT_DOME = 818
DISTRICT_OUT = 819

BND_TYPE_DIRICHLET_ZERO = -3

TWO_PI = 2.0 * math.pi

def fortran_nint(value: float) -> int:
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return -int(math.floor(-value + 0.5))


@dataclass
class CircularParams:
    spacing_f: float
    size_neighbor: int
    size_ghost_layer: int
    rhomin: float
    rhomax: float
    bnd_type_core: int = BND_TYPE_DIRICHLET_ZERO
    phi: float = 0.0

    @property
    def xmin(self) -> float:
        return -BOXFAC * self.rhomax

    @property
    def xmax(self) -> float:
        return BOXFAC * self.rhomax

    @property
    def ymin(self) -> float:
        return -BOXFAC * self.rhomax

    @property
    def ymax(self) -> float:
        return BOXFAC * self.rhomax

    @property
    def nx_f(self) -> int:
        return fortran_nint((self.xmax - self.xmin) / self.spacing_f)

    @property
    def ny_f(self) -> int:
        return fortran_nint((self.ymax - self.ymin) / self.spacing_f)


@dataclass
class MeshLevel:
    lvl: int
    lvst: int
    spacing_f: float
    spacing_c: float
    size_neighbor: int
    size_ghost_layer: int
    cart_i: np.ndarray
    cart_j: np.ndarray
    index_neighbor: np.ndarray
    inner_indices: np.ndarray
    boundary_indices: np.ndarray
    ghost_indices: np.ndarray
    pinfo: np.ndarray
    district: np.ndarray
    redblack_indices: np.ndarray
    n_points_red: int
    n_points_black: int

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


def default_params() -> CircularParams:
    """Return the hardcoded finest-mesh parameters used by the MMS tests."""
    return CircularParams(
        spacing_f=1.0e-3,
        size_neighbor=2,
        size_ghost_layer=2,
        rhomin=0.2,
        rhomax=0.4,
    )


def point_coordinates(cart_i: np.ndarray, cart_j: np.ndarray, params: CircularParams) -> tuple[np.ndarray, np.ndarray]:
    x = params.xmin + (cart_i.astype(np.float64) - 1.0) * params.spacing_f
    y = params.ymin + (cart_j.astype(np.float64) - 1.0) * params.spacing_f
    return x, y


def classify_outside_radius_array(rho: np.ndarray, params: CircularParams) -> np.ndarray:
    midpoint = 0.5 * (params.rhomin + params.rhomax)
    out = np.empty(rho.shape, dtype=np.int32)
    out[rho < params.rhomin] = DISTRICT_CORE
    out[rho > params.rhomax] = DISTRICT_WALL
    inside = (rho >= params.rhomin) & (rho <= params.rhomax)
    out[inside & (rho <= midpoint)] = DISTRICT_CORE
    out[inside & (rho > midpoint)] = DISTRICT_WALL
    return out


def theta(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.mod(np.arctan2(y, x), TWO_PI)


def build_cartesian_points(params: CircularParams, lvst: int) -> tuple[np.ndarray, np.ndarray]:
    spacing_c = params.spacing_f * lvst
    i = np.arange(1, params.nx_f + 1, lvst, dtype=np.int32)
    j = np.arange(1, params.ny_f + 1, lvst, dtype=np.int32)
    ii, jj = np.meshgrid(i, j, indexing="xy")

    x = params.xmin + (ii.astype(np.float64) - 1.0) * params.spacing_f
    y = params.ymin + (jj.astype(np.float64) - 1.0) * params.spacing_f

    shifts = np.array([-0.5, 0.0, 0.5], dtype=np.float64) * spacing_c
    x_s = x[..., None, None] + shifts[None, None, None, :]
    y_s = y[..., None, None] + shifts[None, None, :, None]

    rho_s = np.hypot(x_s, y_s)
    include = np.any((rho_s >= params.rhomin) & (rho_s <= params.rhomax), axis=(2, 3))
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
        [
            inner_i[:, None] + offsets[None, :, 0],
            inner_j[:, None] + offsets[None, :, 1],
        ],
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


def boundary_normals(index_neighbor: np.ndarray, pinfo: np.ndarray, size_neighbor: int, boundary_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local = index_neighbor[
        boundary_indices - 1,
        size_neighbor - 1:size_neighbor + 2,
        size_neighbor - 1:size_neighbor + 2,
    ]
    disc = np.zeros_like(local, dtype=np.int32)
    mask = local > 0
    pvals = np.zeros(np.count_nonzero(mask), dtype=np.int32)
    pinfo_vals = pinfo[local[mask] - 1]
    pvals[pinfo_vals == PINFO_INNER] = 2
    pvals[pinfo_vals == PINFO_BOUNDARY] = 1
    disc[mask] = pvals
    nbx = disc[:, 1, 2] - disc[:, 1, 0]
    nby = disc[:, 2, 1] - disc[:, 0, 1]
    return nbx.astype(np.int32), nby.astype(np.int32)


def build_district(
    cart_i: np.ndarray,
    cart_j: np.ndarray,
    index_neighbor: np.ndarray,
    inner_indices: np.ndarray,
    boundary_indices: np.ndarray,
    ghost_indices: np.ndarray,
    pinfo: np.ndarray,
    params: CircularParams,
    spacing_c: float,
    size_neighbor: int,
) -> np.ndarray:
    district = np.zeros(cart_i.size, dtype=np.int32)
    x, y = point_coordinates(cart_i, cart_j, params)
    rho = np.hypot(x, y)
    district[inner_indices - 1] = DISTRICT_CLOSED

    if ghost_indices.size > 0:
        district[ghost_indices - 1] = classify_outside_radius_array(rho[ghost_indices - 1], params)

    if boundary_indices.size > 0:
        bidx = boundary_indices - 1
        nbx, nby = boundary_normals(index_neighbor, pinfo, size_neighbor, boundary_indices)
        zero_mask = (nbx == 0) & (nby == 0)
        if np.any(zero_mask):
            district[bidx[zero_mask]] = classify_outside_radius_array(rho[bidx[zero_mask]], params)

        nz_mask = ~zero_mask
        if np.any(nz_mask):
            nbn = np.hypot(nbx[nz_mask], nby[nz_mask])
            x_shift = x[bidx[nz_mask]] - nbx[nz_mask] / nbn * spacing_c * INV_SQRT2
            y_shift = y[bidx[nz_mask]] - nby[nz_mask] / nbn * spacing_c * INV_SQRT2
            rho_shift = np.hypot(x_shift, y_shift)
            district[bidx[nz_mask]] = classify_outside_radius_array(rho_shift, params)

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


def build_redblack_placeholder(
    inner_indices: np.ndarray,
    boundary_indices: np.ndarray,
    ghost_indices: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Return a schema-compatible placeholder for readers that expect red-black metadata. 
    """
    redblack = np.concatenate([inner_indices, boundary_indices, ghost_indices]).astype(np.int32)
    return redblack, int(inner_indices.size), 0


def build_finest_mesh(params: CircularParams) -> MeshLevel:
    lvl = 1
    lvst = 1
    spacing_c = params.spacing_f

    cart_i, cart_j = build_cartesian_points(params, lvst)
    index_neighbor = build_connectivity(cart_i, cart_j, lvst, params.size_neighbor)
    inner_indices, boundary_indices = build_boundary_and_inner(index_neighbor, params.size_neighbor)

    cart_i, cart_j, ghost_indices = build_ghost_layer(
        cart_i, cart_j, inner_indices, lvst, params.size_ghost_layer)
    index_neighbor = build_connectivity(cart_i, cart_j, lvst, params.size_neighbor)
    pinfo = build_pinfo(cart_i.size, inner_indices, boundary_indices, ghost_indices)
    district = build_district(
        cart_i,
        cart_j,
        index_neighbor,
        inner_indices,
        boundary_indices,
        ghost_indices,
        pinfo,
        params,
        spacing_c,
        params.size_neighbor,
    )
    boundary_indices, ghost_indices, pinfo = apply_patch(
        boundary_indices, ghost_indices, index_neighbor, pinfo, params.size_neighbor)
    redblack_indices, n_points_red, n_points_black = build_redblack_placeholder(
        inner_indices, boundary_indices, ghost_indices)

    return MeshLevel(
        lvl=lvl,
        lvst=lvst,
        spacing_f=params.spacing_f,
        spacing_c=spacing_c,
        size_neighbor=params.size_neighbor,
        size_ghost_layer=params.size_ghost_layer,
        cart_i=cart_i,
        cart_j=cart_j,
        index_neighbor=index_neighbor,
        inner_indices=inner_indices,
        boundary_indices=boundary_indices,
        ghost_indices=ghost_indices,
        pinfo=pinfo,
        district=district,
        redblack_indices=redblack_indices,
        n_points_red=n_points_red,
        n_points_black=n_points_black,
    )


def create_dimension(group: Dataset, name: str, size: int) -> None:
    group.createDimension(name, None if size == 0 else size)


def write_mesh_group(group: Dataset, mesh: MeshLevel, params: CircularParams) -> None:
    group.setncattr("phi", np.float64(params.phi))
    group.setncattr("lvl", np.int32(mesh.lvl))
    group.setncattr("lvst", np.int32(mesh.lvst))
    group.setncattr("spacing_f", np.float64(mesh.spacing_f))
    group.setncattr("spacing_c", np.float64(mesh.spacing_c))
    group.setncattr("xmin", np.float64(params.xmin))
    group.setncattr("ymin", np.float64(params.ymin))
    group.setncattr("nx_f", np.int32(params.nx_f))
    group.setncattr("ny_f", np.int32(params.ny_f))
    group.setncattr("size_ghost_layer", np.int32(mesh.size_ghost_layer))
    group.setncattr("yperiodic", np.int32(0))
    group.setncattr("extend_beyond_wall", np.int32(0))
    group.setncattr("n_points_red", np.int32(mesh.n_points_red))
    group.setncattr("n_points_black", np.int32(mesh.n_points_black))

    sis = 2 * mesh.size_neighbor + 1
    create_dimension(group, "n_points", mesh.n_points)
    create_dimension(group, "n_points_inner", mesh.n_points_inner)
    create_dimension(group, "n_points_boundary", mesh.n_points_boundary)
    create_dimension(group, "n_points_ghost", mesh.n_points_ghost)
    create_dimension(group, "size_neighbor", sis)

    group.createVariable("cart_i", "i4", ("n_points",))[:] = mesh.cart_i
    group.createVariable("cart_j", "i4", ("n_points",))[:] = mesh.cart_j
    group.createVariable("pinfo", "i4", ("n_points",))[:] = mesh.pinfo
    group.createVariable("district", "i4", ("n_points",))[:] = mesh.district
    group.createVariable("inner_indices", "i4", ("n_points_inner",))[:] = mesh.inner_indices
    group.createVariable("boundary_indices", "i4", ("n_points_boundary",))[:] = mesh.boundary_indices
    group.createVariable("ghost_indices", "i4", ("n_points_ghost",))[:] = mesh.ghost_indices
    group.createVariable("index_neighbor", "i4", ("n_points", "size_neighbor", "size_neighbor"))[:] = mesh.index_neighbor
    group.createVariable("redblack_indices", "i4", ("n_points",))[:] = mesh.redblack_indices


def compute_fields(mesh: MeshLevel, params: CircularParams) -> dict[str, np.ndarray | int]:
    """Evaluate the level-1 manufactured fields needed by helmholtz_data.nc."""
    x, y = point_coordinates(mesh.cart_i, mesh.cart_j, params)
    rho = np.hypot(x, y)
    theta_vals = theta(x, y)
    rhon = (rho - params.rhomin) / (params.rhomax - params.rhomin)

    sol = np.cos(1.5 * math.pi * rhon) * np.sin(4.0 * theta_vals) + 1.3
    co = 1.1 + np.cos(0.5 * math.pi * rhon) * np.cos(3.0 * theta_vals)

    inner = mesh.inner_indices - 1
    rho_inner = rho[inner]
    theta_inner = theta_vals[inner]

    lambda_inner = rho_inner * np.sin(theta_inner)
    xi_inner = np.sqrt(rho_inner) * (2.0 + np.cos(theta_inner))
    lambda_full = np.zeros(mesh.n_points, dtype=np.float64)
    lambda_full[inner] = lambda_inner

    delta_r = params.rhomax - params.rhomin
    r = rho_inner
    t = theta_inner
    rmin = params.rhomin

    rhs_inner = (
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

    rhs = sol.copy()
    rhs[inner] = rhs_inner
    guess = np.zeros(mesh.n_points, dtype=np.float64)

    return {
        "co": co.astype(np.float64),
        "lambda": lambda_inner.astype(np.float64),
        "lambda_full": lambda_full,
        "xi": xi_inner.astype(np.float64),
        "rhs": rhs.astype(np.float64),
        "guess": guess,
        "sol": sol.astype(np.float64),
        "bnd_type_core": params.bnd_type_core,
        "bnd_type_wall": BND_TYPE_DIRICHLET_ZERO,
        "bnd_type_dome": BND_TYPE_DIRICHLET_ZERO,
        "bnd_type_out": BND_TYPE_DIRICHLET_ZERO,
    }


def compute_schrodinger_fields(
    mesh: MeshLevel,
    params: CircularParams,
    original_fields: dict[str, np.ndarray | int],
) -> dict[str, np.ndarray | int]:
    """Transform the original elliptic MMS into stationary Schrödinger form.

    Starting from

        lambda * u - xi * div(co * grad(u)) = rhs,

    write it as

        -1/W div(P grad(u)) + Q u = F

    with P = co, W = 1/xi, Q = lambda, F = rhs, and apply the Liouville
    transform psi = sqrt(P) * u. The transformed equation is

        -Delta psi + V_eff * psi = S,

    where

        V_eff = U_P + lambda / (xi * co),
        U_P   = Delta sqrt(P) / sqrt(P),
        S     = rhs / (xi * sqrt(P)).

    To keep the downstream NetCDF schema aligned with helmholtz_data.nc, we
    store this transformed problem again in the generic elliptic form with
    co=1 and xi=1:

        lambda_s * psi - div(grad(psi)) = rhs_s.

    Boundary/ghost entries follow the same convention as the original file:
    rhs stores Dirichlet data there rather than a forcing term.
    """
    x, y = point_coordinates(mesh.cart_i, mesh.cart_j, params)
    rho = np.hypot(x, y)
    theta_vals = theta(x, y)

    co = np.asarray(original_fields["co"], dtype=np.float64)
    sol = np.asarray(original_fields["sol"], dtype=np.float64)
    rhs = np.asarray(original_fields["rhs"], dtype=np.float64)
    lambda_inner = np.asarray(original_fields["lambda"], dtype=np.float64)
    xi_inner = np.asarray(original_fields["xi"], dtype=np.float64)

    delta_r = params.rhomax - params.rhomin
    alpha = 0.5 * math.pi / delta_r
    phase = alpha * (rho - params.rhomin)
    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)
    cos3 = np.cos(3.0 * theta_vals)
    sin3 = np.sin(3.0 * theta_vals)
    rho_safe = np.where(rho > 1.0e-12, rho, 1.0e-12)

    # P(r, theta) = co(r, theta)
    p_r = -alpha * sin_phase * cos3
    p_rr = -(alpha**2) * cos_phase * cos3
    p_theta = -3.0 * cos_phase * sin3
    p_thetatheta = -9.0 * cos_phase * cos3

    lap_p = p_rr + p_r / rho_safe + p_thetatheta / (rho_safe**2)
    grad_p_sq = p_r**2 + (p_theta**2) / (rho_safe**2)
    u_geom = 0.5 * lap_p / co - 0.25 * grad_p_sq / (co**2)

    psi = np.sqrt(co) * sol
    inner = mesh.inner_indices - 1
    reaction_inner = lambda_inner / (xi_inner * co[inner])
    potential_inner = u_geom[inner] + reaction_inner
    potential = u_geom.copy()
    potential[inner] = potential_inner

    rhs_s = psi.copy()
    rhs_s[inner] = rhs[inner] / (xi_inner * np.sqrt(co[inner]))
    guess = np.zeros(mesh.n_points, dtype=np.float64)

    return {
        "co": np.ones(mesh.n_points, dtype=np.float64),
        "lambda": potential_inner.astype(np.float64),
        "xi": np.ones(mesh.n_points_inner, dtype=np.float64),
        "rhs": rhs_s.astype(np.float64),
        "sol": psi.astype(np.float64),
        "guess": guess,
        "potential": potential.astype(np.float64),
        "geometry_potential": u_geom.astype(np.float64),
        "reaction": reaction_inner.astype(np.float64),
        "transform_weight": np.sqrt(co).astype(np.float64),
        "bnd_type_core": original_fields["bnd_type_core"],
        "bnd_type_wall": original_fields["bnd_type_wall"],
        "bnd_type_dome": original_fields["bnd_type_dome"],
        "bnd_type_out": original_fields["bnd_type_out"],
    }


def write_helmholtz(path: Path, mesh: MeshLevel, fields: dict[str, np.ndarray | int]) -> None:
    with Dataset(path, "w", format="NETCDF4") as nc:
        nc.setncattr("bnd_type_core", np.int32(fields["bnd_type_core"]))
        nc.setncattr("bnd_type_wall", np.int32(fields["bnd_type_wall"]))
        nc.setncattr("bnd_type_dome", np.int32(fields["bnd_type_dome"]))
        nc.setncattr("bnd_type_out", np.int32(fields["bnd_type_out"]))

        create_dimension(nc, "n_points", mesh.n_points)
        create_dimension(nc, "n_points_inner", mesh.n_points_inner)
        create_dimension(nc, "n_points_boundary", mesh.n_points_boundary)
        create_dimension(nc, "n_points_ghost", mesh.n_points_ghost)

        nc.createVariable("co", "f8", ("n_points",))[:] = fields["co"]
        nc.createVariable("lambda", "f8", ("n_points_inner",))[:] = fields["lambda"]
        nc.createVariable("xi", "f8", ("n_points_inner",))[:] = fields["xi"]
        nc.createVariable("rhs", "f8", ("n_points",))[:] = fields["rhs"]
        nc.createVariable("sol", "f8", ("n_points",))[:] = fields["sol"]
        nc.createVariable("guess", "f8", ("n_points",))[:] = fields["guess"]


def write_schrodinger(path: Path, mesh: MeshLevel, fields: dict[str, np.ndarray | int]) -> None:
    with Dataset(path, "w", format="NETCDF4") as nc:
        nc.setncattr("bnd_type_core", np.int32(fields["bnd_type_core"]))
        nc.setncattr("bnd_type_wall", np.int32(fields["bnd_type_wall"]))
        nc.setncattr("bnd_type_dome", np.int32(fields["bnd_type_dome"]))
        nc.setncattr("bnd_type_out", np.int32(fields["bnd_type_out"]))
        nc.setncattr("transform", "liouville_schrodinger")

        create_dimension(nc, "n_points", mesh.n_points)
        create_dimension(nc, "n_points_inner", mesh.n_points_inner)
        create_dimension(nc, "n_points_boundary", mesh.n_points_boundary)
        create_dimension(nc, "n_points_ghost", mesh.n_points_ghost)

        nc.createVariable("co", "f8", ("n_points",))[:] = fields["co"]
        nc.createVariable("lambda", "f8", ("n_points_inner",))[:] = fields["lambda"]
        nc.createVariable("xi", "f8", ("n_points_inner",))[:] = fields["xi"]
        nc.createVariable("rhs", "f8", ("n_points",))[:] = fields["rhs"]
        nc.createVariable("sol", "f8", ("n_points",))[:] = fields["sol"]
        nc.createVariable("guess", "f8", ("n_points",))[:] = fields["guess"]
        nc.createVariable("potential", "f8", ("n_points",))[:] = fields["potential"]
        nc.createVariable("geometry_potential", "f8", ("n_points",))[:] = fields["geometry_potential"]
        nc.createVariable("reaction", "f8", ("n_points_inner",))[:] = fields["reaction"]
        nc.createVariable("transform_weight", "f8", ("n_points",))[:] = fields["transform_weight"]


def _scatter_to_full_grid(mesh: MeshLevel, params: CircularParams, values: np.ndarray) -> np.ndarray:
    field = np.full((params.ny_f, params.nx_f), np.nan, dtype=np.float64)
    ii = mesh.cart_i.astype(np.int64) - 1
    jj = mesh.cart_j.astype(np.int64) - 1
    field[jj, ii] = values
    return field


def plot_fields(
    mesh: MeshLevel,
    params: CircularParams,
    fields: dict[str, np.ndarray | int],
    *,
    form: str,
    output_dir: Path,
) -> Path:
    if form == "helmholtz":
        specs = [
            ("sol", "Manufactured Solution"),
            ("rhs", "Right-Hand Side"),
            ("co", "Diffusion Coefficient (co)"),
            ("lambda_full", "Reaction Coefficient (lambda)"),
        ]
        filename = output_dir / "helmholtz_fields.png"
        title = "Helmholtz-Form MMS Fields"
    else:
        specs = [
            ("sol", "Schrodinger Solution (psi)"),
            ("rhs", "Schrodinger Source"),
            ("potential", "Effective Potential"),
            ("transform_weight", "Liouville Weight sqrt(co)"),
        ]
        filename = output_dir / "schrodinger_fields.png"
        title = "Schrodinger-Form MMS Fields"

    extent = [params.xmin, params.xmax, params.ymin, params.ymax]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, (key, label) in zip(axes.flat, specs):
        field = _scatter_to_full_grid(mesh, params, np.asarray(fields[key], dtype=np.float64))
        im = ax.imshow(field, origin="lower", extent=extent, aspect="equal", cmap="coolwarm")
        fig.colorbar(im, ax=ax, shrink=0.85, format="%.2e")
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    for ax in list(axes.flat)[len(specs):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(filename, dpi=180)
    plt.close(fig)
    return filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a minimal multigrid.nc containing only the finest mesh.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where multigrid.nc will be written.",
    )
    parser.add_argument(
        "--output-form",
        choices=("helmholtz", "schrodinger", "both"),
        default="both",
        help="Which field dataset to emit.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save field plots for the selected output form(s) into the output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params = default_params()
    mesh = build_finest_mesh(params)
    fields = compute_fields(mesh, params)
    schrodinger_fields = compute_schrodinger_fields(mesh, params, fields)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    multigrid_path = output_dir / "multigrid.nc"
    helmholtz_path = output_dir / "helmholtz_data.nc"
    schrodinger_path = output_dir / "schrodinger_data.nc"

    with Dataset(multigrid_path, "w", format="NETCDF4") as nc:
        nc.setncattr("nlvls", np.int32(1))
        write_mesh_group(nc.createGroup("mesh_lvl_001"), mesh, params)
    if args.output_form in ("helmholtz", "both"):
        write_helmholtz(helmholtz_path, mesh, fields)
    if args.output_form in ("schrodinger", "both"):
        write_schrodinger(schrodinger_path, mesh, schrodinger_fields)

    print(f"Wrote {multigrid_path}")
    if args.output_form in ("helmholtz", "both"):
        print(f"Wrote {helmholtz_path}")
    if args.output_form in ("schrodinger", "both"):
        print(f"Wrote {schrodinger_path}")
    if args.plot:
        if args.output_form in ("helmholtz", "both"):
            helmholtz_plot = plot_fields(mesh, params, fields, form="helmholtz", output_dir=output_dir)
            print(f"Wrote {helmholtz_plot}")
        if args.output_form in ("schrodinger", "both"):
            schrodinger_plot = plot_fields(
                mesh,
                params,
                schrodinger_fields,
                form="schrodinger",
                output_dir=output_dir,
            )
            print(f"Wrote {schrodinger_plot}")
    print(
        f"Level 1: n_points={mesh.n_points}, "
        f"n_inner={mesh.n_points_inner}, "
        f"n_boundary={mesh.n_points_boundary}, "
        f"n_ghost={mesh.n_points_ghost}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
