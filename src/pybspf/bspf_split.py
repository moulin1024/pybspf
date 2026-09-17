"""! @file bspf_split.py
@brief B-spline + Fourier (BSPF) KKT decomposition of sampled fields.

A sampled scalar field is written as a non-periodic B-spline part plus a
periodic residual.  The B-spline coefficients come from a quadrature-weighted
ridge fit augmented with endpoint-derivative-matching equality constraints,
assembled and solved through the package KKT helpers
(:func:`pybspf.kkt.assemble_kkt_matrix`, :class:`pybspf.kkt.KKTLUCache`) -- the
same constrained-fit machinery used by the BSPF1D integration routines.

The 2D directional split applies the 1D decomposition column-wise then row-wise
to produce a complete BSPF representation:

* ``A1`` -- B-spline (x) x B-spline (y) coefficients,
* ``A2`` -- B-spline (y) x Fourier (x) coefficients,
* ``A3`` -- B-spline (x) x Fourier (y) coefficients,
* ``f_per`` -- a doubly periodic residual.

Unlike :class:`pybspf.boundary.EndpointOps1D` (which provides exact
finite-difference endpoint stencils only), this module supports both the exact
finite-difference (``'fd'``) and least-squares local-polynomial
(``'local-poly-qr'``) endpoint-derivative estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Dict, Optional

import numpy as np

from .basis import BSplineValues
from .kkt import KKTLUCache
from .ops.integration import trapezoid_weights_1d

Bc = Optional[Dict[str, float]]


# ---------------------------------------------------------------------------
# Endpoint-derivative stencils
# ---------------------------------------------------------------------------
def fd_coeffs_local(xi, m: int) -> np.ndarray:
    """! @brief Exact finite-difference stencil for the ``m``-th derivative."""
    xi = np.asarray(xi, dtype=float).ravel()
    n = xi.size
    V = np.vstack([xi**k for k in range(n)])
    rhs = np.zeros(n)
    rhs[m] = factorial(m)
    return np.linalg.solve(V, rhs)


def local_poly_endpoint_coeffs(xi, m: int, degree: int) -> np.ndarray:
    """! @brief Least-squares local-polynomial endpoint stencil."""
    xi = np.asarray(xi, dtype=float).ravel()
    n = xi.size
    degree = min(max(degree, m), n - 1)
    V = np.vstack([xi**p for p in range(degree + 1)]).T
    rhs = np.zeros(degree + 1)
    rhs[m] = factorial(m)
    return rhs @ np.linalg.pinv(V)


def endpoint_derivative_matrix_for_grid(N, t, kmax, r, which_end, method, degree) -> np.ndarray:
    """! @brief Map grid samples to endpoint derivatives ``0..kmax``.

    @return Stencil matrix of shape ``(kmax+1, N)``.
    """
    t = np.asarray(t, dtype=float).ravel()
    dt = t[1] - t[0]
    D = np.zeros((kmax + 1, N))
    if str(which_end).lower() == "left":
        xi = np.arange(0, 2 * r + 1)
        idx = xi
        D[0, 0] = 1.0
    else:
        xi = np.arange(0, -(2 * r + 1), -1)
        idx = (N - 1) + xi
        D[0, N - 1] = 1.0
    for m in range(1, kmax + 1):
        if str(method).lower() == "local-poly-qr":
            a = local_poly_endpoint_coeffs(xi, m, degree) / dt**m
        else:
            a = fd_coeffs_local(xi, m) / dt**m
        D[m, idx] = a.ravel()
    return D


def endpoint_matrix_for_basis(bsp: BSplineValues, kmax: int, r: int):
    """! @brief Endpoint jets (value + derivatives ``0..kmax``) of each basis function.

    Uses analytic B-spline derivatives at the domain endpoints.  Returns
    ``(E0, E1)``, each of shape ``(kmax+1, nbasis)``.
    """
    nb = bsp.nbasis
    E0 = np.zeros((kmax + 1, nb))
    E1 = np.zeros((kmax + 1, nb))
    E0[0, :] = bsp.B0[:, 0]
    E1[0, :] = bsp.B0[:, -1]
    for m in range(1, kmax + 1):
        Bk = np.asarray(bsp.basis.BkT(m).T, dtype=float)
        E0[m, :] = Bk[:, 0]
        E1[m, :] = Bk[:, -1]
    return E0, E1


# ---------------------------------------------------------------------------
# 1D BSPF-KKT decomposition
# ---------------------------------------------------------------------------
@dataclass
class KKT1DCache:
    Bmat: np.ndarray          # (N, nb)
    cMap: np.ndarray          # (nb, N) fast no-constraint coefficient map
    solver: KKTLUCache
    rhsMap: np.ndarray        # (nb, N)  =  B * w
    Dpair: np.ndarray         # (2(kmax+1), N) endpoint-derivative stencils
    nE: int
    N: int
    nb: int
    kmax: int
    r: int
    lambda_kkt: float


def bspf_kkt_1d_decompose_precompute(
    bsp: BSplineValues, kmax: int, r: int, lambda_kkt: float, opt: dict
) -> KKT1DCache:
    """! @brief Precompute the cached 1D BSPF-KKT decomposition operator.

    Builds the constrained-fit KKT system ``[[2(Q+lam I), -E^T], [E, 0]]`` (via
    :func:`pybspf.kkt.assemble_kkt_matrix`) whose primal solution is the
    quadrature-weighted ridge B-spline fit subject to endpoint-derivative
    matching ``E c = D v``.
    """
    t = bsp.t
    N = t.size
    B0 = bsp.B0                       # (nb, N)
    nb = B0.shape[0]
    Bmat = B0.T.copy()               # (N, nb)
    w = trapezoid_weights_1d(t)      # (N,)
    BW = B0 * w                      # (nb, N)
    Q = BW @ B0.T                    # (nb, nb)

    E0, E1 = endpoint_matrix_for_basis(bsp, kmax, r)
    E = np.zeros((2 * (kmax + 1), nb))
    E[0::2, :] = E0
    E[1::2, :] = E1

    method = opt.get("endpointDerivativeMethod", "fd")
    radius = opt.get("endpointDerivativeRadius", r)
    degree = opt.get("endpointDerivativeDegree", kmax + 1)
    D0 = endpoint_derivative_matrix_for_grid(N, t, kmax, radius, "left", method, degree)
    D1 = endpoint_derivative_matrix_for_grid(N, t, kmax, radius, "right", method, degree)
    Dpair = np.zeros((2 * (kmax + 1), N))
    Dpair[0::2, :] = D0
    Dpair[1::2, :] = D1

    rhsMap = BW                       # (nb, N), top rhs block before the factor 2
    solver = KKTLUCache(Q, E)

    # No-constraint fast map: solve once against the full sample-to-rhs operator
    # [2 BW; Dpair] so each later application is a single matrix product.
    rhsFullMap = np.vstack([2.0 * rhsMap, Dpair])
    solMap = solver.solve(rhsFullMap, lambda_kkt, overwrite_b=False)
    cMap = solMap[:nb, :]

    return KKT1DCache(
        Bmat=Bmat, cMap=cMap, solver=solver, rhsMap=rhsMap, Dpair=Dpair,
        nE=E.shape[0], N=N, nb=nb, kmax=kmax, r=r, lambda_kkt=lambda_kkt,
    )


def _apply_endpoint_constraint_rhs(rhs: np.ndarray, pc: KKT1DCache, bc: Bc) -> np.ndarray:
    rhs = rhs.copy()
    nEndpoint = 2 * (pc.kmax + 1)
    first = rhs.size - nEndpoint

    def row(m: int, side: int) -> int:  # m, side 1-based
        return first + 2 * (m - 1) + (side - 1)

    if "value_left" in bc:
        rhs[row(1, 1)] = bc["value_left"]
    if "value_right" in bc:
        rhs[row(1, 2)] = bc["value_right"]
    if pc.kmax >= 1:
        if "d1_left" in bc:
            rhs[row(2, 1)] = bc["d1_left"]
        if "d1_right" in bc:
            rhs[row(2, 2)] = bc["d1_right"]
    return rhs


def bspf_kkt_1d_decompose_apply(v: np.ndarray, pc: KKT1DCache, bc: Bc = None):
    """! @brief Decompose a single sample vector into ``(f_nonper, c, f_per)``."""
    v = np.asarray(v, dtype=float).ravel()
    if v.size != pc.N:
        raise ValueError("Cached 1D BSPF-KKT split: vector length mismatch.")
    if not bc:
        c = pc.cMap @ v
    else:
        rhs = np.concatenate([2.0 * (pc.rhsMap @ v), pc.Dpair @ v])
        rhs = _apply_endpoint_constraint_rhs(rhs, pc, bc)
        sol = pc.solver.solve(rhs, pc.lambda_kkt, overwrite_b=False)
        c = sol[: pc.nb]
    f_nonper = pc.Bmat @ c
    f_per = v - f_nonper
    edge = 0.5 * (f_per[0] + f_per[-1])
    f_per[0] = edge
    f_per[-1] = edge
    return f_nonper, c, f_per


def directional_endpoint_bc(bc: Bc, direction: str) -> Bc:
    """! @brief Project a 2D boundary spec onto the x or y 1D split."""
    if not bc or not bc.get("use_kkt_endpoint_constraints", False):
        return None
    out: Dict[str, float] = {}
    if direction.lower() == "x":
        for k in ("value_left", "value_right", "d1_left", "d1_right"):
            if k in bc:
                out[k] = bc[k]
    else:
        mapping = {
            "value_bottom": "value_left",
            "value_top": "value_right",
            "d1_bottom": "d1_left",
            "d1_top": "d1_right",
        }
        for src, dst in mapping.items():
            if src in bc:
                out[dst] = bc[src]
    return out or None


# ---------------------------------------------------------------------------
# 2D directional split
# ---------------------------------------------------------------------------
@dataclass
class SplitResult:
    f_per: np.ndarray
    A1: np.ndarray            # (nbx, nby) real
    A2: np.ndarray            # (nby, Nx0) complex
    A3: np.ndarray            # (nbx, Ny0) complex
    reconstruction_linf: float = float("nan")


def split2d_kkt_directional(f: np.ndarray, cache, bc: Bc = None) -> SplitResult:
    """! @brief 2D directional BSPF-KKT split of ``f`` (shape ``(Ny, Nx)``).

    ``cache`` must expose ``Nx0``, ``Ny0``, ``nbasis``, ``Bvals_x``, ``Bvals_y``
    and the cached 1D operators ``splitX`` / ``splitY``.
    """
    f = np.asarray(f, dtype=float)
    Ny, Nx = f.shape
    Nx0, Ny0 = cache.Nx0, cache.Ny0
    nbx = nby = cache.nbasis
    splitX, splitY = cache.splitX, cache.splitY
    bcY = directional_endpoint_bc(bc, "y")
    bcX = directional_endpoint_bc(bc, "x")

    if bcY is None:
        Cy = splitY.cMap @ f
        Ry = f - splitY.Bmat @ Cy
        edge = 0.5 * (Ry[0, :] + Ry[-1, :])
        Ry[0, :] = edge
        Ry[-1, :] = edge
    else:
        Cy = np.zeros((nby, Nx))
        Ry = np.zeros((Ny, Nx))
        for ix in range(Nx):
            _fy, cy, fy_per = bspf_kkt_1d_decompose_apply(f[:, ix], splitY, bcY)
            Cy[:, ix] = cy
            Ry[:, ix] = fy_per

    A1 = np.zeros((nbx, nby))
    Cxy = splitX.cMap @ Cy.T
    A1 += Cxy
    cx_per = Cy.T - splitX.Bmat @ Cxy
    edge = 0.5 * (cx_per[0, :] + cx_per[-1, :])
    cx_per[0, :] = edge
    cx_per[-1, :] = edge
    cx_per[-1, :] = cx_per[0, :]
    A2 = np.fft.fft(cx_per[:Nx0, :], axis=0).T / Nx0

    if bcX is None:
        Cx = splitX.cMap @ Ry.T
        Rxy = Ry - (splitX.Bmat @ Cx).T
        edge = 0.5 * (Rxy[:, 0] + Rxy[:, -1])
        Rxy[:, 0] = edge
        Rxy[:, -1] = edge
    else:
        Cx = np.zeros((nbx, Ny))
        Rxy = np.zeros((Ny, Nx))
        for iy in range(Ny):
            _fxnp, cx, fx_per = bspf_kkt_1d_decompose_apply(Ry[iy, :], splitX, bcX)
            Cx[:, iy] = cx
            Rxy[iy, :] = fx_per

    Cyx = splitY.cMap @ Cx.T
    A1 += Cyx.T
    cy_per = Cx.T - splitY.Bmat @ Cyx
    edge = 0.5 * (cy_per[0, :] + cy_per[-1, :])
    cy_per[0, :] = edge
    cy_per[-1, :] = edge
    cy_per[-1, :] = cy_per[0, :]
    A3 = np.fft.fft(cy_per[:Ny0, :], axis=0).T / Ny0

    f_per = Rxy.copy()
    f_per[Ny - 1, :Nx0] = f_per[0, :Nx0]
    f_per[:Ny0, Nx - 1] = f_per[:Ny0, 0]
    f_per[Ny - 1, Nx - 1] = f_per[0, 0]

    res = SplitResult(f_per=f_per, A1=A1, A2=A2, A3=A3)
    try:
        recon = reconstruct_nonper_from_A123(A1, A2, A3, cache.Bvals_x, cache.Bvals_y)
        res.reconstruction_linf = float(np.max(np.abs(recon + f_per - f)))
    except Exception:
        res.reconstruction_linf = float("nan")
    return res


def reconstruct_nonper_from_A123(A1, A2, A3, Bx, By) -> np.ndarray:
    """! @brief Reconstruct the non-periodic part from BSPF coefficients (debug check)."""
    nby, Nx0 = A2.shape
    nbx, Ny0 = A3.shape
    Ny = By.shape[1]
    Nx = Bx.shape[1]

    fBB = By.T @ A1.T @ Bx

    fA2 = np.zeros((Ny, Nx))
    for jy in range(nby):
        tmpx0 = np.real(np.fft.ifft(A2[jy, :] * Nx0))
        tmpx = np.zeros(Nx)
        tmpx[:Nx0] = tmpx0
        tmpx[-1] = tmpx0[0]
        fA2 += np.outer(By[jy, :], tmpx)

    fA3 = np.zeros((Ny, Nx))
    for jx in range(nbx):
        tmpy0 = np.real(np.fft.ifft(A3[jx, :] * Ny0))
        tmpy = np.zeros(Ny)
        tmpy[:Ny0] = tmpy0
        tmpy[-1] = tmpy0[0]
        fA3 += np.outer(tmpy, Bx[jx, :])

    return fBB + fA2 + fA3


def embed_periodic_full(A0: np.ndarray) -> np.ndarray:
    """! @brief Copy first row/column to last (periodic-block completion)."""
    Ny0, Nx0 = A0.shape
    out = np.zeros((Ny0 + 1, Nx0 + 1), dtype=A0.dtype)
    out[:Ny0, :Nx0] = A0
    out[-1, :Nx0] = A0[0, :]
    out[:Ny0, -1] = A0[:, 0]
    out[-1, -1] = A0[0, 0]
    return out


__all__ = [
    "BSplineValues",
    "KKT1DCache",
    "SplitResult",
    "bspf_kkt_1d_decompose_apply",
    "bspf_kkt_1d_decompose_precompute",
    "directional_endpoint_bc",
    "embed_periodic_full",
    "endpoint_derivative_matrix_for_grid",
    "endpoint_matrix_for_basis",
    "fd_coeffs_local",
    "local_poly_endpoint_coeffs",
    "reconstruct_nonper_from_A123",
    "split2d_kkt_directional",
]
