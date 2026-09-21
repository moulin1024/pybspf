"""BSPF-KKT Neumann Poisson solver (Leray projection engine).

This is the projection core used by the time integrator.  It solves

    lap U = f ,    dU/dn = q   on the four edges

by splitting ``f`` into a complete BSPF representation, responding to each
coefficient block with a precomputed particular-solution kernel, adding the FFT
periodic-residual particular solution, and closing the Neumann data with a
Trefftz harmonic-polynomial correction.

Ports the second half of ``ns_solver.m``:
``default_bspf_kkt_neumann_params``, ``build_1d_modes_symmetric_two_sided_nozm``
and its helpers, ``build_U_basis_new_inline``,
``build_newbasis_exp_kernel_1d_inline``, ``build_quadratic_particular``,
``laplace_rect_solver_trefftz_neumann_*``, ``boundary_flux_rect_cached``,
``bspf_kkt_poisson_neumann_precompute`` and ``..._apply_cached``.

All ``reshape``/``(:)`` operations use Fortran (column-major) order to match
MATLAB semantics exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import null_space

from ..basis import make_bspline_basis_values
from ..bspf_split import (
    bspf_kkt_1d_decompose_precompute,
    embed_periodic_full,
    split2d_kkt_directional,
)
from ..ops.integration import high_order_quad_weights_vector
from ..spectral import (
    periodic_gradient_2d_complex,
    periodic_gradient_2d_real,
    poisson_fft_periodic_2d_zero_mean,
    poisson_fft_periodic_2d_zero_mean_complex,
    spectral_poisson_2d_uniform_precompute,
    spectral_poisson_2d_uniform_with_grad_cached,
)

EPS = np.finfo(float).eps


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
def default_bspf_kkt_neumann_params(Nx: int = 100, Ny: int | None = None) -> dict:
    if Ny is None:
        Ny = Nx
    return dict(
        verbose=False,
        doWaitbar=False,
        optSplit=dict(kmax=5, r=3, lambda_kkt=1e-10),
        degB=6,
        nbasis=18,
        s_newbasis=5,
        delta_newbasis=0.60,
        reportSeam=False,
        optLapN=dict(
            K=min(44, Nx // 2),
            lambda_=1e-14,
            remove_flux_mean=True,
            zero_mean=True,
            flux_correction_tol=1e-9,
        ),
        solution_mean=0,
    )


# ---------------------------------------------------------------------------
# 1D symmetric two-sided boundary modes (B-spline jet / null reflection basis)
# ---------------------------------------------------------------------------
def _build_boundary_derivative_matrix(sp1D, idx, xbd, s) -> np.ndarray:
    n = len(idx)
    D = np.zeros((s + 1, n))
    for r in range(s + 1):
        for a in range(n):
            D[r, a] = sp1D[idx[a]].derivative(r)(xbd)
    return D


def _jet_and_null_basis_square(D):
    nB = D.shape[1]
    r = D.shape[0]
    row_scale = np.max(np.abs(D), axis=1)
    row_scale[row_scale == 0] = 1.0
    Dscaled = D / row_scale[:, None]
    Cjet = np.zeros((nB, r))
    for k in range(r):
        rhs = np.zeros(r)
        rhs[k] = 1.0 / row_scale[k]
        Cjet[:, k], *_ = np.linalg.lstsq(Dscaled, rhs, rcond=None)
    Nnull = null_space(D)
    if Nnull.size == 0:
        Nnull = np.zeros((nB, 0))
    return Cjet, Nnull


def _reflect_left_mode_to_right(cL, idxL, idxR, scale):
    localL = cL[idxL]
    localR = scale * localL[::-1]
    cR = np.zeros_like(cL)
    cR[idxR] = localR
    return cR


def _assemble_square_basis_from_left_reflection(nbasis, idxL, idxI, idxR, CjetL, NnullL, s):
    nL, nR = len(idxL), len(idxR)
    if nL != nR:
        raise ValueError("Left/right boundary DOF counts differ.")
    T = np.zeros((nbasis, nbasis))
    meta: List[dict] = []
    col = 0

    leftJetCols = np.zeros(s + 1, dtype=int)
    for r in range(s + 1):
        c = np.zeros(nbasis)
        c[idxL] = CjetL[:, r]
        T[:, col] = c
        meta.append(dict(type="jet", side="L", r=r))
        leftJetCols[r] = col
        col += 1

    nNullL = NnullL.shape[1]
    leftNullCols = np.zeros(nNullL, dtype=int)
    for k in range(nNullL):
        c = np.zeros(nbasis)
        c[idxL] = NnullL[:, k]
        T[:, col] = c
        meta.append(dict(type="null", side="L", r=-1))
        leftNullCols[k] = col
        col += 1

    for k in idxI:
        T[k, col] = 1.0
        meta.append(dict(type="int", side="I", r=-1))
        col += 1

    for r in range(s + 1):
        cL = T[:, leftJetCols[r]]
        T[:, col] = _reflect_left_mode_to_right(cL, idxL, idxR, (-1.0) ** r)
        meta.append(dict(type="jet", side="R", r=r))
        col += 1

    for k in range(nNullL):
        cL = T[:, leftNullCols[k]]
        T[:, col] = _reflect_left_mode_to_right(cL, idxL, idxR, 1.0)
        meta.append(dict(type="null", side="R", r=-1))
        col += 1

    if col != nbasis:
        raise ValueError("assemble_square_basis: final column count is not nbasis.")
    return T, meta


def _eval_mode_on_interval(sp1D, c, x):
    phi = np.zeros_like(np.asarray(x, dtype=float))
    for j in range(c.size):
        if abs(c[j]) > 0:
            phi = phi + c[j] * sp1D[j](x)
    return phi


def _extend_mode_two_sided(sp1D, c, xext, L, meta):
    xext = np.asarray(xext, dtype=float)
    phi_ext = np.zeros_like(xext)
    maskIn = (xext >= 0) & (xext <= L)
    phi_ext[maskIn] = _eval_mode_on_interval(sp1D, c, xext[maskIn])

    maskL = xext < 0
    if meta["type"] == "jet" and meta["side"] == "L":
        xr = -xext[maskL]
        phi_ext[maskL] = ((-1.0) ** meta["r"]) * _eval_mode_on_interval(sp1D, c, xr)

    maskR = xext > L
    if meta["type"] == "jet" and meta["side"] == "R":
        xr = 2 * L - xext[maskR]
        phi_ext[maskR] = ((-1.0) ** meta["r"]) * _eval_mode_on_interval(sp1D, c, xr)
    return phi_ext


def build_1d_modes_symmetric_two_sided(knots, nbasis, deg, s, xext):
    if s > deg:
        raise ValueError("Require s <= deg.")
    tol = 1e-12
    order = deg + 1
    L = knots[-1]

    sp1D = []
    supp = np.zeros((nbasis, 2))
    for j in range(nbasis):
        coefs = np.zeros(nbasis)
        coefs[j] = 1.0
        sp1D.append(BSpline(knots, coefs, deg))
        supp[j, 0] = knots[j]
        supp[j, 1] = knots[j + order]

    idxL = np.where(np.abs(supp[:, 0] - knots[0]) < tol)[0]
    idxR = np.where(np.abs(supp[:, 1] - knots[-1]) < tol)[0]
    idxI = np.array(sorted(set(range(nbasis)) - set(idxL) - set(idxR)), dtype=int)

    if len(set(idxL) & set(idxR)):
        raise ValueError("Left/right boundary spaces overlap.")
    if idxL.size != idxR.size:
        raise ValueError("Left/right boundary DOF counts differ.")

    DL = _build_boundary_derivative_matrix(sp1D, idxL, knots[0], s)
    if np.linalg.matrix_rank(DL, tol=1e-10) < s + 1:
        raise ValueError("Left boundary derivative matrix rank is insufficient.")

    CjetL, NnullL = _jet_and_null_basis_square(DL)
    T, meta = _assemble_square_basis_from_left_reflection(nbasis, idxL, idxI, idxR, CjetL, NnullL, s)

    nmodes = T.shape[1]
    Phi_ext = np.zeros((nmodes, xext.size))
    for k in range(nmodes):
        Phi_ext[k, :] = _extend_mode_two_sided(sp1D, T[:, k], xext, L, meta[k])

    return SimpleNamespace(T=T, meta=meta, Phi_ext=Phi_ext, knots=knots, L=L, sp1D=sp1D)


# ---------------------------------------------------------------------------
# Quadratic particular solution for the constant (mean) forcing
# ---------------------------------------------------------------------------
def build_quadratic_particular(xext, yext, hx, hy, c0):
    Nx = xext.size
    Ny = yext.size
    Px = Nx * hx
    Py = Ny * hy
    X, Y = np.meshgrid(xext, yext)
    xc = xext[0] + 0.5 * Px
    yc = yext[0] + 0.5 * Py
    Q = (c0 / 4.0) * ((X - xc) ** 2 + (Y - yc) ** 2)
    return Q - np.mean(Q)


# ---------------------------------------------------------------------------
# BB response kernel: Poisson response of every B-spline x B-spline product
# ---------------------------------------------------------------------------
def build_U_basis_new(opt: dict) -> SimpleNamespace:
    deg = opt["deg"]
    nbasis = opt["nbasis"]
    s = opt["s"]
    Lx = opt["Lx"]
    Ly = opt["Ly"]
    Nx = opt["Nx"]
    Ny = opt["Ny"]
    delta = opt["delta"]

    if s > deg:
        raise ValueError("Require s <= deg.")
    if nbasis < 2 * (deg + 1):
        raise ValueError("Need nbasis >= 2*(deg+1).")

    hx = Lx / (Nx - 1)
    hy = Ly / (Ny - 1)
    Nx_pad = int(round(delta / hx))
    Ny_pad = int(round(delta / hy))
    Nx_ext = Nx + 2 * Nx_pad
    Ny_ext = Ny + 2 * Ny_pad

    xext = np.arange(-Nx_pad, Nx + Nx_pad) * hx
    yext = np.arange(-Ny_pad, Ny + Ny_pad) * hy
    ix0 = Nx_pad + np.arange(Nx)
    iy0 = Ny_pad + np.arange(Ny)

    Px = Nx_ext * hx
    Py = Ny_ext * hy

    Xext, Yext = np.meshgrid(xext, yext)
    Qx_unit = 0.5 * (Xext - (xext[0] + 0.5 * Px))
    Qy_unit = 0.5 * (Yext - (yext[0] + 0.5 * Py))

    order = deg + 1
    nseg = nbasis - deg
    t_int_x = np.arange(1, nseg) / nseg
    t_int_y = np.arange(1, nseg) / nseg
    knots_x = np.concatenate([np.zeros(order), Lx * t_int_x, Lx * np.ones(order)])
    knots_y = np.concatenate([np.zeros(order), Ly * t_int_y, Ly * np.ones(order)])

    modesX = build_1d_modes_symmetric_two_sided(knots_x, nbasis, deg, s, xext)
    modesY = build_1d_modes_symmetric_two_sided(knots_y, nbasis, deg, s, yext)

    U_basis_new = np.zeros((nbasis, nbasis, Ny, Nx))
    qL_BB = np.zeros((nbasis, nbasis, Ny))
    qR_BB = np.zeros((nbasis, nbasis, Ny))
    qB_BB = np.zeros((nbasis, nbasis, Nx))
    qT_BB = np.zeros((nbasis, nbasis, Nx))

    for ixb in range(nbasis):
        px = modesX.Phi_ext[ixb, :]
        for iyb in range(nbasis):
            py = modesY.Phi_ext[iyb, :]
            Fext = np.outer(py, px)
            mean_disc = float(np.mean(Fext))
            Feff = Fext - mean_disc
            Vext = poisson_fft_periodic_2d_zero_mean(Feff, Px, Py)
            Qext = build_quadratic_particular(xext, yext, hx, hy, mean_disc)
            Uext = Vext + Qext
            U_basis_new[ixb, iyb, :, :] = Uext[np.ix_(iy0, ix0)]

            Vext_x, Vext_y = periodic_gradient_2d_real(Vext, Px, Py)
            Uext_x = Vext_x + mean_disc * Qx_unit
            Uext_y = Vext_y + mean_disc * Qy_unit

            qL_BB[ixb, iyb, :] = -Uext_x[iy0, ix0[0]]
            qR_BB[ixb, iyb, :] = Uext_x[iy0, ix0[-1]]
            qB_BB[ixb, iyb, :] = -Uext_y[iy0[0], ix0]
            qT_BB[ixb, iyb, :] = Uext_y[iy0[-1], ix0]

    return SimpleNamespace(
        U_basis_new=U_basis_new,
        qL_BB=qL_BB, qR_BB=qR_BB, qB_BB=qB_BB, qT_BB=qT_BB,
        Tx=modesX.T, Ty=modesY.T,
    )


# ---------------------------------------------------------------------------
# Mixed B-spline / Fourier modal response kernel
# ---------------------------------------------------------------------------
def build_newbasis_exp_kernel_1d(opt: dict) -> SimpleNamespace:
    deg = opt["deg"]
    nbasis = opt["nbasis"]
    s = opt["s"]
    basis_L = opt["basis_L"]
    mode_L = opt["mode_L"]
    basis_N = opt["basis_N"]
    mode_N = opt["mode_N"]
    delta = opt["delta"]
    n_modes = opt["n_modes"]

    if s > deg:
        raise ValueError("Require s <= deg.")
    if nbasis < 2 * (deg + 1):
        raise ValueError("Need nbasis >= 2*(deg+1).")
    if n_modes < 0:
        raise ValueError("n_modes cannot be negative.")

    hx = basis_L / (basis_N - 1)
    Nx_pad = int(round(delta / hx))
    xext = np.arange(-Nx_pad, basis_N + Nx_pad) * hx
    Nx_ext = xext.size
    ix0 = Nx_pad + np.arange(basis_N)

    Ny0 = mode_N - 1
    hy = mode_L / Ny0
    y0 = np.arange(Ny0) * hy

    Px = Nx_ext * hx
    Py = mode_L

    order = deg + 1
    nseg = nbasis - deg
    t_int = np.arange(1, nseg) / nseg
    knots = np.concatenate([np.zeros(order), basis_L * t_int, basis_L * np.ones(order)])

    modes = build_1d_modes_symmetric_two_sided(knots, nbasis, deg, s, xext)
    Phi_ext = modes.Phi_ext

    ell_list = (2 * np.pi / mode_L) * np.arange(1, n_modes + 1)

    G1_new = np.zeros((n_modes, nbasis, basis_N))
    G1_d_left = np.zeros((n_modes, nbasis))
    G1_d_right = np.zeros((n_modes, nbasis))

    for iL in range(n_modes):
        ell = ell_list[iL]
        phase_y = np.exp(1j * ell * y0)
        for ib in range(nbasis):
            phi_x = Phi_ext[ib, :]
            Fxy = np.outer(phase_y, phi_x)
            Uxy = poisson_fft_periodic_2d_zero_mean_complex(Fxy, Px, Py)
            Uxy_x, _ = periodic_gradient_2d_complex(Uxy, Px, Py)
            phase_rep = phase_y[:, None]
            amp_x = np.mean(Uxy / phase_rep, axis=0)
            amp_x_dx = np.mean(Uxy_x / phase_rep, axis=0)
            G1_new[iL, ib, :] = np.real(amp_x[ix0])
            G1_d_left[iL, ib] = np.real(amp_x_dx[ix0[0]])
            G1_d_right[iL, ib] = np.real(amp_x_dx[ix0[-1]])

    return SimpleNamespace(
        G1_new=G1_new, G1_d_left=G1_d_left, G1_d_right=G1_d_right,
        ell_list=ell_list, T=modes.T,
    )


# ---------------------------------------------------------------------------
# Trefftz harmonic-polynomial Neumann correction
# ---------------------------------------------------------------------------
def laplace_rect_solver_trefftz_neumann_precompute(x, y, opt: dict) -> SimpleNamespace:
    K = opt.get("K", 24)
    lam = opt.get("lambda_", 0.0)
    zero_mean = opt.get("zero_mean", True)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    Nx, Ny = x.size, y.size
    xmin, xmax, ymin, ymax = x[0], x[-1], y[0], y[-1]
    xc = 0.5 * (xmin + xmax)
    yc = 0.5 * (ymin + ymax)
    R = 0.5 * np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)

    XL = xmin * np.ones(Ny); YL = y.copy(); nXL = -np.ones(Ny); nYL = np.zeros(Ny)
    XR = xmax * np.ones(Ny); YR = y.copy(); nXR = np.ones(Ny); nYR = np.zeros(Ny)
    XB = x.copy(); YB = ymin * np.ones(Nx); nXB = np.zeros(Nx); nYB = -np.ones(Nx)
    XT = x.copy(); YT = ymax * np.ones(Nx); nXT = np.zeros(Nx); nYT = np.ones(Nx)
    Xbd = np.concatenate([XL, XR, XB, XT])
    Ybd = np.concatenate([YL, YR, YB, YT])
    nx = np.concatenate([nXL, nXR, nXB, nXT])
    ny = np.concatenate([nYL, nYR, nYB, nYT])

    wy = high_order_quad_weights_vector(y)
    wx = high_order_quad_weights_vector(x)
    wbd = np.concatenate([wy, wy, wx, wx])
    sw = np.sqrt(wbd)

    M = Xbd.size
    nbasis = 2 * K
    A = np.zeros((M, nbasis))
    zeta = ((Xbd - xc) + 1j * (Ybd - yc)) / R
    for k in range(1, K + 1):
        dzdx = k * zeta ** (k - 1) / R
        dzdy = 1j * k * zeta ** (k - 1) / R
        dRe_dn = nx * np.real(dzdx) + ny * np.real(dzdy)
        dIm_dn = nx * np.imag(dzdx) + ny * np.imag(dzdy)
        A[:, 2 * k - 2] = dRe_dn
        A[:, 2 * k - 1] = dIm_dn

    Aw = A * sw[:, None]
    colScale = np.sqrt(np.sum(np.abs(Aw) ** 2, axis=0))
    colScale[colScale < EPS] = 1.0
    As = Aw / colScale[None, :]
    if lam > 0:
        solverMat = np.linalg.solve(As.T @ As + lam * np.eye(nbasis), As.T)
    else:
        solverMat = np.linalg.pinv(As)

    Xg, Yg = np.meshgrid(x, y)
    zeta_grid = ((Xg - xc) + 1j * (Yg - yc)) / R
    Hgrid = np.zeros((Nx * Ny, nbasis))
    for k in range(1, K + 1):
        zk = zeta_grid ** k
        Hgrid[:, 2 * k - 2] = np.real(zk).ravel(order="F")
        Hgrid[:, 2 * k - 1] = np.imag(zk).ravel(order="F")

    return SimpleNamespace(
        K=K, zero_mean=zero_mean, Nx=Nx, Ny=Ny,
        A=A, sw=sw, colScale=colScale, solverMat=solverMat, Hgrid=Hgrid,
    )


def laplace_rect_solver_trefftz_neumann_apply(q0v, q1v, h0v, h1v, lap) -> np.ndarray:
    qbd = np.concatenate([np.ravel(q0v), np.ravel(q1v), np.ravel(h0v), np.ravel(h1v)])
    cs = lap.solverMat @ (qbd * lap.sw)
    coef = cs / lap.colScale
    U = (lap.Hgrid @ coef).reshape(lap.Ny, lap.Nx, order="F")
    if lap.zero_mean:
        U = U - np.mean(U)
    return U


def boundary_flux_rect(qL, qR, qB, qT, boundary) -> float:
    return float(
        np.sum(boundary.wy * np.ravel(qL))
        + np.sum(boundary.wy * np.ravel(qR))
        + np.sum(boundary.wx * np.ravel(qB))
        + np.sum(boundary.wx * np.ravel(qT))
    )


# ---------------------------------------------------------------------------
# Full precompute and cached apply
# ---------------------------------------------------------------------------
def _merge(in_dict: Optional[dict], defaults: dict) -> dict:
    out = dict(defaults)
    if in_dict:
        out.update(in_dict)
    return out


def bspf_kkt_poisson_neumann_precompute(x, y, params: Optional[dict] = None) -> SimpleNamespace:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    Nx, Ny = x.size, y.size

    p0 = default_bspf_kkt_neumann_params(Nx, Ny)
    params = _merge(params, p0)
    params["optSplit"] = _merge(params.get("optSplit"), p0["optSplit"])
    params["optLapN"] = _merge(params.get("optLapN"), p0["optLapN"])

    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]
    Nx0, Ny0 = Nx - 1, Ny - 1

    degB = params["degB"]
    nbasis = params["nbasis"]
    s_newbasis = params["s_newbasis"]
    delta_newbasis = params["delta_newbasis"]
    optSplit = dict(params["optSplit"])
    optSplit["degB"] = degB
    optLapN = params["optLapN"]

    bx = make_bspline_basis_values(x, degB, nbasis)
    by = make_bspline_basis_values(y, degB, nbasis)
    splitX = bspf_kkt_1d_decompose_precompute(bx, optSplit["kmax"], optSplit["r"], optSplit["lambda_kkt"], optSplit)
    splitY = bspf_kkt_1d_decompose_precompute(by, optSplit["kmax"], optSplit["r"], optSplit["lambda_kkt"], optSplit)

    Bx0 = bx.B0[:, :Nx0]
    By0 = by.B0[:, :Ny0]
    lam = 1e-12
    dx_const = np.linalg.solve(Bx0 @ Bx0.T + lam * np.eye(nbasis), Bx0 @ np.ones(Nx0))
    dy_const = np.linalg.solve(By0 @ By0.T + lam * np.eye(nbasis), By0 @ np.ones(Ny0))

    kerBB = build_U_basis_new(dict(
        deg=degB, nbasis=nbasis, s=s_newbasis, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        delta=delta_newbasis,
    ))

    kerExpX = build_newbasis_exp_kernel_1d(dict(
        deg=degB, nbasis=nbasis, s=s_newbasis, basis_L=Lx, mode_L=Ly,
        basis_N=Nx, mode_N=Ny, delta=delta_newbasis, n_modes=Ny // 2 - 1,
    ))
    kerExpY = build_newbasis_exp_kernel_1d(dict(
        deg=degB, nbasis=nbasis, s=s_newbasis, basis_L=Ly, mode_L=Lx,
        basis_N=Ny, mode_N=Nx, delta=delta_newbasis, n_modes=Nx // 2 - 1,
    ))

    xw = x.copy(); xw[-1] = xw[0]
    yw = y.copy(); yw[-1] = yw[0]
    ell_list_y = kerExpX.ell_list
    ell_list_x = kerExpY.ell_list
    phaseY = np.exp(1j * np.outer(yw, ell_list_y))            # (Ny, nLy)
    phaseX = np.exp(1j * np.outer(ell_list_x, xw))            # (nLx, Nx)

    fftCache = spectral_poisson_2d_uniform_precompute(Nx0, Ny0, Lx, Ly)
    lapCache = laplace_rect_solver_trefftz_neumann_precompute(x, y, optLapN)

    boundary = SimpleNamespace(
        wx=high_order_quad_weights_vector(x),
        wy=high_order_quad_weights_vector(y),
    )
    boundary.measure = 2 * np.sum(boundary.wx) + 2 * np.sum(boundary.wy)

    cache = SimpleNamespace(
        x=x, y=y, Nx=Nx, Ny=Ny, Nx0=Nx0, Ny0=Ny0, Lx=Lx, Ly=Ly,
        params=params, optSplit=optSplit, optLapN=optLapN,
        degB=degB, nbasis=nbasis,
        Bvals_x=bx.B0, Bvals_y=by.B0,
        splitX=splitX, splitY=splitY,
        dx_const=dx_const, dy_const=dy_const,
        # BB kernel, reshaped column-major (MATLAB reshape order)
        U_BB_mat=kerBB.U_basis_new.reshape(nbasis * nbasis, Ny * Nx, order="F"),
        qL_BB_mat=kerBB.qL_BB.reshape(nbasis * nbasis, Ny, order="F"),
        qR_BB_mat=kerBB.qR_BB.reshape(nbasis * nbasis, Ny, order="F"),
        qB_BB_mat=kerBB.qB_BB.reshape(nbasis * nbasis, Nx, order="F"),
        qT_BB_mat=kerBB.qT_BB.reshape(nbasis * nbasis, Nx, order="F"),
        Tx=kerBB.Tx, Ty=kerBB.Ty,
        Gx_new=kerExpX.G1_new, Gy_new=kerExpY.G1_new,
        Gx_dx_left=kerExpX.G1_d_left, Gx_dx_right=kerExpX.G1_d_right,
        Gy_dy_bottom=kerExpY.G1_d_left, Gy_dy_top=kerExpY.G1_d_right,
        T1x=kerExpX.T, T1y=kerExpY.T,
        ell_list_y=ell_list_y, ell_list_x=ell_list_x,
        phaseY=phaseY, phaseX=phaseX,
        Kpos_x=(Nx0 - 1) // 2, Kpos_y=(Ny0 - 1) // 2,
        fft=fftCache, laplace=lapCache, boundary=boundary,
    )
    return cache


def bspf_kkt_poisson_neumann_apply(f, qL, qR, qB, qT, cache, solution_mean: float = 0.0):
    f = np.asarray(f, dtype=float)
    Ny, Nx = f.shape
    if Ny != cache.Ny or Nx != cache.Nx:
        raise ValueError(f"Cached Poisson apply: f size must be [{cache.Ny},{cache.Nx}].")

    nbasis = cache.nbasis
    Nx0, Ny0 = cache.Nx0, cache.Ny0

    # 1. BSPF-KKT split of the forcing.
    split = split2d_kkt_directional(f, cache)
    f_per = split.f_per
    A1_f = split.A1.astype(float).copy()
    A2_f = split.A2.copy()
    A3_f = split.A3.copy()

    A3_dc = A3_f[:, 0].copy()
    A2_dc = A2_f[:, 0].copy()
    A1_f = A1_f + np.real(np.outer(A3_dc, cache.dy_const))
    A1_f = A1_f + np.real(np.outer(cache.dx_const, A2_dc))
    A3_f[:, 0] = 0.0
    A2_f[:, 0] = 0.0

    mu_per = float(np.mean(f_per[:Ny0, :Nx0]))
    f0 = f_per[:Ny0, :Nx0] - mu_per
    f_per = embed_periodic_full(f0)
    A1_f = A1_f + mu_per * np.outer(cache.dx_const, cache.dy_const)

    # 2. Coefficient transforms into the modal bases.
    A1_new = np.linalg.solve(cache.Tx, A1_f)
    A1_new = np.linalg.solve(cache.Ty, A1_new.T).T
    A3_new = np.linalg.solve(cache.T1x, A3_f)
    A2_new = np.linalg.solve(cache.T1y, A2_f)

    # 3. Particular solution + normal-derivative traces from BB kernel.
    A1vec = A1_new.ravel(order="F")
    U_total = (A1vec @ cache.U_BB_mat).reshape(Ny, Nx, order="F")
    qP_L = (A1vec @ cache.qL_BB_mat).ravel()
    qP_R = (A1vec @ cache.qR_BB_mat).ravel()
    qP_B = (A1vec @ cache.qB_BB_mat).ravel()
    qP_T = (A1vec @ cache.qT_BB_mat).ravel()

    Kx_use = int(min(cache.ell_list_x.size, cache.Kpos_x, A2_new.shape[1] - 1))
    Ky_use = int(min(cache.ell_list_y.size, cache.Kpos_y, A3_new.shape[1] - 1))

    # B_x * exp(i ell y) part.
    if Ky_use > 0:
        coeff_x = A3_new[:, 1:Ky_use + 1]                         # (nbasis, K)
        sx_modes = np.einsum("bk,kbn->kn", coeff_x, cache.Gx_new[:Ky_use])
        phase_y = cache.phaseY[:, :Ky_use]                        # (Ny, K)
        U_total = U_total + 2 * np.real(phase_y @ sx_modes)

        sx_left = np.einsum("bk,kb->k", coeff_x, cache.Gx_dx_left[:Ky_use])
        sx_right = np.einsum("bk,kb->k", coeff_x, cache.Gx_dx_right[:Ky_use])
        qP_L = qP_L - 2 * np.real(phase_y @ sx_left)
        qP_R = qP_R + 2 * np.real(phase_y @ sx_right)

        ell_y = cache.ell_list_y[:Ky_use]
        qP_B = qP_B - 2 * np.real(((1j * ell_y * phase_y[0, :])[:, None] * sx_modes).sum(axis=0))
        qP_T = qP_T + 2 * np.real(((1j * ell_y * phase_y[-1, :])[:, None] * sx_modes).sum(axis=0))

    # B_y * exp(i ell x) part.
    if Kx_use > 0:
        coeff_y = A2_new[:, 1:Kx_use + 1]                         # (nbasis, K)
        sy_modes = np.einsum("bk,kbn->kn", coeff_y, cache.Gy_new[:Kx_use])
        phase_x = cache.phaseX[:Kx_use, :]                        # (K, Nx)
        U_total = U_total + 2 * np.real(sy_modes.T @ phase_x)

        ell_x = cache.ell_list_x[:Kx_use]
        qP_L = qP_L - 2 * np.real(sy_modes.T @ (1j * ell_x * phase_x[:, 0]))
        qP_R = qP_R + 2 * np.real(sy_modes.T @ (1j * ell_x * phase_x[:, -1]))

        sy_bottom = np.einsum("bk,kb->k", coeff_y, cache.Gy_dy_bottom[:Kx_use])
        sy_top = np.einsum("bk,kb->k", coeff_y, cache.Gy_dy_top[:Kx_use])
        qP_B = qP_B - 2 * np.real(sy_bottom @ phase_x)
        qP_T = qP_T + 2 * np.real(sy_top @ phase_x)

    # 3b. Periodic FFT particular solution.
    f0 = f_per[:Ny0, :Nx0]
    Phi0, Phi0x, Phi0y = spectral_poisson_2d_uniform_with_grad_cached(f0, cache.fft)
    Phi = embed_periodic_full(Phi0)
    Phi_x = embed_periodic_full(Phi0x)
    Phi_y = embed_periodic_full(Phi0y)

    U_particular = U_total + Phi
    qP_L = qP_L - Phi_x[:, 0]
    qP_R = qP_R + Phi_x[:, -1]
    qP_B = qP_B - Phi_y[0, :]
    qP_T = qP_T + Phi_y[-1, :]

    # 4. Trefftz-Neumann correction for the residual Neumann data.
    qL_corr = np.ravel(qL) - qP_L
    qR_corr = np.ravel(qR) - qP_R
    qB_corr = np.ravel(qB) - qP_B
    qT_corr = np.ravel(qT) - qP_T

    flux_before = boundary_flux_rect(qL_corr, qR_corr, qB_corr, qT_corr, cache.boundary)
    flux_after = flux_before
    flux_tol = cache.optLapN.get("flux_correction_tol", 0.0) or 0.0
    if cache.optLapN.get("remove_flux_mean", False) and abs(flux_before) > flux_tol:
        corr = flux_before / cache.boundary.measure
        qL_corr = qL_corr - corr
        qR_corr = qR_corr - corr
        qB_corr = qB_corr - corr
        qT_corr = qT_corr - corr
        flux_after = boundary_flux_rect(qL_corr, qR_corr, qB_corr, qT_corr, cache.boundary)

    U_lap = laplace_rect_solver_trefftz_neumann_apply(qL_corr, qR_corr, qB_corr, qT_corr, cache.laplace)

    U_raw = U_particular + U_lap
    U = U_raw - np.mean(U_raw) + solution_mean

    info = SimpleNamespace(
        U_particular=U_particular, U_lap=U_lap,
        flux_before=flux_before, flux_after=flux_after,
    )
    return U, info
