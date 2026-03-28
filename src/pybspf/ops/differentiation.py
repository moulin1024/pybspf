"""! @file ops/differentiation.py
@brief Package-owned differentiation workflows for BSPF1D.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
from scipy import linalg as sla

from ..backend import _HAS_CUPY, cp, cpla
from ..types import Array


def _solve_spline_system(
    self,
    f: Array,
    lam: float,
    neumann_bc: Optional[Tuple[Optional[float], Optional[float]]],
):
    """Fit spline coefficients and return ``(P, f_spline, residual)``."""
    if self.use_gpu and _HAS_CUPY:
        xp = cp
        la = cpla
        input_was_numpy = isinstance(f, np.ndarray) or not isinstance(f, cp.ndarray)
        is_complex = xp.iscomplexobj(f)
        dtype = xp.complex128 if is_complex else xp.float64
        f_x = xp.asarray(f, dtype=dtype)

        n_b = self._BW_f.shape[0]
        if is_complex:
            rhs = xp.empty(n_b + 2 * self.order, dtype=xp.complex128)
        else:
            rhs = self._rhs_buf
        rhs[:n_b] = 2.0 * (self._BW_f @ f_x)
        rhs[n_b:] = self._BND_f @ f_x

        if neumann_bc is not None:
            if self.order < 1:
                raise ValueError("Neumann BC requires self.order >= 1.")
            left_flux, right_flux = neumann_bc
            if left_flux is not None:
                rhs[n_b + 1] = complex(left_flux) if is_complex else float(left_flux)
            if right_flux is not None:
                rhs[n_b + self.order + 1] = complex(right_flux) if is_complex else float(right_flux)

        lu_cpu, piv_cpu = self._kkt_lu(lam)
        sol = la.lu_solve((xp.asarray(lu_cpu), xp.asarray(piv_cpu)), rhs, overwrite_b=True)
        P = sol[: self.basis.B0.shape[0]]
        f_spline = self._BT0_f @ P
        residual = f_x - f_spline
        return P, f_spline, residual, input_was_numpy

    is_complex = np.iscomplexobj(f)
    dtype = np.complex128 if is_complex else np.float64
    f_x = np.asarray(f, dtype=dtype)
    if f_x.shape[0] != self.grid.n:
        raise ValueError("Length of f must match grid size.")

    n_b = self._BW_f.shape[0]
    if is_complex:
        rhs = np.empty(n_b + 2 * self.order, dtype=np.complex128)
    else:
        rhs = self._rhs_buf
    rhs[:n_b] = 2.0 * (self._BW_f @ f_x)
    rhs[n_b:] = self._BND_f @ f_x

    if neumann_bc is not None:
        if self.order < 1:
            raise ValueError("Neumann BC requires self.order >= 1.")
        left_flux, right_flux = neumann_bc
        if left_flux is not None:
            rhs[n_b + 1] = complex(left_flux) if is_complex else float(left_flux)
        if right_flux is not None:
            rhs[n_b + self.order + 1] = complex(right_flux) if is_complex else float(right_flux)

    lu, piv = self._kkt_lu(lam)
    sol = sla.lu_solve((lu, piv), rhs, overwrite_b=True)
    P = sol[: self.basis.B0.shape[0]]
    f_spline = self._BT0_f @ P
    residual = f_x - f_spline
    return P, f_spline, residual, True


def differentiate(
    self,
    f: Array,
    k: int = 1,
    lam: float = 0.0,
    *,
    neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None,
):
    """Differentiate a sampled signal."""
    if k not in (1, 2, 3):
        raise ValueError("Only 1st/2nd/3rd derivatives are supported.")

    timings: dict = {}
    t_total_start = time.perf_counter()
    P, f_spline, residual, input_was_numpy = _solve_spline_system(self, f, lam, neumann_bc)

    if self.use_gpu and _HAS_CUPY:
        fft = cp.fft
        df_spline = self.basis.BkT(k) @ P
        R = fft.rfft(residual)
        if k == 1:
            corr = fft.irfft(R * self._iomega, n=self.grid.n)
        elif k == 2:
            corr = fft.irfft(R * self._iomega2, n=self.grid.n)
        else:
            corr = fft.irfft(R * self._iomega3, n=self.grid.n)
        out = df_spline + corr
        timings["total"] = time.perf_counter() - t_total_start
        self.last_timing_diff = timings
        return self._bk.ensure_like_input(out, input_was_numpy), self._bk.ensure_like_input(
            f_spline, input_was_numpy
        )

    if np.iscomplexobj(f):
        df_spline = self.basis.BkT(k) @ P
        R = np.fft.fft(residual)
        omega = 2.0 * np.pi * np.fft.fftfreq(self.grid.n, d=self.grid.dx)
        corr = np.fft.ifft(R * (1j * omega) ** k)
        out = (df_spline + corr).astype(np.complex128)
        timings["total"] = time.perf_counter() - t_total_start
        self.last_timing_diff = timings
        return out, f_spline.astype(np.complex128)

    df_spline = self.basis.BkT(k) @ P
    R = np.fft.rfft(residual)
    if k == 1:
        corr = np.fft.irfft(R * self._iomega, n=self.grid.n)
    elif k == 2:
        corr = np.fft.irfft(R * self._iomega2, n=self.grid.n)
    else:
        corr = np.fft.irfft(R * self._iomega3, n=self.grid.n)
    out = (df_spline + corr).astype(np.float64)
    timings["total"] = time.perf_counter() - t_total_start
    self.last_timing_diff = timings
    return out, f_spline.astype(np.float64)


def differentiate_1_2(
    self,
    f: Array,
    lam: float = 0.0,
    *,
    neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None,
):
    """Compute the first and second derivatives together."""
    timings: dict = {}
    t_total_start = time.perf_counter()
    P, f_spline, residual, input_was_numpy = _solve_spline_system(self, f, lam, neumann_bc)

    if self.use_gpu and _HAS_CUPY:
        fft = cp.fft
        df1_spline = self.basis.BkT(1) @ P
        df2_spline = self.basis.BkT(2) @ P
        R = fft.rfft(residual)
        corr1 = fft.irfft(R * self._iomega, n=self.grid.n)
        corr2 = fft.irfft(R * self._iomega2, n=self.grid.n)
        df1 = df1_spline + corr1
        df2 = df2_spline + corr2
        timings["total"] = time.perf_counter() - t_total_start
        self.last_timing_d12 = timings
        return (
            self._bk.ensure_like_input(df1, input_was_numpy),
            self._bk.ensure_like_input(df2, input_was_numpy),
            self._bk.ensure_like_input(f_spline, input_was_numpy),
        )

    if np.iscomplexobj(f):
        df1_spline = self.basis.BkT(1) @ P
        df2_spline = self.basis.BkT(2) @ P
        R = np.fft.fft(residual)
        omega = 2.0 * np.pi * np.fft.fftfreq(self.grid.n, d=self.grid.dx)
        corr1 = np.fft.ifft(R * (1j * omega))
        corr2 = np.fft.ifft(R * (1j * omega) ** 2)
        timings["total"] = time.perf_counter() - t_total_start
        self.last_timing_d12 = timings
        return (
            (df1_spline + corr1).astype(np.complex128),
            (df2_spline + corr2).astype(np.complex128),
            f_spline.astype(np.complex128),
        )

    df1_spline = self._B1T_f @ P
    df2_spline = self._B2T_f @ P
    R = np.fft.rfft(residual)
    corr1 = np.fft.irfft(R * self._iomega, n=self.grid.n)
    corr2 = np.fft.irfft(R * self._iomega2, n=self.grid.n)
    timings["total"] = time.perf_counter() - t_total_start
    self.last_timing_d12 = timings
    return (
        (df1_spline + corr1).astype(np.float64),
        (df2_spline + corr2).astype(np.float64),
        f_spline.astype(np.float64),
    )


def differentiate_1_2_3(
    self,
    f: Array,
    lam: float = 0.0,
    *,
    neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None,
):
    """Compute the first, second, and third derivatives together."""
    timings: dict = {}
    t_total_start = time.perf_counter()
    P, f_spline, residual, input_was_numpy = _solve_spline_system(self, f, lam, neumann_bc)

    if self.use_gpu and _HAS_CUPY:
        fft = cp.fft
        df1_spline = self.basis.BkT(1) @ P
        df2_spline = self.basis.BkT(2) @ P
        df3_spline = self.basis.BkT(3) @ P
        R = fft.rfft(residual)
        corr1 = fft.irfft(R * self._iomega, n=self.grid.n)
        corr2 = fft.irfft(R * self._iomega2, n=self.grid.n)
        corr3 = fft.irfft(R * self._iomega3, n=self.grid.n)
        timings["total"] = time.perf_counter() - t_total_start
        self.last_timing_d123 = timings
        return (
            self._bk.ensure_like_input(df1_spline + corr1, input_was_numpy),
            self._bk.ensure_like_input(df2_spline + corr2, input_was_numpy),
            self._bk.ensure_like_input(df3_spline + corr3, input_was_numpy),
            self._bk.ensure_like_input(f_spline, input_was_numpy),
        )

    if np.iscomplexobj(f):
        df1_spline = self.basis.BkT(1) @ P
        df2_spline = self.basis.BkT(2) @ P
        df3_spline = self.basis.BkT(3) @ P
        R = np.fft.fft(residual)
        omega = 2.0 * np.pi * np.fft.fftfreq(self.grid.n, d=self.grid.dx)
        corr1 = np.fft.ifft(R * (1j * omega))
        corr2 = np.fft.ifft(R * (1j * omega) ** 2)
        corr3 = np.fft.ifft(R * (1j * omega) ** 3)
        timings["total"] = time.perf_counter() - t_total_start
        self.last_timing_d123 = timings
        return (
            (df1_spline + corr1).astype(np.complex128),
            (df2_spline + corr2).astype(np.complex128),
            (df3_spline + corr3).astype(np.complex128),
            f_spline.astype(np.complex128),
        )

    df1_spline = self._B1T_f @ P
    df2_spline = self._B2T_f @ P
    df3_spline = self._B3T_f @ P
    R = np.fft.rfft(residual)
    corr1 = np.fft.irfft(R * self._iomega, n=self.grid.n)
    corr2 = np.fft.irfft(R * self._iomega2, n=self.grid.n)
    corr3 = np.fft.irfft(R * self._iomega3, n=self.grid.n)
    timings["total"] = time.perf_counter() - t_total_start
    self.last_timing_d123 = timings
    return (
        (df1_spline + corr1).astype(np.float64),
        (df2_spline + corr2).astype(np.float64),
        (df3_spline + corr3).astype(np.float64),
        f_spline.astype(np.float64),
    )


def differentiate_1_2_batched(
    self,
    f: Array,
    lam: float = 0.0,
    *,
    neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None,
):
    """Batched first and second derivatives for multiple signals."""
    if f.ndim != 2:
        raise ValueError("Expected f with shape (n, batch).")

    df1 = []
    df2 = []
    f_spline = []
    for i in range(f.shape[1]):
        d1_i, d2_i, s_i = differentiate_1_2(self, f[:, i], lam=lam, neumann_bc=neumann_bc)
        df1.append(d1_i)
        df2.append(d2_i)
        f_spline.append(s_i)

    if self.use_gpu and _HAS_CUPY and isinstance(df1[0], cp.ndarray):
        xp = cp
    else:
        xp = np
    return xp.stack(df1, axis=1), xp.stack(df2, axis=1), xp.stack(f_spline, axis=1)


__all__ = [
    "differentiate",
    "differentiate_1_2",
    "differentiate_1_2_3",
    "differentiate_1_2_batched",
]
