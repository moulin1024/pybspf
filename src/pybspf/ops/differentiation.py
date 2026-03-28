"""! @file ops/differentiation.py
@brief Package-owned differentiation workflows for BSPF1D.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy import linalg as sla

from ..backend import _HAS_CUPY, cp, cpla
from ..types import Array


@dataclass(frozen=True)
class DerivativeResult:
    """Collection of derivative arrays computed from a shared solve/FFT path."""

    values: Dict[int, Array]
    spline: Array

    def __getitem__(self, order: int) -> Array:
        return self.values[order]


def _normalize_orders(orders: int | Iterable[int]) -> Tuple[int, ...]:
    """Return a validated, deduplicated derivative-order tuple."""
    if isinstance(orders, int):
        normalized = (int(orders),)
    else:
        normalized = tuple(sorted({int(order) for order in orders}))

    if not normalized:
        raise ValueError("At least one derivative order must be requested.")
    for order in normalized:
        if order not in (1, 2, 3, 4):
            raise ValueError("Only 1st/2nd/3rd/4th derivatives are supported.")
    return normalized


def _basis_derivative_matrix(self, order: int):
    """Return a cached basis derivative matrix when available."""
    attr_name = f"_B{order}T_f"
    return getattr(self, attr_name, self.basis.BkT(order))


def _spectral_multiplier(self, order: int):
    """Return the cached spectral multiplier for the requested order."""
    attr_name = f"_iomega{'' if order == 1 else order}"
    return getattr(self, attr_name)


def _compute_derivative_values(
    self,
    P: Array,
    residual: Array,
    orders: Tuple[int, ...],
    *,
    input_was_numpy: bool,
):
    """Compute multiple derivative orders from one spline fit and one transform."""
    is_complex = bool(np.iscomplexobj(residual))

    if self.use_gpu and _HAS_CUPY:
        fft = cp.fft
        if cp.iscomplexobj(residual):
            R = fft.fft(residual)
            omega = cp.asarray(self.grid.omega)
            corrections = {
                order: fft.ifft(R * (1j * omega) ** order)
                for order in orders
            }
        else:
            R = fft.rfft(residual)
            corrections = {
                order: fft.irfft(R * _spectral_multiplier(self, order), n=self.grid.n)
                for order in orders
            }

        values = {
            order: self._bk.ensure_like_input(
                _basis_derivative_matrix(self, order) @ P + corrections[order],
                input_was_numpy,
            )
            for order in orders
        }
        return values

    if is_complex:
        R = np.fft.fft(residual)
        omega = 2.0 * np.pi * np.fft.fftfreq(self.grid.n, d=self.grid.dx)
        corrections = {
            order: np.fft.ifft(R * (1j * omega) ** order)
            for order in orders
        }
        return {
            order: (_basis_derivative_matrix(self, order) @ P + corrections[order]).astype(np.complex128)
            for order in orders
        }

    R = np.fft.rfft(residual)
    corrections = {
        order: np.fft.irfft(R * _spectral_multiplier(self, order), n=self.grid.n)
        for order in orders
    }
    return {
        order: (_basis_derivative_matrix(self, order) @ P + corrections[order]).astype(np.float64)
        for order in orders
    }


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
    result = derivatives(self, f, orders=(k,), lam=lam, neumann_bc=neumann_bc)
    return result[k], result.spline


def derivatives(
    self,
    f: Array,
    orders: int | Iterable[int],
    lam: float = 0.0,
    *,
    neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> DerivativeResult:
    """Compute any supported derivative-order combination through one shared path."""
    normalized_orders = _normalize_orders(orders)

    timings: dict = {}
    t_total_start = time.perf_counter()
    P, f_spline, residual, input_was_numpy = _solve_spline_system(self, f, lam, neumann_bc)
    values = _compute_derivative_values(
        self,
        P,
        residual,
        normalized_orders,
        input_was_numpy=input_was_numpy,
    )

    timings["total"] = time.perf_counter() - t_total_start
    self.last_timing_derivatives = timings
    if len(normalized_orders) == 1:
        self.last_timing_diff = timings
    elif normalized_orders == (1, 2):
        self.last_timing_d12 = timings
    elif normalized_orders == (1, 2, 3):
        self.last_timing_d123 = timings

    if self.use_gpu and _HAS_CUPY:
        spline = self._bk.ensure_like_input(f_spline, input_was_numpy)
    elif np.iscomplexobj(f):
        spline = f_spline.astype(np.complex128)
    else:
        spline = f_spline.astype(np.float64)
    return DerivativeResult(values=values, spline=spline)

def derivatives_batched(
    self,
    f: Array,
    orders: int | Iterable[int],
    lam: float = 0.0,
    *,
    neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> DerivativeResult:
    """Batched multi-order derivatives for arrays with shape ``(n, batch)``."""
    if f.ndim != 2:
        raise ValueError("Expected f with shape (n, batch).")

    normalized_orders = _normalize_orders(orders)
    batched_values = {order: [] for order in normalized_orders}
    splines = []
    for i in range(f.shape[1]):
        result_i = derivatives(self, f[:, i], orders=normalized_orders, lam=lam, neumann_bc=neumann_bc)
        for order in normalized_orders:
            batched_values[order].append(result_i[order])
        splines.append(result_i.spline)

    first_order = normalized_orders[0]
    if self.use_gpu and _HAS_CUPY and isinstance(batched_values[first_order][0], cp.ndarray):
        xp = cp
    else:
        xp = np
    return DerivativeResult(
        values={
            order: xp.stack(batched_values[order], axis=1)
            for order in normalized_orders
        },
        spline=xp.stack(splines, axis=1),
    )


__all__ = [
    "DerivativeResult",
    "differentiate",
    "derivatives",
    "derivatives_batched",
]
