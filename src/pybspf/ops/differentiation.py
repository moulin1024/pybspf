"""Single and batched derivatives sharing one spline solve and FFT."""

from __future__ import annotations

import time
from dataclasses import dataclass
from numbers import Integral
from typing import Iterable

from ..backend import get_array_module
from ..types import Array
from ._common import solve_spline


@dataclass(frozen=True)
class DerivativeResult:
    """Derivatives indexed by order, plus the fitted spline on the input grid."""

    values: dict[int, Array]
    spline: Array

    def __getitem__(self, order: int) -> Array:
        return self.values[order]


def _normalize_orders(orders: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(orders, Integral):
        orders = (orders,)
    else:
        try:
            orders = tuple(orders)
        except TypeError as exc:
            raise ValueError("orders must be an integer or iterable of integers.") from exc
    if not orders:
        raise ValueError("At least one derivative order must be requested.")
    if any(isinstance(k, bool) or not isinstance(k, Integral) or k not in (1, 2, 3, 4) for k in orders):
        raise ValueError("Only integer 1st/2nd/3rd/4th derivatives are supported.")
    return tuple(sorted(set(orders)))


def _derivatives(op, f, orders, lam, neumann_bc, *, ndim):
    orders = _normalize_orders(orders)
    start = time.perf_counter()
    coefficients, spline, residual = solve_spline(op, f, lam, neumann_bc, ndim=ndim)
    xp = get_array_module(use_gpu=op.use_gpu)
    values = {k: getattr(op, f"_B{k}T_f") @ coefficients for k in orders}
    if op.correction == "spectral" and residual.size:
        is_complex = xp.iscomplexobj(residual)
        fft = xp.fft.fft if is_complex else xp.fft.rfft
        ifft = xp.fft.ifft if is_complex else xp.fft.irfft
        spectrum = fft(residual, axis=0)
        omega = (2 * xp.pi * xp.fft.fftfreq(op.grid.n, d=op.grid.dx)
                 if is_complex else op.grid.omega)
        multiplier = (1j * omega).reshape((-1,) + (1,) * (ndim - 1))
        for k in orders:
            values[k] += ifft(spectrum * multiplier**k, n=op.grid.n, axis=0)
    timings = {"total": time.perf_counter() - start}
    op.last_timing_derivatives = timings
    if len(orders) == 1:
        op.last_timing_diff = timings
    elif orders == (1, 2):
        op.last_timing_d12 = timings
    elif orders == (1, 2, 3):
        op.last_timing_d123 = timings
    return DerivativeResult(values, spline)


def differentiate(self, f: Array, k: int = 1, lam: float = 0.0, *, neumann_bc=None):
    """Return ``(derivative, spline)`` for one order from 1 through 4."""
    result = derivatives(self, f, orders=(k,), lam=lam, neumann_bc=neumann_bc)
    return result[k], result.spline


def derivatives(self, f: Array, orders: int | Iterable[int], lam: float = 0.0, *, neumann_bc=None) -> DerivativeResult:
    """Compute multiple orders for a real or complex signal of shape ``(n,)``."""
    return _derivatives(self, f, orders, lam, neumann_bc, ndim=1)


def derivatives_batched(self, f: Array, orders: int | Iterable[int], lam: float = 0.0, *, neumann_bc=None) -> DerivativeResult:
    """Compute derivatives of shape ``(n, batch)`` with one matrix solve and FFT."""
    return _derivatives(self, f, orders, lam, neumann_bc, ndim=2)


__all__ = ["DerivativeResult", "differentiate", "derivatives", "derivatives_batched"]
