"""! @file ops/integration.py
@brief Package-owned integration workflows for BSPF1D.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..backend import get_array_module
from ._common import prepare_samples, solve_spline
from ..types import Array


def definite_integral(
    self,
    f: Array,
    a: Optional[float] = None,
    b: Optional[float] = None,
    lam: float = 0.0,
) -> float:
    """Compute a definite integral of the sampled signal."""
    xp = get_array_module(use_gpu=self.use_gpu)
    f = prepare_samples(self, f)
    if xp.iscomplexobj(f):
        raise ValueError("definite_integral currently requires real samples.")
    a = self.grid.a if a is None else float(a)
    b = self.grid.b if b is None else float(b)
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("Integration bounds must be finite.")
    if min(a, b) < self.grid.a or max(a, b) > self.grid.b:
        raise ValueError("Integration bounds must lie inside the grid domain.")
    P, _, residual = solve_spline(self, f, lam)
    spline_integral = self.basis.integrate_basis(a, b) @ P
    # Integrate the piecewise-linear residual over exactly the requested interval.
    lo, hi = min(a, b), max(a, b)
    x = self.grid.x
    interior = x[(x > lo) & (x < hi)]
    nodes = xp.concatenate((xp.asarray([lo]), interior, xp.asarray([hi])))
    values = xp.interp(nodes, x, residual)
    residual_integral = xp.sum(xp.diff(nodes) * (values[:-1] + values[1:]) / 2)
    if b < a:
        residual_integral = -residual_integral
    return float(spline_integral + residual_integral)


def antiderivative(
    self,
    f: Array,
    order: int = 1,
    *,
    left_value: float = 0.0,
    match_right: Optional[float] = None,
    lam: float = 0.0,
):
    """Compute a first or second antiderivative of the sampled signal."""
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")

    xp = get_array_module(use_gpu=self.use_gpu)
    f = prepare_samples(self, f)
    if xp.iscomplexobj(f):
        raise ValueError("antiderivative currently requires real samples.")
    P, f_spline, residual = solve_spline(self, f, lam)
    x = self.grid.x
    F_spline = xp.zeros_like(x)
    for i, spline in enumerate(self.basis._splines):
        F_spline += P[i] * spline.antiderivative(order)(x)
    fft = xp.fft

    om = xp.asarray(self.grid.omega)
    R = fft.rfft(residual)

    if order == 1:
        mask = om != 0.0
        out_hat = xp.zeros_like(R, dtype=xp.complex128)
        out_hat[mask] = R[mask] / (1j * om[mask])
        F_corr = fft.irfft(out_hat, n=self.grid.n)
        xx = xp.asarray(x)
        mean_r = float(xp.mean(residual))
        F_corr = F_corr + mean_r * (xx - float(xx[0]))
        F_corr = F_corr - F_corr[0]
    else:
        mask = om != 0.0
        out_hat = xp.zeros_like(R, dtype=xp.complex128)
        out_hat[mask] = R[mask] / ((1j * om[mask]) ** 2)
        F_corr = fft.irfft(out_hat, n=self.grid.n)
        xx = xp.asarray(x)
        x0 = float(xx[0])
        x1 = float(xx[-1])
        mean_r = float(xp.mean(residual))
        F_corr = F_corr + 0.5 * mean_r * (xx - x0) * (xx - x1)

    xx = xp.asarray(x)
    x0 = float(xx[0])
    x1 = float(xx[-1])
    F = xp.asarray(F_spline) + F_corr
    F = F - (F[0] - float(left_value))

    if match_right is not None:
        if order == 1:
            F = F + (float(match_right) - F[-1])
        else:
            F = F + (float(match_right) - F[-1]) * (xx - x0) / (x1 - x0)

    return F, f_spline


_EPS = np.finfo(float).eps


def trapezoid_weights_1d(t) -> np.ndarray:
    """! @brief Composite trapezoidal quadrature weights for a 1D node vector."""
    t = np.asarray(t, dtype=float).ravel()
    n = t.size
    if n < 2:
        raise ValueError("trapezoid_weights_1d: at least two grid points are required.")
    w = np.zeros(n)
    dt = np.diff(t)
    w[0] = dt[0] / 2
    w[-1] = dt[-1] / 2
    if n > 2:
        w[1:-1] = 0.5 * (dt[:-1] + dt[1:])
    return w


def simple_trapezoid_weights(x) -> np.ndarray:
    """! @brief Robust trapezoidal weights (uniform or non-uniform nodes)."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        return np.array([1.0])
    w = np.zeros(n)
    dx = np.diff(x)
    w[0] = dx[0] / 2
    w[-1] = dx[-1] / 2
    if n > 2:
        w[1:-1] = (dx[:-1] + dx[1:]) / 2
    return w


def high_order_quad_weights_vector(x) -> np.ndarray:
    """! @brief High-order (Simpson 1/3 + 3/8) weights on a uniform grid.

    Falls back to trapezoidal weights for non-uniform grids.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    if N < 2:
        raise ValueError("high_order_quad_weights_vector: N must be >= 2.")

    hvec = np.diff(x)
    h = float(np.mean(hvec))
    if np.max(np.abs(hvec - h)) > 100 * _EPS * max(1.0, abs(h)):
        w = np.zeros(N)
        w[0] = hvec[0] / 2
        w[-1] = hvec[-1] / 2
        if N > 2:
            w[1:-1] = 0.5 * (hvec[:-1] + hvec[1:])
        return w

    m = N - 1
    w = np.zeros(N)
    if m == 1:
        w[:] = h / 2
        return w

    if m % 2 == 0:
        w[0] = h / 3
        w[-1] = h / 3
        w[1:-1:2] = 4 * h / 3
        w[2:-2:2] = 2 * h / 3
    else:
        mSim = m - 3
        if mSim > 0:
            idxEnd = mSim
            w[0] += h / 3
            w[idxEnd] += h / 3
            if idxEnd >= 2:
                w[1:idxEnd:2] += 4 * h / 3
                w[2 : idxEnd - 1 : 2] += 2 * h / 3
        else:
            idxEnd = 0
        ids = np.arange(idxEnd, idxEnd + 4)
        w[ids] += (3 * h / 8) * np.array([1.0, 3.0, 3.0, 1.0])
    return w


def weighted_mean_2d(A, x, y) -> float:
    """! @brief Tensor-product weighted mean of ``A`` (shape ``(Ny, Nx)``)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    wx = simple_trapezoid_weights(x)
    wy = simple_trapezoid_weights(y)
    area = (x[-1] - x[0]) * (y[-1] - y[0])
    return float((wy @ A @ wx) / max(area, _EPS))


__all__ = [
    "antiderivative",
    "definite_integral",
    "high_order_quad_weights_vector",
    "simple_trapezoid_weights",
    "trapezoid_weights_1d",
    "weighted_mean_2d",
]
