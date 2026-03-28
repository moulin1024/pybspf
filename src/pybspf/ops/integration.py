"""! @file ops/integration.py
@brief Package-owned integration workflows for BSPF1D.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import linalg as sla

from ..backend import _HAS_CUPY, cp
from ..types import Array


def definite_integral(
    self,
    f: Array,
    a: Optional[float] = None,
    b: Optional[float] = None,
    lam: float = 0.0,
) -> float:
    """Compute a definite integral of the sampled signal."""
    f = np.asarray(f, dtype=np.float64)
    if f.shape[0] != self.grid.n:
        raise ValueError("Length of f must match grid size.")

    a = self.grid.a if a is None else float(a)
    b = self.grid.b if b is None else float(b)

    rhs_2bw = 2.0 * (self.BW @ f)
    dY = self.end.BND @ f
    rhs = np.concatenate((rhs_2bw, dY))

    lu, piv = self._kkt_lu(lam)
    sol = sla.lu_solve((lu, piv), rhs, overwrite_b=True)
    P = sol[: self.basis.B0.shape[0]]

    basis_integrals = self.basis.integrate_basis(a, b)
    spline_integral = basis_integrals @ P

    residual = f - (self.basis.BT0 @ P)
    residual_integral = np.sum(residual * self.grid.trap)
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

    f = np.asarray(f, dtype=np.float64)
    if f.shape[0] != self.grid.n:
        raise ValueError("Length of f must match grid size.")

    rhs_2bw = 2.0 * (self.BW @ f)
    dY = self.end.BND @ f
    rhs = np.concatenate((rhs_2bw, dY))
    lu, piv = self._kkt_lu(lam)
    sol = sla.lu_solve((lu, piv), rhs, overwrite_b=True)
    P = sol[: self.basis.B0.shape[0]]

    x = self.grid.x
    f_spline = self.basis.BT0 @ P

    F_spline_host = np.zeros_like(x)
    for i, spline in enumerate(self.basis._splines):
        F_spline_host += P[i] * spline.antiderivative(order)(x)

    if self.use_gpu and _HAS_CUPY:
        xp, fft = cp, cp.fft
    else:
        xp, fft = np, np.fft

    residual = xp.asarray(f - f_spline)
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
    F = xp.asarray(F_spline_host) + F_corr
    F = F - (F[0] - float(left_value))

    if match_right is not None:
        if order == 1:
            F = F + (float(match_right) - F[-1])
        else:
            F = F + (float(match_right) - F[-1]) * (xx - x0) / (x1 - x0)

    if self.use_gpu and _HAS_CUPY:
        F_result = cp.asnumpy(F).astype(np.float64)
    else:
        F_result = np.asarray(F, dtype=np.float64)

    return F_result, f_spline.astype(np.float64)


__all__ = ["antiderivative", "definite_integral"]
