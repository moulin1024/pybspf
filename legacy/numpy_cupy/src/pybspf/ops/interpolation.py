"""! @file ops/interpolation.py
@brief Package-owned interpolation and spline-fit workflows for BSPF1D.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import make_interp_spline

from ._common import solve_spline, prepare_samples
from ..types import Array


def enforced_zero_flux(self, f: Array) -> Tuple[float, float]:
    """Repair endpoint values to satisfy a zero-flux condition."""
    if self.use_gpu:
        raise ValueError("enforced_zero_flux currently supports only use_gpu=False.")
    f = prepare_samples(self, f)
    if np.iscomplexobj(f):
        raise ValueError("This operation currently requires real samples.")
    if f.shape[0] != self.grid.n:
        raise ValueError("Length of f must match grid size.")

    n_ghost = self.degree - 1
    if n_ghost < 1:
        raise ValueError(f"degree must be at least 2 for enforced_zero_flux (got {self.degree})")

    x = self.grid.x
    dx = self.grid.dx

    x_left = (x[0] - dx * np.arange(1, n_ghost + 1))[::-1]
    f_left = f[1 : 1 + n_ghost][::-1]
    x_right = x[-1] + dx * np.arange(1, n_ghost + 1)
    f_right = f[-1 - n_ghost : -1][::-1]

    x_extended = np.concatenate([x_left, x, x_right])
    f_extended = np.concatenate([f_left, f, f_right])

    boundary_idx_left = n_ghost
    x_boundary_left = x_extended[boundary_idx_left]
    mask_left = np.ones(len(x_extended), dtype=bool)
    mask_left[boundary_idx_left] = False
    try:
        spline_left = make_interp_spline(
            x_extended[mask_left], f_extended[mask_left], k=self.degree, bc_type="natural"
        )
    except Exception:
        spline_left = make_interp_spline(x_extended[mask_left], f_extended[mask_left], k=self.degree)
    f_left_corrected = float(spline_left(x_boundary_left))

    boundary_idx_right = -(n_ghost + 1)
    x_boundary_right = x_extended[boundary_idx_right]
    mask_right = np.ones(len(x_extended), dtype=bool)
    mask_right[boundary_idx_right] = False
    try:
        spline_right = make_interp_spline(
            x_extended[mask_right], f_extended[mask_right], k=self.degree, bc_type="natural"
        )
    except Exception:
        spline_right = make_interp_spline(x_extended[mask_right], f_extended[mask_right], k=self.degree)
    f_right_corrected = float(spline_right(x_boundary_right))

    return f_left_corrected, f_right_corrected


def fit_spline(
    self,
    f: Array,
    lam: float = 0.0,
    neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None,
):
    """Fit spline coefficients and return the fitted spline and residual."""
    return solve_spline(self, f, lam, neumann_bc)


def interpolate(self, f: Array, lam: float = 0.0, use_fft: bool = False):
    """Interpolate the signal onto a grid with inserted midpoints."""
    if use_fft:
        raise NotImplementedError("FFT interpolation is not implemented; use use_fft=False.")
    if self.use_gpu:
        raise ValueError("interpolate currently supports only use_gpu=False.")

    f = prepare_samples(self, f)
    if np.iscomplexobj(f):
        raise ValueError("This operation currently requires real samples.")
    if f.shape[0] != self.grid.n:
        raise ValueError(f"Length of f ({f.shape[0]}) must match grid size ({self.grid.n})")

    x_old = self.grid.x
    x_new = np.empty(2 * len(x_old) - 1, dtype=np.float64)
    x_new[::2] = x_old
    x_new[1::2] = 0.5 * (x_old[:-1] + x_old[1:])

    P, _f_spline, residual = fit_spline(self, f, lam=lam)

    residual_new = np.interp(x_new, x_old, residual)
    f_spline_new = self.basis._evaluate_splines_vectorized(x_new, deriv_order=0).T @ P
    return x_new, np.asarray(f_spline_new + residual_new, dtype=np.float64)


def interpolate_split_mesh(
    self,
    f: Array,
    refine_factor: int,
    lam: float = 0.0,
    neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None,
):
    """Interpolate onto an arbitrarily refined mesh with spline/residual split."""
    if self.use_gpu:
        raise ValueError("interpolate_split_mesh currently supports only use_gpu=False.")

    f = prepare_samples(self, f)
    if np.iscomplexobj(f):
        raise ValueError("This operation currently requires real samples.")
    N = self.grid.n
    if f.shape[0] != N:
        raise ValueError(f"Length of f ({f.shape[0]}) must match grid size ({N})")

    M = int(refine_factor)
    if M < 1 or M != refine_factor:
        raise ValueError("refine_factor must be a positive integer")

    P, _f_spline, residual = fit_spline(self, f, lam=lam, neumann_bc=neumann_bc)

    dx = self.grid.dx
    x0 = self.grid.a
    N_fine = M * (N - 1) + 1
    x_fine = x0 + dx * (np.arange(N_fine, dtype=np.float64) / M)
    f_spline_fine = self.basis._evaluate_splines_vectorized(x_fine, deriv_order=0).T @ P
    r_fine = np.interp(x_fine, self.grid.x, residual)
    f_fine = f_spline_fine + r_fine
    return x_fine, f_fine, f_spline_fine, r_fine


__all__ = [
    "enforced_zero_flux",
    "fit_spline",
    "interpolate",
    "interpolate_split_mesh",
]
