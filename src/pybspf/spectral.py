"""FFT spectral helpers: wavenumbers, periodic Poisson solves, gradients.

Ports ``spectral_poisson_2d_uniform_precompute`` /
``spectral_poisson_2d_uniform_with_grad_cached``,
``poisson_fft_periodic_2d_zero_mean`` (+ complex variant), and
``periodic_gradient_2d_real`` / ``..._complex``.

The integer wavenumber ordering matches MATLAB exactly
(``[0:floor(N/2), -ceil(N/2)+1:-1]``), which differs from
``numpy.fft.fftfreq`` at the Nyquist mode for even ``N``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def matlab_freqs(N: int) -> np.ndarray:
    """Integer FFT wavenumbers in MATLAB's ordering."""
    pos = np.arange(0, int(np.floor(N / 2)) + 1, dtype=float)
    neg = np.arange(-int(np.ceil(N / 2)) + 1, 0, dtype=float)
    return np.concatenate([pos, neg])


def _wavenumber_grids(Nx, Ny, Lx, Ly):
    kx = (2 * np.pi / Lx) * matlab_freqs(Nx)
    ky = (2 * np.pi / Ly) * matlab_freqs(Ny)
    KX, KY = np.meshgrid(kx, ky)  # (Ny, Nx)
    return KX, KY


@dataclass
class FFTCache:
    Nx: int
    Ny: int
    KX: np.ndarray
    KY: np.ndarray
    K2: np.ndarray
    mask: np.ndarray


def spectral_poisson_2d_uniform_precompute(Nx, Ny, Lx, Ly) -> FFTCache:
    KX, KY = _wavenumber_grids(Nx, Ny, Lx, Ly)
    K2 = KX**2 + KY**2
    return FFTCache(Nx=Nx, Ny=Ny, KX=KX, KY=KY, K2=K2, mask=(K2 != 0))


def spectral_poisson_2d_uniform_with_grad_cached(f, cache: FFTCache):
    """Solve ``-lap Phi = f`` (zero-mean) on the periodic grid, return Phi, Phi_x, Phi_y."""
    f = np.asarray(f, dtype=float)
    Fhat = np.fft.fft2(f)
    Phi_hat = np.zeros_like(Fhat)
    Phi_hat[cache.mask] = -Fhat[cache.mask] / cache.K2[cache.mask]
    Phi_hat[0, 0] = 0.0
    Phi = np.real(np.fft.ifft2(Phi_hat))
    Phix = np.real(np.fft.ifft2(1j * cache.KX * Phi_hat))
    Phiy = np.real(np.fft.ifft2(1j * cache.KY * Phi_hat))
    return Phi, Phix, Phiy


def poisson_fft_periodic_2d_zero_mean(F, Px, Py) -> np.ndarray:
    """Real zero-mean periodic Poisson solve (``poisson_fft_periodic_2d_zero_mean``)."""
    F = np.asarray(F, dtype=float)
    Ny, Nx = F.shape
    Fhat = np.fft.fft2(F)
    Fhat[0, 0] = 0.0
    KX, KY = _wavenumber_grids(Nx, Ny, Px, Py)
    K2 = KX**2 + KY**2
    Vhat = np.zeros_like(Fhat)
    mask = K2 > 0
    Vhat[mask] = -Fhat[mask] / K2[mask]
    return np.real(np.fft.ifft2(Vhat))


def poisson_fft_periodic_2d_zero_mean_complex(F, Px, Py) -> np.ndarray:
    """Complex zero-mean periodic Poisson solve (``..._complex``)."""
    F = np.asarray(F)
    Ny, Nx = F.shape
    Fhat = np.fft.fft2(F)
    Fhat[0, 0] = 0.0
    KX, KY = _wavenumber_grids(Nx, Ny, Px, Py)
    K2 = KX**2 + KY**2
    Uhat = np.zeros_like(Fhat)
    mask = K2 > 0
    Uhat[mask] = -Fhat[mask] / K2[mask]
    return np.fft.ifft2(Uhat)


def periodic_gradient_2d_real(U, Px, Py):
    U = np.asarray(U, dtype=float)
    Ny, Nx = U.shape
    KX, KY = _wavenumber_grids(Nx, Ny, Px, Py)
    Uhat = np.fft.fft2(U)
    Ux = np.real(np.fft.ifft2(1j * KX * Uhat))
    Uy = np.real(np.fft.ifft2(1j * KY * Uhat))
    return Ux, Uy


def periodic_gradient_2d_complex(U, Px, Py):
    U = np.asarray(U)
    Ny, Nx = U.shape
    KX, KY = _wavenumber_grids(Nx, Ny, Px, Py)
    Uhat = np.fft.fft2(U)
    Ux = np.fft.ifft2(1j * KX * Uhat)
    Uy = np.fft.ifft2(1j * KY * Uhat)
    return Ux, Uy
