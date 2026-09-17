"""! @file grid.py
@brief Uniform one-dimensional grid abstractions.
"""

from __future__ import annotations

import numpy as np

from .backend import get_array_module, normalize_backend_array, validate_backend_array
from .types import Array


class Grid1D:
    """! @brief Uniform 1D grid with rFFT frequencies and trapezoid weights.

    @param x Sample coordinates on a uniform 1D mesh.
    @param atol Absolute tolerance used for the uniform-spacing check.
    @param use_gpu Whether the grid should store data on the GPU.
    """

    def __init__(self, x: Array, *, atol: float = 1e-13, use_gpu: bool = False):
        xp = get_array_module(use_gpu=use_gpu)
        validate_backend_array(x, use_gpu=use_gpu, name="Grid1D")
        if not np.isfinite(atol) or atol < 0:
            raise ValueError("atol must be finite and nonnegative.")
        if xp.iscomplexobj(x):
            raise ValueError("x must contain real coordinates.")
        x = normalize_backend_array(x, use_gpu=use_gpu, dtype=np.float64, name="Grid1D")
        if x.ndim != 1:
            raise ValueError("x must be a 1D array.")
        if x.size < 2:
            raise ValueError("x must have at least 2 points.")
        if not bool(xp.all(xp.isfinite(x))):
            raise ValueError("x must contain only finite coordinates.")
        spacing = xp.diff(x)
        if not bool(xp.all(spacing > 0)):
            raise ValueError("x must be strictly increasing.")
        dx = float(spacing[0])
        if not bool(xp.allclose(spacing, dx, rtol=0, atol=atol)):
            raise ValueError("x must be uniformly spaced.")

        self.x: Array = x
        self.dx: float = dx
        self.use_gpu: bool = use_gpu
        self.omega: Array = 2.0 * xp.pi * xp.fft.rfftfreq(x.size, d=dx)
        w = xp.full(x.size, dx, dtype=xp.float64)
        w[0] = w[-1] = dx / 2.0
        self.trap: Array = w

    @property
    def a(self) -> float:
        """! @brief Left endpoint of the grid domain."""
        return float(self.x[0])

    @property
    def b(self) -> float:
        """! @brief Right endpoint of the grid domain."""
        return float(self.x[-1])

    @property
    def n(self) -> int:
        """! @brief Number of grid points."""
        return self.x.size


__all__ = ["Grid1D"]
