"""! @file grid.py
@brief Uniform one-dimensional grid abstractions.
"""

from __future__ import annotations

import numpy as np

from .backend import _HAS_CUPY, cp, is_cupy_array, normalize_backend_array, validate_backend_array
from .types import Array


class Grid1D:
    """! @brief Uniform 1D grid with rFFT frequencies and trapezoid weights.

    @param x Sample coordinates on a uniform 1D mesh.
    @param atol Absolute tolerance used for the uniform-spacing check.
    @param use_gpu Whether the grid should store data on the GPU.
    """

    def __init__(self, x: Array, *, atol: float = 1e-13, use_gpu: bool = False):
        # Detect the array backend from the input so we can enforce that the
        # caller's ``use_gpu`` flag matches the actual storage location.
        is_gpu_array = is_cupy_array(x)
        validate_backend_array(x, use_gpu=use_gpu, name="Grid1D")
        x = normalize_backend_array(x, use_gpu=use_gpu, dtype=np.float64, name="Grid1D")

        if x.size < 2:
            raise ValueError("x must have at least 2 points.")

        # Store ``dx`` as a Python float because downstream code expects scalar
        # arithmetic to behave identically on CPU and GPU paths.
        dx = float(x[1] - x[0])

        if is_gpu_array:
            if not cp.allclose(cp.diff(x), dx, rtol=0, atol=atol):
                raise ValueError("x must be uniformly spaced.")
        elif not np.allclose(np.diff(x), dx, rtol=0, atol=atol):
            raise ValueError("x must be uniformly spaced.")

        self.x: Array = x
        self.dx: float = dx
        self.use_gpu: bool = use_gpu

        if is_gpu_array:
            # Use rFFT frequencies because the current operators target
            # real-valued sampled data on a uniform mesh.
            self.omega: Array = 2.0 * cp.pi * cp.fft.rfftfreq(x.size, d=dx)
            w = cp.full(x.size, dx, dtype=cp.float64)
            w[0] = w[-1] = dx / 2.0
        else:
            self.omega = 2.0 * np.pi * np.fft.rfftfreq(x.size, d=dx)
            w = np.full(x.size, dx, dtype=np.float64)
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
