"""! @file boundary.py
@brief Endpoint constraint helpers for the BSPF package.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import linalg as sla

from .backend import _HAS_CUPY, cp, is_cupy_array
from .basis import BSplineBasis1D
from .types import Array


class EndpointOps1D:
    """! @brief Endpoint constraints and sample-to-endpoint operators.

    @param basis B-spline basis used by the operator.
    @param order Number of endpoint derivatives to constrain.
    @param num_bd Number of sample points used near each boundary.
    """

    def __init__(self, basis: BSplineBasis1D, *, order: int, num_bd: int):
        self.order = int(order)
        self.num_bd = int(num_bd)
        self.grid = basis.grid
        self.use_gpu = basis.use_gpu

        # Use the same backend as the basis matrices so all assembled operators
        # remain on one device without implicit data movement.
        if self.use_gpu and _HAS_CUPY and is_cupy_array(basis.B0):
            xp = cp
            la_solve = cp.linalg.solve
        else:
            xp = np
            la_solve = sla.solve

        B0 = basis.B0
        Bk = {0: B0}
        for k in range(1, order + 1):
            Bk[k] = basis.BkT(k).T

        n_basis, n_points = B0.shape

        # Assemble the map from spline coefficients to endpoint derivatives.
        C = xp.zeros((2 * order, n_basis), dtype=xp.float64)
        for p in range(order):
            C[p, :] = Bk[p][:, 0]
            C[order + p, :] = Bk[p][:, -1]

        # The boundary stencil algebra is built from a small Vandermonde-like
        # system on equally spaced sample points near each endpoint.
        if self.use_gpu and _HAS_CUPY:
            i_np, j_np = np.meshgrid(np.arange(num_bd), np.arange(num_bd), indexing="ij")
            i = cp.asarray(i_np)
            j = cp.asarray(j_np)
        else:
            i, j = np.meshgrid(np.arange(num_bd), np.arange(num_bd), indexing="ij")

        fact = xp.array([math.factorial(k) for k in range(num_bd)], dtype=xp.float64)
        A_left = (j**i) / fact[:, None]
        A_right = xp.flip(A_left * ((-1.0) ** i), axis=(0, 1))

        E_left = xp.eye(num_bd, dtype=xp.float64)[:order, :].T
        idx = xp.arange(num_bd - 1, num_bd - order - 1, -1)
        E_right = xp.eye(num_bd, dtype=xp.float64)[idx, :].T

        X_left = la_solve(A_left, E_left).T
        X_right = la_solve(A_right, E_right).T

        # Scale the finite-difference-like endpoint stencils by powers of the
        # grid spacing so they approximate physical derivatives.
        if self.use_gpu and _HAS_CUPY:
            dx_pows = xp.asarray(self.grid.dx ** np.arange(order, dtype=np.float64))
        else:
            dx_pows = self.grid.dx ** np.arange(order, dtype=np.float64)

        BND = xp.zeros((2 * order, n_points), dtype=xp.float64)
        BND[:order, :num_bd] = X_left / dx_pows[:, None]
        BND[order:, n_points - num_bd:] = X_right / dx_pows[:, None]

        self.C: Array = C.astype(xp.float64)
        self.BND: Array = BND.astype(xp.float64)
        self.X_left: Array = X_left.astype(xp.float64)
        self.X_right: Array = X_right.astype(xp.float64)


__all__ = ["EndpointOps1D"]
