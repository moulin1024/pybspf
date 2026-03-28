"""! @file kkt.py
@brief KKT assembly, factorization, and solve helpers for constrained spline fits.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import linalg as sla

from .backend import _HAS_CUPY, cp, cpla, is_cupy_array
from .types import Array


def assemble_kkt_matrix(Q: Array, C: Array, lam: float, *, use_gpu: bool = False) -> Array:
    """! @brief Assemble the KKT matrix for the constrained spline fit.

    @param Q Weighted Gram-like matrix of the spline basis.
    @param C Endpoint constraint matrix.
    @param lam Tikhonov regularization parameter.
    @param use_gpu Whether to assemble the system on the GPU.
    @return Assembled KKT matrix.
    """
    if use_gpu and _HAS_CUPY and is_cupy_array(Q):
        xp = cp
    else:
        xp = np

    n_b = Q.shape[0]
    m = C.shape[0]
    KKT = xp.zeros((n_b + m, n_b + m), dtype=xp.float64)

    # Top-left block: regularized least-squares term.
    KKT[:n_b, :n_b] = 2.0 * (Q + lam * xp.eye(n_b, dtype=xp.float64))
    # Off-diagonal blocks: equality constraints and their multipliers.
    KKT[:n_b, n_b:] = -C.T
    KKT[n_b:, :n_b] = C
    return KKT


class KKTLUCache:
    """! @brief Cache LU factorizations of KKT systems keyed by ``lam``.

    @param Q Weighted Gram-like matrix of the spline basis.
    @param C Endpoint constraint matrix.
    @param use_gpu Whether factorizations should be computed on the GPU.
    """

    def __init__(self, Q: Array, C: Array, *, use_gpu: bool = False):
        self.Q = Q
        self.C = C
        self.use_gpu = bool(use_gpu)
        self._cache: Dict[float, Tuple[Array, Array]] = {}

    def factorize(self, lam: float) -> Tuple[Array, Array]:
        """! @brief Return an LU factorization for a specific regularization value.

        @param lam Tikhonov regularization parameter.
        @return Tuple ``(lu, piv)`` compatible with ``lu_solve``.
        """
        lam = float(lam)
        if lam in self._cache:
            return self._cache[lam]

        KKT = assemble_kkt_matrix(self.Q, self.C, lam, use_gpu=self.use_gpu)

        if self.use_gpu and _HAS_CUPY and is_cupy_array(KKT):
            lu, piv = cpla.lu_factor(KKT)
        else:
            if is_cupy_array(KKT):
                raise ValueError(
                    "Cannot convert CuPy array to NumPy in CPU mode. "
                    "When use_gpu=False, internal arrays should be NumPy."
                )
            lu, piv = sla.lu_factor(KKT)

        self._cache[lam] = (lu, piv)
        return lu, piv

    def solve(self, rhs: Array, lam: float, *, overwrite_b: bool = True) -> Array:
        """! @brief Solve the KKT system for a given right-hand side.

        @param rhs Right-hand side vector or matrix.
        @param lam Tikhonov regularization parameter.
        @param overwrite_b Forwarded to the LU solve implementation when supported.
        @return Solution with the same backend as the configured cache.
        """
        lu, piv = self.factorize(lam)

        if self.use_gpu and _HAS_CUPY and is_cupy_array(lu):
            rhs_dev = rhs if is_cupy_array(rhs) else cp.asarray(rhs)
            return cpla.lu_solve((lu, piv), rhs_dev, overwrite_b=overwrite_b)

        if is_cupy_array(rhs):
            raise ValueError(
                "Cannot solve a CPU KKT system with a CuPy right-hand side. "
                "Convert the RHS to NumPy or enable GPU mode."
            )
        return sla.lu_solve((lu, piv), rhs, overwrite_b=overwrite_b)


__all__ = ["KKTLUCache", "assemble_kkt_matrix"]
