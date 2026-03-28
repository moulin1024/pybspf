"""! @file basis.py
@brief B-spline basis helpers for the BSPF package.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import BSpline

from .backend import _HAS_CUPY, cp, cp_interp, is_cupy_array, normalize_backend_array
from .grid import Grid1D
from .types import Array


class BSplineBasis1D:
    """! @brief B-spline basis on a uniform grid with lazy derivative matrices.

    @param degree Polynomial degree of the basis functions.
    @param knots Knot vector defining the basis.
    @param grid Uniform grid used for basis evaluation.
    @param use_gpu Whether the basis should be built using CuPy/CuPyX objects.
    """

    def __init__(self, degree: int, knots: Array, grid: Grid1D, use_gpu: bool = False):
        self.degree = int(degree)
        self.use_gpu = bool(use_gpu)

        # Keep the knot vector on the same backend as the operator so later
        # basis evaluations do not trigger implicit host/device conversions.
        self.knots: Array = normalize_backend_array(
            knots,
            use_gpu=use_gpu,
            dtype=np.float64,
            name="BSplineBasis1D knots",
        )

        self.grid = grid

        self._splines = self._mk_splines()
        n_basis = len(self._splines)

        # Pre-evaluate the basis on the operator grid because this matrix is
        # used repeatedly by fitting, differentiation, and interpolation paths.
        if use_gpu and _HAS_CUPY:
            xp = cp
            B0 = xp.empty((n_basis, grid.n), dtype=xp.float64)
            x_eval = grid.x
        else:
            xp = np
            B0 = xp.empty((n_basis, grid.n), dtype=xp.float64)
            x_eval = grid.x

        for i, spline in enumerate(self._splines):
            B0[i, :] = spline(x_eval)

        self._B0: Array = B0
        self._BT0: Array = B0.T.copy()

        # Cache derivative evaluations because several public methods need the
        # same basis derivative matrices on the original grid.
        self._BkT: Dict[int, Array] = {}
        self._eval_cache: Dict[Tuple[float, int], Array] = {}

    def _mk_splines(self):
        """! @brief Build one spline object per basis function.

        @return List of SciPy or CuPyX spline objects.
        @throws ValueError If CPU mode is asked to consume CuPy knot data.
        """
        n_basis = len(self.knots) - self.degree - 1

        if self.use_gpu and _HAS_CUPY:
            coeffs = cp.eye(n_basis, dtype=cp.float64)
            return [cp_interp.BSpline(self.knots, coeffs[i], self.degree) for i in range(n_basis)]

        if _HAS_CUPY and isinstance(self.knots, cp.ndarray):
            raise ValueError(
                "Cannot convert CuPy knots to NumPy in CPU mode. "
                "When use_gpu=False, provide NumPy arrays. "
                "Either: (1) convert knots to NumPy before creating operator, or (2) use use_gpu=True."
            )

        coeffs = np.eye(n_basis, dtype=np.float64)
        return [BSpline(self.knots, coeffs[i], self.degree) for i in range(n_basis)]

    def _evaluate_splines_vectorized(self, x: Array, deriv_order: int = 0) -> Array:
        """! @brief Evaluate the whole basis or one derivative order on a grid.

        @param x Evaluation points.
        @param deriv_order Derivative order to evaluate.
        @return Matrix with shape ``(n_basis, len(x))``.
        """
        # The code currently only evaluates on uniform grids, so the first
        # coordinate plus derivative order is enough to key the cache.
        cache_key = (float(x[0]), deriv_order)
        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]

        n_basis = len(self._splines)

        if self.use_gpu and _HAS_CUPY:
            xp = cp
            result = xp.empty((n_basis, len(x)), dtype=xp.float64)
            x_eval = cp.asarray(x) if isinstance(x, np.ndarray) else x
        else:
            xp = np
            result = xp.empty((n_basis, len(x)), dtype=xp.float64)
            if is_cupy_array(x):
                raise ValueError(
                    "Cannot convert CuPy array to NumPy in CPU mode. "
                    "When use_gpu=False, provide NumPy arrays. "
                    "Either: (1) convert input to NumPy before calling, or (2) use use_gpu=True."
                )
            x_eval = x

        for i, spline in enumerate(self._splines):
            result[i, :] = (spline.derivative(deriv_order) if deriv_order else spline)(x_eval)

        self._eval_cache[cache_key] = result
        return result

    @property
    def B0(self) -> Array:
        """! @brief Basis values evaluated on the operator grid."""
        return self._B0

    @property
    def BT0(self) -> Array:
        """! @brief Transpose of the basis matrix on the operator grid."""
        return self._BT0

    def BkT(self, k: int) -> Array:
        """! @brief Transpose of the ``k``th derivative basis matrix.

        @param k Derivative order.
        @return Matrix with shape ``(n_grid, n_basis)``.
        """
        if k == 0:
            return self._BT0
        if k > self.degree:
            n_basis = len(self._splines)
            if self.use_gpu and _HAS_CUPY:
                return cp.zeros((self.grid.n, n_basis), dtype=cp.float64)
            return np.zeros((self.grid.n, n_basis), dtype=np.float64)
        if k not in self._BkT:
            Bk = self._evaluate_splines_vectorized(self.grid.x, deriv_order=k)
            self._BkT[k] = Bk.T.copy()
        return self._BkT[k]

    def integrate_basis(self, a: float, b: float) -> Array:
        """! @brief Integrate each basis function on a physical interval.

        @param a Left integration bound.
        @param b Right integration bound.
        @return One integral per basis function.
        """
        # Basis integration is currently kept on CPU because SciPy's spline API
        # provides the reference behavior used by the legacy implementation.
        return np.array([spline.integrate(a, b) for spline in self._splines])


__all__ = ["BSplineBasis1D"]
