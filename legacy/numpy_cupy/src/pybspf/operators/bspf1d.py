"""! @file operators/bspf1d.py
@brief Primary 1D BSPF operator implemented inside the package.

The constructor and core state assembly live in the package and compose the
grid, basis, boundary, correction, and KKT helpers. Operation methods are also
implemented inside the package modules.
"""

from __future__ import annotations

from typing import Optional, Tuple
from numbers import Integral

import numpy as np

from ..backend import _Backend, _HAS_CUPY, cp, is_cupy_array, normalize_backend_array
from ..basis import BSplineBasis1D
from ..boundary import EndpointOps1D
from ..grid import Grid1D
from ..kkt import KKTLUCache
from ..knots import _Knot
from ..ops.differentiation import (
    derivatives,
    derivatives_batched,
    differentiate,
)
from ..ops.integration import antiderivative, definite_integral
from ..ops.interpolation import enforced_zero_flux, fit_spline, interpolate, interpolate_split_mesh
from ..types import Array


class BSPF1D:
    """! @brief Facade for 1D BSPF operations on a uniform grid.

    @param grid Uniform grid object.
    @param degree B-spline degree.
    @param knots Knot vector for the spline basis.
    @param order Number of endpoint derivative constraints.
    @param num_boundary_points Number of boundary samples used in endpoint stencils.
    @param correction Residual correction strategy name.
    @param use_gpu Whether to use the GPU backend.
    """

    def __init__(
        self,
        *,
        grid: Grid1D,
        degree: int,
        knots: Array,
        order: Optional[int] = None,
        num_boundary_points: Optional[int] = None,
        correction: str = "spectral",
        use_gpu: bool = False,
    ):
        # Build the backend adapter first because all later arrays must obey the
        # same host/device contract.
        self.use_gpu = bool(use_gpu)
        self._bk = _Backend(self.use_gpu) if self.use_gpu else None

        self.grid = grid
        self.degree = int(degree)
        self.order = self.degree - 1 if order is None else int(order)
        self.num_bd = self.degree if num_boundary_points is None else int(num_boundary_points)

        if grid.use_gpu != self.use_gpu:
            raise ValueError("grid and operator must use the same backend.")
        if correction not in ("spectral", "none"):
            raise ValueError("correction must be 'spectral' or 'none'.")
        if self.degree < 1 or self.degree != degree:
            raise ValueError("degree must be a positive integer.")
        if not 0 <= self.order <= self.degree or (order is not None and self.order != order):
            raise ValueError("order must be an integer between 0 and degree.")
        if not max(1, self.order) <= self.num_bd <= grid.n:
            raise ValueError("num_boundary_points must be between max(1, order) and grid size.")

        self.knots = normalize_backend_array(
            knots,
            use_gpu=self.use_gpu,
            dtype=np.float64,
            name="BSPF1D knots",
        )

        # Compose the package-owned foundational objects extracted in earlier
        # phases instead of inheriting construction from the legacy monolith.
        self.basis = BSplineBasis1D(self.degree, self.knots, self.grid, use_gpu=self.use_gpu)
        trap = self.grid.trap

        self.BW = self.basis.B0 * trap
        self.Q = self.BW @ self.basis.B0.T
        self.end = EndpointOps1D(self.basis, order=self.order, num_bd=self.num_bd)

        # Cache a package-owned KKT factorization helper. ``_kkt_cache`` is kept
        # for compatibility because some tests and legacy-style method bodies
        # expect the attribute to exist.
        self._kkt_solver = KKTLUCache(self.Q, self.end.C, use_gpu=self.use_gpu)
        self._kkt_cache = self._kkt_solver._cache

        layout = (lambda a: a) if self.use_gpu else np.asfortranarray
        self._BW_f = layout(self.BW)
        self._BND_f = layout(self.end.BND)
        self._BT0_f = layout(self.basis.BT0)
        for k in range(1, 5):
            setattr(self, f"_B{k}T_f", layout(self.basis.BkT(k)))
        self.correction = correction

    @classmethod
    def from_grid(
        cls,
        degree: int,
        x: Array,
        *,
        knots: Optional[Array] = None,
        n_basis: Optional[int] = None,
        domain: Optional[Tuple[float, float]] = None,
        use_clustering: bool = False,
        clustering_factor: float = 2.0,
        order: Optional[int] = None,
        num_boundary_points: Optional[int] = None,
        correction: str = "spectral",
        use_gpu: bool = False,
    ) -> "BSPF1D":
        """! @brief Construct the operator from raw grid coordinates.

        @param degree B-spline degree.
        @param x Uniform grid coordinates.
        @param knots Optional explicit knot vector.
        @param n_basis Optional number of basis functions for generated knots.
        @param domain Optional knot-generation domain.
        @param use_clustering Whether generated knots should be boundary clustered.
        @param clustering_factor Strength of the clustering map.
        @param order Optional endpoint derivative constraint order.
        @param num_boundary_points Optional number of boundary stencil points.
        @param correction Residual correction strategy name.
        @param use_gpu Whether to create a GPU-backed operator.
        @return Configured ``BSPF1D`` instance.
        """
        if isinstance(degree, bool) or not isinstance(degree, Integral) or degree < 1:
            raise ValueError("degree must be a positive integer.")
        x = normalize_backend_array(x, use_gpu=use_gpu, dtype=None, name="BSPF1D x")

        grid = Grid1D(x, use_gpu=use_gpu)
        k = _Knot.resolve(
            degree=degree,
            grid=grid,
            knots=knots,
            n_basis=n_basis,
            domain=domain,
            use_clustering=use_clustering,
            clustering_factor=clustering_factor,
        )

        if use_gpu and _HAS_CUPY and not is_cupy_array(k):
            k = cp.asarray(k, dtype=cp.float64)
        elif not use_gpu:
            k = normalize_backend_array(k, use_gpu=False, dtype=np.float64, name="BSPF1D resolved knots")

        return cls(
            grid=grid,
            degree=degree,
            knots=k,
            order=order,
            num_boundary_points=num_boundary_points,
            correction=correction,
            use_gpu=use_gpu,
        )

    # Numerical implementations live in the operation modules.
    differentiate = differentiate
    derivatives = derivatives
    derivatives_batched = derivatives_batched
    definite_integral = definite_integral
    antiderivative = antiderivative
    enforced_zero_flux = enforced_zero_flux
    interpolate = interpolate
    fit_spline = fit_spline
    interpolate_split_mesh = interpolate_split_mesh

    def _kkt_lu(self, lam: float):
        """! @brief Return a cached KKT LU factorization for ``lam``."""
        return self._kkt_solver.factorize(lam)


# Preserve the original lowercase class name so older call sites continue to
# work while the package API is introduced.
bspf1d = BSPF1D

__all__ = ["BSPF1D", "bspf1d"]
