"""! @file operators/bspf2d.py
@brief Separable 2D BSPF operator built from two package-owned 1D operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from ..ops.differentiation import DerivativeResult
from ..types import Array
from .bspf1d import BSPF1D


@dataclass
class BSPF2D:
    """! @brief 2D facade composed from package-owned 1D operators."""

    x: Array
    y: Array
    x_model: BSPF1D
    y_model: BSPF1D
    use_gpu: bool = False

    @classmethod
    def from_grids(
        cls,
        *,
        x: Array,
        y: Array,
        degree_x: int = 10,
        degree_y: Optional[int] = None,
        knots_x: Optional[Array] = None,
        knots_y: Optional[Array] = None,
        n_basis_x: Optional[int] = None,
        n_basis_y: Optional[int] = None,
        domain_x: Optional[tuple[float, float]] = None,
        domain_y: Optional[tuple[float, float]] = None,
        use_clustering_x: bool = False,
        use_clustering_y: bool = False,
        clustering_factor_x: float = 2.0,
        clustering_factor_y: float = 2.0,
        order_x: Optional[int] = None,
        order_y: Optional[int] = None,
        num_boundary_points_x: Optional[int] = None,
        num_boundary_points_y: Optional[int] = None,
        correction: str = "spectral",
        use_gpu: bool = False,
    ) -> "BSPF2D":
        """! @brief Construct a separable 2D operator from orthogonal 1D grids."""
        if degree_y is None:
            degree_y = degree_x

        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        x_model = BSPF1D.from_grid(
            degree=degree_x,
            x=x_arr,
            knots=knots_x,
            n_basis=n_basis_x,
            domain=domain_x,
            use_clustering=use_clustering_x,
            clustering_factor=clustering_factor_x,
            order=order_x,
            num_boundary_points=num_boundary_points_x,
            correction=correction,
            use_gpu=use_gpu,
        )
        y_model = BSPF1D.from_grid(
            degree=degree_y,
            x=y_arr,
            knots=knots_y,
            n_basis=n_basis_y,
            domain=domain_y,
            use_clustering=use_clustering_y,
            clustering_factor=clustering_factor_y,
            order=order_y,
            num_boundary_points=num_boundary_points_y,
            correction=correction,
            use_gpu=use_gpu,
        )
        return cls(x=x_arr, y=y_arr, x_model=x_model, y_model=y_model, use_gpu=use_gpu)

    def _check_shape(self, field: Array) -> tuple[int, int]:
        """! @brief Validate that ``field`` has shape ``(len(y), len(x))``."""
        f_arr = np.asarray(field)
        if f_arr.ndim != 2:
            raise ValueError("F must be 2D with shape (len(y), len(x)).")
        ny, nx = f_arr.shape
        if ny != self.y.size or nx != self.x.size:
            raise ValueError(
                f"F shape {f_arr.shape} must match (len(y), len(x))=({self.y.size}, {self.x.size})."
            )
        return ny, nx

    def derivatives_axis(
        self,
        field: Array,
        *,
        axis: int,
        orders: int | Iterable[int],
        lam: float = 0.0,
        neumann_bc: Optional[tuple[Optional[float], Optional[float]]] = None,
    ) -> DerivativeResult:
        """! @brief Compute requested derivatives along one axis through batched 1D solves."""
        self._check_shape(field)
        f_arr = np.asarray(field)

        if axis == 1:
            result = self.x_model.derivatives_batched(
                f_arr.T,
                orders=orders,
                lam=lam,
                neumann_bc=neumann_bc,
            )
            return DerivativeResult(
                values={order: value.T for order, value in result.values.items()},
                spline=result.spline.T,
            )

        if axis == 0:
            return self.y_model.derivatives_batched(
                f_arr,
                orders=orders,
                lam=lam,
                neumann_bc=neumann_bc,
            )

        raise ValueError("axis must be 0 (y) or 1 (x).")

    def differentiate_axis(
        self,
        field: Array,
        *,
        axis: int,
        k: int = 1,
        lam: float = 0.0,
        neumann_bc: Optional[tuple[Optional[float], Optional[float]]] = None,
    ) -> tuple[Array, Array]:
        """! @brief Compute one derivative order along one axis."""
        result = self.derivatives_axis(field, axis=axis, orders=(k,), lam=lam, neumann_bc=neumann_bc)
        return result[k], result.spline

    def partial_x(
        self,
        field: Array,
        *,
        order: int = 1,
        lam: float = 0.0,
        neumann_bc: Optional[tuple[Optional[float], Optional[float]]] = None,
    ) -> tuple[Array, Array]:
        """! @brief Convenience wrapper for x-direction derivatives."""
        return self.differentiate_axis(field, axis=1, k=order, lam=lam, neumann_bc=neumann_bc)

    def partial_y(
        self,
        field: Array,
        *,
        order: int = 1,
        lam: float = 0.0,
        neumann_bc: Optional[tuple[Optional[float], Optional[float]]] = None,
    ) -> tuple[Array, Array]:
        """! @brief Convenience wrapper for y-direction derivatives."""
        return self.differentiate_axis(field, axis=0, k=order, lam=lam, neumann_bc=neumann_bc)

    def laplacian(
        self,
        field: Array,
        *,
        lam_x: float = 0.0,
        lam_y: float = 0.0,
        neumann_bc_x: Optional[tuple[Optional[float], Optional[float]]] = None,
        neumann_bc_y: Optional[tuple[Optional[float], Optional[float]]] = None,
    ) -> Array:
        """! @brief Return ``d2/dx2 + d2/dy2`` using the separable 1D operators."""
        dxx, _ = self.partial_x(field, order=2, lam=lam_x, neumann_bc=neumann_bc_x)
        dyy, _ = self.partial_y(field, order=2, lam=lam_y, neumann_bc=neumann_bc_y)
        return dxx + dyy


bspf2d = BSPF2D


__all__ = ["BSPF2D", "bspf2d"]
