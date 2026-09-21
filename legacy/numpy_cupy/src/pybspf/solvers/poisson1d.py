"""! @file solvers/poisson1d.py
@brief Direct 1D Poisson solver based on BSPF antiderivatives plus a linear patch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..operators.bspf1d import BSPF1D


@dataclass
class Poisson1DDirichletSolver:
    """Direct solver for ``-u'' = rhs`` on a 1D grid with Dirichlet data."""

    x: np.ndarray
    model: BSPF1D

    @classmethod
    def from_grid(
        cls,
        *,
        x: np.ndarray,
        degree: int = 10,
        knots: Optional[np.ndarray] = None,
        n_basis: Optional[int] = None,
        domain: Optional[tuple[float, float]] = None,
        use_clustering: bool = False,
        clustering_factor: float = 2.0,
    ) -> "Poisson1DDirichletSolver":
        """Construct the 1D solver from a uniform grid."""
        x_arr = np.asarray(x, dtype=np.float64)
        model = BSPF1D.from_grid(
            degree=degree,
            x=x_arr,
            knots=knots,
            n_basis=n_basis,
            domain=domain,
            use_clustering=use_clustering,
            clustering_factor=clustering_factor,
            use_gpu=False,
        )
        return cls(x=x_arr, model=model)

    def solve(
        self,
        rhs: np.ndarray,
        *,
        u_left: float,
        u_right: float,
        lam: float = 0.0,
        return_curvature: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Solve ``-u'' = rhs`` with Dirichlet boundary values.

        The solver first builds a particular solution of ``u'' = -rhs`` by a
        double antiderivative, then adds the homogeneous linear patch that
        matches the requested endpoint values.
        """
        rhs_arr = np.asarray(rhs, dtype=np.float64)
        if rhs_arr.shape != self.x.shape:
            raise ValueError(f"rhs shape {rhs_arr.shape} must match grid shape {self.x.shape}.")

        x0 = float(self.x[0])
        x1 = float(self.x[-1])
        length = x1 - x0
        if length <= 0.0:
            raise ValueError("The grid domain length must be positive.")

        curvature = -rhs_arr
        u_particular, curvature_spline = self.model.antiderivative(
            curvature,
            order=2,
            left_value=0.0,
            match_right=None,
            lam=lam,
        )

        c0 = float(u_left) - float(u_particular[0])
        slope = (float(u_right) - float(u_particular[-1]) - c0) / length
        solution = u_particular + slope * (self.x - x0) + c0
        solution = np.asarray(solution, dtype=np.float64)

        if not return_curvature:
            return solution
        return solution, np.asarray(curvature_spline, dtype=np.float64)


__all__ = ["Poisson1DDirichletSolver"]
