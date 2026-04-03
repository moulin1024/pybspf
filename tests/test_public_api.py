"""! @file test_public_api.py
@brief Smoke tests for the scaffolded public API.
"""

from __future__ import annotations

import numpy as np

from pybspf import (
    BSPF1D,
    BSPF2D,
    DerivativeResult,
    Grid1D,
    PiecewiseBSPF1D,
    Poisson1DDirichletSolver,
    Poisson2DDirichletSolver,
    bspf1d,
    bspf2d,
    integrate_rk4,
)


def test_public_api_exports():
    """! @brief Verify the top-level package exports the expected symbols."""
    assert BSPF1D is bspf1d
    assert BSPF2D is bspf2d
    assert DerivativeResult.__name__ == "DerivativeResult"
    assert Grid1D.__name__ == "Grid1D"
    assert PiecewiseBSPF1D.__name__ == "PiecewiseBSPF1D"
    assert Poisson1DDirichletSolver.__name__ == "Poisson1DDirichletSolver"
    assert Poisson2DDirichletSolver.__name__ == "Poisson2DDirichletSolver"
    assert callable(integrate_rk4)


def test_bspf1d_from_grid_smoke():
    """! @brief Ensure the wrapped operator can still be constructed from a grid."""
    x = np.linspace(0.0, 1.0, 16)
    op = BSPF1D.from_grid(degree=3, x=x)
    assert isinstance(op, BSPF1D)
    assert op.grid.n == x.size
    assert not hasattr(op, "differentiate_1_2")
    assert not hasattr(op, "differentiate_1_2_3")
    assert not hasattr(op, "differentiate_1_2_batched")


def test_bspf2d_from_grids_smoke():
    """! @brief Ensure the 2D wrapper can be constructed from orthogonal grids."""
    x = np.linspace(0.0, 1.0, 16)
    y = np.linspace(-1.0, 1.0, 12)
    op = BSPF2D.from_grids(degree_x=3, degree_y=3, x=x, y=y)
    assert isinstance(op, BSPF2D)
    assert op.x.shape == x.shape
    assert op.y.shape == y.shape


def test_piecewise_constructor_smoke():
    """! @brief Ensure the piecewise wrapper remains constructible."""
    x = np.linspace(0.0, 1.0, 32)
    op = PiecewiseBSPF1D(degree=3, x=x, breakpoints=[0.5])
    assert len(op.segments) >= 1
    assert not hasattr(op, "differentiate_1_2")
