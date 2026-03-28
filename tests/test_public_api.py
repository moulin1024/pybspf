"""! @file test_public_api.py
@brief Smoke tests for the scaffolded public API.
"""

from __future__ import annotations

import numpy as np

from pybspf import BSPF1D, Grid1D, PiecewiseBSPF1D, bspf1d, integrate_rk4


def test_public_api_exports():
    """! @brief Verify the top-level package exports the expected symbols."""
    assert BSPF1D is bspf1d
    assert Grid1D.__name__ == "Grid1D"
    assert PiecewiseBSPF1D.__name__ == "PiecewiseBSPF1D"
    assert callable(integrate_rk4)


def test_bspf1d_from_grid_smoke():
    """! @brief Ensure the wrapped operator can still be constructed from a grid."""
    x = np.linspace(0.0, 1.0, 16)
    op = BSPF1D.from_grid(degree=3, x=x)
    assert isinstance(op, BSPF1D)
    assert op.grid.n == x.size


def test_piecewise_constructor_smoke():
    """! @brief Ensure the piecewise wrapper remains constructible."""
    x = np.linspace(0.0, 1.0, 32)
    op = PiecewiseBSPF1D(degree=3, x=x, breakpoints=[0.5])
    assert len(op.segments) >= 1
