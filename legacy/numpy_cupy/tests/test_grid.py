"""! @file test_grid.py
@brief Regression tests for the extracted Grid1D implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybspf.grid import Grid1D
from bspf1d import Grid1D as LegacyGrid1D


def test_grid_matches_legacy_metadata():
    """! @brief The extracted grid should match the legacy implementation exactly."""
    x = np.linspace(-2.0, 3.0, 17)

    new_grid = Grid1D(x)
    old_grid = LegacyGrid1D(x)

    assert new_grid.n == old_grid.n == x.size
    assert new_grid.dx == old_grid.dx
    assert new_grid.a == old_grid.a
    assert new_grid.b == old_grid.b
    np.testing.assert_allclose(new_grid.x, old_grid.x)
    np.testing.assert_allclose(new_grid.omega, old_grid.omega)
    np.testing.assert_allclose(new_grid.trap, old_grid.trap)


def test_grid_rejects_nonuniform_spacing():
    """! @brief Nonuniform input grids must still be rejected."""
    x = np.array([0.0, 0.5, 1.1, 1.5], dtype=np.float64)

    with pytest.raises(ValueError, match="uniformly spaced"):
        Grid1D(x)


def test_grid_rejects_too_few_points():
    """! @brief A valid grid needs at least two sample points."""
    with pytest.raises(ValueError, match="at least 2 points"):
        Grid1D(np.array([0.0], dtype=np.float64))
