"""! @file test_knots.py
@brief Regression tests for the extracted knot-generation helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybspf.grid import Grid1D
from pybspf.knots import _Knot
from bspf1d import Grid1D as LegacyGrid1D
from bspf1d import _Knot as LegacyKnot


def test_knot_generate_matches_legacy_without_clustering():
    """! @brief Generated uncluttered knots should match legacy output."""
    new_knots = _Knot._generate(
        degree=5,
        domain=(-1.0, 2.0),
        n_basis=12,
        use_clustering=False,
        clustering_factor=2.0,
    )
    old_knots = LegacyKnot._generate(
        degree=5,
        domain=(-1.0, 2.0),
        n_basis=12,
        use_clustering=False,
        clustering_factor=2.0,
    )

    np.testing.assert_allclose(new_knots, old_knots)


def test_knot_generate_matches_legacy_with_clustering():
    """! @brief Generated clustered knots should match legacy output."""
    new_knots = _Knot._generate(
        degree=4,
        domain=(0.0, 1.0),
        n_basis=11,
        use_clustering=True,
        clustering_factor=3.5,
    )
    old_knots = LegacyKnot._generate(
        degree=4,
        domain=(0.0, 1.0),
        n_basis=11,
        use_clustering=True,
        clustering_factor=3.5,
    )

    np.testing.assert_allclose(new_knots, old_knots)


def test_knot_resolve_matches_legacy_defaults():
    """! @brief Default knot resolution should remain byte-for-byte compatible."""
    x = np.linspace(0.0, 2.0, 33)
    grid = Grid1D(x)
    legacy_grid = LegacyGrid1D(x)

    new_knots = _Knot.resolve(
        degree=3,
        grid=grid,
        knots=None,
        n_basis=None,
        domain=None,
        use_clustering=False,
        clustering_factor=2.0,
    )
    old_knots = LegacyKnot.resolve(
        degree=3,
        grid=legacy_grid,
        knots=None,
        n_basis=None,
        domain=None,
        use_clustering=False,
        clustering_factor=2.0,
    )

    np.testing.assert_allclose(new_knots, old_knots)


def test_knot_resolve_rejects_non_1d_explicit_knots():
    """! @brief Explicit knots must remain one-dimensional."""
    grid = Grid1D(np.linspace(0.0, 1.0, 9))

    with pytest.raises(ValueError, match="1D array"):
        _Knot.resolve(
            degree=3,
            grid=grid,
            knots=np.zeros((2, 2), dtype=np.float64),
            n_basis=None,
            domain=None,
            use_clustering=False,
            clustering_factor=2.0,
        )
