"""! @file test_basis.py
@brief Regression tests for the extracted B-spline basis implementation.
"""

from __future__ import annotations

import numpy as np

from pybspf.basis import BSplineBasis1D
from pybspf.grid import Grid1D
from pybspf.knots import _Knot
from bspf1d import BSplineBasis1D as LegacyBSplineBasis1D
from bspf1d import Grid1D as LegacyGrid1D
from bspf1d import _Knot as LegacyKnot


def _make_basis_pair():
    """! @brief Build matching new and legacy basis objects for regression tests."""
    x = np.linspace(-1.0, 1.0, 21)
    degree = 3

    grid = Grid1D(x)
    legacy_grid = LegacyGrid1D(x)
    knots = _Knot.resolve(
        degree=degree,
        grid=grid,
        knots=None,
        n_basis=10,
        domain=None,
        use_clustering=True,
        clustering_factor=2.5,
    )
    legacy_knots = LegacyKnot.resolve(
        degree=degree,
        grid=legacy_grid,
        knots=None,
        n_basis=10,
        domain=None,
        use_clustering=True,
        clustering_factor=2.5,
    )

    new_basis = BSplineBasis1D(degree=degree, knots=knots, grid=grid, use_gpu=False)
    old_basis = LegacyBSplineBasis1D(degree=degree, knots=legacy_knots, grid=legacy_grid, use_gpu=False)
    return new_basis, old_basis


def test_basis_matches_legacy_on_grid():
    """! @brief The extracted basis matrix should match the legacy implementation."""
    new_basis, old_basis = _make_basis_pair()

    np.testing.assert_allclose(new_basis.B0, old_basis.B0)
    np.testing.assert_allclose(new_basis.BT0, old_basis.BT0)


def test_basis_derivative_matrices_match_legacy():
    """! @brief Derivative basis matrices should match the legacy implementation."""
    new_basis, old_basis = _make_basis_pair()

    np.testing.assert_allclose(new_basis.BkT(1), old_basis.BkT(1))
    np.testing.assert_allclose(new_basis.BkT(2), old_basis.BkT(2))


def test_basis_integrals_match_legacy():
    """! @brief Basis integrals should remain compatible with the legacy implementation."""
    new_basis, old_basis = _make_basis_pair()

    np.testing.assert_allclose(new_basis.integrate_basis(-0.25, 0.75), old_basis.integrate_basis(-0.25, 0.75))


def test_basis_derivative_cache_reuses_result():
    """! @brief Repeated derivative requests should reuse the cached matrix object."""
    basis, _ = _make_basis_pair()

    first = basis.BkT(1)
    second = basis.BkT(1)

    assert first is second
