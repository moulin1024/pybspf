"""! @file test_boundary.py
@brief Regression tests for the extracted endpoint operator implementation.
"""

from __future__ import annotations

import numpy as np

from pybspf.basis import BSplineBasis1D
from pybspf.boundary import EndpointOps1D
from pybspf.grid import Grid1D
from pybspf.knots import _Knot
from bspf1d import BSplineBasis1D as LegacyBSplineBasis1D
from bspf1d import EndpointOps1D as LegacyEndpointOps1D
from bspf1d import Grid1D as LegacyGrid1D
from bspf1d import _Knot as LegacyKnot


def _make_endpoint_pair():
    """! @brief Build matching new and legacy endpoint operators."""
    x = np.linspace(0.0, 2.0, 25)
    degree = 4
    order = 3
    num_bd = 4

    grid = Grid1D(x)
    legacy_grid = LegacyGrid1D(x)
    knots = _Knot.resolve(
        degree=degree,
        grid=grid,
        knots=None,
        n_basis=12,
        domain=None,
        use_clustering=False,
        clustering_factor=2.0,
    )
    legacy_knots = LegacyKnot.resolve(
        degree=degree,
        grid=legacy_grid,
        knots=None,
        n_basis=12,
        domain=None,
        use_clustering=False,
        clustering_factor=2.0,
    )

    basis = BSplineBasis1D(degree=degree, knots=knots, grid=grid, use_gpu=False)
    legacy_basis = LegacyBSplineBasis1D(degree=degree, knots=legacy_knots, grid=legacy_grid, use_gpu=False)

    new_ops = EndpointOps1D(basis, order=order, num_bd=num_bd)
    old_ops = LegacyEndpointOps1D(legacy_basis, order=order, num_bd=num_bd)
    return new_ops, old_ops


def test_endpoint_constraint_matrix_matches_legacy():
    """! @brief Endpoint constraint matrix C should match the legacy implementation."""
    new_ops, old_ops = _make_endpoint_pair()

    np.testing.assert_allclose(new_ops.C, old_ops.C)


def test_endpoint_boundary_operator_matches_legacy():
    """! @brief Sample-to-endpoint derivative operator should match the legacy implementation."""
    new_ops, old_ops = _make_endpoint_pair()

    np.testing.assert_allclose(new_ops.BND, old_ops.BND)
    np.testing.assert_allclose(new_ops.X_left, old_ops.X_left)
    np.testing.assert_allclose(new_ops.X_right, old_ops.X_right)


def test_endpoint_operator_shapes_are_consistent():
    """! @brief The extracted endpoint operators should have the expected shapes."""
    ops, _ = _make_endpoint_pair()

    assert ops.C.shape == (2 * ops.order, 12)
    assert ops.BND.shape == (2 * ops.order, ops.grid.n)
    assert ops.X_left.shape == (ops.order, ops.num_bd)
    assert ops.X_right.shape == (ops.order, ops.num_bd)
