"""! @file test_kkt.py
@brief Regression tests for the extracted KKT assembly and solve helpers.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg as sla

from pybspf.basis import BSplineBasis1D
from pybspf.boundary import EndpointOps1D
from pybspf.grid import Grid1D
from pybspf.kkt import KKTLUCache, assemble_kkt_matrix
from pybspf.knots import _Knot
from bspf1d import bspf1d as LegacyBSPF1D


def _make_kkt_fixture():
    """! @brief Build a package-side KKT system alongside the legacy operator."""
    x = np.linspace(0.0, 1.0, 17)
    degree = 3

    grid = Grid1D(x)
    knots = _Knot.resolve(
        degree=degree,
        grid=grid,
        knots=None,
        n_basis=None,
        domain=None,
        use_clustering=False,
        clustering_factor=2.0,
    )
    basis = BSplineBasis1D(degree=degree, knots=knots, grid=grid, use_gpu=False)
    end = EndpointOps1D(basis, order=degree - 1, num_bd=degree)
    BW = basis.B0 * grid.trap
    Q = BW @ basis.B0.T

    legacy_op = LegacyBSPF1D.from_grid(degree=degree, x=x)
    return grid, basis, end, BW, Q, legacy_op


def test_kkt_matrix_matches_legacy_structure():
    """! @brief The package KKT matrix assembly should match the legacy operator."""
    _, _, end, _, Q, legacy_op = _make_kkt_fixture()
    lam = 0.125

    new_kkt = assemble_kkt_matrix(Q, end.C, lam, use_gpu=False)
    n_b = legacy_op.basis.B0.shape[0]
    m = 2 * legacy_op.order
    old_kkt = np.zeros((n_b + m, n_b + m), dtype=np.float64)
    old_kkt[:n_b, :n_b] = 2.0 * (legacy_op.Q + lam * np.eye(n_b, dtype=np.float64))
    old_kkt[:n_b, n_b:] = -legacy_op.end.C.T
    old_kkt[n_b:, :n_b] = legacy_op.end.C

    np.testing.assert_allclose(new_kkt, old_kkt)


def test_kkt_solve_matches_legacy_operator():
    """! @brief Solving the extracted KKT system should match the legacy operator."""
    grid, basis, end, BW, Q, legacy_op = _make_kkt_fixture()
    lam = 0.05
    f = np.sin(2.0 * np.pi * grid.x) + 0.2 * grid.x

    # Match the legacy right-hand-side construction exactly.
    rhs_top = 2.0 * (BW @ f)
    dY = end.BND @ f
    rhs = np.concatenate((rhs_top, dY))

    cache = KKTLUCache(Q, end.C, use_gpu=False)
    new_sol = cache.solve(rhs.copy(), lam, overwrite_b=True)

    lu_old, piv_old = legacy_op._kkt_lu(lam)
    old_sol = sla.lu_solve((lu_old, piv_old), rhs.copy(), overwrite_b=True)

    np.testing.assert_allclose(new_sol, old_sol)


def test_kkt_factorization_cache_reuses_objects():
    """! @brief Repeated factorization requests for the same lam should hit the cache."""
    _, _, end, _, Q, _ = _make_kkt_fixture()
    cache = KKTLUCache(Q, end.C, use_gpu=False)

    first = cache.factorize(0.1)
    second = cache.factorize(0.1)

    # The first call returns a fresh tuple, while later calls return the cached
    # tuple; the important contract is reuse of the stored LU/pivot arrays.
    assert first[0] is second[0]
    assert first[1] is second[1]
