"""! @file test_operator.py
@brief Regression tests for the package-owned BSPF1D operator.
"""

from __future__ import annotations

import numpy as np

from pybspf import BSPF1D, PiecewiseBSPF1D
from bspf1d import PiecewiseBSPF1D as LegacyPiecewiseBSPF1D
from bspf1d import bspf1d as LegacyBSPF1D


def _make_operator_pair():
    """! @brief Build matching package and legacy operators for regression tests."""
    x = np.linspace(0.0, 2.0 * np.pi, 65)
    degree = 5

    new_op = BSPF1D.from_grid(degree=degree, x=x, use_clustering=True, clustering_factor=2.0)
    old_op = LegacyBSPF1D.from_grid(degree=degree, x=x, use_clustering=True, clustering_factor=2.0)
    return x, new_op, old_op


def test_package_operator_is_not_legacy_subclass():
    """! @brief The package operator should no longer inherit from the legacy class."""
    assert BSPF1D is not LegacyBSPF1D
    assert not issubclass(BSPF1D, LegacyBSPF1D)
    assert BSPF1D.differentiate is not LegacyBSPF1D.differentiate
    assert BSPF1D.fit_spline is not LegacyBSPF1D.fit_spline
    assert BSPF1D.differentiate.__module__ == "pybspf.ops.differentiation"
    assert BSPF1D.fit_spline.__module__ == "pybspf.ops.interpolation"


def test_fit_and_differentiate_match_legacy():
    """! @brief Package fitting and differentiation should match legacy results."""
    x, new_op, old_op = _make_operator_pair()
    f = np.sin(x) + 0.2 * np.cos(3.0 * x) + 0.1 * x

    new_df, new_fs = new_op.differentiate(f, k=1, lam=0.01)
    old_df, old_fs = old_op.differentiate(f, k=1, lam=0.01)
    np.testing.assert_allclose(new_df, old_df)
    np.testing.assert_allclose(new_fs, old_fs)

    new_d1, new_d2, new_fs2 = new_op.differentiate_1_2(f, lam=0.01)
    old_d1, old_d2, old_fs2 = old_op.differentiate_1_2(f, lam=0.01)
    np.testing.assert_allclose(new_d1, old_d1)
    np.testing.assert_allclose(new_d2, old_d2)
    np.testing.assert_allclose(new_fs2, old_fs2)

    new_P, new_fit, new_residual = new_op.fit_spline(f, lam=0.01)
    old_P, old_fit, old_residual = old_op.fit_spline(f, lam=0.01)
    np.testing.assert_allclose(new_P, old_P)
    np.testing.assert_allclose(new_fit, old_fit)
    np.testing.assert_allclose(new_residual, old_residual)


def test_integral_and_antiderivative_match_legacy():
    """! @brief Package integration helpers should match legacy results."""
    x, new_op, old_op = _make_operator_pair()
    f = np.cos(2.0 * x) + 0.25 * np.sin(4.0 * x) + 0.05

    new_int = new_op.definite_integral(f, lam=0.02)
    old_int = old_op.definite_integral(f, lam=0.02)
    assert np.isclose(new_int, old_int)

    new_F, new_fs = new_op.antiderivative(f, order=1, left_value=1.25, lam=0.02)
    old_F, old_fs = old_op.antiderivative(f, order=1, left_value=1.25, lam=0.02)
    np.testing.assert_allclose(new_F, old_F)
    np.testing.assert_allclose(new_fs, old_fs)


def test_piecewise_wrapper_matches_legacy():
    """! @brief Package piecewise differentiation should match the legacy wrapper."""
    x = np.linspace(0.0, 1.0, 96)
    f = np.where(x < 0.45, np.sin(6.0 * x), np.sin(6.0 * x) + 1.5)

    new_pw = PiecewiseBSPF1D(degree=5, x=x, breakpoints=[0.45], min_points_per_seg=20)
    old_pw = LegacyPiecewiseBSPF1D(degree=5, x=x, breakpoints=[0.45], min_points_per_seg=20)

    new_d1, new_d2, new_fs = new_pw.differentiate_1_2(f, lam=0.01)
    old_d1, old_d2, old_fs = old_pw.differentiate_1_2(f, lam=0.01)

    np.testing.assert_allclose(new_d1, old_d1)
    np.testing.assert_allclose(new_d2, old_d2)
    np.testing.assert_allclose(new_fs, old_fs)


def test_piecewise_segments_use_package_operator_and_expected_ranges():
    """! @brief Piecewise segmentation should build package operators on expected slices."""
    x = np.linspace(0.0, 1.0, 21)
    pw = PiecewiseBSPF1D(degree=3, x=x, breakpoints=[0.3, 0.7], min_points_per_seg=4)

    assert len(pw.segments) == 3
    assert [seg["i0"] for seg in pw.segments] == [0, 6, 14]
    assert [seg["i1"] for seg in pw.segments] == [5, 13, 20]
    assert all(isinstance(seg["op"], BSPF1D) for seg in pw.segments)


def test_piecewise_skips_segments_shorter_than_threshold():
    """! @brief Segments shorter than the minimum size should be omitted."""
    x = np.linspace(0.0, 1.0, 21)
    pw = PiecewiseBSPF1D(degree=3, x=x, breakpoints=[0.05, 0.5], min_points_per_seg=5)

    assert len(pw.segments) == 2
    assert [seg["i0"] for seg in pw.segments] == [1, 10]
    assert [seg["i1"] for seg in pw.segments] == [9, 20]
