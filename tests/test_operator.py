"""! @file test_operator.py
@brief Regression tests for the package-owned BSPF1D operator.
"""

from __future__ import annotations

import numpy as np

from pybspf import BSPF1D, BSPF2D, PiecewiseBSPF1D
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

    old_d1, old_d2, old_fs2 = old_op.differentiate_1_2(f, lam=0.01)

    multi = new_op.derivatives(f, orders=(1, 2), lam=0.01)
    np.testing.assert_allclose(multi[1], old_d1)
    np.testing.assert_allclose(multi[2], old_d2)
    np.testing.assert_allclose(multi.spline, old_fs2)

    new_P, new_fit, new_residual = new_op.fit_spline(f, lam=0.01)
    old_P, old_fit, old_residual = old_op.fit_spline(f, lam=0.01)
    np.testing.assert_allclose(new_P, old_P)
    np.testing.assert_allclose(new_fit, old_fit)
    np.testing.assert_allclose(new_residual, old_residual)


def test_multi_order_derivatives_reuse_shared_path_for_nonconsecutive_orders():
    """! @brief The generalized API should support arbitrary supported order sets."""
    x, op, _ = _make_operator_pair()
    f = np.sin(x) + 0.15 * np.cos(2.0 * x) - 0.1 * np.sin(5.0 * x)

    d1, fs1 = op.differentiate(f, k=1, lam=0.01)
    d2, fs2 = op.differentiate(f, k=2, lam=0.01)
    d3, fs3 = op.differentiate(f, k=3, lam=0.01)

    result_13 = op.derivatives(f, orders=(1, 3), lam=0.01)
    np.testing.assert_allclose(result_13[1], d1)
    np.testing.assert_allclose(result_13[3], d3)
    np.testing.assert_allclose(result_13.spline, fs1)

    result_23 = op.derivatives(f, orders=(2, 3), lam=0.01)
    np.testing.assert_allclose(result_23[2], d2)
    np.testing.assert_allclose(result_23[3], d3)
    np.testing.assert_allclose(result_23.spline, fs2)
    np.testing.assert_allclose(fs1, fs2)
    np.testing.assert_allclose(fs2, fs3)


def test_fourth_order_derivative_matches_polynomial_shape():
    """! @brief Fourth-order differentiation should be available through the shared API."""
    x = np.linspace(0.0, 2.0 * np.pi, 257)
    op = BSPF1D.from_grid(degree=6, x=x, use_clustering=True, clustering_factor=2.0)
    f = np.sin(2.0 * x) + 0.25 * np.cos(3.0 * x)
    expected_d4 = 16.0 * np.sin(2.0 * x) + 20.25 * np.cos(3.0 * x)

    d4, fs = op.differentiate(f, k=4, lam=1.0e-6)
    result = op.derivatives(f, orders=(2, 4), lam=1.0e-6)

    np.testing.assert_allclose(result[4], d4)
    np.testing.assert_allclose(result.spline, fs)
    np.testing.assert_allclose(d4[8:-8], expected_d4[8:-8], atol=1.5e-1, rtol=1.0e-2)


def test_batched_multi_order_derivatives_match_columnwise_calls():
    """! @brief Batched generalized differentiation should match per-column evaluation."""
    x, op, _ = _make_operator_pair()
    batch = np.stack(
        [
            np.sin(x),
            np.cos(2.0 * x) + 0.1 * x,
            np.sin(3.0 * x) - 0.2 * np.cos(x),
        ],
        axis=1,
    )

    batched = op.derivatives_batched(batch, orders=(1, 4), lam=0.01)
    for idx in range(batch.shape[1]):
        single = op.derivatives(batch[:, idx], orders=(1, 4), lam=0.01)
        np.testing.assert_allclose(batched[1][:, idx], single[1])
        np.testing.assert_allclose(batched[4][:, idx], single[4])
        np.testing.assert_allclose(batched.spline[:, idx], single.spline)


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

    old_d1, old_d2, old_fs = old_pw.differentiate_1_2(f, lam=0.01)
    new_result = new_pw.derivatives(f, orders=(1, 2), lam=0.01)

    np.testing.assert_allclose(new_result[1], old_d1)
    np.testing.assert_allclose(new_result[2], old_d2)
    np.testing.assert_allclose(new_result.spline, old_fs)


def test_piecewise_generalized_derivatives_match_wrapper_for_shared_orders():
    """! @brief Piecewise generalized derivatives should return the expected shapes."""
    x = np.linspace(0.0, 1.0, 96)
    f = np.where(x < 0.45, np.sin(6.0 * x), np.sin(6.0 * x) + 1.5)
    pw = PiecewiseBSPF1D(degree=5, x=x, breakpoints=[0.45], min_points_per_seg=20)

    generalized = pw.derivatives(f, orders=(1, 2), lam=0.01)

    assert generalized[1].shape == f.shape
    assert generalized[2].shape == f.shape
    assert generalized.spline.shape == f.shape


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


def test_bspf2d_axis_derivatives_match_separable_analytical_field():
    """! @brief The 2D facade should reproduce separable derivatives along each axis."""
    x = np.linspace(0.0, 2.0 * np.pi, 65)
    y = np.linspace(0.0, 1.0, 49)
    xx, yy = np.meshgrid(x, y)
    field = np.sin(xx) + yy**3

    op = BSPF2D.from_grids(
        degree_x=5,
        degree_y=5,
        x=x,
        y=y,
        use_clustering_x=True,
        use_clustering_y=True,
        clustering_factor_x=2.0,
        clustering_factor_y=2.0,
    )

    dx_res = op.derivatives_axis(field, axis=1, orders=(1, 2), lam=1.0e-6)
    dy_res = op.derivatives_axis(field, axis=0, orders=(1, 2), lam=1.0e-6)
    lap = op.laplacian(field, lam_x=1.0e-6, lam_y=1.0e-6)

    expected_dx = np.cos(xx)
    expected_dxx = -np.sin(xx)
    expected_dy = 3.0 * yy**2
    expected_dyy = 6.0 * yy

    np.testing.assert_allclose(dx_res[1][6:-6, 6:-6], expected_dx[6:-6, 6:-6], atol=8.0e-2, rtol=1.0e-2)
    np.testing.assert_allclose(dx_res[2][6:-6, 6:-6], expected_dxx[6:-6, 6:-6], atol=1.2e-1, rtol=1.0e-2)
    np.testing.assert_allclose(dy_res[1][6:-6, 6:-6], expected_dy[6:-6, 6:-6], atol=1.2e-1, rtol=2.0e-2)
    np.testing.assert_allclose(dy_res[2][6:-6, 6:-6], expected_dyy[6:-6, 6:-6], atol=2.0e-1, rtol=2.0e-2)
    np.testing.assert_allclose(lap[6:-6, 6:-6], (expected_dxx + expected_dyy)[6:-6, 6:-6], atol=2.5e-1, rtol=2.0e-2)
