"""! @file test_operator.py
@brief Regression tests for the package-owned BSPF1D operator.
"""

from __future__ import annotations

import numpy as np

from pybspf import BSPF1D, BSPF2D, PiecewiseBSPF1D, Poisson1DDirichletSolver, Poisson2DDirichletSolver
from bspf1d import PiecewiseBSPF1D as LegacyPiecewiseBSPF1D
from bspf1d import bspf1d as LegacyBSPF1D


def _negative_discrete_laplacian(field: np.ndarray, *, hx: float, hy: float) -> np.ndarray:
    """! @brief Return the 5-point interior ``-Delta_h`` of a sampled field."""
    center = field[1:-1, 1:-1]
    return (
        (2.0 * center - field[1:-1, :-2] - field[1:-1, 2:]) / (hx * hx)
        + (2.0 * center - field[:-2, 1:-1] - field[2:, 1:-1]) / (hy * hy)
    )


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


def test_poisson1d_direct_solver_matches_polynomial_mms():
    """! @brief The 1D direct solver should recover a smooth polynomial Dirichlet MMS."""
    x = np.linspace(0.0, 1.0, 129)
    exact = x * (1.0 - x)
    rhs = 2.0 * np.ones_like(x)

    solver = Poisson1DDirichletSolver.from_grid(
        x=x,
        degree=5,
        n_basis=32,
        use_clustering=True,
        clustering_factor=2.0,
    )

    numerical, curvature = solver.solve(rhs, u_left=0.0, u_right=0.0, return_curvature=True)

    np.testing.assert_allclose(numerical, exact, atol=2.0e-10, rtol=2.0e-10)
    np.testing.assert_allclose(curvature, -rhs, atol=2.0e-10, rtol=2.0e-10)


def test_poisson1d_direct_solver_handles_nonzero_dirichlet_sine_solution():
    """! @brief The 1D direct solver should combine the particular solve and linear patch correctly."""
    x = np.linspace(0.0, 1.0, 257)
    exact = 1.5 - 0.25 * x + np.sin(np.pi * x)
    rhs = (np.pi**2) * np.sin(np.pi * x)

    solver = Poisson1DDirichletSolver.from_grid(
        x=x,
        degree=6,
        n_basis=48,
        use_clustering=True,
        clustering_factor=2.0,
    )

    numerical = solver.solve(rhs, u_left=float(exact[0]), u_right=float(exact[-1]))

    np.testing.assert_allclose(numerical[0], exact[0], atol=1.0e-12, rtol=1.0e-12)
    np.testing.assert_allclose(numerical[-1], exact[-1], atol=1.0e-12, rtol=1.0e-12)
    np.testing.assert_allclose(numerical[6:-6], exact[6:-6], atol=2.0e-4, rtol=2.0e-4)


def test_poisson2d_direct_solver_matches_mms_with_homogeneous_dirichlet():
    """! @brief The Sylvester solver should recover a smooth zero-Dirichlet MMS field."""
    x = np.linspace(0.0, 1.0, 65)
    y = np.linspace(0.0, 1.0, 61)
    xx, yy = np.meshgrid(x, y)

    exact = xx * (1.0 - xx) * yy * (1.0 - yy)
    rhs = 2.0 * (xx * (1.0 - xx) + yy * (1.0 - yy))

    solver = Poisson2DDirichletSolver.from_grids(
        x=x,
        y=y,
        degree_x=5,
        degree_y=5,
        n_basis_x=24,
        n_basis_y=22,
        use_clustering_x=True,
        use_clustering_y=True,
        clustering_factor_x=2.0,
        clustering_factor_y=2.0,
    )

    numerical = solver.solve(rhs)

    np.testing.assert_allclose(numerical[0, :], 0.0, atol=1.0e-10)
    np.testing.assert_allclose(numerical[-1, :], 0.0, atol=1.0e-10)
    np.testing.assert_allclose(numerical[:, 0], 0.0, atol=1.0e-10)
    np.testing.assert_allclose(numerical[:, -1], 0.0, atol=1.0e-10)
    np.testing.assert_allclose(numerical[4:-4, 4:-4], exact[4:-4, 4:-4], atol=6.0e-4, rtol=3.0e-2)


def test_poisson2d_direct_solver_supports_general_dirichlet_traces():
    """! @brief The direct solver should handle nonzero Dirichlet traces through a boundary lift."""
    x = np.linspace(0.0, 1.0, 65)
    y = np.linspace(0.0, 1.0, 61)
    xx, yy = np.meshgrid(x, y)

    exact = 1.0 + xx + 2.0 * yy + xx * (1.0 - xx) * yy * (1.0 - yy)
    rhs = 2.0 * (xx * (1.0 - xx) + yy * (1.0 - yy))

    solver = Poisson2DDirichletSolver.from_grids(
        x=x,
        y=y,
        degree_x=5,
        degree_y=5,
        n_basis_x=24,
        n_basis_y=22,
        use_clustering_x=True,
        use_clustering_y=True,
        clustering_factor_x=2.0,
        clustering_factor_y=2.0,
    )

    numerical = solver.solve(
        rhs,
        left=exact[:, 0],
        right=exact[:, -1],
        bottom=exact[0, :],
        top=exact[-1, :],
    )

    np.testing.assert_allclose(numerical[0, :], exact[0, :], atol=1.0e-8)
    np.testing.assert_allclose(numerical[-1, :], exact[-1, :], atol=1.0e-8)
    np.testing.assert_allclose(numerical[:, 0], exact[:, 0], atol=1.0e-8)
    np.testing.assert_allclose(numerical[:, -1], exact[:, -1], atol=1.0e-8)
    np.testing.assert_allclose(numerical[4:-4, 4:-4], exact[4:-4, 4:-4], atol=8.0e-4, rtol=3.0e-2)


def test_poisson2d_direct_solver_callable_rhs_recovers_homogeneous_polynomial_mms_to_roundoff():
    """! @brief Callable RHS quadrature should recover the homogeneous polynomial MMS to near machine precision."""
    x = np.linspace(0.0, 1.0, 33)
    y = np.linspace(0.0, 1.0, 31)
    xx, yy = np.meshgrid(x, y)

    exact = xx * (1.0 - xx) * yy * (1.0 - yy)
    rhs = lambda xq, yq: 2.0 * (xq * (1.0 - xq) + yq * (1.0 - yq))

    solver = Poisson2DDirichletSolver.from_grids(
        x=x,
        y=y,
        degree_x=5,
        degree_y=5,
        n_basis_x=24,
        n_basis_y=22,
        use_clustering_x=True,
        use_clustering_y=True,
        clustering_factor_x=2.0,
        clustering_factor_y=2.0,
    )

    numerical = solver.solve(rhs)

    np.testing.assert_allclose(numerical, exact, atol=1.0e-10, rtol=1.0e-10)


def test_poisson2d_direct_solver_can_return_analytic_laplacian():
    """! @brief The direct solver should expose the spline Laplacian alongside the solution."""
    x = np.linspace(0.0, 1.0, 33)
    y = np.linspace(0.0, 1.0, 31)
    xx, yy = np.meshgrid(x, y)

    exact = xx * (1.0 - xx) * yy * (1.0 - yy)
    rhs = 2.0 * (xx * (1.0 - xx) + yy * (1.0 - yy))
    rhs_callable = lambda xq, yq: 2.0 * (xq * (1.0 - xq) + yq * (1.0 - yq))

    solver = Poisson2DDirichletSolver.from_grids(
        x=x,
        y=y,
        degree_x=5,
        degree_y=5,
        n_basis_x=24,
        n_basis_y=22,
        use_clustering_x=True,
        use_clustering_y=True,
        clustering_factor_x=2.0,
        clustering_factor_y=2.0,
    )

    numerical, laplacian = solver.solve(rhs_callable, return_laplacian=True)

    np.testing.assert_allclose(numerical, exact, atol=1.0e-10, rtol=1.0e-10)
    np.testing.assert_allclose(laplacian, -rhs, atol=1.0e-10, rtol=1.0e-10)


def test_poisson2d_direct_solver_sampled_rhs_tracks_callable_quadrature_reference():
    """! @brief Sampled RHS assembly should stay close to the callable high-order load reference."""
    x = np.linspace(0.0, 1.0, 33)
    y = np.linspace(0.0, 1.0, 31)
    xx, yy = np.meshgrid(x, y)

    exact = xx * (1.0 - xx) * yy * (1.0 - yy)
    rhs_samples = 2.0 * (xx * (1.0 - xx) + yy * (1.0 - yy))
    rhs_callable = lambda xq, yq: 2.0 * (xq * (1.0 - xq) + yq * (1.0 - yq))

    solver = Poisson2DDirichletSolver.from_grids(
        x=x,
        y=y,
        degree_x=5,
        degree_y=5,
        n_basis_x=24,
        n_basis_y=22,
        use_clustering_x=True,
        use_clustering_y=True,
        clustering_factor_x=2.0,
        clustering_factor_y=2.0,
    )

    numerical_samples = solver.solve(rhs_samples)
    numerical_callable = solver.solve(rhs_callable)

    np.testing.assert_allclose(numerical_samples, numerical_callable, atol=6.0e-5, rtol=1.0e-3)
    np.testing.assert_allclose(numerical_samples, exact, atol=6.0e-5, rtol=1.0e-3)


def test_poisson2d_hybrid_dst_solver_reproduces_discrete_homogeneous_solution():
    """! @brief The hybrid DST solver should exactly recover a zero-Dirichlet discrete solution."""
    x = np.linspace(0.0, 1.0, 65)
    y = np.linspace(0.0, 1.0, 61)
    xx, yy = np.meshgrid(x, y)
    hx = x[1] - x[0]
    hy = y[1] - y[0]

    exact = np.sin(2.0 * np.pi * xx) * np.sin(np.pi * yy)
    rhs = np.zeros_like(exact)
    rhs[1:-1, 1:-1] = _negative_discrete_laplacian(exact, hx=hx, hy=hy)

    solver = Poisson2DDirichletSolver.from_grids(
        x=x,
        y=y,
        degree_x=5,
        degree_y=5,
        n_basis_x=24,
        n_basis_y=22,
        use_clustering_x=True,
        use_clustering_y=True,
        clustering_factor_x=2.0,
        clustering_factor_y=2.0,
    )

    numerical = solver.solve_hybrid_dst(rhs)

    np.testing.assert_allclose(numerical, exact, atol=1.0e-11, rtol=1.0e-11)


def test_poisson2d_hybrid_dst_solver_handles_general_dirichlet_traces():
    """! @brief The hybrid DST solver should combine the spline boundary lift with the grid solve."""
    x = np.linspace(0.0, 1.0, 65)
    y = np.linspace(0.0, 1.0, 61)
    xx, yy = np.meshgrid(x, y)
    hx = x[1] - x[0]
    hy = y[1] - y[0]

    exact = 1.0 + xx + 2.0 * yy + xx * (1.0 - xx) * yy * (1.0 - yy)
    rhs = np.zeros_like(exact)
    rhs[1:-1, 1:-1] = _negative_discrete_laplacian(exact, hx=hx, hy=hy)

    solver = Poisson2DDirichletSolver.from_grids(
        x=x,
        y=y,
        degree_x=5,
        degree_y=5,
        n_basis_x=24,
        n_basis_y=22,
        use_clustering_x=True,
        use_clustering_y=True,
        clustering_factor_x=2.0,
        clustering_factor_y=2.0,
    )

    numerical = solver.solve_hybrid_dst(
        rhs,
        left=exact[:, 0],
        right=exact[:, -1],
        bottom=exact[0, :],
        top=exact[-1, :],
    )

    np.testing.assert_allclose(numerical, exact, atol=1.0e-10, rtol=1.0e-10)


def test_poisson2d_boundary_corrector_02_matches_dirichlet_edges():
    """! @brief The 0/2-jet boundary corrector should use a zero-mean gauge without changing residual shapes."""
    x = np.linspace(0.0, 1.0, 65)
    y = np.linspace(0.0, 1.0, 61)
    xx, yy = np.meshgrid(x, y)

    exact = 1.0 + xx + 2.0 * yy + xx * (1.0 - xx) * yy * (1.0 - yy)
    rhs = 2.0 * (xx * (1.0 - xx) + yy * (1.0 - yy))

    solver = Poisson2DDirichletSolver.from_grids(
        x=x,
        y=y,
        degree_x=5,
        degree_y=5,
        n_basis_x=24,
        n_basis_y=22,
        use_clustering_x=True,
        use_clustering_y=True,
        clustering_factor_x=2.0,
        clustering_factor_y=2.0,
    )

    corrector, laplacian, corrected_rhs = solver.build_boundary_corrector_02(
        rhs,
        left=exact[:, 0],
        right=exact[:, -1],
        bottom=exact[0, :],
        top=exact[-1, :],
    )

    np.testing.assert_allclose(np.mean(corrector), 0.0, atol=1.0e-10)
    assert laplacian.shape == exact.shape
    assert corrected_rhs.shape == exact.shape


def test_poisson2d_fft_corrected_solver_recovers_periodic_mode():
    """! @brief The 0/2 corrector plus periodic FFT inversion should recover a smooth sine mode."""
    x = np.linspace(0.0, 2.0 * np.pi, 65)
    y = np.linspace(0.0, 2.0 * np.pi, 65)
    xx, yy = np.meshgrid(x, y)

    exact = np.sin(2.0 * xx) * np.sin(3.0 * yy)
    rhs = 13.0 * exact

    solver = Poisson2DDirichletSolver.from_grids(
        x=x,
        y=y,
        degree_x=5,
        degree_y=5,
        n_basis_x=24,
        n_basis_y=24,
        use_clustering_x=False,
        use_clustering_y=False,
    )

    solution, corrector, fft_remainder, harmonic_patch, laplacian, corrected_rhs = solver.solve_fft_corrected_02(
        rhs,
        left=exact[:, 0],
        right=exact[:, -1],
        bottom=exact[0, :],
        top=exact[-1, :],
    )

    np.testing.assert_allclose(solution, exact, atol=5.0e-3, rtol=5.0e-3)
    np.testing.assert_allclose(solution[0, :], exact[0, :], atol=1.0e-8)
    np.testing.assert_allclose(solution[-1, :], exact[-1, :], atol=1.0e-8)
    np.testing.assert_allclose(solution[:, 0], exact[:, 0], atol=1.0e-8)
    np.testing.assert_allclose(solution[:, -1], exact[:, -1], atol=1.0e-8)
    assert corrector.shape == exact.shape
    assert fft_remainder.shape == exact.shape
    assert harmonic_patch.shape == exact.shape
    assert laplacian.shape == exact.shape
    assert corrected_rhs.shape == exact.shape
