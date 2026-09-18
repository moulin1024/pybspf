"""Checks of the unchanged physical discretization and the new solver algebra."""

import jax
import numpy as np
import pytest
import scipy.linalg as la

from bspf_jax.convex_poisson import box_h2_root, trace_transform
from bspf_jax.convex_poisson_tensor import (
    TensorConvexPoissonPlan,
    PoissonIterationError,
    trace_adjoint,
)
from bspf_jax.embedded_poisson import benchmark_domains

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    return TensorConvexPoissonPlan(
        benchmark_domains()[0],
        nodes=17,
        coarse_nodes=17,
        volume_order=8,
        boundary_count=128,
        regularization=1e-8,
        coarse_rcond=1e-13,
        cache_dir=tmp_path_factory.mktemp("tensor_cache"),
    )


def explicit_data_matrix(p):
    # Ordinary paired Kronecker rows, independent of the grouped contractions.
    x, xx = p.bx[p.index], p.hx[p.index]
    lap = -(
        np.einsum("pi,pj->pij", xx, p.by) + np.einsum("pi,pj->pij", x, p.hy)
    ).reshape(len(p.points), -1)
    physical = np.empty_like(lap)
    physical[p.order] = lap
    trace = np.einsum("pi,pj->pij", p.edge_x, p.edge_y).reshape(p.boundary_count, -1)
    return np.vstack(
        (
            p.sqrt_weights[:, None] * physical,
            trace_transform(trace, p.arc.length, p.sobolev_order),
        )
    )


def test_same_quadrature_and_exact_adjoint(plan):
    p, w, _ = plan.domain.volume_rule(8)
    np.testing.assert_array_equal(plan.points, p)
    np.testing.assert_array_equal(plan.weights, w)
    matrix = explicit_data_matrix(plan)
    rng = np.random.default_rng(17)
    c = rng.normal(size=plan.ndofs)
    z = rng.normal(size=plan.data_rows)
    np.testing.assert_allclose(plan.operator @ c, matrix @ c, rtol=2e-11, atol=2e-10)
    np.testing.assert_allclose(
        plan.operator.rmatvec(z), matrix.T @ z, rtol=2e-11, atol=2e-10
    )
    np.testing.assert_allclose(
        plan.operator.matmat(np.column_stack((c, 2 * c))),
        matrix @ np.column_stack((c, 2 * c)),
        rtol=2e-11,
        atol=4e-10,
    )
    np.testing.assert_allclose(
        plan.operator.rmatmat(np.column_stack((z, 2 * z))),
        matrix.T @ np.column_stack((z, 2 * z)),
        rtol=2e-11,
        atol=4e-10,
    )
    a, y = rng.normal(size=plan.ndofs), rng.normal(size=plan.augmented.shape[0])
    np.testing.assert_allclose(
        np.dot(plan.augmented @ a, y),
        np.dot(a, plan.augmented.rmatvec(y)),
        rtol=2e-12,
        atol=2e-12,
    )
    edge = rng.normal(size=128)
    e = rng.normal(size=130)
    for order in (0.0, 1.5, 2.0):
        np.testing.assert_allclose(
            np.dot(trace_transform(edge, plan.arc.length, order), e),
            np.dot(edge, trace_adjoint(e, plan.arc.length, 128, order)),
            rtol=1e-12,
        )


def test_tensor_h2_equals_dense_cholesky(plan):
    rng = np.random.default_rng(81)
    c = rng.normal(size=(17, 17))
    root = box_h2_root(plan.line)
    np.testing.assert_allclose(
        plan.h2.gram(c).ravel(), root.T @ (root @ c.ravel()), rtol=2e-11, atol=1e-7
    )
    np.testing.assert_allclose(
        np.linalg.norm(plan.h2.factor(c)) ** 2,
        np.linalg.norm(root @ c.ravel()) ** 2,
        rtol=2e-11,
    )
    np.testing.assert_allclose(
        plan.h2.adjoint(plan.h2.factor(c)).ravel(),
        root.T @ (root @ c.ravel()),
        rtol=2e-10,
        atol=1e-6,
    )
    # Coarse correction is a projection of the FINE augmented operator.
    for j in (0, len(plan.coarse_singular) // 2, len(plan.coarse_singular) - 1):
        np.testing.assert_allclose(
            plan.augmented @ plan.coarse_right[:, j],
            plan.coarse_left[:, j] * plan.coarse_singular[j],
            atol=2e-12,
            rtol=2e-9,
        )


def test_regularized_coarse_solve_against_independent_dense_svd(plan):
    def exact(p):
        return 1 + 0.2 * p[:, 0] + 0.3 * p[:, 1] + 0.1 * p[:, 0] ** 2

    def f(p):
        return np.full(len(p), -0.2)

    solution = plan.solve(f, exact)
    assert solution.diagnostics["converged"]
    assert (
        solution.diagnostics["iterations"] == 0
    )  # Coarse span is the whole small space.
    A = explicit_data_matrix(plan)
    rhs = np.r_[
        plan.sqrt_weights * f(plan.points),
        trace_transform(exact(plan.boundary), plan.arc.length),
    ]
    R = box_h2_root(plan.line)
    # A different square-root factor of the SAME regularized objective.
    matrix = np.vstack(
        (
            la.solve_triangular(R.T, A.T, lower=True).T,
            plan.regularization * np.eye(plan.ndofs),
        )
    )
    a = la.lstsq(
        matrix, np.r_[rhs, np.zeros(plan.ndofs)], cond=1e-14, lapack_driver="gelsd"
    )[0]
    reference = la.solve_triangular(R, a)
    points = np.array([[0, 0], [-0.5, 0.2], [0.6, -0.1], [0.1, 0.6]])
    from bspf_jax.smooth_extension import evaluate_field

    np.testing.assert_allclose(
        solution.evaluate(points)[0],
        evaluate_field(plan.line, points, reference)[0],
        atol=2e-9,
    )
    np.testing.assert_allclose(solution.evaluate(points)[0], exact(points), atol=2e-8)
    zero = plan.solve(lambda p: np.zeros(len(p)), lambda p: np.zeros(len(p)))
    np.testing.assert_array_equal(zero.coefficients, 0)


def test_unconverged_iterates_are_not_silently_accepted(plan):
    # Keep the physical operator fixed; disable only the coarse acceleration.
    from copy import copy

    p = copy(plan)
    p.coarse_left = np.empty((p.augmented.shape[0], 0))
    p.coarse_right = np.empty((p.ndofs, 0))
    p.coarse_singular = np.empty(0)
    p.maxiter = 1
    with pytest.raises(PoissonIterationError) as error:
        p.solve(lambda x: np.sin(7 * x[:, 0]), lambda x: np.cos(3 * x[:, 1]))
    assert error.value.diagnostics["stop_code"] == 7
    assert not error.value.diagnostics["converged"]


def test_persistent_cache_and_fixed_grid(tmp_path, monkeypatch):
    from bspf_jax import ConvexPoissonGridPlan
    from bspf_jax.convex_poisson_tensor import _line

    settings = dict(
        nodes=17,
        coarse_nodes=17,
        volume_order=4,
        boundary_count=64,
        regularization=1e-7,
        coarse_rcond=1e-13,
        cache_dir=tmp_path,
        coarse_space="spectral",
    )
    domain = benchmark_domains()[0]
    first = TensorConvexPoissonPlan(domain, **settings)
    _line.cache_clear()  # Force the next construction to use the disk cache.

    def unexpected(*args, **kwargs):
        pytest.fail("A warm cache must not repeat MPFR assembly or coarse SVD")

    monkeypatch.setattr("bspf_jax.convex_poisson_tensor._stream_line", unexpected)
    monkeypatch.setattr(
        "bspf_jax.convex_poisson_tensor.stream_evaluate_line", unexpected
    )
    monkeypatch.setattr("bspf_jax.convex_poisson_tensor.la.svd", unexpected)
    warm = TensorConvexPoissonPlan(domain, **settings)
    assert warm.geometry_cache_hit and warm.coarse_cache_hit
    np.testing.assert_array_equal(warm.points, first.points)
    np.testing.assert_array_equal(warm.coarse_right, first.coarse_right)
    np.testing.assert_array_equal(warm.tx, first.tx)
    grid = ConvexPoissonGridPlan.from_plan(
        warm, np.linspace(-1.2, 1.2, 21), np.linspace(-1.2, 1.2, 25)
    )
    result = grid.solve(lambda p: np.zeros(len(p)), lambda p: np.ones(len(p)))
    assert result.diagnostics["backend"] == "tensor_two_level_lsmr"
    assert result.diagnostics["converged"]
    np.testing.assert_allclose(result.values[result.inside], 1, atol=2e-8)
    assert np.isnan(result.values[~result.inside]).all()


def test_explicit_physical_residual_target(plan):
    from copy import copy

    p = copy(plan)
    p.data_tolerance = 1e-6
    result = p.solve(lambda x: np.zeros(len(x)), lambda x: np.ones(len(x)))
    assert result.diagnostics["data_target_met"]
    assert result.diagnostics["iterations"] == 0
    assert result.diagnostics["convergence_criterion"] == "physical_data_residual"
    # A numerical least-squares stop must NOT overrule an unmet requested target.
    p.data_tolerance = 1e-16
    with pytest.raises(PoissonIterationError):
        p.solve(lambda x: np.zeros(len(x)), lambda x: np.ones(len(x)))
