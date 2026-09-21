"""Physical-domain data, original BSPF inverse, and near-wall regression tests."""

import jax
import numpy as np
import pytest

from pybspf.tensor import tensor_elliptic_solve
from bspf_models.elliptic.immersed_poisson import EllipticHole
from bspf_models.elliptic.immersed_poisson import ImmersedPoissonPlan

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def plan():
    # N=17 is the supported underresolved baseline; its original rectangular
    # endpoint construction does not yet reproduce this quartic to roundoff.
    return ImmersedPoissonPlan(nodes=25)


def polynomial(points):
    x, y = np.asarray(points).T
    value = (1 - x * x) * (1 - y * y)
    gradient = np.column_stack((-2 * x * (1 - y * y), -2 * y * (1 - x * x)))
    return value, gradient, 4 - 2 * (x * x + y * y)


def test_exact_geometry():
    hole = EllipticHole()
    t = np.linspace(0, 4, 71)
    normal, tangent = hole.normal(t), hole.curve(t, 1)
    np.testing.assert_allclose(hole.level(hole.curve(t)), 1, atol=2e-15)
    np.testing.assert_allclose(np.sum(normal * tangent, axis=1), 0, atol=2e-15)
    np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1, atol=2e-15)
    assert np.all(hole.level(hole.curve(t) + 1e-4 * normal) > 1)
    with pytest.raises(ValueError):
        EllipticHole(axes=(0.0, 0.2))
    with pytest.raises(ValueError):
        ImmersedPoissonPlan(hole=EllipticHole(center=(0.95, 0)))


def test_physical_data_only_and_polynomial(plan):
    def forcing(points):
        assert np.all(plan.hole.level(points) > 1)
        return polynomial(points)[2]

    def wall(points):
        np.testing.assert_allclose(plan.hole.level(points), 1, atol=2e-14)
        return polynomial(points)[0]

    solution = plan.solve(forcing, wall)
    boundary, t = plan.arc.sample(74, offset=0.319)
    points = np.vstack(
        (boundary, boundary + 1e-6 * plan.hole.normal(t), [[-0.7, 0.4], [0.5, 0.8]])
    )
    value, gradient, laplace = solution.evaluate(points)
    exact, dexact, forcing = polynomial(points)
    np.testing.assert_allclose(value, exact, atol=2e-9)
    np.testing.assert_allclose(gradient, dexact, atol=2e-8)
    np.testing.assert_allclose(-laplace, forcing, atol=2e-7)
    assert solution.diagnostics["physical_data_only"]
    assert solution.diagnostics["training_relative"] < 2e-9


def test_rectangular_inverse_grid_and_reuse(plan):
    solution = plan.solve(lambda p: polynomial(p)[2], lambda p: polynomial(p)[0])
    shape = plan.denominator.shape
    rhs = np.random.default_rng(31).normal(size=shape)
    np.testing.assert_allclose(
        tensor_elliptic_solve(rhs, plan.denominator).ravel(), rhs.ravel() / plan.scale
    )
    # Non-square output grid catches x/y transposition errors.
    x, y = np.linspace(-1, 1, 19), np.linspace(-1, 1, 23)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    values = solution.grid(x, y)
    paired = solution.evaluate(points)
    for grid, point in zip(values, paired):
        np.testing.assert_allclose(grid.reshape(point.shape), point, atol=2e-13)
    assert np.max(abs(values[0][[0, -1]])) < 1e-13
    assert np.max(abs(values[0][:, [0, -1]])) < 1e-13
    factor = plan.left
    doubled = plan.solve(lambda p: 2 * polynomial(p)[2], lambda p: 2 * polynomial(p)[0])
    assert plan.left is factor
    np.testing.assert_allclose(
        doubled.coefficients, 2 * solution.coefficients, atol=2e-13
    )
    old_misses = plan._cached_factors.cache_info().misses
    doubled.grid(x, y)
    assert plan._cached_factors.cache_info().misses == old_misses
    with pytest.raises(ValueError, match="background"):
        solution.evaluate(np.array([[1.01, 0.0]]))
    with pytest.raises(ValueError, match="one value"):
        plan.solve(lambda p: 1.0, lambda p: np.zeros(len(p)))


def test_resolved_nonentire_poisson_near_wall():
    # This reference log is singular INSIDE the hole and is never passed there
    # to the solver. Correct outer zero trace comes from the polynomial bubble.
    plan = ImmersedPoissonPlan(nodes=49)

    def exact(p):
        b, db, fb = polynomial(p)
        d = p - plan.hole.center
        r2 = np.sum(d * d, axis=1)
        w, dw = 0.5 * np.log(r2), d / r2[:, None]
        return (
            b * w,
            b[:, None] * dw + w[:, None] * db,
            w * fb - 2 * np.sum(db * dw, axis=1),
        )

    def forcing(p):
        assert np.all(plan.hole.level(p) > 1)
        return exact(p)[2]

    solution = plan.solve(forcing, lambda p: exact(p)[0])
    boundary, t = plan.arc.sample(150, offset=0.437)
    points = boundary + 1e-6 * plan.hole.normal(t)
    value, gradient, laplace = solution.evaluate(points)
    u, g, f = exact(points)
    assert np.max(abs(value - u)) < 4e-6
    assert np.max(np.linalg.norm(gradient - g, axis=1)) < 3e-4
    assert np.max(abs(laplace + f)) < 2e-2
    center_value = solution.evaluate(np.array([plan.hole.center]))[0]
    assert np.all(np.isfinite(center_value))
    assert abs(center_value[0]) < 10
