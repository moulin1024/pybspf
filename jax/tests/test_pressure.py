"""JAX masked pressure projection against the NumPy extraction and invariants."""

from functools import partial
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bspf_jax as b

# Cross-backend references are test-only; the JAX package has no pybspf dependency.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pybspf import PressurePoisson2D


@pytest.fixture(scope="module", params=["taylor", "chebyshev"])
def setup(request):
    options = dict(q=2, n_basis=5, degree=4, baseline_points=4)
    if request.param == "chebyshev":
        options.update(
            endpoint_method="chebyshev", baseline_points=6, chebyshev_modes=5
        )
    x, y = np.linspace(-1, 2, 7), np.linspace(0, 2, 8)
    return b.plan_pressure_poisson2d(x, y, **options), PressurePoisson2D(
        x, y, **options
    )


def test_numpy_equivalence_and_jit(setup):
    plan, reference = setup
    raw = np.random.default_rng(5).normal(size=plan.mask.shape + (2,))
    expected_v, expected = reference.project(raw.swapaxes(0, 1))
    eager_v, eager = b.project_pressure2d(plan, raw)
    v, result = jax.jit(b.project_pressure2d)(plan, raw)
    assert bool(result.converged)
    np.testing.assert_allclose(v, expected_v.swapaxes(0, 1), atol=1e-9)
    np.testing.assert_allclose(result.pressure, expected.pressure.T, atol=1e-9)
    np.testing.assert_allclose(v, eager_v, atol=1e-10)
    np.testing.assert_allclose(result.pressure, eager.pressure, atol=1e-10)
    assert float(jnp.max(abs(b.pressure_divergence(plan, v)))) < 1e-9
    np.testing.assert_array_equal(np.asarray(v)[np.asarray(plan.mask) == 0], 0)
    assert abs(float(jnp.sum(plan.weights * result.pressure))) < 1e-12


def test_scalar_solve_completion_and_idempotence(setup):
    plan, _ = setup
    p = jax.random.normal(jax.random.key(2), plan.mask.shape)
    gradient = b.pressure_gradient(plan, p)
    result = jax.jit(b.solve_pressure_poisson2d)(
        plan, b.pressure_schur(plan, p), wall_gradient=gradient
    )
    assert bool(result.converged)
    np.testing.assert_allclose(
        result.pressure, b.pressure_remove_mean(plan, p), atol=1e-9
    )
    raw = jax.random.normal(jax.random.key(3), gradient.shape)
    v, _ = b.project_pressure2d(plan, raw)
    vv, _ = b.project_pressure2d(plan, v)
    v0, result0 = jax.jit(partial(b.project_pressure2d, completion=False))(plan, raw)
    assert bool(result0.converged)
    assert bool(jnp.isnan(result0.wall_gradient_fit_linf))
    np.testing.assert_allclose(vv, v, atol=1e-9)
    np.testing.assert_allclose(v0, v, atol=1e-9)


def test_vmap_and_reverse_mode(setup):
    plan, _ = setup
    raw = jax.random.normal(jax.random.key(4), plan.mask.shape + (2,))
    direction = jax.random.normal(jax.random.key(6), raw.shape)
    batch = jnp.stack([raw, 2 * raw])
    v, results = jax.jit(jax.vmap(b.project_pressure2d, in_axes=(None, 0)))(plan, batch)
    assert bool(jnp.all(results.converged))
    np.testing.assert_allclose(v[1], 2 * v[0], atol=1e-10)
    np.testing.assert_allclose(results.pressure[1], 2 * results.pressure[0], atol=1e-10)

    def loss(field):
        projected, _ = b.project_pressure2d(plan, field)
        return jnp.sum(projected**2)

    grad = jax.jit(jax.grad(loss))(raw)
    projected_direction, _ = b.project_pressure2d(plan, direction)
    np.testing.assert_allclose(
        jnp.vdot(grad, direction),
        2 * jnp.vdot(v[0], projected_direction),
        rtol=1e-9,
        atol=1e-8,
    )


def test_failure_flags_under_jit(setup):
    plan, _ = setup
    solve = jax.jit(b.solve_pressure_poisson2d)
    zero = jnp.zeros_like(plan.mask)
    assert bool(solve(plan, zero).converged)
    for bad in [zero.at[0, 0].set(1), zero.at[2, 3].set(1), zero.at[1, 1].set(jnp.nan)]:
        assert not bool(solve(plan, bad).converged)
    assert not bool(solve(plan, zero, atol=-1.0).converged)
    assert not bool(solve(plan, zero, rtol=jnp.nan).converged)
    with pytest.raises(ValueError, match="shape"):
        solve(plan, jnp.zeros((3, 3)))
    with pytest.raises(ValueError, match="real"):
        solve(plan, zero.astype(complex))


def test_chebyshev_improvement_and_numpy_at_production_order():
    x, y = np.linspace(0, 1, 64), np.linspace(0, 1, 65)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    a, c = 5.3 * np.pi, 3.7 * np.pi
    p = np.sin(a * xx + 0.2) * np.cos(c * yy - 0.1)
    raw = np.stack(
        [
            a * np.cos(a * xx + 0.2) * np.cos(c * yy - 0.1),
            -c * np.sin(a * xx + 0.2) * np.sin(c * yy - 0.1),
        ],
        axis=-1,
    )
    errors = []
    for options in [
        {},
        dict(endpoint_method="chebyshev", chebyshev_modes=14, baseline_points=18),
    ]:
        plan = b.plan_pressure_poisson2d(x, y, **options)
        _, result = jax.jit(b.project_pressure2d)(plan, raw)
        assert bool(result.converged)
        expected = b.pressure_remove_mean(plan, p)
        errors.append(
            float(
                jnp.linalg.norm(result.pressure - expected) / jnp.linalg.norm(expected)
            )
        )
        reference = PressurePoisson2D(x, y, **options)
        _, nr = reference.project(raw.swapaxes(0, 1))
        np.testing.assert_allclose(result.pressure, nr.pressure.T, atol=1e-9)
    assert errors[1] < errors[0] / 100
    assert errors[1] < 1e-8


@pytest.mark.parametrize(
    "options",
    [
        dict(endpoint_method="bad"),
        dict(q=True),
        dict(endpoint_method="chebyshev", chebyshev_modes=20),
        dict(endpoint_method="chebyshev", endpoint_regularization=-1),
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        b.plan_pressure_poisson2d(
            np.linspace(0, 1, 40), np.linspace(0, 1, 40), **options
        )


def test_invalid_grid_and_precision():
    with pytest.raises(ValueError, match="uniform"):
        b.plan_pressure_poisson2d(np.arange(40) ** 2, np.arange(40))
    with pytest.raises(ValueError, match="grid size"):
        b.plan_pressure_poisson2d(np.arange(16), np.arange(16))
    previous = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(ValueError, match="x64"):
            b.plan_pressure_poisson2d(np.arange(40), np.arange(40))
    finally:
        jax.config.update("jax_enable_x64", previous)
