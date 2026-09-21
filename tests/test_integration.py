"""Analytic nonperiodic integration checks on unequal tensor grids."""

import pybspf.calculus as bspf_calculus
import pybspf.plans as bspf_plans

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pybspf as b

DOMAINS = np.array([[-0.4, 1.3], [0.2, 1.7], [-0.7, 0.9]])
RATES = np.array([[0.4, 0.4 + 2.3j], [-0.3, -0.3 + 1.7j], [0.2, 0.2 - 2.1j]])


def manufactured(dimension, sizes):
    coordinates = tuple(
        jnp.linspace(*ab, n) for ab, n in zip(DOMAINS[:dimension], sizes)
    )
    axes = tuple(
        bspf_plans.plan_1d(
            x,
            degree=9,
            n_basis=18,
            lam=1e-6,
            endpoint_method="chebyshev",
            boundary_points=16,
            chebyshev_modes=12,
        )
        for x in coordinates
    )
    plan = axes[0] if dimension == 1 else bspf_plans.tensor_plan(*axes)
    mesh = jnp.meshgrid(*coordinates, indexing="ij")
    exponent = sum(x[..., None] * rate for x, rate in zip(mesh, RATES))
    return plan, coordinates, jnp.exp(exponent)


def exact_box(bounds):
    rates = RATES[: len(bounds)]
    lo, hi = np.asarray(bounds).T
    return np.prod(
        np.exp(rates * lo[:, None]) * np.expm1(rates * (hi - lo)[:, None]) / rates,
        axis=0,
    )


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_nonperiodic_tensor_integration(dimension):
    plan, coordinates, field = manufactured(dimension, (33, 41, 49))
    full = DOMAINS[:dimension]
    sub = full + np.array([0.13, -0.17])
    integrate_box = jax.jit(bspf_calculus.integrate_box)
    reference = exact_box(full)
    np.testing.assert_allclose(
        bspf_calculus.integrate_box(plan, field), reference, atol=2e-10, rtol=2e-10
    )
    np.testing.assert_allclose(
        integrate_box(plan, field), reference, atol=2e-10, rtol=2e-10
    )
    np.testing.assert_allclose(
        integrate_box(plan, field.real), reference.real, atol=2e-10, rtol=2e-10
    )
    for bounds in (sub, sub[:, ::-1]):
        result = integrate_box(plan, field, jnp.asarray(bounds))
        np.testing.assert_allclose(result, exact_box(bounds), atol=2e-10, rtol=2e-10)
        assert result.shape == (2,)
    zero = sub.copy()
    zero[0, 1] = zero[0, 0]
    np.testing.assert_allclose(
        integrate_box(plan, field, jnp.asarray(zero)), 0, atol=1e-14
    )
    invalid = sub.copy()
    invalid[0, 0] = full[0, 0] - 0.1
    assert np.isnan(integrate_box(plan, field, jnp.asarray(invalid))).all()

    for axis, x in enumerate(coordinates):
        rate = jnp.asarray(RATES[axis])
        at_left = jnp.take(field, 0, axis=axis)
        lo, hi = sub[axis]
        exact = (
            at_left * jnp.exp(rate * (lo - x[0])) * jnp.expm1(rate * (hi - lo)) / rate
        )
        actual = jax.jit(partial(bspf_calculus.integrate, axis=axis))(plan, field, a=lo, b=hi)
        np.testing.assert_allclose(actual, exact, atol=2e-10, rtol=2e-10)
        # First and second primitives, including nonzero constants.
        t = (x - x[0]).reshape((1,) * axis + (x.size,) + (1,) * (dimension - axis))
        base = jnp.expand_dims(at_left, axis)
        first = base * jnp.expm1(rate * t) / rate + 0.7
        second = base * (jnp.expm1(rate * t) - rate * t) / rate**2 + 0.7 - 0.2 * t
        for order, expected in ((1, first), (2, second)):
            actual = jax.jit(partial(bspf_calculus.antiderivative, axis=axis, order=order))(
                plan, field, left_value=0.7, left_slope=-0.2
            )
            np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)

    # The upper-bound derivative is the integral over the corresponding face.
    def volume(upper):
        bounds = jnp.asarray(sub).at[0, 1].set(upper)
        return bspf_calculus.integrate_box(plan, field, bounds)

    factor = (
        np.exp(RATES[0] * sub[0, 0])
        * np.expm1(RATES[0] * np.diff(sub[0])[0])
        / RATES[0]
    )
    exact_face = exact_box(sub) / factor * np.exp(RATES[0] * sub[0, 1])
    np.testing.assert_allclose(
        jax.jit(jax.jacfwd(volume))(sub[0, 1]), exact_face, atol=2e-9, rtol=2e-9
    )
