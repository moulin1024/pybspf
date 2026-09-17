"""Projected cubic dynamics: linear limit, nonlinear phase, and time order."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import bspf_jax as b


def weak_form():
    plan = b.plan_1d(jnp.linspace(-2., 2., 17), degree=3, n_basis=8, boundary_points=5)
    return b.galerkin_1d(plan, quadrature_order=6)


def test_linear_limit_matches_exact_evolution():
    weak = weak_form()
    initial = jnp.exp(-jnp.linspace(-2., 2., 17)**2).astype(complex)
    times = jnp.array([.2, .3, .7])
    result = jax.jit(lambda w, y: b.integrate_nlse(
        w, y, times, coupling=0., substeps=3))(weak, initial)
    exact = b.integrate_schrodinger(weak.mass, weak.stiffness, initial, times)
    np.testing.assert_allclose(result, exact, atol=2e-13)
    np.testing.assert_array_equal(result[0], initial)


@pytest.mark.parametrize('coupling', [2., -2.])
def test_cubic_sign_and_fourth_order(coupling):
    weak = weak_form()
    weak = replace(weak, stiffness=jnp.zeros_like(weak.stiffness))
    initial = jnp.full(17, .7+.2j)
    times = jnp.array([0., .5])
    exact = initial*jnp.exp(1j*coupling*jnp.abs(initial)**2*.5)
    errors = []
    for steps in (5, 10):
        result = b.integrate_nlse(weak, initial, times, coupling=coupling, substeps=steps)
        errors.append(float(jnp.max(jnp.abs(result[-1]-exact))))
    assert 13 < errors[0]/errors[1] < 19
    assert errors[-1] < 1e-6


def test_single_time_and_validation():
    weak = weak_form()
    initial = jnp.ones(17)
    out = b.integrate_nlse(weak, initial, jnp.array([0.]))
    np.testing.assert_array_equal(out[0], initial)
    for steps in (0, True, 1.5):
        with pytest.raises(ValueError, match='substeps'):
            b.integrate_nlse(weak, initial, jnp.array([0., 1.]), substeps=steps)
    with pytest.raises(ValueError, match='coupling'):
        b.integrate_nlse(weak, initial, jnp.array([0., 1.]), coupling=1j)
