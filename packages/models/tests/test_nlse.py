"""Projected cubic dynamics: linear limit, nonlinear phase, and time order."""

import bspf_models.waves.schrodinger as bspf_schrodinger
import pybspf.galerkin as bspf_galerkin
import pybspf.plans as bspf_plans
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def weak_form():
    plan = bspf_plans.plan_1d(jnp.linspace(-2., 2., 17), degree=3, n_basis=8, boundary_points=5)
    return bspf_galerkin.galerkin_1d(plan, quadrature_order=6)


def test_linear_limit_matches_exact_evolution():
    weak = weak_form()
    initial = jnp.exp(-jnp.linspace(-2., 2., 17)**2).astype(complex)
    times = jnp.array([.2, .3, .7])
    result = jax.jit(lambda w, y: bspf_schrodinger.integrate_nlse(
        w, y, times, coupling=0., substeps=3))(weak, initial)
    exact = bspf_schrodinger.integrate_schrodinger(weak.mass, weak.stiffness, initial, times)
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
        result = bspf_schrodinger.integrate_nlse(weak, initial, times, coupling=coupling, substeps=steps)
        errors.append(float(jnp.max(jnp.abs(result[-1]-exact))))
    assert 13 < errors[0]/errors[1] < 19
    assert errors[-1] < 1e-6


def test_single_time_and_validation():
    weak = weak_form()
    initial = jnp.ones(17)
    out = bspf_schrodinger.integrate_nlse(weak, initial, jnp.array([0.]))
    np.testing.assert_array_equal(out[0], initial)
    for steps in (0, True, 1.5):
        with pytest.raises(ValueError, match='substeps'):
            bspf_schrodinger.integrate_nlse(weak, initial, jnp.array([0., 1.]), substeps=steps)
    with pytest.raises(ValueError, match='coupling'):
        bspf_schrodinger.integrate_nlse(weak, initial, jnp.array([0., 1.]), coupling=1j)
