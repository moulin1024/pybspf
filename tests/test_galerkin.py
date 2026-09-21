"""Weak-form algebra, physical boundary constraints, and conservative evolution."""

import pybspf.calculus as bspf_calculus
import pybspf.galerkin as bspf_galerkin
import pybspf.operators as bspf_operators
import pybspf.plans as bspf_plans
import pybspf.time_integration as bspf_time_integration
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_weak_form_energy_and_clamp():
    x = jnp.linspace(0., 1., 33)
    plan = bspf_plans.plan_1d(x, degree=5, n_basis=16, boundary_points=7)
    weak = bspf_galerkin.galerkin_1d(plan, derivative_order=2, constraints=((0, 0), (0, 1)))
    q = x[weak.free]**2
    field = jax.jit(lambda w, v: w.extension@v)(weak, q)
    np.testing.assert_allclose(field, x**2, atol=1e-12)
    assert abs(float(bspf_operators.differentiate(plan, field)[0])) < 1e-11
    # Integral (d²x²/dx²)² dx = 4, and integral x^4 dx ~ 1/5.
    np.testing.assert_allclose(q@weak.stiffness@q, 4., rtol=1e-7)
    np.testing.assert_allclose(q@weak.mass@q, .2, atol=4e-4)
    assert np.linalg.eigvalsh(np.array(weak.mass)).min() > 0
    assert np.linalg.eigvalsh(np.array(weak.stiffness)).min() > 0
    np.testing.assert_allclose(weak.stiffness, weak.stiffness.T, rtol=1e-13, atol=1e-7)


def test_neumann_constant_and_schrodinger_norm():
    plan = bspf_plans.plan_1d(jnp.linspace(0., 1., 17), degree=3, n_basis=8, boundary_points=5)
    weak = bspf_galerkin.galerkin_1d(plan)
    np.testing.assert_allclose(weak.stiffness@jnp.ones(17), 0., atol=1e-10)
    operator = -1j*jnp.linalg.solve(weak.mass, weak.stiffness)
    initial = jnp.exp(2j*plan.x)
    history = jax.jit(lambda y: bspf_time_integration.integrate_linear_midpoint(
        operator, y, jnp.array([0., .01, .04]), substeps=10))(initial)
    norm = jnp.real(jnp.einsum('ti,ij,tj->t', history.conj(), weak.mass, history))
    np.testing.assert_allclose(norm, norm[0], atol=1e-12)


def test_midpoint_order_batch_and_single_time():
    operator = jnp.array([[0., -1.], [1., 0.]])
    initial = jnp.eye(2)
    errors = []
    for n in (10, 20):
        result = bspf_time_integration.integrate_linear_midpoint(operator, initial, jnp.array([0., 1.]), substeps=n)
        errors.append(abs(float(result[-1, 0, 0])-np.cos(1.)))
        np.testing.assert_allclose(result[-1].T@result[-1], initial, atol=1e-13)
    assert 3.9 < errors[0]/errors[1] < 4.1
    np.testing.assert_array_equal(bspf_time_integration.integrate_linear_midpoint(operator, initial, jnp.array([0.]))[0], initial)


@pytest.mark.parametrize('constraints', [((2, 0),), ((0, -1),), ((0, 0), (0, 0))])
def test_bad_constraints(constraints):
    plan = bspf_plans.plan_1d(jnp.linspace(0., 1., 17), degree=3, n_basis=8)
    with pytest.raises(ValueError):
        bspf_galerkin.galerkin_1d(plan, constraints=constraints)


def test_nonlinear_noise_plan_rejected():
    plan = bspf_plans.plan_1d(jnp.linspace(0., 1., 17), degree=3, n_basis=8, noise_std=.01)
    with pytest.raises(ValueError, match='clean'):
        bspf_galerkin.galerkin_1d(plan)


def test_resolved_quadrature_beam_static_and_polynomial_mass():
    x = jnp.linspace(0., 1., 33)
    plan = bspf_plans.plan_1d(x, degree=5, n_basis=16, boundary_points=7)
    weak = bspf_galerkin.galerkin_1d(plan, derivative_order=2, constraints=((0, 0), (0, 1)), quadrature_order=8)
    q = x[weak.free]**2
    np.testing.assert_allclose(q@weak.mass@q, .2, atol=1e-12)
    np.testing.assert_allclose(q@weak.stiffness@q, 4., rtol=1e-7)
    # Exact integrals of each BSPF trial function supply consistent loading.
    force = bspf_calculus.integrate(plan, weak.extension)
    solution = weak.extension@jnp.linalg.solve(weak.stiffness, force)
    np.testing.assert_allclose(solution, x**2*(x*x-4*x+6)/24, atol=1e-8)
