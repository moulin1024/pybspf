"""Weak-form algebra, physical boundary constraints, and conservative evolution."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import bspf_jax as b


def test_weak_form_energy_and_clamp():
    x = jnp.linspace(0., 1., 33)
    plan = b.plan_1d(x, degree=5, n_basis=16, boundary_points=7)
    weak = b.galerkin_1d(plan, derivative_order=2, constraints=((0, 0), (0, 1)))
    q = x[weak.free]**2
    field = jax.jit(lambda w, v: w.extension@v)(weak, q)
    np.testing.assert_allclose(field, x**2, atol=1e-12)
    assert abs(float(b.differentiate(plan, field)[0])) < 1e-11
    # Integral (d²x²/dx²)² dx = 4, and integral x^4 dx ~ 1/5.
    np.testing.assert_allclose(q@weak.stiffness@q, 4., rtol=1e-7)
    np.testing.assert_allclose(q@weak.mass@q, .2, atol=4e-4)
    assert np.linalg.eigvalsh(np.array(weak.mass)).min() > 0
    assert np.linalg.eigvalsh(np.array(weak.stiffness)).min() > 0
    np.testing.assert_allclose(weak.stiffness, weak.stiffness.T, rtol=1e-13, atol=1e-7)


def test_neumann_constant_and_schrodinger_norm():
    plan = b.plan_1d(jnp.linspace(0., 1., 17), degree=3, n_basis=8, boundary_points=5)
    weak = b.galerkin_1d(plan)
    np.testing.assert_allclose(weak.stiffness@jnp.ones(17), 0., atol=1e-10)
    operator = -1j*jnp.linalg.solve(weak.mass, weak.stiffness)
    initial = jnp.exp(2j*plan.x)
    history = jax.jit(lambda y: b.integrate_linear_midpoint(
        operator, y, jnp.array([0., .01, .04]), substeps=10))(initial)
    norm = jnp.real(jnp.einsum('ti,ij,tj->t', history.conj(), weak.mass, history))
    np.testing.assert_allclose(norm, norm[0], atol=1e-12)


def test_midpoint_order_batch_and_single_time():
    operator = jnp.array([[0., -1.], [1., 0.]])
    initial = jnp.eye(2)
    errors = []
    for n in (10, 20):
        result = b.integrate_linear_midpoint(operator, initial, jnp.array([0., 1.]), substeps=n)
        errors.append(abs(float(result[-1, 0, 0])-np.cos(1.)))
        np.testing.assert_allclose(result[-1].T@result[-1], initial, atol=1e-13)
    assert 3.9 < errors[0]/errors[1] < 4.1
    np.testing.assert_array_equal(b.integrate_linear_midpoint(operator, initial, jnp.array([0.]))[0], initial)


@pytest.mark.parametrize('constraints', [((2, 0),), ((0, -1),), ((0, 0), (0, 0))])
def test_bad_constraints(constraints):
    plan = b.plan_1d(jnp.linspace(0., 1., 17), degree=3, n_basis=8)
    with pytest.raises(ValueError):
        b.galerkin_1d(plan, constraints=constraints)


def test_nonlinear_noise_plan_rejected():
    plan = b.plan_1d(jnp.linspace(0., 1., 17), degree=3, n_basis=8, noise_std=.01)
    with pytest.raises(ValueError, match='clean'):
        b.galerkin_1d(plan)


def test_resolved_quadrature_beam_static_and_polynomial_mass():
    x = jnp.linspace(0., 1., 33)
    plan = b.plan_1d(x, degree=5, n_basis=16, boundary_points=7)
    weak = b.galerkin_1d(plan, derivative_order=2, constraints=((0, 0), (0, 1)), quadrature_order=8)
    q = x[weak.free]**2
    np.testing.assert_allclose(q@weak.mass@q, .2, atol=1e-12)
    np.testing.assert_allclose(q@weak.stiffness@q, 4., rtol=1e-7)
    # Exact integrals of each BSPF trial function supply consistent loading.
    force = b.integrate(plan, weak.extension)
    solution = weak.extension@jnp.linalg.solve(weak.stiffness, force)
    np.testing.assert_allclose(solution, x**2*(x*x-4*x+6)/24, atol=1e-8)


def test_schrodinger_exact_phases_complex_mass_and_jit():
    # Nontrivial complex Hermitian mass and Hamiltonian with known eigenmodes.
    lower = jnp.array([[2., 0.], [0.2j, 1.]])
    mass = lower@lower.conj().T
    energies = jnp.array([2., 7.])
    hamiltonian = lower@jnp.diag(energies)@lower.conj().T
    initial = jnp.array([1.+2j, -0.5j])
    times = jnp.array([.3, .7, 2.8])
    result = jax.jit(b.integrate_schrodinger)(mass, hamiltonian, initial, times)
    exact = jnp.linalg.solve(lower.conj().T,
        (jnp.exp(-1j*(times-.3)[:, None]*energies)*(lower.conj().T@initial)).T).T
    np.testing.assert_allclose(result, exact, atol=1e-13)
    norm = jnp.real(jnp.einsum('ti,ij,tj->t', result.conj(), mass, result))
    np.testing.assert_allclose(norm, norm[0], atol=1e-13)


def test_schrodinger_resolved_neumann_mode():
    x = jnp.linspace(0., 1., 33)
    plan = b.plan_1d(x, degree=5, n_basis=16, boundary_points=7)
    weak = b.galerkin_1d(plan, quadrature_order=8)
    initial = jnp.cos(jnp.pi*x)
    times = jnp.array([0., .1, .4])
    result = b.integrate_schrodinger(weak.mass, weak.stiffness, initial, times)
    exact = jnp.exp(-1j*jnp.pi**2*times[:, None])*initial
    np.testing.assert_allclose(result, exact, atol=1e-7)
