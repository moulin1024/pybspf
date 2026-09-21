import bspf_models.waves.schrodinger as bspf_schrodinger
import pybspf.calculus as bspf_calculus
import pybspf.galerkin as bspf_galerkin
import pybspf.operators as bspf_operators
import pybspf.plans as bspf_plans
import pybspf.time_integration as bspf_time_integration
import jax
import jax.numpy as jnp
import numpy as np
import pytest

def test_schrodinger_exact_phases_complex_mass_and_jit():
    # Nontrivial complex Hermitian mass and Hamiltonian with known eigenmodes.
    lower = jnp.array([[2., 0.], [0.2j, 1.]])
    mass = lower@lower.conj().T
    energies = jnp.array([2., 7.])
    hamiltonian = lower@jnp.diag(energies)@lower.conj().T
    initial = jnp.array([1.+2j, -0.5j])
    times = jnp.array([.3, .7, 2.8])
    result = jax.jit(bspf_schrodinger.integrate_schrodinger)(mass, hamiltonian, initial, times)
    exact = jnp.linalg.solve(lower.conj().T,
        (jnp.exp(-1j*(times-.3)[:, None]*energies)*(lower.conj().T@initial)).T).T
    np.testing.assert_allclose(result, exact, atol=1e-13)
    norm = jnp.real(jnp.einsum('ti,ij,tj->t', result.conj(), mass, result))
    np.testing.assert_allclose(norm, norm[0], atol=1e-13)


def test_schrodinger_resolved_neumann_mode():
    x = jnp.linspace(0., 1., 33)
    plan = bspf_plans.plan_1d(x, degree=5, n_basis=16, boundary_points=7)
    weak = bspf_galerkin.galerkin_1d(plan, quadrature_order=8)
    initial = jnp.cos(jnp.pi*x)
    times = jnp.array([0., .1, .4])
    result = bspf_schrodinger.integrate_schrodinger(weak.mass, weak.stiffness, initial, times)
    exact = jnp.exp(-1j*jnp.pi**2*times[:, None])*initial
    np.testing.assert_allclose(result, exact, atol=1e-7)
