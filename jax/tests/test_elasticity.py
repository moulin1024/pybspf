from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import bspf_jax as b
from bspf_jax.references import cantilever_modes, cantilever_spectrum, cantilever_step_response


def test_forced_oscillators_nonuniform_times_and_jit():
    weights = jnp.array([2., 3.])
    gradient = jnp.diag(jnp.array([2., 3.]))
    weak = b.Galerkin1D(mass=jnp.diag(weights), stiffness=gradient.T@jnp.diag(weights)@gradient,
                       extension=jnp.eye(2), free=jnp.arange(2), values=jnp.eye(2),
                       quadrature_weights=weights, derivative_values=gradient)
    initial, velocity, force = jnp.array([.2, -.1]), jnp.array([.4, .3]), jnp.array([1., -.5])
    times = jnp.array([.3, .31, .8, 1.7])
    q, v = jax.jit(lambda w: b.integrate_elastic(w, initial, velocity, times,
                          force=force, density=2., rigidity=3.))(weak)
    frequency = jnp.sqrt(1.5)*jnp.array([2., 3.])
    t = (times-times[0])[:, None]
    equilibrium = force/(3*weights*jnp.array([4., 9.]))
    expected = equilibrium+(initial-equilibrium)*jnp.cos(t*frequency)+velocity*jnp.sin(t*frequency)/frequency
    expected_v = -(initial-equilibrium)*frequency*jnp.sin(t*frequency)+velocity*jnp.cos(t*frequency)
    np.testing.assert_allclose(q, expected, atol=1e-14)
    np.testing.assert_allclose(v, expected_v, atol=1e-14)
    np.testing.assert_array_equal(q[0], initial)


def test_zero_frequency_constant_acceleration():
    x = jnp.linspace(0., 1., 17)
    weak = b.galerkin_1d(b.plan_1d(x, degree=3, n_basis=8, boundary_points=5), quadrature_order=6)
    weak = replace(weak, derivative_values=jnp.zeros_like(weak.derivative_values), stiffness=jnp.zeros_like(weak.stiffness))
    initial, velocity = x*x, x
    force = weak.mass@jnp.ones_like(x)
    times = jnp.array([0., .1, 1.])
    q, v = b.integrate_elastic(weak, initial, velocity, times, force=force)
    np.testing.assert_allclose(q, initial+times[:, None]*velocity+.5*times[:, None]**2, atol=1e-13)
    np.testing.assert_allclose(v, velocity+times[:, None], atol=1e-13)


def test_cantilever_reference_independent_integrals_and_boundaries():
    roots, sigma = cantilever_spectrum(8)
    np.testing.assert_allclose(jnp.cos(roots)+1/jnp.cosh(roots), 0., atol=3e-15)
    g, w = np.polynomial.legendre.leggauss(160)
    phi = np.asarray(cantilever_modes(jnp.asarray((g+1)/2), mode_count=8))
    np.testing.assert_allclose((phi*phi)@(w/2), 1., atol=1e-13)
    np.testing.assert_allclose(phi@(w/2), 2*sigma/roots, atol=1e-13)
    shape = lambda x: cantilever_modes(jnp.array([x]), mode_count=8)[:, 0]
    np.testing.assert_allclose(shape(0.), 0., atol=1e-15)
    np.testing.assert_allclose(jax.jacfwd(shape)(0.), 0., atol=1e-13)
    np.testing.assert_allclose(jax.jacfwd(jax.jacfwd(shape))(1.), 0., atol=1e-10)
    np.testing.assert_allclose(jax.jacfwd(jax.jacfwd(jax.jacfwd(shape)))(1.), 0., atol=1e-9)
    x = jnp.linspace(0., 1., 65)
    a = cantilever_step_response(x, jnp.array([0., .1, 1., 3.]), mode_count=256)
    np.testing.assert_array_equal(a[0], jnp.zeros_like(x))
    np.testing.assert_allclose(a, cantilever_step_response(x, jnp.array([0., .1, 1., 3.]), mode_count=512), atol=1e-12)


@pytest.mark.parametrize("strong_free", [False, True])
def test_beam_static_fundamental_frequency_and_transient(strong_free):
    x = jnp.linspace(0., 1., 65)
    plan = b.plan_1d(x, degree=5, n_basis=16, boundary_points=7)
    constraints = ((0, 0), (0, 1))+(((1, 2), (1, 3)) if strong_free else ())
    weak = b.galerkin_1d(plan, derivative_order=2, constraints=constraints, quadrature_order=8)
    force = weak.values.T@weak.quadrature_weights
    frequency, modes = b.elastic_modes(weak)
    static = weak.extension@(modes@((modes.T@force)/frequency**2))
    np.testing.assert_allclose(static, x*x*(x*x-4*x+6)/24, atol=2e-11)
    np.testing.assert_allclose(frequency[0], 1.875104068711961**2, atol=1e-9)
    times = jnp.linspace(0., 3., 31)
    q, _ = b.integrate_elastic(weak, jnp.zeros(weak.mass.shape[0]), jnp.zeros(weak.mass.shape[0]), times, force=force)
    np.testing.assert_allclose(q@weak.extension.T, cantilever_step_response(x, times), atol=3e-8)


def test_argument_validation():
    with pytest.raises(ValueError): cantilever_spectrum(0)
    with pytest.raises(ValueError): cantilever_spectrum(True)
