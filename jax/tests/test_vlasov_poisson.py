"""Integration-based Poisson signs, grounded fields and reservoir equilibrium."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import bspf_jax as b


def grids():
    z = jnp.linspace(-2., 3., 25)
    v = jnp.linspace(-6., 6., 49)
    zp = b.plan_1d(z, degree=5, n_basis=12, boundary_points=7)
    vp = b.plan_1d(v, degree=5, n_basis=16, boundary_points=7)
    return z, v, zp, vp


def test_poisson_primitives_nonzero_dirichlet_and_batch():
    z, _, zp, _ = grids()
    # phi=z^3-2z+1, phi''=6z; independently test electric-field sign.
    exact = z**3-2*z+1
    sources = jnp.stack((6*z, 12*z), axis=1)
    phi, field = jax.jit(lambda s: b.poisson_dirichlet(
        zp, s, left=exact[0]*jnp.array([1., 2.]),
        right=exact[-1]*jnp.array([1., 2.])))(sources)
    np.testing.assert_allclose(phi, exact[:, None]*jnp.array([1., 2.]), atol=2e-11)
    np.testing.assert_allclose(field, (2-3*z*z)[:, None]*jnp.array([1., 2.]), atol=2e-11)


def test_maxwellian_reservoir_is_stationary_under_jit():
    z, v, zp, vp = grids()
    p = b.plan_vlasov_poisson(zp, vp)
    f0 = jnp.broadcast_to(p.background, (z.size, v.size))
    times = jnp.array([0., .1, .3])
    run = jax.jit(lambda p, f: b.integrate_vlasov_poisson(p, f, times, substeps=50))
    f, phi, e = run(p, f0)
    np.testing.assert_array_equal(f, jnp.broadcast_to(f0, f.shape))
    np.testing.assert_array_equal(phi, 0.)
    np.testing.assert_array_equal(e, 0.)
    np.testing.assert_allclose(p.velocity_weights@p.background, 1., atol=2e-15)


def test_density_to_field_sign_and_boundaries():
    z, v, zp, vp = grids()
    p = b.plan_vlasov_poisson(zp, vp)
    f = (1+.01*z[:, None])*p.background
    phi, e = b.vlasov_poisson_fields(p, f)
    expected, expected_e = b.poisson_dirichlet(zp, .01*z)
    np.testing.assert_allclose(phi, expected, atol=2e-14)
    np.testing.assert_allclose(e, expected_e, atol=2e-14)
    np.testing.assert_allclose(phi[jnp.array([0, -1])], 0., atol=1e-14)
    with pytest.raises(ValueError, match='temperature'):
        b.plan_vlasov_poisson(zp, vp, temperature=0.)


def test_perturbation_evolves_with_self_consistent_electron_force():
    z, v, zp, vp = grids()
    p = b.plan_vlasov_poisson(zp, vp)
    shape = jnp.sin(jnp.pi*(z-z[0])/(z[-1]-z[0]))
    f0 = p.background*(1+1e-3*shape[:, None])
    times = jnp.array([0., .05])
    f, phi, e = b.integrate_vlasov_poisson(p, f0, times, substeps=50)
    refined, _, _ = b.integrate_vlasov_poisson(p, f0, times, substeps=100)
    np.testing.assert_allclose(f, refined, atol=2e-11)
    expected_phi, expected_e = b.vlasov_poisson_fields(p, f)
    np.testing.assert_allclose(e, expected_e, atol=2e-14)
    np.testing.assert_allclose(phi, expected_phi, atol=2e-14)
    assert float(jnp.max(jnp.abs(e[-1]-e[0]))) > 1e-8
    energy = b.integrate(zp, (e*e).T)
    assert float(energy[-1]) < float(energy[0])
