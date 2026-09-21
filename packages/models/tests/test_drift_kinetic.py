"""Mirror characteristics, flux accounting, invariant mu and JIT contracts."""

import bspf_models.kinetic.drift_kinetic as bspf_drift_kinetic
import bspf_models.kinetic.parallel_kinetic as bspf_parallel_kinetic
import pybspf.plans as bspf_plans
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def system(n=17, *, field=lambda z: 1+z*z/2, gradient=lambda z: z, mass=1.):
    z, v = jnp.linspace(-2., 2., n), jnp.linspace(-2., 2., n+4)
    zp = bspf_plans.plan_1d(z, degree=5, n_basis=10, boundary_points=7)
    vp = bspf_plans.plan_1d(v, degree=5, n_basis=10, boundary_points=7)
    return bspf_drift_kinetic.plan_drift_kinetic(zp, vp, magnetic_field=field,
        magnetic_gradient=gradient, mu_max=2., n_mu=4, mass=mass, backend="dense")


def initial_coordinates(t, z, v, mu, mass=1.):
    omega = jnp.sqrt(mu/mass)
    c, s = jnp.cos(omega*t), jnp.sin(omega*t)
    return z*c-v*t*jnp.sinc(omega*t/jnp.pi), v*c+omega*z*s


def grid(p):
    return p.transport.z[:, None, None], p.transport.v[None, :, None], p.mu[None, None, :]


@pytest.mark.parametrize('mass', [1., 2.])
def test_harmonic_mirror_characteristics_and_all_face_balances_under_jit(mass):
    p = system(mass=mass)
    def exact(t, z, v, mu):
        z0, v0 = initial_coordinates(t, z, v, mu, mass)
        return 2+.04*z0+.03*v0+.02*z0*z0+.01*mu
    z, v, mu = grid(p)
    times = jnp.linspace(0., 2., 9)
    f, transfer = jax.jit(lambda p, f: bspf_drift_kinetic.integrate_drift_kinetic(
        p, f, times, inflow=exact, substeps=100))(p, exact(0., z, v, mu))
    reference = exact(times[:, None, None, None], z[None], v[None], mu[None])
    np.testing.assert_allclose(f, reference, atol=2e-8, rtol=0)
    moments = bspf_drift_kinetic.drift_kinetic_moments(p, f)
    balance = moments[:, jnp.array([0, 3])]-moments[0, jnp.array([0, 3])]-transfer.sum(axis=-1)
    assert float(jnp.max(jnp.abs(balance))) < 2e-7
    # Both signs of B' and v are active; finite velocity faces cannot be ignored.
    assert np.all(np.abs(np.asarray(transfer[-1])) > .01)


def test_energy_equilibrium_and_nonzero_mirror_force():
    p = system()
    z, v, mu = grid(p)
    equilibrium = lambda t, z, v, mu: 3+.02*(v*v/2+mu*(1+z*z/2))
    f = equilibrium(0., z, v, mu)
    df, rates = bspf_drift_kinetic.drift_kinetic_rhs(p, 0., f, inflow=equilibrium)
    np.testing.assert_allclose(df, 0., atol=2e-9)
    np.testing.assert_allclose(rates.sum(axis=-1), 0., atol=2e-9)
    # A v slope is accelerated by -mu*z, including its mu dependence.
    tilted = lambda t, z, v, mu: 3+.1*v+0*z+0*mu
    df, _ = bspf_drift_kinetic.drift_kinetic_rhs(p, 0., tilted(0., z, v, mu), inflow=tilted)
    np.testing.assert_allclose(df, jnp.broadcast_to(.1*mu*z, df.shape), atol=2e-9)


def test_uniform_field_reduces_to_parallel_transport():
    p = system(field=lambda z: 2., gradient=lambda z: 0.)
    z, v, mu = grid(p)
    f = jnp.exp(-z*z-v*v)*(1+mu)
    times = jnp.array([0., .2])
    history, _ = bspf_drift_kinetic.integrate_drift_kinetic(p, f, times, inflow=lambda t,z,v,mu: 0., substeps=100)
    old = bspf_parallel_kinetic.integrate_parallel_kinetic(p.transport, f[:, :, 0], times,
                                       inflow=lambda t,z,v: 0., substeps=100)
    np.testing.assert_allclose(history[..., 0], old, atol=2e-11)
    np.testing.assert_allclose(history/(1+mu),
        jnp.broadcast_to((history[..., :1]/(1+mu[..., :1])), history.shape), atol=2e-11)


def test_validation():
    with pytest.raises(ValueError, match='magnetic_field'):
        system(field=lambda z: -1.)
    p = system()
    with pytest.raises(ValueError, match='initial'):
        bspf_drift_kinetic.integrate_drift_kinetic(p, jnp.ones((2, 3)), jnp.array([0.]), inflow=lambda t,z,v,mu: 0.)


def test_nonquadratic_field_energy_equilibrium():
    field = lambda z: 1+.2*z*z+.01*z**4
    p = system(field=field, gradient=lambda z: .4*z+.04*z**3)
    z, v, mu = grid(p)
    exact = lambda t, z, v, mu: 2+.03*(v*v/2+mu*field(z))
    df, rates = bspf_drift_kinetic.drift_kinetic_rhs(p, 0., exact(0., z, v, mu), inflow=exact)
    np.testing.assert_allclose(df, 0., atol=2e-9)
    np.testing.assert_allclose(rates.sum(axis=-1), 0., atol=2e-9)


def test_log_transport_is_positive_between_nodes_and_has_physical_fluxes():
    p = system()
    z, v, mu = grid(p)
    def log_exact(t, z, v, mu):
        z0, v0 = initial_coordinates(t, z, v, mu)
        return -.8*z0*z0-.6*(v0-.5)**2-.2*mu
    times = jnp.linspace(0., 1., 6)
    g, transfers = jax.jit(lambda p, g: bspf_drift_kinetic.integrate_log_drift_kinetic(
        p, g, times, log_inflow=log_exact, substeps=100))(p, log_exact(0., z, v, mu))
    reference = log_exact(times[:, None, None, None], z[None], v[None], mu[None])
    np.testing.assert_allclose(g, reference, atol=4e-8, rtol=0)
    np.testing.assert_allclose(jnp.exp(g), jnp.exp(reference), atol=2e-9, rtol=0)
    moments, minimum, qminimum = bspf_drift_kinetic.log_drift_kinetic_diagnostics(p, g)
    assert np.all(np.asarray(minimum) > 0)
    assert np.all(np.asarray(qminimum) > 0)
    nh = moments[:, jnp.array([0, 3])]
    residual = nh-nh[0]-transfers.sum(axis=-1)
    assert float(jnp.max(jnp.abs(residual)/nh[0])) < 1e-8
    # Independent exact physical boundary integration: substantial flow at ALL
    # faces, to distinguish exp(log trace) from (incorrect) flux of log(f).
    from bspf_models.kinetic.drift_kinetic import _boundary_fluxes
    from bspf_models.kinetic.drift_kinetic import _boundary_rates
    gf = reference[-1]
    rates = _boundary_rates(p, _boundary_fluxes(p, times[-1], gf, log_exact, logarithmic=True))
    x, vel = p.transport.z_points[:, None], p.transport.v_points[:, None]
    mu2 = p.mu[None, :]
    fluxes = (vel*jnp.exp(log_exact(times[-1], p.transport.z[0], vel, mu2)),
              vel*jnp.exp(log_exact(times[-1], p.transport.z[-1], vel, mu2)),
              -mu2*x*jnp.exp(log_exact(times[-1], x, p.transport.v[0], mu2)),
              -mu2*x*jnp.exp(log_exact(times[-1], x, p.transport.v[-1], mu2)))
    np.testing.assert_allclose(rates, _boundary_rates(p, fluxes), atol=2e-9)
    assert np.all(np.abs(np.asarray(transfers[-1])) > .001)


def test_log_representation_does_not_interpolate_exponentiated_samples():
    p = system()
    z, v, mu = grid(p)
    g = -20*(z-.3)**2-10*(v-.4)**2-0*mu
    moments, minimum, qminimum = bspf_drift_kinetic.log_drift_kinetic_diagnostics(p, g)
    assert float(minimum) > 0 and float(qminimum) > 0
    # Direct Gaussian quadrature of the known continuous exponential.
    x, vel = p.transport.z_points[:, None, None], p.transport.v_points[None, :, None]
    weights = p.z_weights[:, None, None]*p.v_weights[None, :, None]*p.mu_weights[None, None, :]
    values = jnp.exp(-20*(x-.3)**2-10*(vel-.4)**2)
    np.testing.assert_allclose(moments[0], jnp.sum(weights*values), atol=2e-10)
    # The old linear-f reconstruction actually oscillates below zero here.
    old = jnp.einsum('ai,ijm,bj->abm', p.z_values, jnp.exp(g), p.v_values)
    assert float(jnp.min(old)) < -1e-5
    with pytest.raises(ValueError, match='initial_log'):
        bspf_drift_kinetic.integrate_log_drift_kinetic(p, g[..., 0], jnp.array([0.]), log_inflow=lambda t,z,v,mu: 0.)
