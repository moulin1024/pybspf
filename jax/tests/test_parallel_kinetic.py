"""Open kinetic boundary fluxes, both acceleration signs and transport accuracy."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import bspf_jax as b


def system(a):
    z = jnp.linspace(0., 1., 17)
    v = jnp.linspace(-2., 2., 25)
    zp = b.plan_1d(z, degree=5, n_basis=10, boundary_points=7)
    vp = b.plan_1d(v, degree=5, n_basis=12, boundary_points=7)
    return z, v, b.plan_parallel_kinetic(zp, vp, acceleration=a)


@pytest.mark.parametrize('a', [-.3, 0., .3])
def test_affine_characteristic_solution_and_jit(a):
    z, v, p = system(a)
    # Both velocity signs and the acceleration inflow face are active.
    exact = lambda t, z, v: 3+.1*(z-v*t+.5*a*t*t)+.2*(v-a*t)
    initial = exact(0., z[:, None], v[None, :])
    times = jnp.array([0., .1, .3])
    run = jax.jit(lambda p, f: b.integrate_parallel_kinetic(
        p, f, times, inflow=exact, substeps=100))
    result = run(p, initial)
    np.testing.assert_allclose(result, exact(times[:, None, None], z[None, :, None], v[None, None, :]), atol=2e-9)


def test_stationary_constant_and_validation():
    z, v, p = system(.3)
    initial = jnp.ones((z.size, v.size))
    result = b.integrate_parallel_kinetic(p, initial, jnp.array([0., .2]),
                                        inflow=lambda t, z, v: 1., substeps=100)
    np.testing.assert_allclose(result, 1., atol=2e-10)
    with pytest.raises(ValueError, match='initial'):
        b.integrate_parallel_kinetic(p, initial.T, jnp.array([0.]), inflow=lambda t, z, v: 1.)
    with pytest.raises(ValueError, match='substeps'):
        b.integrate_parallel_kinetic(p, initial, jnp.array([0.]), inflow=lambda t, z, v: 1., substeps=0)


def test_open_homogeneous_inflow_dissipates_phase_space_norm():
    z, v, p = system(.3)
    initial = jnp.exp(-30*(z[:, None]-.5)**2-2*(v[None, :]-1.)**2)
    times = jnp.linspace(0., .5, 11)
    history = b.integrate_parallel_kinetic(p, initial, times,
                                          inflow=lambda t, z, v: 0., substeps=50)
    zp = b.plan_1d(z, degree=5, n_basis=10, boundary_points=7)
    vp = b.plan_1d(v, degree=5, n_basis=12, boundary_points=7)
    mz = b.galerkin_1d(zp, quadrature_order=8).mass
    mv = b.galerkin_1d(vp, quadrature_order=8).mass
    norm = jax.vmap(lambda f: jnp.sum(f*(mz@f@mv)))(history)
    assert np.all(np.diff(norm) <= 1e-10)
    assert float(norm[-1]) < .8*float(norm[0])
    # Outgoing traces must not be reset to the zero incoming reservoir.
    assert float(jnp.max(jnp.abs(history[:, -1, v > 0]))) > .1
