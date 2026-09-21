"""Nonperiodic wave accuracy, boundary acceleration lifting and JIT support."""

import bspf_models.waves.sine_gordon as bspf_sine_gordon
import pybspf.galerkin as bspf_galerkin
import pybspf.plans as bspf_plans
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def system(n=33):
    x = jnp.linspace(-3., 3., n)
    p = bspf_plans.plan_1d(x, degree=5, n_basis=16, boundary_points=7)
    return x, bspf_galerkin.galerkin_1d(p, quadrature_order=8)


def test_boundary_acceleration_lift_under_jit():
    x, weak = system()
    # Isolate the inertial boundary coupling: M_ii q_tt = -M_ib g_tt.
    inertial = replace(weak, stiffness=jnp.zeros_like(weak.stiffness),
                       values=jnp.zeros_like(weak.values))
    times = jnp.array([.2, .3, .5, .7])
    boundary = lambda t: jnp.array([(t-.2)**2, jnp.sin(t-.2)])
    rate = jax.jacfwd(boundary)
    initial = jnp.zeros_like(x)
    velocity = jnp.zeros_like(x).at[-1].set(1.)
    run = jax.jit(lambda w, u, v: bspf_sine_gordon.integrate_sine_gordon(
        w, u, v, times, boundary=boundary, substeps=20))
    u, v = run(inertial, initial, velocity)
    g, gt = jax.vmap(boundary)(times), jax.vmap(rate)(times)
    coupling = np.linalg.solve(np.asarray(weak.mass[1:-1, 1:-1]),
                               np.asarray(weak.mass[1:-1, jnp.array([0, -1])]))
    reference = -(g-g[0]-(times-times[0])[:, None]*gt[0])@coupling.T
    np.testing.assert_allclose(u[:, 1:-1], reference, atol=2e-12)
    np.testing.assert_allclose(v[:, 1:-1], -(gt-gt[0])@coupling.T, atol=2e-12)
    np.testing.assert_allclose(u[:, [0, -1]], g, atol=0)
    np.testing.assert_allclose(v[:, [0, -1]], gt, atol=0)


@pytest.mark.parametrize('speed', [0., .6])
def test_exact_kink(speed):
    x, weak = system(65)
    gamma = 1/jnp.sqrt(1-speed**2)
    exact = lambda x, t: 4*jnp.arctan(jnp.exp(gamma*(x-speed*t)))
    velocity = lambda x, t: -2*speed*gamma/jnp.cosh(gamma*(x-speed*t))
    times = jnp.array([0., .2, .5])
    u, v = bspf_sine_gordon.integrate_sine_gordon(
        weak, exact(x, 0.), velocity(x, 0.), times,
        boundary=lambda t: exact(x[jnp.array([0, -1])], t), substeps=150)
    np.testing.assert_allclose(u, exact(x[None], times[:, None]), atol=3e-7)
    np.testing.assert_allclose(v, velocity(x[None], times[:, None]), atol=3e-6)


def test_single_time_and_invalid_arguments():
    x, weak = system()
    initial = jnp.ones_like(x)
    boundary = lambda t: jnp.ones(2)
    u, v = bspf_sine_gordon.integrate_sine_gordon(weak, initial, 0*initial, jnp.array([0.]), boundary=boundary)
    np.testing.assert_array_equal(u[0], initial)
    np.testing.assert_array_equal(v[0], 0*initial)
    for steps in (0, True, 1.5):
        with pytest.raises(ValueError, match='substeps'):
            bspf_sine_gordon.integrate_sine_gordon(weak, initial, 0*initial, jnp.array([0.]),
                                    boundary=boundary, substeps=steps)
    with pytest.raises(ValueError, match='two real'):
        bspf_sine_gordon.integrate_sine_gordon(weak, initial, 0*initial, jnp.array([0.]), boundary=lambda t: jnp.ones(3))
    with pytest.raises(ValueError, match='velocity'):
        bspf_sine_gordon.integrate_sine_gordon(weak, initial, jnp.ones(2), jnp.array([0.]), boundary=boundary)


def test_kink_antikink_collision():
    x, weak = system(65)
    speed, center, collision = .6, .4, .5
    gamma = 1/jnp.sqrt(1-speed**2)

    def exact(x, t):
        return 4*jnp.arctan(jnp.sinh(gamma*speed*(t-collision))
                            /(speed*jnp.cosh(gamma*(x-center))))

    dt = jax.grad(exact, argnums=1)
    dtt = jax.grad(dt, argnums=1)
    dxx = jax.grad(jax.grad(exact, argnums=0), argnums=0)
    points = jnp.array([-2., -.7, center, 1.1, 2.8])
    samples = jnp.array([0., .4, collision, .6, 1.])
    residual = jax.vmap(lambda x, t: dtt(x, t)-dxx(x, t)+jnp.sin(exact(x, t)))(points, samples)
    np.testing.assert_allclose(residual, 0., atol=2e-14)
    velocity = lambda x, t: jax.jvp(lambda tau: exact(x, tau), (t,), (jnp.ones_like(t),))[1]
    times = jnp.linspace(0., 1., 11)
    u, v = bspf_sine_gordon.integrate_sine_gordon(
        weak, exact(x, times[0]), velocity(x, times[0]), times,
        boundary=lambda t: exact(x[jnp.array([0, -1])], t), substeps=50)
    np.testing.assert_allclose(u, exact(x[None], times[:, None]), atol=5e-7)
    np.testing.assert_allclose(v, velocity(x[None], times[:, None]), atol=3e-6)
    # The field crosses zero smoothly, while kinetic energy remains nonzero.
    assert float(jnp.max(jnp.abs(u[5]))) < 5e-7
    assert float(jnp.max(jnp.abs(v[5]))) > 4.
