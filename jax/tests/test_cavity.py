"""Continuous manufactured NS solution and physical cavity boundary checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial import Polynomial

from bspf_jax.cavity import (
    plan_cavity,
    plan_cavity_stepper,
    cavity_step,
    cavity_fields,
    cavity_lift,
    cavity_ramp,
    cavity_rhs,
)
from bspf_jax.stream_navier_stokes import (
    stream_ns_load,
    stream_ns_velocity,
    stream_ns_divergence,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def cavity():
    pytest.importorskip("gmpy2")
    return plan_cavity(n=33)


def test_physical_walls_and_divergence(cavity):
    c, p = cavity, cavity.spatial
    a = jnp.asarray(np.random.default_rng(42).normal(size=p.denominator.shape) * 1e-5)
    for t in (0.0, 0.37, 2.0):
        psi, velocity, _ = map(np.asarray, cavity_fields(c, a, t))
        lid = (
            float(cavity_ramp(t, 1.0)[0])
            * 16
            * np.asarray(p.x.x) ** 2
            * (1 - np.asarray(p.x.x)) ** 2
        )
        np.testing.assert_allclose(velocity[[0, -1]], 0, atol=2e-11)
        np.testing.assert_allclose(velocity[:, 0], 0, atol=2e-11)
        np.testing.assert_allclose(velocity[:, -1, 1], 0, atol=2e-11)
        np.testing.assert_allclose(velocity[:, -1, 0], lid, atol=2e-11)
        assert max(abs(psi[[0, -1]]).max(), abs(psi[:, [0, -1]]).max()) < 2e-11
        assert np.max(abs(stream_ns_divergence(p, a))) < 2e-11
    zero = jnp.zeros_like(a)
    np.testing.assert_array_equal(cavity_rhs(c, zero, 0.0), zero)
    np.testing.assert_array_equal(cavity_fields(c, zero, 0.0)[1], 0.0)


def manufactured(c):
    """psi=s(t)*(lift+.2*f(x)*g(y)), p=-|u|²/2, with continuous force.

    Polynomial derivatives are evaluated independently of BSPF differentiation.
    The physical forcing is u_t - u cross omega - nu*Laplacian(u), with the
    kinetic-pressure gradient absorbed in the specified pressure.
    """
    p = c.spatial
    x, y = np.asarray(p.x.points)[:, None], np.asarray(p.y.points)[None, :]
    f, g = Polynomial([0, 0, 16, -32, 16]), Polynomial([0, 0, 1, -2, 1])
    u = 0.2 * f(x) * g.deriv()(y)
    v = -0.2 * f.deriv()(x) * g(y)
    perturbation = jnp.asarray(np.stack((u, v), axis=-1))
    omega = -0.2 * (f.deriv(2)(x) * g(y) + f(x) * g.deriv(2)(y))
    lap = 0.2 * np.stack(
        (
            f.deriv(2)(x) * g.deriv()(y) + f(x) * g.deriv(3)(y),
            -f.deriv(3)(x) * g(y) - f.deriv()(x) * g.deriv(2)(y),
        ),
        axis=-1,
    )
    _, lift, lift_omega, lift_lap = cavity_lift(p.x.points, p.y.points)
    total = perturbation + lift
    omega = omega + lift_omega
    mass = stream_ns_load(p, total)
    rotational = stream_ns_load(
        p, jnp.stack((total[..., 1] * omega, -total[..., 0] * omega), axis=-1)
    )
    diffusion = stream_ns_load(p, lap + lift_lap)
    a0 = stream_ns_load(p, perturbation) / p.denominator

    def load(t):
        s, rate = cavity_ramp(t, c.ramp_time)
        return rate * mass - s**2 * rotational - p.nu * s * diffusion

    return a0, load, perturbation


def test_continuous_manufactured_residual(cavity):
    c, p = cavity, cavity.spatial
    a0, load, exact_velocity = manufactured(c)
    np.testing.assert_allclose(stream_ns_velocity(p, a0), exact_velocity, atol=2e-10)
    for t in (0.3, 2.0):
        s, rate = cavity_ramp(t, c.ramp_time)
        residual = cavity_rhs(c, s * a0, t) + load(t) / p.denominator - rate * a0
        assert np.max(abs(stream_ns_velocity(p, residual))) < 2e-7


def test_second_order_time_and_boundary_inertia(cavity):
    c, p = cavity, cavity.spatial
    a0, load, _ = manufactured(c)
    errors = []
    for dt in (0.04, 0.02, 0.01):
        stepper = plan_cavity_stepper(c, dt)

        @jax.jit
        def run():
            return jax.lax.fori_loop(
                0,
                round(0.4 / dt),
                lambda k, a: cavity_step(c, stepper, a, k * dt, load),
                jnp.zeros_like(a0),
            )

        error = stream_ns_velocity(p, run() - cavity_ramp(0.4, 1.0)[0] * a0)
        errors.append(
            float(
                jnp.sqrt(
                    jnp.sum(
                        p.x.weights[:, None]
                        * p.y.weights[None, :]
                        * jnp.sum(error**2, axis=-1)
                    )
                )
            )
        )
    assert errors[0] / errors[1] > 3.3, errors
    assert errors[1] / errors[2] > 3.3, errors
    assert errors[-1] < 2e-5, errors


@pytest.mark.parametrize("kwargs", [{"n": 32}, {"reynolds": 0}, {"ramp_time": 0}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        plan_cavity(**kwargs)
