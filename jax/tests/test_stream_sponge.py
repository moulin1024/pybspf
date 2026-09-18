"""External sponge support and dissipative Galerkin load checks."""

import jax
import jax.numpy as jnp
import numpy as np
from bspf_jax.stream_navier_stokes import (
    plan_stream_navier_stokes2d,
    plan_stream_sponge,
    stream_ns_sponge_load,
    stream_ns_velocity,
    stream_ns_load,
    stream_kh_initial,
    _smooth_step,
)

jax.config.update("jax_enable_x64", True)


def test_external_sponge_energy_and_direct_load():
    p = plan_stream_navier_stokes2d(
        np.linspace(-5, 5, 40), np.linspace(-1, 1, 40), x_boundary="open"
    )
    s = plan_stream_sponge(p, interior=(-3, 3), strength=4)
    assert np.all(np.asarray(s.sigma)[abs(np.asarray(p.x.points)) <= 3] == 0)
    assert np.min(s.sigma) >= 0 and np.max(s.sigma) <= 4
    a = jnp.asarray(np.random.default_rng(3).normal(size=p.denominator.shape) * 1e-4)
    u = stream_ns_velocity(p, a)
    direct = stream_ns_sponge_load(p, a, s)
    quadrature = stream_ns_load(p, -s.sigma[:, None, None] * u)
    np.testing.assert_allclose(direct, quadrature, rtol=1e-11, atol=1e-11)
    work = jnp.sum(a * direct)
    exact = -jnp.sum(
        p.x.weights[:, None]
        * p.y.weights[None, :]
        * s.sigma[:, None]
        * jnp.sum(u * u, axis=-1)
    )
    np.testing.assert_allclose(work, exact, rtol=1e-12, atol=1e-12)
    assert work < 0
    np.testing.assert_array_equal(stream_ns_sponge_load(p, jnp.zeros_like(a), s), 0)
    assert np.all(np.isfinite(stream_kh_initial(p, extension_cutoff=(3, 4))))


def test_smooth_taper_support_and_derivative():
    x = jnp.array([-1.0, 0.0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0, 2.0])
    value, derivative = _smooth_step(x)
    np.testing.assert_array_equal(np.asarray(value)[[0, 1, 7, 8]], [0, 0, 1, 1])
    np.testing.assert_array_equal(np.asarray(derivative)[[0, 1, 7, 8]], 0)
    actual = jax.vmap(jax.grad(lambda s: _smooth_step(s)[0]))(x)
    np.testing.assert_allclose(actual, derivative, atol=1e-13)
