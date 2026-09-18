"""Dynamic open boundary: direct mass inverse, energy, analytic transient PDE."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bspf_jax.stream_navier_stokes import (
    plan_stream_navier_stokes2d,
    with_stream_dynamic_boundary,
    stream_ns_inertia_apply,
    stream_ns_inertia_solve,
    stream_ns_rhs,
    stream_ns_open_velocity,
    stream_ns_velocity,
    stream_ns_load,
    stream_ns_boundary_load,
)
from test_stream_open_boundary import manufactured

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def base():
    return plan_stream_navier_stokes2d(
        np.linspace(-1, 1, 40), np.linspace(-1, 1, 41), x_boundary="open"
    )


@pytest.mark.parametrize("D0", [0.0, 1.0, 2.0])
def test_direct_mass_inverse_and_energy(base, D0):
    p = with_stream_dynamic_boundary(base, D0=D0)
    a = jnp.asarray(np.random.default_rng(71).normal(size=p.denominator.shape) * 1e-4)
    np.testing.assert_allclose(
        stream_ns_inertia_solve(p, stream_ns_inertia_apply(p, a)),
        a,
        atol=1e-13,
        rtol=1e-10,
    )
    edge = stream_ns_open_velocity(p, a)
    boundary_mass = p.boundary_inertia * jnp.sum(
        p.y.weights * jnp.sum(edge**2, axis=-1)
    )
    np.testing.assert_allclose(
        jnp.sum(a * stream_ns_inertia_apply(p, a)),
        jnp.sum(p.denominator * a * a) + boundary_mass,
        atol=1e-12,
    )
    rhs = jax.jit(stream_ns_rhs)(p, a)
    diss = -p.nu * jnp.sum(
        a
        * (
            p.x.bending @ a
            + a @ p.y.bending.T
            + 2 * p.x.lam[:, None] * a * p.y.lam[None, :]
        )
    )
    un = jnp.array([-1.0, 1.0])[:, None] * edge[..., 0]
    flux = -0.5 * jnp.sum(p.y.weights * jnp.abs(un) * jnp.sum(edge**2, axis=-1))
    np.testing.assert_allclose(
        jnp.sum(stream_ns_inertia_apply(p, a) * rhs),
        diss + flux,
        atol=2e-12,
        rtol=2e-12,
    )


def test_transient_continuous_pde_and_boundary_inertia(base):
    p = with_stream_dynamic_boundary(base, D0=2.0)
    # u(t)=exp(-t)*curl(polynomial), checked at t=0; analytic p nonzero.
    vel, force, _, _, _ = manufactured(p.x.points, p.y.points)
    a = stream_ns_load(p, vel) / p.denominator
    edge, _, pressure, ux, vx = manufactured([-1, 1], p.y.points)
    normal = np.array([-1.0, 1.0])[:, None]
    traction = np.stack(
        (0.002 * normal * ux - normal * pressure, 0.002 * normal * vx), axis=-1
    )
    un = normal * edge[..., 0]
    E = 0.5 * un[..., None] * edge
    E[..., 0] += 0.5 * normal * np.sum(edge**2, axis=-1)
    E *= (un < 0)[..., None]
    fb = traction - E - p.boundary_inertia * edge
    load = stream_ns_load(p, force - vel) + stream_ns_boundary_load(p, fb)
    derivative = stream_ns_rhs(p, a, load)
    expected = -manufactured(p.x.x, p.y.x)[0]
    assert np.max(abs(stream_ns_velocity(p, derivative, nodes=True) - expected)) < 2e-9


def test_combined_dynamic_boundary_and_external_sponge_energy(base):
    from bspf_jax.stream_navier_stokes import plan_stream_sponge

    p = with_stream_dynamic_boundary(base, D0=1.0)
    sponge = plan_stream_sponge(p, interior=(-0.6, 0.6), strength=4.0)
    a = jnp.asarray(np.random.default_rng(81).normal(size=p.denominator.shape) * 1e-4)
    derivative = stream_ns_rhs(p, a, sponge=sponge)
    edge = stream_ns_open_velocity(p, a)
    un = jnp.array([-1.0, 1.0])[:, None] * edge[..., 0]
    flux = -0.5 * jnp.sum(p.y.weights * jnp.abs(un) * jnp.sum(edge**2, axis=-1))
    u = stream_ns_velocity(p, a)
    absorption = -jnp.sum(
        p.x.weights[:, None]
        * p.y.weights[None, :]
        * sponge.sigma[:, None]
        * jnp.sum(u**2, axis=-1)
    )
    viscous = -p.nu * jnp.sum(
        a
        * (
            p.x.bending @ a
            + a @ p.y.bending.T
            + 2 * p.x.lam[:, None] * a * p.y.lam[None, :]
        )
    )
    np.testing.assert_allclose(
        jnp.sum(stream_ns_inertia_apply(p, a) * derivative),
        viscous + flux + absorption,
        rtol=2e-12,
        atol=2e-12,
    )
    assert np.all(np.asarray(sponge.sigma)[abs(np.asarray(p.x.points)) <= 0.6] == 0)
