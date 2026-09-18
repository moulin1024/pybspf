"""Open-face energy flux and independent continuous manufactured PDE checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bspf_jax.stream_navier_stokes import (
    plan_stream_navier_stokes2d,
    stream_ns_rhs,
    stream_ns_load,
    stream_ns_velocity,
    stream_ns_open_velocity,
    stream_ns_boundary_load,
    stream_ns_divergence,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def plan():
    return plan_stream_navier_stokes2d(
        np.linspace(-1, 1, 40), np.linspace(-1, 1, 41), x_boundary="open"
    )


def test_open_energy_flux_and_divergence(plan):
    a = jnp.asarray(np.random.default_rng(9).normal(size=plan.denominator.shape) * 1e-4)
    f = jax.jit(stream_ns_rhs)(plan, a)
    viscous = -plan.nu * jnp.sum(
        a
        * (
            plan.x.bending @ a
            + a @ plan.y.bending.T
            + 2 * plan.x.lam[:, None] * a * plan.y.lam[None, :]
        )
    )
    edge = stream_ns_open_velocity(plan, a)
    un = jnp.array([-1.0, 1.0])[:, None] * edge[..., 0]
    # Zero reservoir: both incoming and outgoing boundary energy is dissipative.
    flux = -0.5 * jnp.sum(plan.y.weights * jnp.abs(un) * jnp.sum(edge**2, axis=-1))
    np.testing.assert_allclose(
        jnp.sum(plan.denominator * a * f), viscous + flux, rtol=2e-12, atol=2e-12
    )
    assert float(jnp.max(abs(stream_ns_divergence(plan, a)))) < 1e-11
    vel = stream_ns_velocity(plan, a, nodes=True)
    assert float(jnp.max(abs(vel[:, jnp.array([0, -1])]))) < 1e-11
    assert float(jnp.max(abs(edge))) > 1e-3


def manufactured(x, y):
    x, y = np.asarray(x)[:, None], np.asarray(y)[None, :]
    f, fp, fpp = 1 + 0.2 * x + 0.1 * x * x, 0.2 + 0.2 * x, 0 * x + 0.2
    g, gp, gpp, gppp = (1 - y * y) ** 2, -4 * y + 4 * y**3, -4 + 12 * y * y, 24 * y
    u, v = f * gp, -fp * g
    ux, uy, vx, vy = fp * gp, f * gpp, -fpp * g, -fp * gp
    lapu, lapv = fpp * gp + f * gppp, -fp * gpp
    pressure = np.exp(0.3 * x) * np.cos(x + 2 * y)
    px = np.exp(0.3 * x) * (0.3 * np.cos(x + 2 * y) - np.sin(x + 2 * y))
    py = -2 * np.exp(0.3 * x) * np.sin(x + 2 * y)
    force = np.stack(
        (u * ux + v * uy - 0.002 * lapu + px, u * vx + v * vy - 0.002 * lapv + py),
        axis=-1,
    )
    return np.stack((u, v), axis=-1), force, pressure, ux, vx


def test_continuous_pde_with_nonzero_open_velocity_and_pressure(plan):
    # Polynomial streamfunction, analytic advection/diffusion/pressure/traction.
    # This checks open weak boundary signs, rather than manufacturing a discrete RHS.
    vel, force, _, _, _ = manufactured(plan.x.points, plan.y.points)
    a = stream_ns_load(plan, vel) / plan.denominator
    exact = manufactured(plan.x.x, plan.y.x)[0]
    assert np.max(abs(stream_ns_velocity(plan, a, nodes=True) - exact)) < 1e-10
    edge, _, pressure, ux, vx = manufactured([-1, 1], plan.y.points)
    normal = np.array([-1.0, 1.0])[:, None]
    traction = np.stack(
        (0.002 * normal * ux - normal * pressure, 0.002 * normal * vx), axis=-1
    )
    # Replace default incoming reservoir traction with the exact prescribed traction.
    incoming = np.minimum(normal * edge[..., 0], 0)[..., None] * edge
    forcing = stream_ns_load(plan, force) + stream_ns_boundary_load(
        plan, traction - incoming
    )
    residual = stream_ns_rhs(plan, a, forcing)
    assert np.max(abs(stream_ns_velocity(plan, residual, nodes=True))) < 2e-9
