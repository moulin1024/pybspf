"""Manufactured NS acceleration and fixed-boundary time stepping."""

import bspf_models.elliptic.pressure as bspf_pressure

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bspf_models.fluids.navier_stokes import plan_navier_stokes2d
from bspf_models.fluids.navier_stokes import ns_rhs
from bspf_models.fluids.navier_stokes import ns_raw_rhs
from bspf_models.fluids.navier_stokes import ns_rk4_step
from bspf_models.fluids.navier_stokes import ns_vorticity


@pytest.fixture(scope="module")
def plan():
    p = bspf_pressure.plan_pressure_poisson2d(
        jnp.linspace(-1, 1, 12),
        jnp.linspace(-1, 1, 14),
        q=3,
        degree=5,
        n_basis=8,
        baseline_points=8,
        endpoint_method="chebyshev",
        chebyshev_modes=6,
        endpoint_regularization=0.0,
    )
    return plan_navier_stokes2d(p, degree=5, viscosity=0.02)


def test_manufactured_acceleration(plan):
    x, y = jnp.meshgrid(plan.pressure.x.x, plan.pressure.y.x, indexing="ij")

    def h(t):
        return (1 - t * t) ** 2

    def hp(t):
        return -4 * t * (1 - t * t)

    def hpp(t):
        return -4 + 12 * t * t

    u = jnp.stack([h(x) * hp(y), -hp(x) * h(y)], axis=-1)
    dx = jnp.stack([hp(x) * hp(y), -hpp(x) * h(y)], axis=-1)
    dy = jnp.stack([h(x) * hpp(y), -hp(x) * hp(y)], axis=-1)
    lap = jnp.stack(
        [hpp(x) * hp(y) + h(x) * 24 * y, -24 * x * h(y) - hp(x) * hpp(y)], axis=-1
    )
    gradp = jnp.stack([2 * x, 2 * y], axis=-1)
    force = (
        -u + u[..., 0, None] * dx + u[..., 1, None] * dy - plan.viscosity * lap + gradp
    )
    acceleration, diag = jax.jit(ns_rhs)(plan, u, force)
    assert bool(diag.converged)
    np.testing.assert_allclose(acceleration, -u, atol=1e-9)
    np.testing.assert_allclose(
        ns_vorticity(plan, u), dx[..., 1] - dy[..., 0], atol=1e-9
    )


def test_fixed_base_and_boundaries(plan):
    p = plan.pressure
    x, y = jnp.meshgrid(p.x.x, p.y.x, indexing="ij")
    base = jnp.stack([jnp.tanh(y), jnp.zeros_like(x)], axis=-1)
    force = -ns_raw_rhs(plan, base)
    updated, diag = jax.jit(ns_rk4_step)(plan, base, 0.001, force)
    assert bool(diag.converged)
    np.testing.assert_allclose(updated, base, atol=1e-13)
    seed, _ = bspf_pressure.project_pressure2d(
        p, jax.random.normal(jax.random.key(12), base.shape) * 0.001
    )
    velocity = base + seed
    updated, diag = jax.jit(ns_rk4_step)(
        plan, velocity, 0.001, force, reference=base, sponge=jnp.ones_like(x)
    )
    assert bool(diag.converged)
    np.testing.assert_array_equal(
        np.asarray(updated)[np.asarray(p.mask) == 0],
        np.asarray(base)[np.asarray(p.mask) == 0],
    )
    assert float(jnp.max(abs(bspf_pressure.pressure_divergence(p, updated)))) < 1e-9


def test_invalid_ns_parameters(plan):
    with pytest.raises(ValueError, match="positive"):
        plan_navier_stokes2d(plan.pressure, degree=5, viscosity=-1)
    with pytest.raises(ValueError, match="match"):
        plan_navier_stokes2d(plan.pressure, degree=4)
