"""Algebraic energy laws and independent reference for the SBP NS option."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import lstsq

import bspf_jax as b
from bspf_jax._energy_stable import sbp84_derivative, sbp84_diffusion


@pytest.fixture(scope="module")
def plan():
    pressure = b.plan_pressure_poisson2d(
        np.linspace(-1, 1, 24),
        np.linspace(-1, 1, 25),
        q=3,
        degree=5,
        n_basis=8,
        baseline_points=8,
        endpoint_method="chebyshev",
        chebyshev_modes=6,
    )
    return b.plan_navier_stokes2d(pressure, closure="sbp84", viscosity=0.02)


@pytest.mark.parametrize("n", [24, 41, 80])
def test_sbp_identity_and_exactness(n):
    x = np.linspace(-1, 2, n)
    d, h = map(np.asarray, sbp84_derivative(x))
    boundary = np.zeros(n)
    boundary[0], boundary[-1] = -1, 1
    np.testing.assert_allclose(h[:, None] * d + d.T * h, np.diag(boundary), atol=2e-15)
    for k in range(5):
        np.testing.assert_allclose(
            d @ x**k, 0 if k == 0 else k * x ** (k - 1), atol=2e-10
        )
    for k in range(5, 9):
        np.testing.assert_allclose(
            (d @ x**k)[8:-8], (k * x ** (k - 1))[8:-8], atol=2e-10
        )
    diffusion = np.asarray(sbp84_diffusion(d, h))
    if n > 24:
        nyquist = (-1.0) ** np.arange(n)
        np.testing.assert_allclose(
            (diffusion @ nyquist)[12:-12],
            (-4 / (x[1] - x[0]) ** 2 * nyquist)[12:-12],
            rtol=1e-12,
        )
    for speed in [-1.0, 1.0]:
        a = (-speed * d + 0.002 * diffusion)[1:-1, 1:-1]
        root = np.sqrt(h[1:-1])
        scaled = root[:, None] * a / root[None, :]
        assert np.linalg.eigvalsh((scaled + scaled.T) / 2).max() < 0


def test_boundary_convergence():
    errors = []
    for n in [40, 80, 160]:
        x = np.linspace(-1, 1, n)
        d, _ = sbp84_derivative(x)
        errors.append(np.max(abs(np.asarray(d) @ np.exp(x) - np.exp(x))))
    assert errors[0] / errors[1] > 12
    assert errors[1] / errors[2] > 12


def test_direct_projection_against_dense_constraint_svd(plan):
    nx, ny = plan.pressure.mask.shape
    root = np.asarray(plan.energy.sqrt_weights)
    # Explicit divergence matrix only in this small independent reference test.
    ex, ey = np.eye(nx)[:, 1:-1], np.eye(ny)[:, 1:-1]
    bx = np.kron(np.asarray(plan.dx)[:, 1:-1], ey)
    by = np.kron(ex, np.asarray(plan.dy)[:, 1:-1])
    mass_root = np.tile(root[1:-1, 1:-1].ravel(), 2)
    constraint = np.hstack([bx, by]) / mass_root
    raw = np.random.default_rng(7).normal(size=(nx, ny, 2))
    projected, diagnostics = jax.jit(b.ns_project_velocity)(plan, jnp.asarray(raw))
    assert bool(diagnostics.converged)
    r = np.concatenate([raw[1:-1, 1:-1, k].ravel() for k in range(2)]) * mass_root
    correction = lstsq(constraint, constraint @ r, cond=1e-12, lapack_driver="gelsd")[0]
    expected = r - correction
    actual = (
        np.concatenate([np.asarray(projected)[1:-1, 1:-1, k].ravel() for k in range(2)])
        * mass_root
    )
    np.testing.assert_allclose(actual, expected, atol=2e-12)
    assert np.max(abs(np.asarray(b.ns_divergence(plan, projected)))) < 1e-10
    again, _ = b.ns_project_velocity(plan, projected)
    np.testing.assert_allclose(again, projected, atol=2e-12)
    assert abs(np.dot(actual, r - actual)) < 1e-11
    assert np.linalg.norm(actual) <= np.linalg.norm(r)
    np.testing.assert_array_equal(
        np.asarray(projected)[np.asarray(plan.pressure.mask) == 0], 0.0
    )


def test_nonlinear_energy_law_and_uniform_flow(plan):
    raw = jax.random.normal(jax.random.key(3), plan.pressure.mask.shape + (2,))
    u, _ = b.ns_project_velocity(plan, raw)
    acceleration, diag = jax.jit(b.ns_rhs)(plan, u)
    assert bool(diag.converged)
    weights = plan.energy.sqrt_weights**2
    dx = jnp.einsum("ij,jkc->ikc", plan.dx, u)
    dy = jnp.einsum("ij,kjc->kic", plan.dy, u)
    rate = jnp.sum(weights[..., None] * u * acceleration)
    dissipation = -plan.viscosity * jnp.sum(weights[..., None] * (dx**2 + dy**2))
    # Independent fifth-difference energy remainder (not the assembled L).
    kx, ky = np.diff(np.asarray(u), n=5, axis=0), np.diff(np.asarray(u), n=5, axis=1)
    hx, hy = plan.energy.x.weights, plan.energy.y.weights
    dissipation -= plan.viscosity * (
        jnp.sum(hy[None, :, None] * kx**2) / (256 * hx[8])
        + jnp.sum(hx[:, None, None] * ky**2) / (256 * hy[8])
    )
    np.testing.assert_allclose(rate, dissipation, rtol=2e-12, atol=1e-10)
    base = jnp.zeros_like(u).at[..., 0].set(1.0)
    linear = jax.jvp(lambda v: b.ns_rhs(plan, v)[0], (base,), (u,))[1]
    np.testing.assert_allclose(
        jnp.sum(weights[..., None] * u * linear), dissipation, rtol=2e-12, atol=1e-10
    )
    force = -b.ns_raw_rhs(plan, base)
    updated, _ = jax.jit(b.ns_rk4_step)(plan, base, 1e-3, force)
    np.testing.assert_allclose(updated, base, atol=1e-14)


def test_invalid_grid_and_closure(plan):
    with pytest.raises(ValueError, match="24"):
        sbp84_derivative(np.linspace(0, 1, 23))
    with pytest.raises(ValueError, match="uniform"):
        sbp84_derivative(np.linspace(0, 1, 24) ** 2)
    with pytest.raises(ValueError, match="closure"):
        b.plan_navier_stokes2d(plan.pressure, closure="invalid")
