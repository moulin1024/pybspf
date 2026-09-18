"""Exact geometry, analytic NS forcing, buffer work and independent wall checks."""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la
import pytest
from numpy.polynomial import Polynomial

from bspf_jax.immersed_flow import (
    ImmersedFlowPlan,
    channel_quadrature,
    channel_lift,
    elliptic_wall_factor,
)
from bspf_jax.immersed_poisson import EllipticHole

jax.config.update("jax_enable_x64", True)


def test_curved_quadrature_moments():
    hole = EllipticHole()
    p, w = channel_quadrature((-1, 5, 1), hole, 49, 25, 1.5, (3.0,))
    x, y = p.T
    cx, cy = hole.center
    a, b = hole.axes
    area = np.pi * a * b
    assert np.all(hole.level(p) > 1)
    assert np.all(w > 0)
    np.testing.assert_allclose(
        [w.sum(), w @ x, w @ y, w @ (x * x), w @ (y * y)],
        [
            12 - area,
            24 - area * cx,
            -area * cy,
            84 - area * (cx * cx + a * a / 4),
            4 - area * (cy * cy + b * b / 4),
        ],
        atol=5e-12,
        rtol=5e-14,
    )


@pytest.fixture(scope="module")
def channel():
    return ImmersedFlowPlan(nx=33, ny=25, hole=None)


def test_poiseuille_and_buffer_support(channel):
    p = channel
    np.testing.assert_array_equal(p.sponge_profile(np.array([-1.0, 0.0, 2.0, 3.0])), 0)
    sigma = p.sponge_profile(np.linspace(3, 5, 151))
    assert np.all(np.diff(sigma) >= 0)
    assert sigma[-1] == 3
    velocity = [op @ p.stokes_state for op in p.operators_fluid[:2]]
    assert np.max(np.hypot(*velocity)) < 1e-8
    assert np.linalg.norm(p.explicit(np.zeros(p.dofs))) < 1e-7
    random = np.random.default_rng(8).normal(size=p.dofs) * 1e-3
    u, v = (a @ random for a in p.operators_fluid[:2])
    direct = np.sum(p.weights * p.sigma * (u * u + v * v))
    np.testing.assert_allclose(random @ p.sponge @ random, direct, rtol=2e-12)
    assert direct > 0


def manufactured(p):
    """Analytic polynomial streamfunction perturbation; pressure linear in x.

    Inlet/wall velocity fixed. At outlet perturbation value, first and second
    derivatives vanish, so the physical traction is exactly zero.
    """
    x, y = p.points.T
    left, right, h = p.bounds
    length = right - left
    t = (x - left) / length
    pp = 5 * Polynomial([0, 0, 1, -3, 3, -1])
    qq = Polynomial([1, 0, -2, 0, 1])
    px = [pp.deriv(k)(t) / length**k for k in range(4)]
    qy = [qq.deriv(k)(y / h) / h**k for k in range(4)]
    du, dv = px[0] * qy[1], -px[1] * qy[0]
    exact_coeff = la.cho_solve(p.mass_factor, p.force_load(np.column_stack((du, dv))))

    def load(time):
        s, rate = np.sin(time), np.cos(time)
        base = channel_lift(p.points, h, p.peak)
        u, v = base[1] + s * du, s * dv
        ux, uy = s * px[1] * qy[1], base[4] + s * px[0] * qy[2]
        vx, vy = -s * px[2] * qy[0], -ux
        lapu = -2 * p.peak / h**2 + s * (px[2] * qy[1] + px[0] * qy[3])
        lapv = -s * (px[3] * qy[0] + px[1] * qy[2])
        fx = (
            rate * du
            + u * ux
            + v * uy
            - p.nu * lapu
            - 2 * p.nu * p.peak / h**2
            + p.sigma * s * du
        )
        fy = rate * dv + u * vx + v * vy - p.nu * lapv + p.sigma * s * dv
        return p.force_load(np.column_stack((fx, fy)))

    return exact_coeff, load


def test_continuous_ns_residual_and_second_order(channel):
    p = channel
    coefficient, load = manufactured(p)
    at = 0.31
    residual = (
        p.explicit(np.sin(at) * coefficient)
        + load(at)
        - p.linear @ (np.sin(at) * coefficient)
        - p.mass @ (np.cos(at) * coefficient)
    )
    assert np.linalg.norm(residual) < 1e-7
    errors = []
    for dt in (0.08, 0.04, 0.02):
        step = p.stepper(dt)
        state = np.zeros(p.dofs)
        for k in range(round(0.4 / dt)):
            state = step.step(state, k * dt, load)
        error = state - np.sin(0.4) * coefficient
        errors.append(np.sqrt(error @ p.mass @ error))
    assert errors[0] / errors[1] > 3.5, errors
    assert errors[1] / errors[2] > 3.5, errors
    assert errors[-1] < 1e-5, errors


@pytest.fixture(scope="module")
def factored_hole():
    return ImmersedFlowPlan(nx=33, ny=25, wall_method="factor", quadrature_factor=4)


@pytest.mark.parametrize("method", ["svd", "factor"])
def test_hole_constraints_and_physical_flux(method, factored_hole):
    p = factored_hole if method == "factor" else ImmersedFlowPlan(nx=33, ny=25)
    a = p.stokes_state.copy()
    step = p.stepper(0.02)
    for k in range(5):
        a = step.step(a, k * 0.02)
    boundary, t = p.arc.sample(190, offset=0.381)
    _, u, v, *_ = p.evaluate(a, boundary)
    assert np.max(np.hypot(u, v)) < 2e-7
    x, y = np.linspace(-1, 5, 51), np.linspace(-1, 1, 45)
    fields = p.grid(a, x, y)
    np.testing.assert_allclose(fields["u"][[0, -1]], 0, atol=2e-11)
    np.testing.assert_allclose(fields["v"][[0, -1]], 0, atol=2e-11)
    np.testing.assert_allclose(fields["u"][:, 0], 1 - y * y, atol=2e-11)
    np.testing.assert_allclose(fields["v"][:, 0], 0, atol=2e-11)
    np.testing.assert_allclose(
        p.out_weights @ (p.out_ops[0] @ a + p.out_lift[0]), 4 / 3, atol=2e-9
    )
    assert np.all(p.hole.level(p.points) > 1)


def test_wall_factor_derivatives():
    hole = EllipticHole()
    bounds = (-1, 5, 1)
    points = np.random.default_rng(412).uniform([-1, -1], [5, 1], (67, 2))

    def value(z):
        x, y = z
        (cx, cy), (a, b) = hole.center, hole.axes
        f = ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 - 1
        g = (x + 1) * (5 - x) * (1 - y * y) / ((cx + 1) * (5 - cx) * (1 - cy * cy))
        return f * f / (f * f + 4 * g * g)

    q, qx, qy, qxx, qxy, qyy = elliptic_wall_factor(points, bounds, hole)
    d = np.asarray(jax.vmap(jax.grad(value))(points))
    dd = np.asarray(jax.vmap(jax.hessian(value))(points))
    np.testing.assert_allclose(q, jax.vmap(value)(points), atol=2e-15)
    np.testing.assert_allclose(np.array([qx, qy]).T, d, atol=2e-14)
    np.testing.assert_allclose(
        np.array([qxx, qxy, qyy]).T, dd[:, (0, 0, 1), (0, 1, 1)], atol=2e-13
    )


def test_factored_continuous_ns_mms(factored_hole):
    """Independent continuous forcing, including outlet traction and pressure.

    The exact field contains the rational geometry and a nonzero unknown hole
    streamfunction constant. It is not a force manufactured from matrix actions.
    """
    p = factored_hole
    left, right, h = p.bounds
    (cx, cy), (a, b) = p.hole.center, p.hole.axes
    constant = 0.61

    def psi(z):
        x, y = z
        f = ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 - 1
        g = (
            (x - left)
            * (right - x)
            * (1 - (y / h) ** 2)
            / ((cx - left) * (right - cx) * (1 - (cy / h) ** 2))
        )
        q = f * f / (f * f + p.wall_width**2 * g * g)
        return q * p.peak * (y - y**3 / (3 * h * h) + 2 * h / 3) + (1 - q) * constant

    grad = jax.grad(psi)
    hess = jax.jacfwd(grad)
    third = jax.jacfwd(hess)

    @jax.jit
    def exact(points):
        d = jax.vmap(grad)(points)
        dd = jax.vmap(hess)(points)
        ddd = jax.vmap(third)(points)
        return jnp.stack(
            (
                d[:, 1],
                -d[:, 0],
                dd[:, 0, 1],
                dd[:, 1, 1],
                -dd[:, 0, 0],
                ddd[:, 1, 0, 0] + ddd[:, 1, 1, 1],
                -ddd[:, 0, 0, 0] - ddd[:, 0, 1, 1],
            )
        )

    u, v, ux, uy, vx, lapu, lapv = np.asarray(exact(p.points))
    force = np.column_stack(
        (
            u * ux
            + v * uy
            - p.nu * lapu
            - 2 * p.nu * p.peak / h**2
            + p.sigma * (u - p.peak * (1 - (p.points[:, 1] / h) ** 2)),
            u * vx - v * ux - p.nu * lapv + p.sigma * v,
        )
    )
    load = p.force_load(force)
    _, _, ox, _, ovx, *_ = np.asarray(exact(p.out_points))
    load += p.nu * (
        p.out_ops[0].T @ (p.out_weights * ox) + p.out_ops[1].T @ (p.out_weights * ovx)
    )
    state = la.cho_solve(
        p.mass_factor,
        p.force_load(
            np.column_stack(
                (
                    u - p.lift_fields[1],
                    v - p.lift_fields[2],
                )
            )
        ),
    )
    residual = p.explicit(state) + load - p.linear @ state
    assert la.norm(residual) < 1e-6
    velocity_error = np.hypot(
        p.operators_fluid[0] @ state + p.lift_fields[1] - u,
        p.operators_fluid[1] @ state + p.lift_fields[2] - v,
    )
    assert np.max(velocity_error) < 1e-10
    boundary, _ = p.arc.sample(194, offset=0.31)
    fields = p.evaluate(state, boundary)
    np.testing.assert_allclose(fields[0], constant, atol=2e-11)
    assert np.max(np.hypot(fields[1], fields[2])) < 2e-13
