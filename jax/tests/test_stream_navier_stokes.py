"""Independent accuracy and conservation checks for compatible enriched BSPF."""

from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import BSpline
from bspf_jax._weak_basis import mp_trial_values
from bspf_jax.stream_navier_stokes import (
    plan_stream_navier_stokes2d,
    stream_ns_velocity,
    stream_ns_vorticity,
    stream_ns_load,
    stream_ns_rhs,
    stream_ns_divergence,
    stream_evaluate_line,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def plan():
    pytest.importorskip("gmpy2")
    return plan_stream_navier_stokes2d(
        np.linspace(-1, 1, 40), np.linspace(-1, 1, 41), x_layers=(0.002, 0.008, 0.032)
    )


def test_energy_law_and_uniform_advection(plan):
    a = jnp.asarray(np.random.default_rng(4).normal(size=plan.denominator.shape) * 1e-4)
    f = jax.jit(stream_ns_rhs)(plan, a)
    diss = -plan.nu * jnp.sum(
        a
        * (
            plan.x.bending @ a
            + a @ plan.y.bending.T
            + 2 * plan.x.lam[:, None] * a * plan.y.lam[None, :]
        )
    )
    np.testing.assert_allclose(
        jnp.sum(plan.denominator * a * f), diss, atol=2e-12, rtol=2e-12
    )
    # Linearized rotational convection around U=(1,0), independent of nonlinear RHS.
    omega = stream_ns_vorticity(plan, a)
    linear = stream_ns_load(plan, jnp.stack((jnp.zeros_like(omega), -omega), axis=-1))
    assert abs(float(jnp.sum(a * linear))) < 2e-11


def test_exact_divergence_and_clamped_walls(plan):
    a = np.random.default_rng(5).normal(size=plan.denominator.shape) * 1e-4
    assert np.max(abs(stream_ns_divergence(plan, a))) < 1e-11
    vel = np.asarray(stream_ns_velocity(plan, a, nodes=True))
    assert max(np.max(abs(vel[[0, -1]])), np.max(abs(vel[:, [0, -1]]))) < 1e-11
    # Mixed derivatives evaluated at independent, non-nodal physical points.
    bx, gx, _ = stream_evaluate_line(plan.x, [-0.9997, -0.913, 0.127, 0.68, 0.999])
    by, gy, _ = stream_evaluate_line(plan.y, [-0.93, -0.31, 0.17, 0.91])
    ux = (gx @ a) @ gy.T
    vy = -gx @ (a @ gy.T)
    assert np.max(abs(ux + vy)) < 1e-11
    assert np.all(np.isfinite(bx)) and np.all(np.isfinite(by))


def test_pressure_robustness(plan):
    x = plan.x.points[:, None]
    y = plan.y.points[None, :]
    # Exact gradient of exp(.3*x)*cos(x+2*y), no discrete manufacture.
    e = jnp.exp(0.3 * x)
    c = jnp.cos(x + 2 * y)
    s = jnp.sin(x + 2 * y)
    grad = jnp.stack((e * (0.3 * c - s), -2 * e * s), axis=-1)
    acceleration = stream_ns_load(plan, grad) / plan.denominator
    assert np.max(abs(stream_ns_velocity(plan, acceleration, nodes=True))) < 2e-10


def test_mp_second_derivative_polynomial_and_endpoint(plan):
    for line in (plan.x, plan.y):
        x = np.asarray(line.x)
        knots = np.r_[
            np.repeat(x[0], 14),
            np.linspace(x[0], x[-1], 20)[1:-1],
            np.repeat(x[-1], 14),
        ]
        points = np.r_[x[0], np.linspace(-0.93, 0.93, 15), x[-1]]
        b, g, h = mp_trial_values(
            SimpleNamespace(x=x, P=np.asarray(line.projector)),
            BSpline(knots, np.eye(32), 13),
            points,
            second=True,
        )
        u = (1 - x * x) ** 2
        np.testing.assert_allclose(b @ u, (1 - points * points) ** 2, atol=2e-11)
        np.testing.assert_allclose(
            g @ u, -4 * points * (1 - points * points), atol=2e-10
        )
        np.testing.assert_allclose(h @ u, 12 * points * points - 4, atol=2e-8)


def test_enrichment_keeps_smooth_clamped_polynomial(plan):
    x = plan.x.points[:, None]
    y = plan.y.points[None, :]
    u = (1 - x * x) ** 2 * (-4 * y * (1 - y * y))
    v = 4 * x * (1 - x * x) * (1 - y * y) ** 2
    a = stream_ns_load(plan, jnp.stack((u, v), axis=-1)) / plan.denominator
    xx = plan.x.x[:, None]
    yy = plan.y.x[None, :]
    exact = jnp.stack(
        (
            (1 - xx * xx) ** 2 * (-4 * yy * (1 - yy * yy)),
            4 * xx * (1 - xx * xx) * (1 - yy * yy) ** 2,
        ),
        axis=-1,
    )
    assert np.max(abs(stream_ns_velocity(plan, a, nodes=True) - exact)) < 2e-10
