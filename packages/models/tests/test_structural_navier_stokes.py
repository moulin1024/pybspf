"""Incompressibility is a separate acceptance criterion, not a pressure penalty."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bspf_models.fluids.embedded_navier_stokes import plan_embedded_navier_stokes2d
from bspf_models.fluids._embedded.incompressible import ConstrainedSpace


def exact(p):
    x, y = np.asarray(p).T
    return np.column_stack(
        (
            0.02 * x * x,
            -0.04 * x * y,
            0.3 + x + y,
            0.04 * x,
            0 * x,
            -0.04 * y,
            -0.04 * x,
        )
    )


def boundary(p, tag):
    return exact(p)[:, :2]


def traction(p, n, viscosity):
    f = exact(p)
    return (
        viscosity
        * np.column_stack(
            (
                f[:, 3] * n[:, 0] + f[:, 4] * n[:, 1],
                f[:, 5] * n[:, 0] + f[:, 6] * n[:, 1],
            )
        )
        - f[:, 2, None] * n
    )


@pytest.fixture(scope="module", params=[(0.38, False), (0.00038, False), (0.38, True)])
def plan(request):
    viscosity, enriched = request.param
    return plan_embedded_navier_stokes2d(
        boundary,
        dt=0.001,
        viscosity=viscosity,
        cells=3,
        degree=3,
        order=48,
        convection_order=48,
        outflow=True,
        linear_backend="dense",
        prior_levels=1 if enriched else 0,
        prior_modulation=1,
        prior_surface_samples=8,
        prior_velocity_halo=0.5,
    )


def start(plan):
    nu = plan.spatial.viscosity
    return plan.stokes_initial_state(
        np.tile([1 - 0.04 * nu, 1.0], (len(plan.force_points), 1)),
        traction(plan.traction_points, plan.traction_normals, nu),
    )


def test_dense_wall_constraints_and_jit_audit():
    p = plan_embedded_navier_stokes2d(
        boundary, dt=.001, viscosity=.38, cells=3, degree=3,
        order=40, convection_order=40, outflow=True,
        linear_backend="dense", wall_enforcement="constraint",
    )
    state = start(p)
    assert float(jax.jit(p.wall_slip_error)(state.coefficients)) < 1e-10
    points = np.array([[-.8, .4], [.7, .6]])
    np.testing.assert_allclose(p.evaluate(points, state), exact(points), atol=2e-8)


def test_mms_velocity_pressure_and_exact_constraints(plan):
    state = start(plan)
    x, y = plan.force_points.T
    nu = plan.spatial.viscosity
    force = jnp.column_stack(
        (1 - 0.04 * nu + 2 * 0.02**2 * x**3, 1 + 2 * 0.02**2 * x * x * y)
    )
    load = plan.load(force, traction(plan.traction_points, plan.traction_normals, nu))
    final, history = jax.jit(
        lambda s: jax.lax.scan(lambda q, _: plan.step(q, load), s, None, length=5)
    )(state)
    points = np.array([[-0.8, 0.4], [0.7, 0.6], [0.6, -0.7]])
    np.testing.assert_allclose(plan.evaluate(points, final), exact(points), atol=2e-8)
    assert np.all(history.valid)
    assert np.max(history.divergence_l2) < 1e-10
    assert np.max(history.normal_jump_l2) < 1e-10
    assert np.max(history.boundary_normal_l2) < 1e-10
    n = 2 * plan.nv
    assert plan.spatial.A[n:, n:].nnz == 0
    assert plan.info["unconstrained_velocity_dofs"] > 0
    assert not plan.info["pressure_stabilization_active"]


def test_initial_data_and_forged_constraint_load_rejected(plan):
    state = start(plan)
    corrupt = np.array(state.coefficients)
    corrupt[5] += 0.1
    with pytest.raises(ValueError, match="structural incompressibility"):
        plan.initialize(corrupt)
    bad_load = plan.boundary_load.at[2 * plan.nv].add(1.0)
    _, diagnostics = jax.jit(plan.step)(state, bad_load)
    assert not bool(diagnostics.valid)
    with pytest.raises(RuntimeError):
        plan.advance(state, 1, load=lambda t: bad_load)


def test_gradient_force_does_not_drive_velocity(plan):
    nu = plan.spatial.viscosity
    f = np.tile([1 - 0.04 * nu, 1.0], (len(plan.force_points), 1))
    t = traction(plan.traction_points, plan.traction_normals, nu)
    state = plan.stokes_initial_state(f, t)
    x, y = np.asarray(plan.traction_points).T
    added = 0.7 * x - 0.2 * y
    changed = plan.stokes_initial_state(
        f + np.array([0.7, -0.2]), t - added[:, None] * plan.traction_normals
    )
    np.testing.assert_allclose(
        changed.coefficients[: 2 * plan.nv],
        state.coefficients[: 2 * plan.nv],
        atol=2e-9,
    )


def test_closed_gauge_and_sparse_runtime():
    p = plan_embedded_navier_stokes2d(
        boundary,
        dt=0.001,
        viscosity=0.38,
        cells=3,
        degree=3,
        order=40,
        convection_order=40,
        linear_backend="sparse",
    )
    s = start(p)
    points = np.array([[-0.8, 0.4], [0.7, 0.6], [0.6, -0.7]])
    expected = exact(points)
    area = 4 - np.pi * 0.27 * 0.19
    mean = 0.3 - np.pi * 0.27 * 0.19 * (0.13 - 0.07) / area
    expected[:, 2] -= mean
    np.testing.assert_allclose(p.evaluate(points, s), expected, atol=2e-8)
    _, d = jax.jit(p.step)(s)
    assert bool(d.valid)
    assert float(d.divergence_l2) < 1e-10
    with pytest.raises(ValueError, match="no stabilized fallback"):
        ConstrainedSpace(p.spatial.base, boundary, max_velocity_dofs=1)


def test_incompatible_closed_flux_rejected():
    def expanding(p, tag):
        return np.asarray(p)

    with pytest.raises(ValueError, match="Incompatible"):
        plan_embedded_navier_stokes2d(
            expanding,
            dt=0.01,
            viscosity=1,
            cells=3,
            degree=3,
            order=32,
            linear_backend="dense",
        )


def test_second_order_on_constrained_time_dependent_flow():
    from bspf_models.fluids.embedded_navier_stokes import EmbeddedNavierStokes2D

    p = plan_embedded_navier_stokes2d(
        boundary,
        dt=0.01,
        viscosity=0.38,
        cells=3,
        degree=3,
        order=40,
        convection_order=40,
        outflow=True,
        linear_backend="dense",
    )
    s = p.spatial
    c = start(p).coefficients
    # A local polynomial curl with zero normal trace on all four cell faces.
    root = next(
        r
        for r in s.roots
        if r in s.grid.full and sum(a == r for a in s.owner.values()) == 1
    )
    points, w = s.volume[root]
    basis = s.vbasis[root]
    x, y = ((points - basis.origin) / basis.scale).T
    hx, hy = basis.scale
    field = np.column_stack(
        (x * (1 - x) * (1 - 2 * y) / hy, -(1 - 2 * x) * y * (1 - y) / hx)
    )
    v = basis.evaluate(points, 0)[0]
    local = np.linalg.lstsq(
        np.sqrt(w[:, None]) * v, np.sqrt(w[:, None]) * field, rcond=None
    )[0]
    bubble = np.zeros(p.size)
    bubble[s.vi[root]] = local[:, 0]
    bubble[s.nv + s.vi[root]] = local[:, 1]
    bubble = jnp.asarray(bubble) * 0.01
    errors = []
    for dt in (0.01, 0.005, 0.0025):
        q = EmbeddedNavierStokes2D(s, boundary, dt, 40, "dense", 1e-9)

        def load(t):
            co = c + jnp.sin(t) * bubble
            rhs = q.operator @ co + q.mass @ (jnp.cos(t) * bubble)
            return rhs.at[: 2 * q.nv].add(q.convection(co))

        state = q.advance(q.initialize(c), round(0.04 / dt), load=load)
        errors.append(
            float(
                jnp.linalg.norm(
                    state.coefficients[: 2 * q.nv]
                    - (c + jnp.sin(0.04) * bubble)[: 2 * q.nv]
                )
            )
        )
    rates = np.log2(np.asarray(errors[:-1]) / errors[1:])
    assert np.min(rates) > 1.8, (errors, rates)
