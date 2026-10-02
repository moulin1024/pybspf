"""Physical MMS, temporal accuracy, JIT/restart and sparse solve validation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from bspf_models.fluids.embedded_navier_stokes import plan_embedded_navier_stokes2d
from bspf_models.fluids._embedded.runtime import factor_solve


def _legacy_plan(*args, **kwargs):
    return plan_embedded_navier_stokes2d(
        *args, incompressibility="stabilized", **kwargs
    )


@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_factor_permutations_and_jit(backend):
    rng = np.random.default_rng(4)
    a = rng.normal(size=(23, 23))
    a[abs(a) < 0.8] = 0
    a[0, 0] = 0
    a = sp.csc_matrix(a)
    solve = jax.jit(factor_solve(a, backend))
    b = rng.normal(size=23)
    np.testing.assert_allclose(solve(jnp.asarray(b)), spsolve(a, b), atol=2e-11)


def exact(p):
    x, y = p.T
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


def force(p):
    x, y = p.T
    return jnp.column_stack(
        (1 - 0.04 * 0.38 + 2 * 0.02**2 * x**3, 1 + 2 * 0.02**2 * x * x * y)
    )


def traction(p, n):
    x, y = p.T
    return (
        0.38
        * jnp.column_stack(
            (0.04 * x * n[:, 0], -0.04 * y * n[:, 0] - 0.04 * x * n[:, 1])
        )
        - (0.3 + x + y)[:, None] * n
    )


@pytest.fixture(scope="module", params=[False, True])
def plan(request):
    return _legacy_plan(
        boundary,
        dt=0.025,
        viscosity=0.38,
        cells=3,
        degree=3 if request.param else 2,
        prior_levels=1 if request.param else 0,
        prior_modulation=1,
        prior_surface_samples=8,
        prior_velocity_halo=0.5,
        order=48 if request.param else 24,
        convection_order=48 if request.param else 24,
        outflow=True,
        linear_backend="dense",
    )


def initial(plan):
    s = plan.spatial

    def f(p):
        return np.tile([1 - 0.04 * 0.38, 1.0], (len(p), 1))

    return plan.initialize(spsolve(s.A, s.rhs(boundary, f, traction)))


def test_manufactured_solution_and_loads(plan):
    state = initial(plan)
    load = plan.load(
        force(plan.force_points), traction(plan.traction_points, plan.traction_normals)
    )
    expected = plan.spatial.rhs(
        boundary,
        lambda p: np.asarray(force(p)),
        lambda p, n: np.asarray(traction(p, n)),
    )
    np.testing.assert_allclose(load, expected, atol=3e-13)
    before = state.coefficients
    state = plan.advance(state, 4, load=lambda t: load)
    points = np.array([[-0.8, 0.4], [0.7, 0.6], [0.6, -0.7]])
    np.testing.assert_allclose(plan.evaluate(points, state), exact(points), atol=2e-8)
    np.testing.assert_allclose(state.coefficients, before, atol=2e-8)
    assert plan.size == 2 * plan.nv + plan.spatial.np


def test_jit_scan_and_restart(plan):
    state = initial(plan)
    load = plan.load(
        force(plan.force_points), traction(plan.traction_points, plan.traction_normals)
    )

    def body(s, _):
        return plan.step(s, load)

    final, diagnostics = jax.jit(lambda s: jax.lax.scan(body, s, None, length=4))(state)
    checkpoint = plan.advance(state, 2, load=lambda t: load)
    # Emulate a serialized/reloaded complete state, including AB2 history.
    restored = type(checkpoint)(*[jnp.asarray(np.array(x)) for x in checkpoint])
    resumed = plan.advance(restored, 2, load=lambda t: load)
    np.testing.assert_allclose(final.coefficients, resumed.coefficients, atol=2e-11)
    assert np.all(diagnostics.valid)
    assert int(resumed.step) == 4


def test_bad_data_flag_and_validation(plan):
    state = initial(plan)
    _, diag = jax.jit(plan.step)(state, jnp.full(plan.size, jnp.nan))
    assert not bool(diag.valid)
    with pytest.raises(RuntimeError):
        plan.advance(state, 1, load=lambda t: jnp.full(plan.size, jnp.nan))
    with pytest.raises(ValueError):
        plan.initialize(np.zeros(3))
    with pytest.raises(ValueError):
        plan.load(jnp.zeros((3, 2)))
    with pytest.raises(ValueError):
        _legacy_plan(boundary, dt=-1, viscosity=1)


def test_closed_domain_gauge_and_sparse_step():
    def zero(p, tag):
        return np.zeros_like(p)

    plan = _legacy_plan(
        zero,
        dt=0.01,
        viscosity=1,
        cells=3,
        degree=2,
        order=18,
        convection_order=18,
        linear_backend="sparse",
    )
    assert plan.size == 2 * plan.nv + plan.spatial.np + 1
    state = plan.advance(plan.initialize(np.zeros(plan.size)), 2)
    np.testing.assert_array_equal(state.coefficients, 0)


def test_temporal_second_order(plan):
    # Mixed semidiscrete MMS with analytic solution exp(t)*c. The full load
    # includes the exact algebraic constraint; this isolates time accuracy.
    c = initial(plan).coefficients * 0.01
    errors = []
    for dt in (0.02, 0.01, 0.005):
        s = plan.spatial
        from bspf_models.fluids.embedded_navier_stokes import EmbeddedNavierStokes2D

        p = EmbeddedNavierStokes2D(s, boundary, dt, 24, "dense", 1e-9)

        def load(t):
            exact = jnp.exp(t) * c
            result = p.operator @ exact + p.mass @ exact
            return result.at[: 2 * p.nv].add(p.convection(exact))

        result = p.advance(p.initialize(c), round(0.1 / dt), load=load)
        errors.append(
            float(
                jnp.linalg.norm(
                    result.coefficients[: 2 * p.nv] - jnp.exp(0.1) * c[: 2 * p.nv]
                )
            )
        )
    rates = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.min(rates) > 1.8, (errors, rates)


def test_nonzero_sparse_step_matches_dense():
    options = dict(
        dt=0.025,
        viscosity=0.38,
        cells=3,
        degree=2,
        order=24,
        convection_order=24,
        outflow=True,
    )
    dense = _legacy_plan(boundary, linear_backend="dense", **options)
    sparse = _legacy_plan(boundary, linear_backend="sparse", **options)
    state = dense.stokes_initial_state()
    a = dense.advance(state, 3)
    b = sparse.advance(state, 3)
    np.testing.assert_allclose(a.coefficients, b.coefficients, atol=2e-9, rtol=2e-9)
