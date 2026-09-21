"""Direct compressed application, including actual low-rank blocks and JIT AD."""

import bspf_models.elliptic.pressure as bspf_pressure

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.fft import idct

from bspf_models._numerics._compressed_transform import build_transforms
from bspf_models._numerics._compressed_transform import apply_transform
from bspf_models._numerics._compressed_transform import transform_storage
from bspf_models._numerics._compressed_transform import LayeredBlocks
from bspf_models._numerics._compressed_transform import _blocks


@pytest.mark.parametrize("complex_basis", [False, True])
@pytest.mark.parametrize("layout", ["grouped", "layered"])
def test_factored_transform_and_protected_modes(complex_basis, layout):
    n = 67
    rng = np.random.default_rng(31)
    c = np.eye(n) + 0.003 * rng.normal(size=(n, 3)) @ rng.normal(size=(3, n))
    v = idct(c, type=2, norm="ortho", axis=0)
    if complex_basis:
        v = v * np.exp(1j * np.linspace(0, 1, n))
    vi = np.linalg.inv(v)
    plan = build_transforms(
        v, vi, tolerance=1e-12, leaf_size=8, protected_modes=5, layout=layout
    )
    blocks = (
        plan.forward.levels if isinstance(plan.forward, LayeredBlocks) else plan.forward
    )
    assert any(g.right is not None for g in blocks)
    assert transform_storage(plan)["factor_ratio"] < 0.8
    x = jnp.asarray(rng.normal(size=(n, 4)) + 1j * rng.normal(size=(n, 4)))
    if layout == "layered":
        ir = str(
            jax.jit(_blocks).lower(plan.forward, x).compiler_ir(dialect="stablehlo")
        )
        assert "stablehlo.gather" not in ir
        assert "stablehlo.scatter" not in ir
    for inverse, matrix in [(False, v), (True, vi)]:
        run = jax.jit(partial(apply_transform, inverse=inverse))
        np.testing.assert_allclose(run(plan, x), matrix @ x, rtol=1e-10, atol=1e-11)
    e = np.eye(n)[:, :5]
    np.testing.assert_allclose(
        apply_transform(plan, jnp.asarray(e)), v[:, :5], atol=1e-12
    )
    np.testing.assert_allclose(
        apply_transform(plan, jnp.asarray(v[:, :5]), inverse=True), e, atol=1e-12
    )
    batch = jnp.stack([x, 2 * x])
    got = jax.jit(jax.vmap(apply_transform, in_axes=(None, 0)))(plan, batch)
    np.testing.assert_allclose(got[1], 2 * got[0], atol=1e-12)

    def loss(z):
        y = apply_transform(plan, z)
        return jnp.sum(abs(y) ** 2)

    xr = x.real
    grad = jax.jit(jax.grad(loss))(xr)
    expected = 2 * np.real(v.conj().T @ (v @ np.asarray(xr)))
    np.testing.assert_allclose(grad, expected, rtol=1e-10, atol=1e-10)


def test_pressure_rectangular_direct_and_projection():
    dense = bspf_pressure.plan_pressure_poisson2d(np.linspace(0, 1, 40), np.linspace(0, 1, 41))
    compressed = bspf_pressure.compress_pressure_plan(dense, tolerance=1e-12, leaf_size=8)
    assert compressed.x.vectors is None and compressed.y.inverse_vectors is None
    x, y = jnp.meshgrid(dense.x.x, dense.y.x, indexing="ij")
    p = jnp.exp(x + 0.5 * y) + jnp.sin(3 * x) * jnp.cos(2 * y)
    rhs = bspf_pressure.pressure_schur(dense, p)
    gradient = bspf_pressure.pressure_gradient(dense, p)
    solve = jax.jit(partial(bspf_pressure.solve_pressure_poisson2d, refinement_steps=0))
    ref = solve(dense, rhs, wall_gradient=gradient)
    got = solve(compressed, rhs, wall_gradient=gradient)
    assert bool(got.converged)
    np.testing.assert_allclose(got.pressure, ref.pressure, rtol=1e-8, atol=1e-9)
    # Default compressed solve is identically the zero-refinement path.
    default = jax.jit(bspf_pressure.solve_pressure_poisson2d)(
        compressed, rhs, wall_gradient=gradient
    )
    np.testing.assert_allclose(default.pressure, got.pressure, atol=1e-12)
    raw = jnp.asarray(np.random.default_rng(1).normal(size=p.shape + (2,)))
    project = jax.jit(
        partial(bspf_pressure.project_pressure2d, completion=False, refinement_steps=0)
    )
    vr, _ = project(dense, raw)
    vc, result = project(compressed, raw)
    assert bool(result.converged)
    np.testing.assert_allclose(vc, vr, rtol=1e-8, atol=1e-9)
    assert not bool(solve(compressed, jnp.zeros_like(p).at[0, 0].set(1)).converged)
    with pytest.raises(ValueError, match="already compressed"):
        bspf_pressure.compress_pressure_plan(compressed)


def test_factory_and_validation():
    grid = np.linspace(0, 1, 12)
    plan = bspf_pressure.plan_pressure_poisson2d(
        grid,
        grid,
        q=2,
        n_basis=5,
        degree=4,
        baseline_points=4,
        transform_backend="dct_hodlr",
        protected_modes=2,
    )
    assert plan.x.vectors is None
    assert bool(
        jax.jit(bspf_pressure.solve_pressure_poisson2d)(plan, jnp.zeros((12, 12))).converged
    )
    for options in [
        dict(tolerance=0),
        dict(tolerance=np.nan),
        dict(leaf_size=0),
        dict(protected_modes=1),
        dict(layout="invalid"),
    ]:
        with pytest.raises(ValueError):
            build_transforms(np.eye(8), np.eye(8), **options)
    with pytest.raises(ValueError, match="transform_backend"):
        bspf_pressure.plan_pressure_poisson2d(grid, grid, transform_backend="bad")


def test_layered_zero_rank_and_single_leaf():
    from bspf_models._numerics._compressed_transform import _layers

    for n, leaf in [(17, 4), (3, 8)]:
        matrix = np.diag(np.arange(1.0, n + 1))
        factors = _layers(matrix, 1e-12, leaf)
        values = jnp.arange(n * 3, dtype=jnp.float64).reshape(n, 3)
        result = jax.jit(_blocks)(factors, values)
        np.testing.assert_allclose(result, matrix @ np.asarray(values), atol=1e-13)
