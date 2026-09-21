"""Noise-aware differentiation: independent joint-fit references and tensor semantics."""

import pybspf.calculus as bspf_calculus
import pybspf.operators as bspf_operators
import pybspf.plans as bspf_plans

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import BSpline

import pybspf as b


def dense_operators(p, alpha):
    """Independent full Fourier/spline least-squares reference (small grids only)."""
    x, knots = np.asarray(p.x), np.asarray(p.knots)
    n, m = len(x), p.basis[0].shape[1]
    spline = BSpline(knots, np.eye(m), p.degree)
    nodes, weights = np.polynomial.legendre.leggauss(p.degree - 1)  # p=2
    breaks = knots[p.degree : -p.degree]
    width = np.diff(breaks)
    points = (
        (breaks[:-1, None] + breaks[1:, None]) / 2 + width[:, None] * nodes / 2
    ).ravel()
    weights = (width[:, None] * weights / 2 * n / (x[-1] - x[0])).ravel()
    roughness = np.sqrt(weights[:, None]) * spline(points, nu=2)
    F = np.fft.ifft(np.eye(n), axis=0, norm="ortho")[:, 1:]
    omega = 2 * np.pi * np.fft.fftfreq(n, x[1] - x[0])[1:]
    A = np.column_stack([spline(x), F])
    penalty = np.zeros((len(points) + n - 1, m + n - 1), dtype=complex)
    penalty[: len(points), :m] = roughness
    penalty[len(points) :, m:] = np.diag(omega**2)
    projector = np.linalg.lstsq(
        np.vstack([A, np.sqrt(alpha) * penalty]),
        np.vstack([np.eye(n), np.zeros((len(penalty), n))]),
        rcond=None,
    )[0]
    return tuple(
        np.column_stack([spline(x, nu=k), F * (1j * omega) ** k]) @ projector
        for k in range(3)
    )


def along(matrix, f, axis, real):
    value = np.moveaxis(
        np.tensordot(matrix, np.moveaxis(f, axis, 0), axes=(1, 0)), 0, axis
    )
    return value.real if real else value


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("complex_input", [False, True])
def test_joint_fit_against_dense_tensor_reference(dimension, complex_input):
    xs = [np.linspace(-0.2, 1.2, n) for n in (17, 18, 19)[:dimension]]
    axes = [
        bspf_plans.plan_1d(
            x, degree=3, n_basis=7, noise_std=0.01, noise_alphas=[1e-6, 1e-4, 1e-2]
        )
        for x in xs
    ]
    p = axes[0] if dimension == 1 else bspf_plans.tensor_plan(*axes)
    mesh = np.meshgrid(*xs, indexing="ij")
    clean = np.sin(sum((i + 1) * x for i, x in enumerate(mesh)))
    rng = np.random.default_rng(251)
    f = np.stack([clean, 2 * clean], axis=-1) + 0.01 * rng.normal(
        size=clean.shape + (2,)
    )
    if complex_input:
        f = f + 1j * (0.3 * clean[..., None] + 0.01 * rng.normal(size=f.shape))
    selections = jax.jit(bspf_operators.noise_diagnostics)(p, f)
    operators = []
    for axis, pa in enumerate(axes):
        candidates = [dense_operators(pa, float(a)) for a in pa.noise.alphas]
        ratios = np.array(
            [
                np.linalg.norm(along(m[0], f, axis, not complex_input) - f)
                / (0.01 * np.sqrt(f.size))
                for m in candidates
            ]
        )
        index = int(np.argmin(abs(ratios - 1)))
        assert int(selections[axis].index) == index
        np.testing.assert_allclose(
            selections[axis].residual_ratio, ratios[index], rtol=1e-8, atol=1e-9
        )
        operators.append(candidates[index])

    def reference(orders):
        result = f
        for axis, k in enumerate(orders):
            result = along(operators[axis][k], result, axis, not complex_input)
        return result

    mixed = (1,) * dimension
    np.testing.assert_allclose(
        jax.jit(partial(bspf_operators.mixed_partial, orders=mixed))(p, f),
        reference(mixed),
        atol=2e-7,
        rtol=2e-8,
    )
    np.testing.assert_allclose(
        jax.jit(partial(bspf_operators.differentiate, order=0))(p, f),
        reference((0,) * dimension),
        atol=2e-8,
    )
    expected_gradient = np.stack(
        [
            reference(tuple(int(i == j) for j in range(dimension)))
            for i in range(dimension)
        ]
    )
    np.testing.assert_allclose(
        jax.jit(bspf_operators.gradient)(p, f), expected_gradient, atol=2e-7, rtol=2e-8
    )
    expected_lap = sum(
        reference(tuple(2 * int(i == j) for j in range(dimension)))
        for i in range(dimension)
    )
    np.testing.assert_allclose(
        jax.jit(bspf_operators.laplacian)(p, f), expected_lap, atol=2e-7, rtol=2e-8
    )
    assert jax.tree_util.tree_leaves(p)
    jax.clear_caches()


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_noisy_gradient_improves_seeded_smooth_field(dimension):
    xs = [jnp.linspace(0, 1, n) for n in (33, 35, 37)[:dimension]]
    construct = (bspf_plans.plan_1d, bspf_plans.plan_2d, bspf_plans.plan_3d)[dimension - 1]
    options = dict(degree=5, n_basis=12)
    clean_plan = construct(*xs, **options)
    noisy_plan = construct(*xs, **options, noise_std=0.001)
    mesh = jnp.meshgrid(*xs, indexing="ij")
    phase = sum((i + 1) * x for i, x in enumerate(mesh))
    truth = jnp.stack([(i + 1) * jnp.cos(phase) for i in range(dimension)])
    f = jnp.sin(phase) + 0.001 * jnp.asarray(
        np.random.default_rng(45).normal(size=phase.shape)
    )
    raw = jax.jit(bspf_operators.gradient)(clean_plan, f)
    fitted = jax.jit(bspf_operators.gradient)(noisy_plan, f)
    error = float(jnp.linalg.norm(fitted - truth) / jnp.linalg.norm(truth))
    raw_error = float(jnp.linalg.norm(raw - truth) / jnp.linalg.norm(truth))
    print(
        f"{dimension}D gradient relative L2: clean operator={raw_error:.6g}, noisy operator={error:.6g}"
    )
    assert error < 0.7 * raw_error
    # Parallel BLAS reductions can change the last bits without changing the
    # selected operator or numerical result at scientific precision.
    np.testing.assert_allclose(
        fitted, jax.jit(bspf_operators.gradient)(noisy_plan, f), rtol=2e-12, atol=2e-12
    )
    jax.clear_caches()


def test_zero_noise_preserves_clean_path_and_validates_options():
    x = jnp.linspace(0, 1, 33)
    f = jnp.sin(x)
    a = bspf_plans.plan_1d(x)
    z = bspf_plans.plan_1d(x, noise_std=0.0)
    assert z.noise is None
    np.testing.assert_array_equal(bspf_operators.differentiate(a, f), bspf_operators.differentiate(z, f))
    for kwargs in (
        {"noise_std": -1},
        {"noise_std": "auto"},
        {"noise_std": 1j},
        {"noise_std": np.nan},
        {"noise_std": [0.1]},
        {"noise_std": 0.1, "noise_alphas": [1.0, 0.0]},
        {"noise_std": 0.1, "noise_alphas": [1.0]},
        {"noise_std": 0.1, "noise_alphas": [0.0, 1.0]},
        {"noise_std": 0.0, "noise_alphas": [1.0, 2.0]},
        {"noise_std": 0.1, "noise_penalty_order": 6},
        {"noise_std": 0.1, "lam": 1.0},
    ):
        with pytest.raises(ValueError):
            bspf_plans.plan_1d(x, **kwargs)
    p = bspf_plans.plan_1d(x, noise_std=0.1, noise_alphas=[1e-5, 1e-3])
    with pytest.raises(ValueError):
        bspf_plans.tensor_plan(p, a)
    with pytest.raises(ValueError):
        bspf_plans.tensor_plan(p, bspf_plans.plan_1d(x, noise_std=0.2, noise_alphas=[1e-5, 1e-3]))
    with pytest.raises(ValueError):
        bspf_operators.differentiate(p, f, boundary=jnp.zeros((2, 4)))
    with pytest.raises(ValueError):
        bspf_operators.differentiate(p, f, correction=False)
    with pytest.raises(ValueError):
        bspf_calculus.integrate(p, f)
    with pytest.raises(ValueError):
        bspf_plans.with_regularization(p, 0.1)
    with pytest.raises(ValueError):
        bspf_operators.noise_diagnostics(a, f)


def test_noisy_hessian_vectors_and_local_autodiff():
    x = jnp.linspace(0, 1, 17)
    p = bspf_plans.plan_2d(
        x, x, degree=3, n_basis=7, noise_std=0.01, noise_alphas=[1e-6, 1e-4, 1e-2]
    )
    X, Y = jnp.meshgrid(x, x, indexing="ij")
    f = jnp.sin(X + Y)
    H = jax.jit(bspf_operators.hessian)(p, f)
    np.testing.assert_allclose(H[0, 1], H[1, 0], atol=1e-12)
    vector = jnp.stack([f, 2 * f])
    np.testing.assert_allclose(
        jax.jit(bspf_operators.divergence)(p, vector),
        bspf_operators.differentiate(p, f, axis=0) + bspf_operators.differentiate(p, 2 * f, axis=1),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        jax.jit(bspf_operators.curl)(p, vector),
        bspf_operators.differentiate(p, 2 * f, axis=0) - bspf_operators.differentiate(p, f, axis=1),
        atol=1e-10,
    )
    loss = lambda values: jnp.sum(bspf_operators.differentiate(p, values) ** 2)
    direction = jnp.cos(X - Y)
    eps = 1e-6
    plus = bspf_operators.noise_diagnostics(p, f + eps * direction)
    minus = bspf_operators.noise_diagnostics(p, f - eps * direction)
    assert all(int(u.index) == int(v.index) for u, v in zip(plus, minus))
    actual = jnp.vdot(jax.jit(jax.grad(loss))(f), direction)
    expected = (loss(f + eps * direction) - loss(f - eps * direction)) / (2 * eps)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_prior_noise_scaling_and_vmap():
    from dataclasses import replace

    x = jnp.linspace(0, 1, 17)
    p = bspf_plans.plan_1d(
        x, degree=3, n_basis=7, noise_std=0.01, noise_alphas=[1e-6, 1e-4, 1e-2]
    )
    observed = jnp.stack([jnp.sin(x), jnp.cos(2 * x)])
    actual = jax.jit(jax.vmap(lambda values: bspf_operators.differentiate(p, values)))(observed)
    expected = jnp.stack([bspf_operators.differentiate(p, values) for values in observed])
    np.testing.assert_allclose(actual, expected, atol=1e-11)
    scaled_plan = replace(p, noise=replace(p.noise, sigma=7 * p.noise.sigma))
    np.testing.assert_allclose(
        bspf_operators.differentiate(scaled_plan, 7 * observed[0]),
        7 * bspf_operators.differentiate(p, observed[0]),
        atol=1e-10,
    )
    original = bspf_operators.noise_diagnostics(p, observed[0])[0]
    scaled = bspf_operators.noise_diagnostics(scaled_plan, 7 * observed[0])[0]
    assert int(original.index) == int(scaled.index)
    np.testing.assert_allclose(original.noise_std, 0.01)
    np.testing.assert_allclose(scaled.noise_std, 0.07)
