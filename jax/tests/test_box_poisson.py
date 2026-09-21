"""Independent 3D operator, MMS, batch and residency tests for fast diagonalization."""

import numpy as np
import pytest
from bspf_jax import plan_box_poisson, solve_box_poisson
from bspf_jax.rectangle_poisson import plan_rectangle_poisson, solve_rectangle_poisson

import jax


@pytest.fixture(params=["cpu", "gpu"])
def device(request):
    try:
        return jax.devices(request.param)[0]
    except RuntimeError:
        pytest.skip(f"{request.param} backend unavailable")


def axis(n, rng):
    a, b = rng.normal(size=(2, n, n))
    return a.T @ a + np.eye(n), b.T @ b + np.eye(n)


def kron_all(arrays):
    result = arrays[0]
    for a in arrays[1:]:
        result = np.kron(result, a)
    return result


def test_generalized_3d_noncubic_batch_autodiff_and_residency(device):
    rng = np.random.default_rng(73)
    pairs = [axis(n, rng) for n in (3, 4, 5)]
    masses, stiffnesses = zip(*pairs)
    weights = (0.3, 1.1, 2.4)
    plan = plan_box_poisson(
        masses, stiffnesses, device=device, shift=0.7, weights=weights
    )
    matrix = 0.7 * kron_all(masses)
    for i in range(3):
        factors = list(masses)
        factors[i] = stiffnesses[i]
        matrix += weights[i] * kron_all(factors)
    rhs = rng.normal(size=(2, 3, 3, 4, 5))
    expected = np.linalg.solve(matrix, rhs.reshape(-1, 60).T).T.reshape(rhs.shape)
    load = jax.device_put(rhs, device)
    solve_box_poisson(plan, load).block_until_ready()
    with jax.transfer_guard("disallow"):
        actual = solve_box_poisson(plan, load)
        actual.block_until_ready()
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=3e-12)
    np.testing.assert_allclose(
        np.asarray(actual).reshape(-1, 60) @ matrix.T, rhs.reshape(-1, 60), atol=5e-12
    )
    assert actual.devices() == {device}
    assert all(a.devices() == {device} for a in jax.tree_util.tree_leaves(plan))
    assert all(a.ndim <= 2 for a in jax.tree_util.tree_leaves(plan))
    mapped = jax.jit(jax.vmap(lambda f: solve_box_poisson(plan, f)))(load)
    np.testing.assert_allclose(mapped, expected, atol=1e-14, rtol=3e-12)
    derivative = jax.jit(jax.grad(lambda f: solve_box_poisson(plan, f).sum()))(
        load[0, 0]
    )
    np.testing.assert_allclose(
        derivative,
        np.linalg.solve(matrix.T, np.ones(60)).reshape(3, 4, 5),
        atol=1e-14,
        rtol=3e-12,
    )


def test_2d_box_matches_rectangle(device):
    rng = np.random.default_rng(1)
    (mx, kx), (my, ky) = axis(4, rng), axis(5, rng)
    box = plan_box_poisson((mx, my), (kx, ky), device=device, shift=2, weights=(0.5, 2))
    rectangle = plan_rectangle_poisson(
        mx, kx, my, ky, device=device, shift=2, weights=(0.5, 2)
    )
    rhs = jax.device_put(rng.normal(size=(3, 4, 5)), device)
    np.testing.assert_allclose(
        solve_box_poisson(box, rhs), solve_rectangle_poisson(rectangle, rhs), atol=1e-14
    )


def test_continuous_noncubic_mms_convergence(device):
    errors = []
    lengths = (2.0, 3.0, 1.5)
    for shape in ((7, 9, 11), (15, 19, 23)):
        masses, stiffnesses, fields = [], [], []
        for n, length in zip(shape, lengths):
            h = length / (n + 1)
            masses.append(np.eye(n))
            stiffnesses.append(
                (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)) / h**2
            )
            fields.append(np.sin(np.pi * np.arange(1, n + 1) / (n + 1)))
        exact = np.einsum("i,j,k->ijk", *fields)
        # Continuous forcing, not the discrete matrix times the exact samples.
        rhs = sum((np.pi / length) ** 2 for length in lengths) * exact
        plan = plan_box_poisson(masses, stiffnesses, device=device)
        actual = np.asarray(solve_box_poisson(plan, jax.device_put(rhs, device)))
        errors.append(np.linalg.norm(actual - exact) / np.linalg.norm(exact))
    assert 3.9 < errors[0] / errors[1] < 4.1


def test_nullspace_shift_validation_and_shared_axis(device):
    m = np.eye(2)
    k = np.array([[1.0, -1.0], [-1.0, 1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        plan_box_poisson((m,) * 3, (k,) * 3, device=device)
    plan = plan_box_poisson((m,) * 3, (k,) * 3, device=device, shift=2)
    assert plan.rotations[0] is plan.rotations[1] is plan.rotations[2]
    assert plan.shape == (2, 2, 2)
    actual = solve_box_poisson(plan, jax.device_put(np.ones((2, 2, 2)), device))
    np.testing.assert_allclose(actual, 0.5, atol=1e-14)
    for masses, stiffnesses in [
        ((m,), (k,)),
        ((m,) * 3, (k,) * 2),
        ((m,) * 4, (k,) * 4),
    ]:
        with pytest.raises(ValueError, match="2 or 3 axes"):
            plan_box_poisson(masses, stiffnesses, device=device)
    with pytest.raises(ValueError, match="per axis"):
        plan_box_poisson((m,) * 3, (m,) * 3, device=device, weights=(1, 2))
    with pytest.raises(ValueError, match="shapes"):
        plan_box_poisson((m, m, np.eye(3)), (m,) * 3, device=device)
    with pytest.raises(ValueError, match="trailing spatial shape"):
        solve_box_poisson(plan, jax.device_put(np.ones((2, 2)), device))


def test_benchmark_3d_forcing_against_autodiff(monkeypatch):
    from pathlib import Path

    import jax.numpy as jnp

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scratch"))
    import benchmark_box_poisson as benchmark

    def exact(p):
        t, s, r = p / jnp.array([2.0, 3.0, 1.5])
        return (
            t
            * (1 - t)
            * s
            * (1 - s)
            * r
            * (1 - r)
            * (
                jnp.exp(0.7 * t - 0.4 * s - 0.4 * r)
                + 0.2
                * jnp.sin(5 * jnp.pi * t)
                * jnp.cos(3 * jnp.pi * s)
                * jnp.cos(3 * jnp.pi * r)
            )
        )

    for point in np.random.default_rng(5).uniform(size=(6, 3)) * [2, 3, 1.5]:
        value, force = benchmark.mms([np.array([x]) for x in point])
        np.testing.assert_allclose(value.item(), exact(point), atol=1e-15)
        np.testing.assert_allclose(
            force.item(), -jnp.trace(jax.hessian(exact)(point)), atol=2e-14
        )
