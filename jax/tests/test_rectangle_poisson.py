"""Independent operator, continuum, batching, and GPU residency checks."""

import numpy as np
import pytest
from bspf_jax.rectangle_poisson import (
    plan_rectangle_poisson,
    solve_rectangle_poisson,
)

import jax


@pytest.fixture(params=["cpu", "gpu"])
def device(request):
    try:
        return jax.devices(request.param)[0]
    except RuntimeError:
        pytest.skip(f"{request.param} backend unavailable")


def axis(n, seed):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=(2, n, n))
    return a.T @ a + np.eye(n), b.T @ b + np.eye(n)


def test_generalized_batched_inverse_and_derivative(device):
    mx, kx = axis(5, 1)
    my, ky = axis(7, 2)
    plan = plan_rectangle_poisson(
        mx, kx, my, ky, device=device, shift=0.7, weights=(0.3, 1.8)
    )
    # Assemble an independent full Kronecker matrix only in this small test.
    matrix = 0.7 * np.kron(mx, my) + 0.3 * np.kron(kx, my) + 1.8 * np.kron(mx, ky)
    rhs = np.random.default_rng(3).normal(size=(2, 3, 5, 7))
    expected = np.linalg.solve(matrix, rhs.reshape(-1, 35).T).T.reshape(rhs.shape)
    load = jax.device_put(rhs, device)
    # Warm compilation before forbidding implicit host/device transfers.
    solve_rectangle_poisson(plan, load).block_until_ready()
    with jax.transfer_guard("disallow"):
        actual = solve_rectangle_poisson(plan, load)
        actual.block_until_ready()
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-13)
    np.testing.assert_allclose(
        np.asarray(actual).reshape(-1, 35) @ matrix.T,
        rhs.reshape(-1, 35),
        rtol=3e-12,
        atol=3e-12,
    )
    assert all(a.devices() == {device} for a in plan)
    assert actual.devices() == {device}
    mapped = jax.jit(jax.vmap(lambda b: solve_rectangle_poisson(plan, b)))(load)
    np.testing.assert_allclose(mapped, expected, rtol=3e-12, atol=3e-13)
    # d(sum(A^-1 b))/db = A^-T 1, independent of the modal implementation.
    derivative = jax.jit(jax.grad(lambda b: solve_rectangle_poisson(plan, b).sum()))(
        load[0, 0]
    )
    np.testing.assert_allclose(
        derivative,
        np.linalg.solve(matrix.T, np.ones(35)).reshape(5, 7),
        rtol=3e-12,
        atol=3e-13,
    )


def test_nonunit_rectangle_continuous_manufactured_solution(device):
    # Centered differences provide an independent discretization for this
    # generic algebraic solver. The exact sine solution is not a discrete RHS.
    errors = []
    for nx, ny in [(15, 11), (31, 23)]:
        hx, hy = 2.0 / (nx + 1), 3.0 / (ny + 1)

        def stiffness(n, h):
            return (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)) / h**2

        plan = plan_rectangle_poisson(
            np.eye(nx), stiffness(nx, hx), np.eye(ny), stiffness(ny, hy), device=device
        )
        x, y = np.arange(1, nx + 1) * hx, np.arange(1, ny + 1) * hy
        exact = np.sin(np.pi * x[:, None] / 2) * np.sin(np.pi * y[None, :] / 3)
        load = jax.device_put(((np.pi / 2) ** 2 + (np.pi / 3) ** 2) * exact, device)
        result = np.asarray(solve_rectangle_poisson(plan, load))
        errors.append(np.linalg.norm(result - exact) / np.linalg.norm(exact))
    assert 3.9 < errors[0] / errors[1] < 4.1


def test_invalid_operators_and_load(device):
    identity = np.eye(3)

    def make(mx=identity, kx=identity, **kwargs):
        return plan_rectangle_poisson(
            mx, kx, identity, identity, device=device, **kwargs
        )

    with pytest.raises(ValueError, match="symmetric"):
        make(kx=identity + np.eye(3, k=1))
    with pytest.raises(ValueError, match="positive definite"):
        make(mx=-identity)
    with pytest.raises(ValueError, match="shapes"):
        make(mx=np.eye(2))
    with pytest.raises(ValueError, match="finite"):
        make(shift=np.nan)
    with pytest.raises(ValueError, match="positive"):
        make(weights=(0, 1))
    with pytest.raises(ValueError, match="positive definite"):
        plan_rectangle_poisson(
            identity, np.zeros((3, 3)), identity, np.zeros((3, 3)), device=device
        )
    with pytest.raises(ValueError, match="trailing shape"):
        solve_rectangle_poisson(make(), jax.device_put(np.ones((3, 4)), device))


def test_shift_removes_nullspace(device):
    k = np.array([[1.0, -1.0], [-1.0, 1.0]])
    m = np.eye(2)
    with pytest.raises(ValueError, match="positive definite"):
        plan_rectangle_poisson(m, k, m, k, device=device)
    plan = plan_rectangle_poisson(m, k, m, k, shift=2, device=device)
    actual = solve_rectangle_poisson(plan, jax.device_put(np.ones((2, 2)), device))
    np.testing.assert_allclose(actual, 0.5, atol=1e-14)
