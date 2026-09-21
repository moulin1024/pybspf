"""Shared flow integration and device tensor-PCG regression."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pybspf.time_integration import rk4_stages
from pybspf.tensor import tensor_elliptic_solve
from bspf_models._numerics._tensor_pcg import plan_tensor_preconditioner
from bspf_models._numerics._tensor_pcg import tensor_pcg_device


def test_device_pcg_nonseparable_against_direct_solve():
    rng = np.random.default_rng(45)
    def spd(n):
        a = rng.normal(size=(n, n))
        return a.T @ a + np.eye(n)
    mr, kr, mt, kt = spd(5), spd(5), spd(7), spd(7)
    pre = plan_tensor_preconditioner(mr, kr, mt, kt, 1., 1.)
    correction = rng.normal(size=(35, 9))
    matrix = np.kron(kr, mt)+np.kron(mr, kt)+correction @ correction.T
    rhs = rng.normal(size=(5, 7))
    device_matrix = jnp.asarray(matrix)
    solve = jax.jit(lambda r: tensor_pcg_device(
        lambda x: (device_matrix @ x.ravel()).reshape(r.shape), r,
        lambda x: tensor_elliptic_solve(x, pre.denominator, pre.left, pre.right)))
    actual, (iterations, residual, ok) = solve(jnp.asarray(rhs))
    assert ok and iterations > 1 and residual < 2.1e-12
    np.testing.assert_allclose(actual.ravel(), np.linalg.solve(matrix, rhs.ravel()), rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize('case', ['zero', 'nan', 'negative', 'limit'])
def test_device_pcg_status(case):
    rhs = jnp.zeros(4) if case == 'zero' else jnp.ones(4)
    if case == 'nan':
        rhs = rhs.at[0].set(jnp.nan)
    diagonal = -jnp.ones(4) if case == 'negative' else jnp.arange(1., 5.)
    result, (count, residual, ok) = jax.jit(lambda r: tensor_pcg_device(
        lambda x: diagonal*x, r, lambda x: x, maxiter=1))(rhs)
    if case == 'zero':
        assert ok and count == 0 and residual == 0
        np.testing.assert_array_equal(result, 0.)
    else:
        assert not ok
