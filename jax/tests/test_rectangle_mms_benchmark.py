"""Verify analytic MMS forcing and independent FD convergence for the benchmark."""

import importlib.util
import itertools
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

import jax

spec = importlib.util.spec_from_file_location(
    "rectangle_mms",
    Path(__file__).resolve().parents[2] / "scratch/benchmark_rectangle_mms.py",
)
mms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mms)


def test_continuous_forcing_against_autodiff():
    def exact(p):
        t, s = p[0] / 2, p[1] / 3
        return (
            t
            * (1 - t)
            * s
            * (1 - s)
            * (
                jnp.exp(0.7 * t - 0.4 * s)
                + 0.2 * jnp.sin(5 * jnp.pi * t) * jnp.cos(3 * jnp.pi * s)
            )
        )

    points = np.random.default_rng(19).uniform(size=(15, 2)) * [2, 3]
    for point in points:
        u, f = mms.manufactured(point[:1], point[1:])
        np.testing.assert_allclose(u[0, 0], exact(point), rtol=1e-13)
        np.testing.assert_allclose(
            f[0, 0], -jnp.trace(jax.hessian(exact)(point)), atol=2e-14
        )
    u, _ = mms.manufactured(np.linspace(0, 2, 21), np.linspace(0, 3, 23))
    np.testing.assert_array_equal(u[[0, -1]], 0)
    np.testing.assert_array_equal(u[:, [0, -1]], 0)


def test_fd_manufactured_convergence():
    errors = []
    for n in (15, 31, 63):
        mx, kx = mms.fd_axis(n, 2)
        my, ky = mms.fd_axis(n, 3)
        x = np.linspace(0, 2, n + 2)[1:-1]
        y = np.linspace(0, 3, n + 2)[1:-1]
        exact, forcing = mms.manufactured(x, y)
        a = sparse.kron(sparse.csr_matrix(kx), sparse.csr_matrix(my)) + sparse.kron(
            sparse.csr_matrix(mx), sparse.csr_matrix(ky)
        )
        solution = spsolve(a.tocsr(), forcing.ravel()).reshape(n, n)
        errors.append(np.linalg.norm(solution - exact) / np.linalg.norm(exact))
        assert (
            np.linalg.norm(a @ solution.ravel() - forcing.ravel())
            / np.linalg.norm(forcing)
            < 1e-11
        )
    assert all(3.8 < a / b < 4.3 for a, b in itertools.pairwise(errors))


def test_fd_interpolation_preserves_linear_interior_values():
    # Zero Dirichlet interpolation is piecewise linear, including boundary cells.
    n = 7
    grid = np.linspace(0, 2, n + 2)
    nodal = grid * (2 - grid)
    points = np.linspace(0, 2, 70)
    expected = np.interp(points, grid, nodal)
    np.testing.assert_allclose(
        mms.fd_evaluation(n, 2, points) @ nodal[1:-1], expected, atol=1e-15
    )
