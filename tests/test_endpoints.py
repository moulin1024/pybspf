"""Optional endpoint fits: independent references and public API behavior."""

import pybspf.operators as bspf_operators
import pybspf.plans as bspf_plans

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial import chebyshev as cheb
from scipy.linalg import lu_solve

import pybspf as b


@pytest.mark.parametrize("alpha", [0., 1e-3])
def test_chebyshev_map_matches_independent_least_squares(alpha):
    x = jnp.linspace(-2, 3, 33)
    modes, points, q = 7, 12, 4
    p = bspf_plans.plan_1d(x, degree=5, n_basis=12, constraint_order=q,
                  endpoint_method="chebyshev", boundary_points=points,
                  chebyshev_modes=modes, chebyshev_alpha=alpha)
    xi = np.linspace(-1, 1, points)
    v = cheb.chebvander(xi, modes-1)
    penalty = (np.arange(modes)/(modes-1))**4
    penalty[:2] = 0
    A = np.vstack([v, np.sqrt(alpha)*np.diag(penalty)])
    projector = np.linalg.lstsq(A, np.vstack([np.eye(points), np.zeros((modes, points))]), rcond=None)[0]
    expected = np.zeros((2*q, x.size))
    width = float(x[points-1]-x[0])
    for side, t in enumerate([-1., 1.]):
        for k in range(q):
            row = cheb.chebval(t, cheb.chebder(np.eye(modes), m=k, axis=0))
            weights = (2/width)**k*row@projector
            sl = slice(0, points) if side == 0 else slice(-points, None)
            expected[side*q+k, sl] = weights
    expected[0] = 0; expected[0, 0] = 1
    expected[q] = 0; expected[q, -1] = 1
    np.testing.assert_allclose(p.boundary_blocks,
                               np.stack((expected[:q, :points], expected[q:, -points:])),
                               rtol=1e-9, atol=2e-10)


def test_polynomial_jets_and_complex_jit_grad():
    x = jnp.linspace(-1, 1, 33)
    p = bspf_plans.plan_1d(x, degree=5, n_basis=12, endpoint_method="chebyshev",
                  chebyshev_modes=6, boundary_points=10, chebyshev_alpha=0)
    f = jnp.stack([x**3, 2j*x**3], axis=-1)
    ends = jnp.array([-1., 1.])
    exact = jnp.stack([ends**3, 3*ends**2, 6*ends, jnp.full(2, 6.)], axis=1)
    np.testing.assert_allclose(jax.jit(bspf_operators.endpoint_jets)(p, f),
                               jnp.stack([exact, 2j*exact], axis=-1), atol=1e-9)
    df = jax.jit(bspf_operators.differentiate)(p, f)
    np.testing.assert_allclose(df, jnp.stack([3*x*x, 6j*x*x], axis=-1), atol=1e-9)
    grad = jax.jit(jax.grad(lambda values: jnp.sum(bspf_operators.differentiate(p, values))))(x**3)
    np.testing.assert_allclose(jnp.vdot(grad, x**3), jnp.sum(3*x*x), atol=1e-8)
    # Prescribed boundary data bypasses the chosen estimator.
    fd = bspf_plans.plan_1d(x, degree=5, n_basis=12)
    np.testing.assert_allclose(bspf_operators.differentiate(p, x**3, boundary=exact),
                               bspf_operators.differentiate(fd, x**3, boundary=exact), atol=1e-12)


@pytest.mark.parametrize("dimension", [2, 3])
def test_tensor_factories_accept_estimator(dimension):
    x = jnp.linspace(-1, 1, 17)
    factory = bspf_plans.plan_2d if dimension == 2 else bspf_plans.plan_3d
    p = factory(*([x]*dimension), degree=3, n_basis=8,
                endpoint_method="chebyshev", chebyshev_modes=4,
                boundary_points=7, chebyshev_alpha=0)
    mesh = jnp.meshgrid(*([x]*dimension), indexing="ij")
    f = sum(t*t for t in mesh)
    np.testing.assert_allclose(jax.jit(partial(bspf_operators.differentiate, axis=dimension-1))(p, f),
                               2*mesh[-1], atol=1e-9)


def test_degree9_endpoint_option_improves_smooth_field():
    # Same manufactured field that exposed the early endpoint-error tail.
    x = jnp.linspace(0, 2*jnp.pi, 129)
    kx = jnp.array([1, 2, 3, 1, 2, 4], dtype=float)
    ky = jnp.array([2, 1, 1, 3, 3, 2], dtype=float)
    a = jax.random.normal(jax.random.key(123), (6,))/(kx*kx+ky*ky)
    def field(x, y):
        r2 = (x-jnp.pi)**2+(y-jnp.pi)**2
        return jnp.sum(a*jnp.cos(kx*x)*jnp.cos(ky*y)) + .5*(1+jnp.tanh((r2-1)/.6))
    sample = lambda fun: jax.vmap(jax.vmap(fun, in_axes=(None, 0)), in_axes=(0, None))(x, x)
    f, exact = sample(field), sample(jax.grad(field, 0))
    opts = dict(degree=9, n_basis=18, lam=1e-6)
    fd = bspf_plans.plan_1d(x, boundary_points=9, **opts)
    local = bspf_plans.plan_1d(x, endpoint_method="chebyshev", boundary_points=16,
                      chebyshev_modes=12, **opts)
    error_fd = jnp.linalg.norm(bspf_operators.differentiate(fd, f)-exact)/jnp.linalg.norm(exact)
    error_local = jnp.linalg.norm(jax.jit(bspf_operators.differentiate)(local, f)-exact)/jnp.linalg.norm(exact)
    assert error_local < 1e-10
    assert error_local < .2*error_fd


@pytest.mark.parametrize("method", ["finite_difference", "chebyshev"])
@pytest.mark.parametrize("n", [17, 33])
def test_compact_blocks_match_dense_reference(method, n):
    # N=17 deliberately makes the two 12-point endpoint windows overlap.
    x = jnp.linspace(-1, 1, n)
    options = {"chebyshev_modes": 7} if method == "chebyshev" else {}
    p = bspf_plans.plan_1d(x, degree=5, n_basis=12, boundary_points=12,
                  endpoint_method=method, **options)
    assert p.boundary_blocks.shape == (2, 4, 12)
    blocks = np.asarray(p.boundary_blocks)
    dense = np.zeros((8, n))
    dense[:4, :12] = blocks[0]
    dense[4:, -12:] = blocks[1]
    rng = np.random.default_rng(123)
    f = rng.normal(size=(n, 3, 2)) + 1j*rng.normal(size=(n, 3, 2))
    expected_jets = (dense@f.reshape(n, -1)).reshape(2, 4, 3, 2)
    np.testing.assert_allclose(jax.jit(bspf_operators.endpoint_jets)(p, jnp.asarray(f)),
                               expected_jets, rtol=2e-12, atol=1e-8)
    # Independent dense application of the same mathematical BSPF operator.
    flat = f.reshape(n, -1)
    rhs = np.concatenate((2*np.asarray(p.weighted_basis)@flat, dense@flat))
    coeff = lu_solve(tuple(np.asarray(a) for a in p.lu), rhs)[:12]
    residual = flat-np.asarray(p.basis[0])@coeff
    expected = np.asarray(p.basis[1])@coeff + np.fft.ifft(
        1j*np.asarray(p.omega)[:, None]*np.fft.fft(residual, axis=0), axis=0)
    np.testing.assert_allclose(jax.jit(bspf_operators.differentiate)(p, jnp.asarray(f)),
                               expected.reshape(f.shape), rtol=2e-11, atol=1e-8)
    # Boundary extraction also works on a non-leading physical axis.
    other = bspf_plans.plan_1d(jnp.linspace(0, 1, 3), degree=1, n_basis=2)
    tensor = bspf_plans.tensor_plan(other, p)
    np.testing.assert_allclose(bspf_operators.endpoint_jets(tensor, jnp.asarray(f).swapaxes(0, 1), axis=1),
                               expected_jets, rtol=2e-12, atol=1e-8)


@pytest.mark.parametrize("method", ["finite_difference", "chebyshev"])
def test_empty_compact_jets(method):
    x = jnp.linspace(-1, 1, 17)
    p = bspf_plans.plan_1d(x, degree=3, n_basis=8, constraint_order=0,
                  endpoint_method=method)
    assert p.boundary_blocks.shape[:2] == (2, 0)
    assert jax.jit(bspf_operators.endpoint_jets)(p, x).shape == (2, 0)
    np.testing.assert_allclose(bspf_operators.differentiate(p, jnp.ones_like(x)), 0, atol=1e-10)


@pytest.mark.parametrize("kwargs", [
    {"endpoint_method": "ldc"},
    {"chebyshev_modes": 6},
    {"endpoint_method": "chebyshev", "chebyshev_modes": 3},
    {"endpoint_method": "chebyshev", "chebyshev_modes": 10, "boundary_points": 9},
    {"endpoint_method": "chebyshev", "boundary_points": 34},
    {"endpoint_method": "chebyshev", "chebyshev_alpha": -1},
    {"endpoint_method": "chebyshev", "chebyshev_alpha": float("nan")},
    {"endpoint_method": "chebyshev", "chebyshev_penalty_power": -1},
])
def test_invalid_estimator_options(kwargs):
    with pytest.raises(ValueError):
        bspf_plans.plan_1d(jnp.linspace(-1, 1, 33), **kwargs)
