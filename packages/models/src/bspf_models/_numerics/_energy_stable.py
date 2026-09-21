"""Diagonal-norm SBP(8,4) and an orthogonal, tensor-direct projection.

The SBP norm is the classical eighth-order diagonal norm (Mattsson &
Nordstrom, JCP 199, 2004). Boundary coefficients are derived from SBP and
polynomial exactness, choosing the minimum-norm skew boundary block.
No global two-dimensional matrix or iterative solve is used.
"""

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import legder, legval, legvander


class EnergyLine(NamedTuple):
    weights: jax.Array
    derivative: jax.Array
    boundary_null: jax.Array
    tangent_divergence: jax.Array
    vectors: jax.Array
    eigenvalues: jax.Array


class EnergyClosure2D(NamedTuple):
    x: EnergyLine
    y: EnergyLine
    sqrt_weights: jax.Array
    inverse_sum: jax.Array


@lru_cache(maxsize=1)
def _boundary_coefficients():
    """Solve the fixed 40-by-28 polynomial constraints once on the host."""
    n = 32
    w = np.ones(n)
    w[:8] = [
        1498139 / 5080320,
        1107307 / 725760,
        20761 / 80640,
        1304999 / 725760,
        299527 / 725760,
        103097 / 80640,
        670091 / 725760,
        5127739 / 5080320,
    ]
    q = np.zeros((n, n))
    q[0, 0] = -0.5
    for i in range(8, n - 8):
        for k, a in enumerate((4 / 5, -1 / 5, 4 / 105, -1 / 280), 1):
            q[i, i + k], q[i, i - k] = a, -a
            q[i + k, i], q[i - k, i] = -a, a
    # Local scaling avoids ill-conditioned powers of the physical coordinates.
    z = np.arange(n) * (2 / 11) - 1
    v = legvander(z, 4)
    vp = np.column_stack([legval(z, legder(np.eye(5)[k])) * (2 / 11) for k in range(5)])
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    columns = []
    for i, j in pairs:
        column = np.zeros((8, 5))
        column[i], column[j] = v[j], -v[i]
        columns.append(column.ravel())
    matrix = np.array(columns).T
    rhs = (w[:, None] * vp - q @ v)[:8].ravel()
    coefficients = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    if np.max(abs(matrix @ coefficients - rhs)) > 1e-12:
        raise RuntimeError("Inconsistent SBP boundary moment equations")
    for (i, j), a in zip(pairs, coefficients):
        q[i, j], q[j, i] = a, -a
    return q[:8, :12] / w[:8, None], w[:8]


def sbp84_derivative(coordinates):
    """Uniform-grid D and diagonal H: HD + D.T H = diag(-1,0,...,1).

    D has eighth-order interior and fourth-order boundary truncation error.
    D@D is energy dissipative with strong homogeneous Dirichlet data;
    its pointwise boundary accuracy is generally only third order.
    """
    x = np.asarray(coordinates, dtype=float)
    if x.ndim != 1 or x.size < 24 or not np.all(np.isfinite(x)):
        raise ValueError("sbp84 requires at least 24 finite nodes on each axis")
    h = (x[-1] - x[0]) / (x.size - 1)
    if h <= 0 or not np.allclose(np.diff(x), h, rtol=1e-10, atol=1e-14 * h):
        raise ValueError("sbp84 requires strictly increasing uniform nodes")
    left, weights = _boundary_coefficients()
    n = x.size
    d = np.zeros((n, n))
    d[:8, :12] = left
    d[-8:, -12:] = -left[::-1, ::-1]
    for i in range(8, n - 8):
        for k, a in enumerate((4 / 5, -1 / 5, 4 / 105, -1 / 280), 1):
            d[i, i + k], d[i, i - k] = a, -a
    w = np.ones(n)
    w[:8], w[-8:] = weights, weights[::-1]
    return jnp.asarray(d / h), jnp.asarray(w * h)


def sbp84_diffusion(derivative, weights):
    """Compatible viscosity with an eighth-order Nyquist-damping remainder.

    L = D^2 - H^-1 K5.T K5/(256 h), with K5 the unscaled fifth forward
    difference. The remainder is positive semidefinite, annihilates degree-4
    polynomials, and has O(h^8) interior/O(h^3) boundary consistency error.
    Its periodic Nyquist symbol is -4/h^2, avoiding the D^2 odd/even null mode.
    """
    n = derivative.shape[0]
    h = weights[8]  # unit interior quadrature weight, n >= 24
    k5 = jnp.asarray(np.diff(np.eye(n), n=5, axis=0))
    return derivative @ derivative - (k5.T @ k5) / (256 * h * weights[:, None])


def _energy_line(d, weights):
    d, weights = np.asarray(d), np.asarray(weights)
    root = np.sqrt(weights)
    b = root[:, None] * d[:, 1:-1] / root[None, 1:-1]
    # First enforce the two face-divergence constraints by an orthogonal
    # projection. Interior pressure then solves the remaining tensor sum.
    _, singular, vh = np.linalg.svd(b[[0, -1]], full_matrices=True)
    if singular[-1] <= singular[0] * 1e-12:
        raise ValueError("Degenerate SBP face constraints")
    z = vh[2:].T
    c = z @ z.T
    reduced = b[1:-1] @ z
    u, s, _ = np.linalg.svd(reduced, full_matrices=True)
    if s[-1] <= s[0] * 1e-12:
        raise ValueError("Unexpected extra pressure null modes")
    # Two exact zeros per axis; their tensor products are pressure gauge modes.
    values = np.r_[s * s, 0.0, 0.0]
    return EnergyLine(*map(jnp.asarray, (weights, d, c, reduced @ z.T, u, values)))


def energy_closure2d(x, y):
    dx, hx = sbp84_derivative(x)
    dy, hy = sbp84_derivative(y)
    lx, ly = _energy_line(dx, hx), _energy_line(dy, hy)
    values = np.asarray(lx.eigenvalues)[:, None] + np.asarray(ly.eigenvalues)[None, :]
    inverse = np.zeros_like(values)
    np.divide(1.0, values, out=inverse, where=values > 0)
    return EnergyClosure2D(
        lx, ly, jnp.sqrt(hx[:, None] * hy[None, :]), jnp.asarray(inverse)
    )


def energy_divergence(closure, velocity):
    return (
        closure.x.derivative @ velocity[..., 0]
        + velocity[..., 1] @ closure.y.derivative.T
    )


def energy_project(closure, raw):
    """H-orthogonal projection onto zero-wall, full-grid divergence-free fields.

    Mask first, eliminate face constraints, diagonalize the interior tensor
    sum, and subtract its weighted adjoint gradient. Four gauge modes use the
    pseudoinverse. This is one direct solve, with zero refinement steps.
    """
    x, y = closure.x, closure.y
    root = closure.sqrt_weights[1:-1, 1:-1]
    fx, fy = raw[1:-1, 1:-1, 0] * root, raw[1:-1, 1:-1, 1] * root
    fx, fy = x.boundary_null @ fx, fy @ y.boundary_null
    rhs = x.tangent_divergence @ fx + fy @ y.tangent_divergence.T
    modal = (x.vectors.T @ rhs @ y.vectors) * closure.inverse_sum
    pressure = x.vectors @ modal @ y.vectors.T
    fx -= x.tangent_divergence.T @ pressure
    fy -= pressure @ y.tangent_divergence
    return (
        jnp.zeros_like(raw)
        .at[1:-1, 1:-1]
        .set(jnp.stack((fx / root, fy / root), axis=-1))
    )
