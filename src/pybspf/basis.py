"""Cox–de Boor basis evaluation and exact spline calculus, entirely in JAX.

Matrices use ``(evaluation points, basis functions)``. Degree and derivative
order determine shapes and must be static under ``jit``.
"""
from __future__ import annotations

import jax.numpy as jnp


def open_knots(a, b, *, degree: int, n_basis: int, clustering: float = 0.0):
    """Clamped knots with optional tanh clustering toward both endpoints."""
    u = jnp.linspace(-1.0, 1.0, n_basis - degree + 1)
    # Safe denominator also gives a finite derivative at zero clustering.
    scale = jnp.where(clustering == 0, 1.0, clustering)
    warped = jnp.tanh(scale * u) / jnp.tanh(scale)
    u = jnp.where(clustering == 0, u, warped)
    breaks = a + (b - a) * (u + 1.0) / 2.0
    return jnp.concatenate((jnp.repeat(a, degree), breaks, jnp.repeat(b, degree)))


def _divide(a, b):
    """Zero for a repeated-knot denominator, without NaNs in reverse mode."""
    return jnp.where(b != 0, a / jnp.where(b != 0, b, 1), 0)


def basis_matrix(knots, x, *, degree: int, derivative: int = 0):
    """Evaluate clamped B-splines or their physical derivatives.

    Values at the right endpoint use the left-hand limit. Outside the knot
    interval the basis is zero; extrapolation is deliberately not implicit.
    """
    knots, x = jnp.asarray(knots), jnp.atleast_1d(x)
    if degree < 0 or derivative < 0:
        raise ValueError("degree and derivative must be nonnegative")
    n_basis = knots.size - degree - 1
    if derivative > degree:
        return jnp.zeros((x.size, n_basis), dtype=jnp.result_type(knots, x))
    if derivative:
        lower = basis_matrix(knots, x, degree=degree - 1, derivative=derivative - 1)
        left = _divide(degree, knots[degree:-1] - knots[:-(degree + 1)])
        right = _divide(degree, knots[degree + 1:] - knots[1:-degree])
        return lower[:, :-1] * left - lower[:, 1:] * right
    xx = x[:, None]
    values = ((xx >= knots[:-1]) & (xx < knots[1:])).astype(jnp.result_type(knots, x))
    # Seed the final nonempty interval at the clamped endpoint. This remains
    # correct when evaluating lower-degree bases on a higher-multiplicity knot vector.
    last_interval = jnp.max(jnp.where(knots[:-1] < knots[-1], jnp.arange(knots.size - 1), -1))
    endpoint = jnp.arange(knots.size - 1) == last_interval
    values = jnp.where(xx == knots[-1], endpoint[None, :], values)
    for p in range(1, degree + 1):
        left = _divide(xx - knots[:-(p + 1)], knots[p:-1] - knots[:-(p + 1)])
        right = _divide(knots[p + 1:] - xx, knots[p + 1:] - knots[1:-p])
        values = left * values[:, :-1] + right * values[:, 1:]
    return values


def primitive_coefficients(knots, coefficients, *, degree: int):
    """Return knots/coefficients of a degree+1 primitive zero at the left end.

    Coefficients have shape ``(n_basis, ...)``; trailing dimensions are batches.
    """
    width = (knots[degree + 1:] - knots[:-(degree + 1)]) / (degree + 1)
    increments = coefficients * width.reshape((-1,) + (1,) * (coefficients.ndim - 1))
    c = jnp.concatenate((jnp.zeros_like(coefficients[:1]), jnp.cumsum(increments, axis=0)))
    t = jnp.concatenate((knots[:1], knots, knots[-1:]))
    return t, c


def spline_primitive(knots, coefficients, x, *, degree: int, order: int = 1):
    """Evaluate an exact spline primitive (integration constants initially zero)."""
    for _ in range(order):
        knots, coefficients = primitive_coefficients(knots, coefficients, degree=degree)
        degree += 1
    values = basis_matrix(knots, x, degree=degree)
    return jnp.tensordot(values, coefficients, axes=(1, 0))
