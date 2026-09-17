"""Small 1D weak-form systems assembled from full BSPF derivative matrices.

These dense O(N^2) operators are intended for modest 1D PDE examples, not
large tensor grids. Natural boundary conditions belong to the weak form;
``constraints`` impose homogeneous essential conditions on the full field.
"""
from dataclasses import dataclass
from functools import partial
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from .operators import derivatives, decompose
from .basis import basis_matrix
from .plans import Plan1D


@partial(jax.tree_util.register_dataclass,
         data_fields=['mass', 'stiffness', 'extension', 'free', 'values', 'quadrature_weights', 'derivative_values'], meta_fields=[])
@dataclass(frozen=True)
class Galerkin1D:
    """Mass/stiffness of BSPF trial functions; physical samples are E @ q."""
    mass: jax.Array
    stiffness: jax.Array
    extension: jax.Array
    free: jax.Array
    values: jax.Array  # physical trial functions at quadrature points
    quadrature_weights: jax.Array
    derivative_values: jax.Array  # differentiated trial functions at quadrature points


def _quadrature_rule(plan, quadrature_order):
    """Gauss nodes and weights split at all sample locations and spline knots."""
    if isinstance(quadrature_order, bool) or not isinstance(quadrature_order, Integral) or quadrature_order < 1:
        raise ValueError('quadrature_order must be a positive integer')
    # Host-only geometry inspection; numerical quadrature assembly uses JAX.
    edges = jnp.asarray(np.unique(np.concatenate((np.asarray(plan.x), np.asarray(plan.knots)))))
    index = jnp.arange(1, quadrature_order, dtype=plan.x.dtype)
    offdiag = index/jnp.sqrt(4*index**2-1)
    nodes, vectors = jnp.linalg.eigh(jnp.diag(offdiag, 1)+jnp.diag(offdiag, -1))
    widths = jnp.diff(edges)
    points = (edges[:-1, None]+widths[:, None]*(nodes+1)/2).reshape(-1)
    weights = (widths[:, None]*vectors[0]**2).reshape(-1)
    return points, weights


def _quadrature_trial(plan, extension, orders, quadrature_order):
    """Evaluate full BSPF trial functions/derivatives on resolved Gauss nodes."""
    points, weights = _quadrature_rule(plan, quadrature_order)
    split = decompose(plan, extension)
    spectrum = jnp.fft.fft(split.residual, axis=0)/plan.x.size
    phase = jnp.exp(1j*(points[:, None]-plan.x[0])*plan.omega)
    values = tuple(
        basis_matrix(plan.knots, points, degree=plan.degree, derivative=k)@split.coefficients
        +(phase@((1j*plan.omega[:, None])**k*spectrum)).real for k in orders)
    return weights, values


def galerkin_1d(plan, *, derivative_order=1, constraints=(), quadrature_order=None):
    """Assemble a quadrature weak form outside jit.

    ``constraints`` is a sequence of (side, derivative order), e.g.
    ``((0, 0), (0, 1))`` for a left clamp; side is 0 (left) or 1 (right).
    Boundary values are zero. Remaining end conditions are natural: with
    derivative_order=1 these are zero flux; with order=2, zero moment/shear.
    This is a quadrature approximation, not an exact spline Galerkin method.
    ``quadrature_order=None`` retains nodal trapezoidal assembly. For accurate
    PDE spectra, supply a positive Gauss order (at least degree+1 recommended):
    integrate actual spline/Fourier trial functions, splitting at nodes and
    knots. This avoids under-integrating oscillatory cardinal functions.
    Positive-noise plans are nonlinear and cannot define these matrices.
    """
    if not isinstance(plan, Plan1D) or plan.noise is not None:
        raise ValueError('galerkin_1d requires a clean Plan1D')
    if isinstance(derivative_order, bool) or not isinstance(derivative_order, Integral) or not 1 <= derivative_order <= plan.max_derivative:
        raise ValueError('derivative_order must be a supported positive integer')
    constraints = tuple(tuple(c) for c in constraints)
    if any(len(c) != 2 or c[0] not in (0, 1) or isinstance(c[1], bool)
           or not isinstance(c[1], Integral) or not 0 <= c[1] <= plan.max_derivative
           for c in constraints) or len(set(constraints)) != len(constraints):
        raise ValueError('constraints must be unique (side=0 or 1, supported order) pairs')
    n = plan.x.size
    if len(constraints) >= n:
        raise ValueError('constraints must leave at least one free degree of freedom')
    identity = jnp.eye(n, dtype=plan.x.dtype)
    orders = tuple(sorted({derivative_order} | {k for _, k in constraints if k}))
    matrices = derivatives(plan, identity, orders=orders)
    matrices[0] = identity
    left = sum(side == 0 for side, _ in constraints)
    right = len(constraints)-left
    pivots = list(range(left)) + list(range(n-right, n))
    free = jnp.arange(left, n-right)
    extension = identity[:, free]
    if constraints:
        rows = jnp.stack([matrices[k][0 if side == 0 else -1] for side, k in constraints])
        # Row scaling avoids comparing value and high-derivative units in rank checks.
        rows = rows / jnp.linalg.norm(rows, axis=1, keepdims=True)
        boundary = rows[:, jnp.array(pivots)]
        if np.linalg.matrix_rank(np.asarray(boundary)) != len(constraints):
            raise ValueError('boundary elimination is singular for these constraints')
        extension = extension.at[jnp.array(pivots)].set(
            -jnp.linalg.solve(boundary, rows[:, free]))
    if quadrature_order is not None:
        if isinstance(quadrature_order, bool) or not isinstance(quadrature_order, Integral) or quadrature_order < 1:
            raise ValueError('quadrature_order must be a positive integer or None')
        weights, (values, gradient) = _quadrature_trial(
            plan, extension, (0, derivative_order), quadrature_order)
        mass = values.T@(weights[:, None]*values)
        stiffness = gradient.T@(weights[:, None]*gradient)
        return Galerkin1D(mass, stiffness, extension, free, values, weights, gradient)
    gradient = matrices[derivative_order] @ extension
    mass = extension.T @ (plan.weights[:, None]*extension)
    stiffness = gradient.T @ (plan.weights[:, None]*gradient)
    return Galerkin1D(mass, stiffness, extension, free, extension, plan.weights, gradient)
