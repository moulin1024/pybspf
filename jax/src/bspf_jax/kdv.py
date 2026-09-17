"""Finite-interval KdV with nonperiodic, time-dependent boundary data."""
from dataclasses import dataclass
from functools import partial
from numbers import Integral

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from .galerkin import _quadrature_trial
from .operators import differentiate
from .plans import Plan1D
from .time_integration import _integrate_matrix_etdrk4


@partial(jax.tree_util.register_dataclass, data_fields=[
    'mass', 'dispersion', 'values', 'gradient', 'quadrature_weights', 'free',
    'boundary_values', 'value_load', 'value_mass', 'right_slope_test'], meta_fields=[])
@dataclass(frozen=True)
class KdVPlan:
    """Weak system with essential endpoint values and natural right-end slope."""
    mass: jax.Array
    dispersion: jax.Array
    values: jax.Array
    gradient: jax.Array
    quadrature_weights: jax.Array
    free: jax.Array
    boundary_values: jax.Array
    value_load: jax.Array
    value_mass: jax.Array
    right_slope_test: jax.Array


def plan_kdv(plan, *, quadrature_order=8):
    """Assemble nonperiodic KdV on a finite closed interval, outside jit.

    For u_t + a u u_x + u_xxx = 0, prescribe u(left,t), u(right,t),
    and u_x(right,t). Endpoint values are strongly lifted into the full BSPF
    field. The right slope is supplied as a natural weak boundary load.
    There is no periodic identification or padding.

    Test functions vanish at both ends. Integration by parts twice gives
    A=-G2ᵀWG1-d_leftᵀd_left. Its symmetric part is
    -(d_leftᵀd_left+d_rightᵀd_right)/2, hence homogeneous linear evolution is
    dissipative in the mass norm. Explicitly retaining that symmetric part
    avoids roundoff spoiling this property. Boundary lifting contributes both
    a spatial load and a mass term involving the time derivative of the data.
    Dense resolved quadrature is intended for small 1D systems only.
    """
    if not isinstance(plan, Plan1D) or plan.noise is not None:
        raise ValueError('plan_kdv requires a clean Plan1D')
    if plan.degree < 2 or plan.max_derivative < 2:
        raise ValueError('KdV weak assembly requires degree and max_derivative >= 2')
    n = plan.x.size
    identity = jnp.eye(n, dtype=plan.x.dtype)
    derivative = differentiate(plan, identity)
    free, ends = jnp.arange(1, n-1), jnp.array([0, n-1])
    weights, (q, g1, g2) = _quadrature_trial(plan, identity, (0, 1, 2), quadrature_order)
    values, gradient, second = q[:, free], g1[:, free], g2[:, free]
    boundary_values, boundary_gradient = q[:, ends], g1[:, ends]
    left, right = derivative[0, free], derivative[-1, free]
    mass = values.T@(weights[:, None]*values)
    dispersion = -second.T@(weights[:, None]*gradient)-jnp.outer(left, left)
    dispersion = (dispersion-dispersion.T)/2-(jnp.outer(left, left)+jnp.outer(right, right))/2
    value_load = (-second.T@(weights[:, None]*boundary_gradient)
                  -jnp.outer(left, derivative[0, ends]))
    value_mass = values.T@(weights[:, None]*boundary_values)
    return KdVPlan(mass, dispersion, values, gradient, weights, free,
                   boundary_values, value_load, value_mass, right)


def integrate_kdv(plan, initial, times, *, boundary, nonlinearity=6., substeps=1):
    """Evolve u_t + a u u_x + u_xxx = 0 with prescribed nonperiodic boundaries.

    ``boundary(t)`` is a differentiable JAX callable returning the real vector
    [u(left,t), u(right,t), u_x(right,t)]. Its Dirichlet time derivatives are
    obtained by JVP for the mass lifting term. Endpoint values are imposed at
    every stage/output; the right slope is weakly imposed and must be measured.
    ``initial`` and each returned state contain all N closed-grid samples;
    initial endpoints must be compatible with boundary(times[0]).

    Fourth-order ETDRK4 uses matrix exponential/phi functions in mass-scaled
    coordinates. Unlike a modal diagonalization, this tolerates the highly
    nonnormal finite-interval third-derivative operator. Nonautonomous boundary
    forcing can reduce the observed temporal order on stiff grids: refine dt.
    Norm/mass need not be conserved when the boundary flux is nonzero.

    Caller supplies finite, UNIFORMLY SPACED increasing real times, a finite
    real nonlinearity, and compatible initial/boundary data. substeps is a
    positive static integer; boundary must be static under outer JIT. Storage
    and matrix-function setup are dense, with O(N³) setup. Returns (ntimes,N).
    """
    if isinstance(substeps, bool) or not isinstance(substeps, Integral) or substeps < 1:
        raise ValueError('substeps must be a positive static integer')
    initial, times, nonlinearity = map(jnp.asarray, (initial, times, nonlinearity))
    n = plan.mass.shape[0]+2
    if initial.ndim != 1 or initial.size != n or jnp.iscomplexobj(initial):
        raise ValueError('initial must be a real vector of all closed-grid samples')
    if times.ndim != 1 or times.size < 1 or jnp.iscomplexobj(times):
        raise ValueError('times must be a nonempty real 1D array')
    if nonlinearity.ndim != 0 or jnp.iscomplexobj(nonlinearity):
        raise ValueError('nonlinearity must be a real scalar')
    initial = initial.astype(plan.mass.dtype)
    times = times.astype(plan.mass.dtype)

    def data(t):
        result = jnp.asarray(boundary(t))
        if result.shape != (3,) or jnp.iscomplexobj(result):
            raise ValueError('boundary(t) must return three real scalars')
        return result.astype(plan.mass.dtype)

    lower = jnp.linalg.cholesky(plan.mass)
    left = solve_triangular(lower, plan.dispersion, lower=True)
    operator = solve_triangular(lower, left.T, lower=True).T
    inverse_lower = solve_triangular(lower, jnp.eye(n-2, dtype=plan.mass.dtype), lower=True)
    physical = inverse_lower.T
    values = plan.values@physical
    projection = (nonlinearity/2)*(inverse_lower@plan.gradient.T)*plan.quadrature_weights
    value_load = inverse_lower@plan.value_load
    value_mass = inverse_lower@plan.value_mass
    slope_load = inverse_lower@plan.right_slope_test

    def nonlinear(t, state):
        prescribed, rate = jax.jvp(data, (t,), (jnp.ones_like(t),))
        g = prescribed[:2]
        field = values@state+plan.boundary_values@g
        return (projection@(field*field)+value_load@g
                +slope_load*prescribed[2]-value_mass@rate[:2])

    state = lower.T@initial[plan.free]
    history = _integrate_matrix_etdrk4(operator, nonlinear, state, times, substeps)
    prescribed = jax.vmap(data)(times)
    result = jnp.zeros((times.size, n), dtype=initial.dtype)
    result = result.at[:, plan.free].set(history@physical.T)
    result = result.at[:, 0].set(prescribed[:, 0]).at[:, -1].set(prescribed[:, 1])
    return result.at[0, plan.free].set(initial[plan.free])
