"""Pure fixed-step time integration for array or PyTree states."""
from numbers import Integral

import jax
import jax.numpy as jnp


def rk4_step(rhs, state, t, dt):
    """Classical RK4 for ``rhs(t, state)``; all state leaves retain their shapes."""
    def add(y, k, h):
        return jax.tree_util.tree_map(lambda a, b: a+h*b, y, k)
    k1 = rhs(t, state)
    k2 = rhs(t+dt/2, add(state, k1, dt/2))
    k3 = rhs(t+dt/2, add(state, k2, dt/2))
    k4 = rhs(t+dt, add(state, k3, dt))
    return jax.tree_util.tree_map(
        lambda y, a, b, c, d: y+dt*(a+2*b+2*c+d)/6, state, k1, k2, k3, k4)


def integrate_rk4(rhs, initial, times, *, substeps=1):
    """Return a PyTree of histories with a leading time axis, including initial.

    Each output interval takes exactly ``substeps`` RK4 steps using ``lax.scan``
    and ``lax.fori_loop``. The caller supplies finite, strictly increasing real
    times and a stable step size; there is no adaptive error or CFL control.
    ``rhs`` and positive integer ``substeps`` must be static when passed to jit.
    ``initial`` leaves must be real/complex floating arrays with RHS-compatible
    dtypes, which are preserved by the scan. Parameters captured by rhs can be
    differentiated through the integration.
    """
    if isinstance(substeps, bool) or not isinstance(substeps, Integral) or substeps < 1:
        raise ValueError("substeps must be a positive static integer")
    times = jnp.asarray(times)
    if times.ndim != 1 or times.size < 1 or jnp.iscomplexobj(times):
        raise ValueError("times must be a nonempty real 1D array")
    initial = jax.tree_util.tree_map(jnp.asarray, initial)
    if any(not jnp.issubdtype(a.dtype, jnp.inexact) for a in jax.tree_util.tree_leaves(initial)):
        raise ValueError("initial state leaves must be floating or complex arrays")

    def interval(state, bounds):
        start, end = bounds
        dt = (end-start)/substeps
        state = jax.lax.fori_loop(0, substeps,
            lambda i, y: rk4_step(rhs, y, start+i*dt, dt), state)
        return state, state

    _, history = jax.lax.scan(interval, initial, (times[:-1], times[1:]))
    return jax.tree_util.tree_map(
        lambda first, rest: jnp.concatenate((first[None], rest), axis=0), initial, history)


def integrate_linear_midpoint(operator, initial, times, *, substeps=1):
    """Implicit midpoint histories for an autonomous dense system y'=A@y.

    A must be square and initial must be a vector (or a matrix of columns).
    Each output interval factors its midpoint system once, then reuses its
    propagation matrix. Setup costs O(N^3), storage O(N^2). This method
    preserves quadratic invariants of the semidiscrete system to roundoff,
    but is only second-order accurate in time; stability does not imply
    resolved oscillatory phases. Caller supplies increasing finite times.
    """
    if isinstance(substeps, bool) or not isinstance(substeps, Integral) or substeps < 1:
        raise ValueError('substeps must be a positive static integer')
    operator, initial, times = jnp.asarray(operator), jnp.asarray(initial), jnp.asarray(times)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError('operator must be square')
    if initial.ndim not in (1, 2) or initial.shape[0] != operator.shape[0]:
        raise ValueError('initial must have shape (N,) or (N, batch)')
    if times.ndim != 1 or times.size < 1 or jnp.iscomplexobj(times):
        raise ValueError('times must be a nonempty real 1D array')
    dtype = jnp.result_type(operator, initial, 1.0)
    operator, initial = operator.astype(dtype), initial.astype(dtype)
    identity = jnp.eye(operator.shape[0], dtype=dtype)

    def interval(state, bounds):
        dt = (bounds[1]-bounds[0])/substeps
        advance = jnp.linalg.solve(identity-dt/2*operator, identity+dt/2*operator)
        state = jax.lax.fori_loop(0, substeps, lambda i, y: advance@y, state)
        return state, state

    _, history = jax.lax.scan(interval, initial, (times[:-1], times[1:]))
    return jnp.concatenate((initial[None], history), axis=0)


def _integrate_modal_rk4(energies, nonlinear, initial, times, substeps):
    """Shared fourth-order interaction-picture stages for a diagonal linear part."""
    def interval(c, bounds):
        dt = (bounds[1]-bounds[0])/substeps
        half = jnp.exp(-0.5j*dt*energies)
        full = half*half
        def step(i, c):
            k1 = nonlinear(c)
            k2 = nonlinear(half*(c+dt/2*k1))
            k3 = nonlinear(half*c+dt/2*k2)
            k4 = nonlinear(full*c+dt*half*k3)
            return full*c+dt/6*(full*k1+2*half*(k2+k3)+k4)
        c = jax.lax.fori_loop(0, substeps, step, c)
        return c, c

    _, history = jax.lax.scan(interval, initial, (times[:-1], times[1:]))
    return jnp.concatenate((initial[None], history), axis=0)


def _etdrk4_coefficients(operator, dt):
    """Matrix phi-functions via block exponentials, without an eigenbasis/inverse A."""
    from jax.scipy.linalg import expm
    n = operator.shape[0]
    identity, zero = jnp.eye(n, dtype=operator.dtype), jnp.zeros_like(operator)
    block = jnp.block([[dt*operator, identity, zero, zero],
                      [zero, zero, identity, zero],
                      [zero, zero, zero, identity],
                      [zero, zero, zero, zero]])
    exponential = expm(block, max_squarings=32)
    full = exponential[:n, :n]
    p1, p2, p3 = (exponential[:n, n:2*n], exponential[:n, 2*n:3*n], exponential[:n, 3*n:])
    half_block = expm(jnp.block([[dt/2*operator, identity], [zero, zero]]), max_squarings=32)
    return (full, half_block[:n, :n], dt/2*half_block[:n, n:],
            dt*(p1-3*p2+4*p3), dt*(p2-2*p3), dt*(-p2+4*p3))


def _integrate_matrix_etdrk4(operator, nonlinear, initial, times, substeps):
    """Fourth-order exponential RK; caller supplies a uniform output time grid."""
    if times.size == 1:
        return initial[None]
    dt = (times[-1]-times[0])/((times.size-1)*substeps)
    full, half, q, f1, f2, f3 = _etdrk4_coefficients(operator, dt)

    def interval(state, start):
        def step(i, state):
            t = start+i*dt
            n1 = nonlinear(t, state)
            a = half@state+q@n1
            n2 = nonlinear(t+dt/2, a)
            b = half@state+q@n2
            n3 = nonlinear(t+dt/2, b)
            c = half@a+q@(2*n3-n1)
            n4 = nonlinear(t+dt, c)
            return full@state+f1@n1+2*f2@(n2+n3)+f3@n4
        state = jax.lax.fori_loop(0, substeps, step, state)
        return state, state

    _, history = jax.lax.scan(interval, initial, times[:-1])
    return jnp.concatenate((initial[None], history), axis=0)

def rk4_stages(state, dt, rhs):
    """Shared flow RK4 for array/PyTree states and per-stage diagnostics.

    Tree mapping preserves NumPy leaves for host callers and JAX leaves for
    compiled callers; coupled fields need not share a shape or be packed.
    """
    from jax.tree_util import tree_map

    def add(y, derivative, h):
        return tree_map(lambda v, k: v+h*k, y, derivative)

    a, da = rhs(state)
    b, db = rhs(add(state, a, dt/2))
    c, dc = rhs(add(state, b, dt/2))
    d, dd = rhs(add(state, c, dt))
    result = tree_map(lambda y, ka, kb, kc, kd: y+dt/6*(ka+2*kb+2*kc+kd),
                      state, a, b, c, d)
    return result, (da, db, dc, dd)


def imex_midpoint(state, time, dt, mass, diffusion, explicit, solve):
    """Cavity IMEX midpoint, with caller-supplied linear operators/factorization.

    diffusion is the positive viscous operator including viscosity; solve
    inverts M + dt/2 D. Works with either NumPy or JAX arrays.
    """
    momentum = mass(state)
    half = solve(momentum + dt / 2 * explicit(state, time))
    return solve(
        momentum - dt / 2 * diffusion(state) + dt * explicit(half, time + dt / 2)
    )
