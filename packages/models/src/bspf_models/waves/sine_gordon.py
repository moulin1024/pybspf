"""Finite-interval sine–Gordon dynamics with moving Dirichlet boundaries."""
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

from pybspf.time_integration import integrate_rk4


def integrate_sine_gordon(weak, initial, velocity, times, *, boundary, substeps=1):
    """Solve u_tt = u_xx - sin(u) with prescribed endpoint values.

    Pass an unconstrained ``galerkin_1d(plan, derivative_order=1,
    quadrature_order=...)`` system. The nonlinear load is evaluated on its
    resolved quadrature nodes. Interior samples evolve by classical RK4;
    endpoint values and velocities are imposed at every stage/output.
    Returns (displacement, velocity), each shaped (ntimes, N).

    ``boundary(t)`` returns [u(left,t), u(right,t)] and must support two JAX
    time derivatives. The mass lifting term -M_ib g_tt is essential for
    nonstationary boundaries. Initial displacement AND velocity must agree
    with g and g_t at times[0]. Caller supplies finite increasing real times,
    compatible real data, and a stable step size (wave CFL); no adaptivity is
    provided. boundary and substeps are static under outer JIT. Assembly and
    mass factorization are dense: intended for modest 1D examples only.
    """
    initial, velocity, times = map(jnp.asarray, (initial, velocity, times))
    n = weak.mass.shape[0]
    if weak.extension.shape != (n, n) or weak.free.size != n:
        raise ValueError('sine-Gordon requires an unconstrained first-derivative weak form')
    for name, a in (('initial', initial), ('velocity', velocity)):
        if a.shape != (n,) or jnp.iscomplexobj(a):
            raise ValueError(f'{name} must be a real vector of all N samples')
    if times.ndim != 1 or times.size < 1 or jnp.iscomplexobj(times):
        raise ValueError('times must be a nonempty real 1D array')
    initial, velocity, times = (a.astype(weak.mass.dtype) for a in (initial, velocity, times))

    def data(t):
        g = jnp.asarray(boundary(t))
        if g.shape != (2,) or jnp.iscomplexobj(g):
            raise ValueError('boundary(t) must return two real scalars')
        return g.astype(weak.mass.dtype)

    def rate(t):
        return jax.jvp(data, (t,), (jnp.ones_like(t),))[1]

    ends = jnp.array([0, n-1])
    factor = cho_factor(weak.mass[1:-1, 1:-1], lower=True)
    stiffness = cho_solve(factor, weak.stiffness[1:-1])
    lift_mass = cho_solve(factor, weak.mass[1:-1, ends])
    projection = cho_solve(factor, weak.values[:, 1:-1].T*weak.quadrature_weights)

    def rhs(t, state):
        q, v = state
        g = data(t)
        field = jnp.concatenate((g[:1], q, g[1:]))
        acceleration = (-stiffness@field-projection@jnp.sin(weak.values@field)
                        -lift_mass@jax.jvp(rate, (t,), (jnp.ones_like(t),))[1])
        return v, acceleration

    q, v = integrate_rk4(rhs, (initial[1:-1], velocity[1:-1]), times, substeps=substeps)
    g, gt = jax.vmap(data)(times), jax.vmap(rate)(times)
    return (jnp.concatenate((g[:, :1], q, g[:, 1:]), axis=1),
            jnp.concatenate((gt[:, :1], v, gt[:, 1:]), axis=1))
