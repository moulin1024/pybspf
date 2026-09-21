"""Static flux-tube 1z2v mirror transport: BSPF in z/v, Gauss nodes in mu.

mu = m*v_perp**2/(2*B); H = m*v_parallel**2/2 + mu*B(z).
The tube has A(z)*B(z)=constant. The gyrovelocity Jacobian B cancels A,
so the reduced particle measure is dz dv_parallel dmu (constant 2*pi*A*B/m
suppressed). This is not a constant-area Cartesian model with varying B.
No electric field, collisions, perpendicular drifts or mu transport.
"""
from dataclasses import dataclass
from functools import partial
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from .galerkin import _quadrature_trial
from .parallel_kinetic import plan_parallel_kinetic, _transport_axis
from .time_integration import integrate_rk4
from .fast_drift_kinetic import (FastDriftKineticPlan, plan_fast_drift_kinetic,
    fast_boundary_fluxes, fast_drift_rhs, fast_moments, fast_log_diagnostics)


@partial(jax.tree_util.register_dataclass,
         data_fields=['transport', 'mu', 'mu_weights', 'z_values', 'v_values',
                      'z_weights', 'v_weights', 'field', 'field_ends',
                      'acceleration', 'force_matrix', 'v_transport', 'v_lift',
                      'number_weight', 'parallel_weight', 'magnetic_weight'],
         meta_fields=['mass'])
@dataclass(frozen=True)
class DriftKineticPlan:
    transport: object
    mu: jax.Array
    mu_weights: jax.Array
    z_values: jax.Array
    v_values: jax.Array
    z_weights: jax.Array
    v_weights: jax.Array
    field: jax.Array
    field_ends: jax.Array
    acceleration: jax.Array
    force_matrix: jax.Array
    v_transport: jax.Array
    v_lift: jax.Array
    number_weight: jax.Array
    parallel_weight: jax.Array
    magnetic_weight: jax.Array
    mass: float


def plan_drift_kinetic(z_plan, v_plan, *, magnetic_field, magnetic_gradient,
                       mu_max, n_mu=8, mass=1., quadrature_order=8, backend="matrix_free"):
    """Assemble outside JIT. B and dB/dz are vectorized, consistent callables.

    Gauss-Legendre mu nodes cover [0,mu_max]; each node is an invariant slice.
    z/v use clean BSPF plans. Positive B is checked at nodes/quadrature/endpoints;
    the caller ensures positivity between them and consistency of B and B'.
    The default matrix_free backend uses shifted FFT quadrature, compact
    spline evaluation, FFT convolution and a fixed-size mass correction. Use
    sample-aligned (preferably boundary-clustered) knots and a bounded spline
    core; see plan_fast_axis for size constraints. backend="dense" retains the
    original split-knot dense reference. No silent dense fallback. Use float64.
    """
    if backend == "matrix_free":
        return plan_fast_drift_kinetic(z_plan,v_plan,magnetic_field=magnetic_field,
            magnetic_gradient=magnetic_gradient,mu_max=mu_max,n_mu=n_mu,mass=mass,
            quadrature_order=quadrature_order)
    if backend != "dense":
        raise ValueError('backend must be dense or matrix_free')
    for name, value in [('mass', mass), ('mu_max', mu_max)]:
        a = np.asarray(value)
        if a.ndim or np.iscomplexobj(a) or not np.isfinite(a) or a <= 0:
            raise ValueError(f'{name} must be a finite positive scalar')
    if isinstance(n_mu, bool) or not isinstance(n_mu, Integral) or n_mu < 1:
        raise ValueError('n_mu must be a positive integer')
    p = plan_parallel_kinetic(z_plan, v_plan, quadrature_order=quadrature_order)
    wz, (qz,) = _quadrature_trial(z_plan, jnp.eye(z_plan.x.size), (0,), quadrature_order)
    wv, (qv,) = _quadrature_trial(v_plan, jnp.eye(v_plan.x.size), (0,), quadrature_order)
    _, _, sv, lv, _ = _transport_axis(v_plan, quadrature_order)
    nodes, weights = np.polynomial.legendre.leggauss(n_mu)
    mu = jnp.asarray((nodes+1)*float(mu_max)/2, dtype=z_plan.x.dtype)
    wm = jnp.asarray(weights*float(mu_max)/2, dtype=z_plan.x.dtype)

    def evaluate(fn, points, name, positive=False):
        a = np.asarray(fn(points))
        if np.iscomplexobj(a) or not np.all(np.isfinite(a)) or (positive and np.any(a <= 0)):
            raise ValueError(f'{name} must be real, finite' + (' and positive' if positive else ''))
        return jnp.broadcast_to(jnp.asarray(a, dtype=z_plan.x.dtype), points.shape)

    field = evaluate(magnetic_field, p.z_points, 'magnetic_field', True)
    bn = evaluate(magnetic_field, p.z, 'magnetic_field', True)
    gradient = evaluate(magnetic_gradient, p.z_points, 'magnetic_gradient')
    acceleration = -gradient/float(mass)
    force = p.z_projection@(acceleration[:, None]*qz)
    zw, vw = wz@qz, wv@qv
    number = zw[:, None]*vw[None, :]
    parallel = zw[:, None]*((wv*(float(mass)*p.v_points**2/2))@qv)[None, :]
    magnetic = ((wz*field)@qz)[:, None]*vw[None, :]
    return DriftKineticPlan(p, mu, wm, qz, qv, wz, wv, field, bn[jnp.array([0, -1])],
                           acceleration, force, sv, lv, number, parallel, magnetic, float(mass))


def _boundary_fluxes(plan, t, f, inflow, *, logarithmic=False):
    """Signed coordinate fluxes at z-left/right, v-left/right, before normals."""
    if isinstance(plan, FastDriftKineticPlan):
        return fast_boundary_fluxes(plan,t,f,inflow,logarithmic=logarithmic)
    p = plan.transport
    mu = plan.mu[None, :]
    value = jnp.exp if logarithmic else lambda a: a

    def data(z, v, shape):
        a = jnp.asarray(inflow(t, z, v, mu))
        if jnp.iscomplexobj(a):
            raise ValueError('inflow must be real')
        return value(jnp.broadcast_to(a, shape))

    v = p.v_points[:, None]
    zl = jnp.maximum(v, 0)*data(p.z[0], v, (v.size, mu.size)) + jnp.minimum(v, 0)*value(plan.v_values@f[0])
    zr = jnp.minimum(v, 0)*data(p.z[-1], v, (v.size, mu.size)) + jnp.maximum(v, 0)*value(plan.v_values@f[-1])
    a = plan.acceleration[:, None]*mu
    z = p.z_points[:, None]
    vl = jnp.maximum(a, 0)*data(z, p.v[0], a.shape) + jnp.minimum(a, 0)*value(plan.z_values@f[:, 0])
    vr = jnp.minimum(a, 0)*data(z, p.v[-1], a.shape) + jnp.maximum(a, 0)*value(plan.z_values@f[:, -1])
    return zl, zr, vl, vr


def _boundary_rates(plan, fluxes):
    """Inward N/H rates, shape (2,4), faces z-left,z-right,v-left,v-right."""
    p = plan.transport
    mu = plan.mu[None, :]
    hz = [plan.mass*p.v_points[:, None]**2/2 + mu*b for b in plan.field_ends]
    hv = [plan.mass*v**2/2 + mu*plan.field[:, None] for v in (p.v[0], p.v[-1])]
    weights = (plan.v_weights, plan.v_weights, plan.z_weights, plan.z_weights)
    rates = []
    for flux, h, w, sign in zip(fluxes, hz+hv, weights, (1., -1., 1., -1.)):
        weighted = sign*w[:, None]*plan.mu_weights[None, :]*flux
        rates.append(jnp.stack((jnp.sum(weighted), jnp.sum(weighted*h))))
    return jnp.stack(rates, axis=-1)


def drift_kinetic_rhs(plan, t, distribution, *, inflow):
    """Return df/dt and independently integrated inward boundary N/H rates."""
    if isinstance(plan, FastDriftKineticPlan):
        return fast_drift_rhs(plan,t,distribution,inflow=inflow)
    p, f = plan.transport, distribution
    # Independent invariant-mu slices; no dense five-dimensional matrix.
    result = jnp.einsum('ij,jkm,lk->ilm', p.z_transport, f, p.velocity)
    result += jnp.einsum('ij,jkm,lk,m->ilm', plan.force_matrix, f,
                         plan.v_transport, plan.mu)
    fluxes = _boundary_fluxes(plan, t, f, inflow)
    zl, zr, vl, vr = fluxes
    result += p.z_lift[:, 0, None, None]*(p.v_projection@zl)[None, :, :]
    result -= p.z_lift[:, 1, None, None]*(p.v_projection@zr)[None, :, :]
    result += (p.z_projection@vl)[:, None, :]*plan.v_lift[None, :, 0, None]
    result -= (p.z_projection@vr)[:, None, :]*plan.v_lift[None, :, 1, None]
    return result, _boundary_rates(plan, fluxes)


def drift_kinetic_moments(plan, distribution):
    """Reduced N, parallel energy, perpendicular energy, total energy.

    Arbitrary leading batch/time axes are supported. These use resolved BSPF
    quadrature, with the same physical measure as boundary diagnostics.
    Perpendicular energy mu*B changes along an orbit although mu is fixed.
    """
    f = jnp.asarray(distribution)
    if isinstance(plan, FastDriftKineticPlan):
        return fast_moments(plan,f)
    n = jnp.einsum('...ijm,ij,m->...', f, plan.number_weight, plan.mu_weights)
    kp = jnp.einsum('...ijm,ij,m->...', f, plan.parallel_weight, plan.mu_weights)
    km = jnp.einsum('...ijm,ij,m->...', f, plan.magnetic_weight, plan.mu_weights*plan.mu)
    return jnp.stack((n, kp, km, kp+km), axis=-1)


def integrate_drift_kinetic(plan, initial, times, *, inflow, substeps=1):
    """Return (f_history, cumulative_inward_N_H_by_face), using coupled RK4.

    f shape is (nz,nv,n_mu); transfers shape is (nt,2,4), initially zero.
    The numerical upwind flux uses reservoir data only on incoming faces.
    Outgoing values evolve freely; no mu-boundary flux exists. Caller supplies
    finite compatible initial/inflow data, increasing times and a stable step.
    No positivity limiter or exact energy-conservation claim. Check N-N0-sum
    transfers[:,0] and H-H0-sum transfers[:,1], plus resolution/step convergence.
    No diagnostic correction is applied to the distribution or transfers.
    """
    f = jnp.asarray(initial)
    expected = (plan.transport.z.size, plan.transport.v.size, plan.mu.size)
    if f.shape != expected or jnp.iscomplexobj(f):
        raise ValueError(f'initial must be real with shape {expected}')
    f = f.astype(plan.transport.z.dtype)
    return integrate_rk4(lambda t, state: drift_kinetic_rhs(plan, t, state[0], inflow=inflow),
                         (f, jnp.zeros((2, 4), dtype=f.dtype)), times, substeps=substeps)


def integrate_log_drift_kinetic(plan, initial_log, times, *, log_inflow, substeps=1):
    """Evolve g=log(f) and return (g_history, physical_N_H_transfers).

    For this divergence-free (z,v) flow, g obeys the same advection equation
    as f. The represented distribution is exp(Qz @ g @ Qv.T), NOT the BSPF
    interpolant of exp(g) at nodes. Thus reconstructed f is nonnegative at
    every point where g is finite (underflow can give zero). No floor, clipping
    or moment repair is applied. initial_log and log_inflow must be finite real
    logarithms; exact vacuum/zero-inflow data are not supported by this path.

    Boundary transfers integrate exp(g) using upwind incoming log data and
    computed outgoing log traces, at the same RK4 stages. The logarithmic
    discretization is not exactly conservative for general distributions:
    validate with log_drift_kinetic_diagnostics and resolution refinement.
    The original linear-f integrator remains available for distributions with
    exact zeros and for comparisons. Inputs and time-step contracts otherwise
    match integrate_drift_kinetic. Returns LOGS explicitly, avoiding underflow
    information loss in diagnostics. Use exp(history) only for nodal output.
    """
    g = jnp.asarray(initial_log)
    expected = (plan.transport.z.size, plan.transport.v.size, plan.mu.size)
    if g.shape != expected or jnp.iscomplexobj(g):
        raise ValueError(f'initial_log must be real with shape {expected}')
    g = g.astype(plan.transport.z.dtype)

    def rhs(t, state):
        dg, _ = drift_kinetic_rhs(plan, t, state[0], inflow=log_inflow)
        fluxes = _boundary_fluxes(plan, t, state[0], log_inflow, logarithmic=True)
        return dg, _boundary_rates(plan, fluxes)

    return integrate_rk4(rhs, (g, jnp.zeros((2, 4), dtype=g.dtype)), times, substeps=substeps)


def log_drift_kinetic_diagnostics(plan, log_distribution):
    """Return (N/Kparallel/Kperp/H, minimum_nodal_f, minimum_quadrature_f).

    Integrate the actual exponential representation, not an interpolation of
    its nodal samples. Arbitrary leading axes are supported; batch states are
    processed sequentially to avoid storing all time/quadrature/mu values.
    Quadrature minima are additional checks, not the basis of the positivity
    guarantee: exp(real reconstruction) is nonnegative between these points too.
    """
    g = jnp.asarray(log_distribution)
    if isinstance(plan, FastDriftKineticPlan):
        return fast_log_diagnostics(plan,g)
    shape = (plan.transport.z.size, plan.transport.v.size, plan.mu.size)
    if g.shape[-3:] != shape or jnp.iscomplexobj(g):
        raise ValueError(f'log_distribution must be real with trailing shape {shape}')
    leading = g.shape[:-3]
    weights = plan.z_weights[:, None, None]*plan.v_weights[None, :, None]*plan.mu_weights[None, None, :]
    kp = plan.mass*plan.transport.v_points[None, :, None]**2/2
    km = plan.field[:, None, None]*plan.mu[None, None, :]

    def one(state):
        values = jnp.exp(jnp.einsum('ai,ijm,bj->abm', plan.z_values, state, plan.v_values))
        wf = weights*values
        n, parallel, perpendicular = jnp.sum(wf), jnp.sum(wf*kp), jnp.sum(wf*km)
        return (jnp.stack((n, parallel, perpendicular, parallel+perpendicular)),
                jnp.exp(jnp.min(state)), jnp.min(values))

    moments, nodal, quadrature = jax.lax.map(one, g.reshape((-1,)+shape))
    return moments.reshape(leading+(4,)), nodal.reshape(leading), quadrature.reshape(leading)
