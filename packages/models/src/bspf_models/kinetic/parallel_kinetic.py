"""Open-boundary 1z1v kinetic transport with weak upwind boundary fluxes."""
from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_factor, cho_solve

from pybspf.galerkin import galerkin_1d
from pybspf.galerkin import _quadrature_rule
from pybspf.time_integration import integrate_rk4


@partial(jax.tree_util.register_dataclass, data_fields=[
    'z', 'v', 'z_points', 'v_points', 'z_transport', 'velocity', 'positive',
    'negative', 'v_transport', 'z_lift', 'v_lift', 'z_projection', 'v_projection'],
    meta_fields=['acceleration'])
@dataclass(frozen=True)
class ParallelKineticPlan:
    z: jax.Array
    v: jax.Array
    z_points: jax.Array
    v_points: jax.Array
    z_transport: jax.Array
    velocity: jax.Array
    positive: jax.Array
    negative: jax.Array
    v_transport: jax.Array
    z_lift: jax.Array
    v_lift: jax.Array
    z_projection: jax.Array
    v_projection: jax.Array
    acceleration: float


def _transport_axis(spatial, quadrature_order):
    weak = galerkin_1d(spatial, derivative_order=1, quadrature_order=quadrature_order)
    points, weights = _quadrature_rule(spatial, quadrature_order)
    n = spatial.x.size
    ends = jnp.eye(n, dtype=spatial.x.dtype)[:, jnp.array([0, n-1])]
    surface = jnp.outer(ends[:, 1], ends[:, 1])-jnp.outer(ends[:, 0], ends[:, 0])
    s = weak.derivative_values.T@(weights[:, None]*weak.values)
    # Enforce the integration-by-parts identity to roundoff; homogeneous
    # upwind inflow then dissipates the mass-weighted phase-space L2 norm.
    s = .5*(s-s.T+surface)
    factor = cho_factor(weak.mass, lower=True)
    solve = lambda a: cho_solve(factor, a)
    projection = solve(weak.values.T*weights)
    return points, weak.values, solve(s), solve(ends), projection


def plan_parallel_kinetic(z_plan, v_plan, *, acceleration=0., quadrature_order=8):
    """Assemble f_t + v f_z + a f_v = 0 on a finite (z,v) rectangle.

    Two clean 1D BSPF plans define sample axes. ``acceleration=qE_parallel/m``
    is a finite constant; no Poisson solve, collisions or magnetic mirror term
    is included. Velocity multiplication is projected with resolved quadrature,
    split into positive/negative parts for incoming/outgoing spatial fluxes.
    Time integration imposes inflow weakly; endpoint samples are not overwritten.
    Setup is outside JIT; the resulting immutable plan is a PyTree. Dense tensor
    operators target modest 1z1v examples, not full gyrokinetic simulations.
    """
    a = np.asarray(acceleration)
    if a.ndim or np.iscomplexobj(a) or not np.isfinite(a):
        raise ValueError('acceleration must be a finite real scalar')
    a = float(a)
    zq, _, sz, lz, pz = _transport_axis(z_plan, quadrature_order)
    vq, qv, sv, lv, pv = _transport_axis(v_plan, quadrature_order)
    positive = (pv*jnp.maximum(vq, 0.))@qv
    negative = (pv*jnp.minimum(vq, 0.))@qv
    v_operator = a*sv
    # Remove the outgoing normal flux on the appropriate velocity face.
    if a > 0:
        v_operator = v_operator.at[:, -1].add(-a*lv[:, 1])
    elif a < 0:
        v_operator = v_operator.at[:, 0].add(a*lv[:, 0])
    return ParallelKineticPlan(z_plan.x, v_plan.x, zq, vq, sz,
        positive+negative, positive, negative, v_operator, lz, lv, pz, pv, a)


def integrate_parallel_kinetic(plan, initial, times, *, inflow, substeps=1):
    """Return (nt,nz,nv) histories using explicit RK4 and weak inflow fluxes.

    ``inflow(t,z,v)`` is a vectorized JAX callable. At z_left only v>0 is
    prescribed; at z_right only v<0. For a>0 the v_min face is incoming, and
    for a<0 the v_max face is incoming. Outgoing fluxes use the computed field.
    The callable must be finite on whole faces even where its weight is zero.
    Initial values must be compatible with the incoming traces.

    Caller supplies real finite data, increasing times and a stable advection
    step (no automatic CFL control). Supports outer JIT with static inflow and
    substeps. This spectral weak scheme is neither positivity preserving nor
    exactly particle conserving; measure negativity and boundary balances.
    No filtering or clipping of f is performed.
    """
    initial = jnp.asarray(initial)
    if initial.shape != (plan.z.size, plan.v.size) or jnp.iscomplexobj(initial):
        raise ValueError('initial must be a real (nz,nv) array')
    initial = initial.astype(plan.z.dtype)

    def data(t, z, v, shape):
        value = jnp.asarray(inflow(t, z, v))
        if jnp.iscomplexobj(value):
            raise ValueError('inflow must return real values')
        return jnp.broadcast_to(value, shape)

    def rhs(t, f):
        left = plan.v_projection@(jnp.maximum(plan.v_points, 0.)*data(
            t, plan.z[0], plan.v_points, plan.v_points.shape))
        right = plan.v_projection@(jnp.minimum(plan.v_points, 0.)*data(
            t, plan.z[-1], plan.v_points, plan.v_points.shape))
        out_left = f[0]@plan.negative.T
        out_right = f[-1]@plan.positive.T
        result = (plan.z_transport@f@plan.velocity.T+f@plan.v_transport.T
            +jnp.outer(plan.z_lift[:, 0], left+out_left)
            -jnp.outer(plan.z_lift[:, 1], right+out_right))
        if plan.acceleration != 0:
            side = 0 if plan.acceleration > 0 else 1
            value = data(t, plan.z_points, plan.v[0 if side == 0 else -1], plan.z_points.shape)
            result += abs(plan.acceleration)*jnp.outer(
                plan.z_projection@value, plan.v_lift[:, side])
        return result

    return integrate_rk4(rhs, initial, times, substeps=substeps)
