"""Linear shear-Alfvén waves on a driven, line-tied finite interval."""
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_factor, cho_solve

from .galerkin import _quadrature_rule, _quadrature_trial
from .operators import differentiate
from .plans import Plan1D
from .time_integration import integrate_rk4


@partial(jax.tree_util.register_dataclass, data_fields=[
    'mass', 'stiffness', 'values', 'gradient', 'weights', 'density',
    'boundary_gradient', 'magnetic_field', 'permeability'], meta_fields=[])
@dataclass(frozen=True)
class AlfvenPlan:
    """Resolved full-field weak form; density is stored at quadrature points."""
    mass: jax.Array
    stiffness: jax.Array
    values: jax.Array
    gradient: jax.Array
    weights: jax.Array
    density: jax.Array
    boundary_gradient: jax.Array
    magnetic_field: jax.Array
    permeability: jax.Array


def plan_alfven(plan, *, density=1., magnetic_field=1., permeability=1., quadrature_order=8):
    """Assemble rho(z) xi_tt = (B0²/mu0) xi_zz outside jit.

    Straight constant guide field, static positive density, uniform equilibrium
    pressure, linear transverse perturbations. ``density`` is a positive scalar
    or vectorized callable evaluated at Gauss points, not an array of nodal
    density samples. Mass is Q.T W rho Q; stiffness is (B0²/mu0) G.T W G.
    Both endpoint displacements will be prescribed by integrate_alfven.
    Dense, resolved-quadrature infrastructure for modest clean 1D BSPF grids.
    """
    if not isinstance(plan, Plan1D) or plan.noise is not None:
        raise ValueError('plan_alfven requires a clean Plan1D')
    for name, value in [('magnetic_field', magnetic_field), ('permeability', permeability)]:
        a = np.asarray(value)
        if a.ndim or np.iscomplexobj(a) or not np.isfinite(a) or (a <= 0 if name == 'permeability' else a == 0):
            raise ValueError(f'{name} must be a finite real scalar, with nonzero B0 and positive mu0')
    points, weights = _quadrature_rule(plan, quadrature_order)
    rho = jnp.asarray(density(points) if callable(density) else density)
    if rho.shape not in ((), points.shape) or jnp.iscomplexobj(rho):
        raise ValueError('density must be a scalar or callable returning quadrature-point values')
    rho = jnp.broadcast_to(rho, points.shape).astype(plan.x.dtype)
    if not np.all(np.isfinite(rho)) or np.any(np.asarray(rho) <= 0):
        raise ValueError('density must be finite and positive')
    identity = jnp.eye(plan.x.size, dtype=plan.x.dtype)
    _, (q, g) = _quadrature_trial(plan, identity, (0, 1), quadrature_order)
    b0, mu0 = jnp.asarray(magnetic_field, dtype=plan.x.dtype), jnp.asarray(permeability, dtype=plan.x.dtype)
    mass = q.T@((weights*rho)[:, None]*q)
    stiffness = (b0*b0/mu0)*(g.T@(weights[:, None]*g))
    ends = differentiate(plan, identity)[jnp.array([0, plan.x.size-1])]
    return AlfvenPlan(mass, stiffness, q, g, weights, rho, ends, b0, mu0)


def integrate_alfven(plan, displacement, velocity, times, *, boundary, substeps=1):
    """Return (xi, xi_t) histories of all samples with driven Dirichlet ends.

    ``boundary(t)`` returns [xi(left,t), xi(right,t)] and must support two JAX
    derivatives. The boundary mass lifting -M_ib g_tt is included; prescribe
    displacement only, not independent magnetic boundary data. Initial values
    and velocities must agree with g and g_t. Caller supplies finite increasing
    real times and a stable wave-CFL step. RK4 is explicit and not exactly
    energy conserving. boundary/substeps are static under JIT; plans are PyTrees.
    """
    displacement, velocity, times = map(jnp.asarray, (displacement, velocity, times))
    n = plan.mass.shape[0]
    for name, a in [('displacement', displacement), ('velocity', velocity)]:
        if a.shape != (n,) or jnp.iscomplexobj(a):
            raise ValueError(f'{name} must be a real vector of all N samples')
    if times.ndim != 1 or times.size < 1 or jnp.iscomplexobj(times):
        raise ValueError('times must be a nonempty real 1D array')
    displacement, velocity, times = (a.astype(plan.mass.dtype) for a in (displacement, velocity, times))

    def data(t):
        g = jnp.asarray(boundary(t))
        if g.shape != (2,) or jnp.iscomplexobj(g):
            raise ValueError('boundary(t) must return two real scalars')
        return g.astype(plan.mass.dtype)

    def rate(t):
        return jax.jvp(data, (t,), (jnp.ones_like(t),))[1]

    factor = cho_factor(plan.mass[1:-1, 1:-1], lower=True)
    operator = cho_solve(factor, plan.stiffness[1:-1])
    lift = cho_solve(factor, plan.mass[1:-1, jnp.array([0, n-1])])

    def rhs(t, state):
        q, v = state
        g = data(t)
        field = jnp.concatenate((g[:1], q, g[1:]))
        acceleration = jax.jvp(rate, (t,), (jnp.ones_like(t),))[1]
        return v, -operator@field-lift@acceleration

    q, v = integrate_rk4(rhs, (displacement[1:-1], velocity[1:-1]), times, substeps=substeps)
    g, gt = jax.vmap(data)(times), jax.vmap(rate)(times)
    return (jnp.concatenate((g[:, :1], q, g[:, 1:]), axis=1),
            jnp.concatenate((gt[:, :1], v, gt[:, 1:]), axis=1))


def alfven_energy(plan, displacement, velocity):
    """Quadrature energy for samples shaped (..., N); return shape (...,)."""
    strain = jnp.asarray(displacement)@plan.gradient.T
    speed = jnp.asarray(velocity)@plan.values.T
    return .5*(plan.density*speed**2+plan.magnetic_field**2/plan.permeability*strain**2)@plan.weights


def alfven_boundary_power(plan, displacement, velocity):
    """Physical net power [B0²/mu0 xi_t xi_z]_left^right, shape (...,).

    Uses independently evaluated full BSPF endpoint derivatives. This need not
    equal a finite-dimensional boundary reaction exactly; compare under mesh
    refinement. No separate magnetic condition is imposed at the endpoints.
    """
    strain = jnp.asarray(displacement)@plan.boundary_gradient.T
    speed = jnp.asarray(velocity)[..., jnp.array([0, plan.mass.shape[0]-1])]
    flux = plan.magnetic_field**2/plan.permeability*speed*strain
    return flux[..., 1]-flux[..., 0]
