"""Self-consistent electrons with Maxwellian reservoirs and grounded potentials."""
from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from pybspf.calculus import antiderivative
from pybspf.calculus import integrate
from pybspf.galerkin import _quadrature_trial
from bspf_models.kinetic.parallel_kinetic import ParallelKineticPlan
from bspf_models.kinetic.parallel_kinetic import plan_parallel_kinetic
from bspf_models.kinetic.parallel_kinetic import _transport_axis
from pybspf.plans import Plan1D
from pybspf.time_integration import integrate_rk4


def poisson_dirichlet(plan, source, *, left=0., right=0.):
    """Solve phi''=source by BSPF integration; return (phi, E=-phi').

    Source samples have shape (nz, ...); boundary potentials broadcast over
    trailing axes. Integrate twice and add the affine function enforcing both
    potentials. The electric field uses the first primitive directly, without
    differentiating the fitted potential or solving a dense Poisson matrix.
    Supports JIT. A clean 1D plan is required.
    """
    if not isinstance(plan, Plan1D) or plan.noise is not None:
        raise ValueError('Poisson integration requires a clean Plan1D')
    source = jnp.asarray(source)
    first = antiderivative(plan, source)
    second = antiderivative(plan, source, order=2)
    length = plan.x[-1]-plan.x[0]
    slope = (jnp.asarray(right)-jnp.asarray(left)-second[-1])/length
    distance = (plan.x-plan.x[0]).reshape((-1,)+(1,)*(source.ndim-1))
    return second+jnp.asarray(left)+distance*slope, -first-slope


@partial(jax.tree_util.register_dataclass, data_fields=[
    'transport', 'z_values', 'v_transport', 'v_lift', 'velocity_weights',
    'background', 'potential_matrix', 'electric_matrix'], meta_fields=[])
@dataclass(frozen=True)
class VlasovPoissonPlan:
    transport: ParallelKineticPlan
    z_values: jax.Array
    v_transport: jax.Array
    v_lift: jax.Array
    velocity_weights: jax.Array
    background: jax.Array
    potential_matrix: jax.Array
    electric_matrix: jax.Array


def plan_vlasov_poisson(z_plan, v_plan, *, temperature=1., quadrature_order=8):
    """Plan nonlinear f_t+v f_z-E f_v=0, phi''=int(f)dv-1 on a finite box.

    Ions are fixed at density one. Both potentials are zero; incoming electrons
    have a fixed Maxwellian of the given temperature. Its discrete velocity
    integral is normalized to one on the finite velocity interval so the
    background is an exact numerical equilibrium. No periodic identification,
    collisions, filtering or positivity limiter. Positive temperature is static.
    Factory setup is outside JIT; dense tensor operators suit small 1z1v tests.
    """
    t = np.asarray(temperature)
    if t.ndim or np.iscomplexobj(t) or not np.isfinite(t) or t <= 0:
        raise ValueError('temperature must be a finite positive scalar')
    transport = plan_parallel_kinetic(z_plan, v_plan, quadrature_order=quadrature_order)
    _, (qz,) = _quadrature_trial(z_plan, jnp.eye(z_plan.x.size), (0,), quadrature_order)
    _, _, sv, lv, _ = _transport_axis(v_plan, quadrature_order)
    weights = integrate(v_plan, jnp.eye(v_plan.x.size))
    background = jnp.exp(-v_plan.x**2/(2*temperature))
    background = background/(weights@background)
    potential, electric = poisson_dirichlet(z_plan, jnp.eye(z_plan.x.size))
    return VlasovPoissonPlan(transport, qz, sv, lv, weights, background, potential, electric)


def vlasov_poisson_fields(plan, distribution):
    """Return grounded (potential, electric field) for (..., nz, nv) samples."""
    delta_density = (jnp.asarray(distribution)-plan.background)@plan.velocity_weights
    return delta_density@plan.potential_matrix.T, delta_density@plan.electric_matrix.T


def integrate_vlasov_poisson(plan, initial, times, *, substeps=1):
    """Return (f, phi, E) histories for nonlinear open-boundary Vlasov–Poisson.

    Incoming spatial traces are the fixed Maxwellian; outgoing traces are free.
    At each z the sign of -E selects the incoming velocity face, also prescribed
    by that Maxwellian. Boundary fluxes are weak upwind fluxes. Grounded Poisson
    is recomputed at EVERY RK stage from the evolving density, using precomputed
    BSPF primitive maps. The nonlinear electric force is projected at resolved
    spatial quadrature points. Internally evolve f-fM to preserve equilibrium.

    Caller supplies compatible real initial data, finite increasing output times
    and stable explicit RK4 substeps. No automatic CFL control. The scheme is not
    positivity preserving or exactly energy conserving: check phase-space flux,
    velocity resolution, numerical recurrence and time/space refinement. The
    equilibrium, reservoir and grounded-potential choices are fixed by the plan.
    Supports outer JIT. Returns f(nt,nz,nv), phi(nt,nz), E(nt,nz).
    """
    p = plan.transport
    initial = jnp.asarray(initial)
    if initial.shape != (p.z.size, p.v.size) or jnp.iscomplexobj(initial):
        raise ValueError('initial must be a real (nz,nv) array')
    delta = initial.astype(p.z.dtype)-plan.background

    def rhs(t, h):
        field = plan.electric_matrix@(h@plan.velocity_weights)
        acceleration = -(plan.z_values@field)
        full = h+plan.background
        result = (p.z_transport@h@p.velocity.T
            +jnp.outer(p.z_lift[:, 0], h[0]@p.negative.T)
            -jnp.outer(p.z_lift[:, 1], h[-1]@p.positive.T))
        # Galerkin multiplication by a(z), rather than pointwise multiplying
        # a dense-mass residual, keeps the tensor weak form consistent.
        result += p.z_projection@(acceleration[:, None]*(plan.z_values@(full@plan.v_transport.T)))
        positive, negative = jnp.maximum(acceleration, 0.), jnp.minimum(acceleration, 0.)
        right = positive*(plan.z_values@full[:, -1])+negative*plan.background[-1]
        left = negative*(plan.z_values@full[:, 0])+positive*plan.background[0]
        result += jnp.outer(p.z_projection@left, plan.v_lift[:, 0])
        result -= jnp.outer(p.z_projection@right, plan.v_lift[:, 1])
        return result

    history = integrate_rk4(rhs, delta, times, substeps=substeps)
    distribution = history+plan.background
    density = history@plan.velocity_weights
    return distribution, density@plan.potential_matrix.T, density@plan.electric_matrix.T
