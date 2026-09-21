"""Regularized unit-square lid-driven cavity using the existing BSPF basis.

u = psi_y, v = -psi_x; psi = a(t) in the clamped tensor basis + s(t)*f(x)*h(y).
The lift imposes u(x,1)=s(t)*16*x**2*(1-x)**2 and zero velocity elsewhere.
Pressure drops out of the divergence-free weak formulation. No body force,
filter, sponge, or artificial viscosity is used in the physical cavity.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np
from pybspf.time_integration import imex_midpoint

from bspf_models.fluids.stream_navier_stokes import StreamNavierStokes2DPlan
from bspf_models.fluids.stream_navier_stokes import plan_stream_navier_stokes2d
from bspf_models.fluids.stream_navier_stokes import stream_ns_load
from bspf_models.fluids.stream_navier_stokes import stream_ns_velocity
from bspf_models.fluids.stream_navier_stokes import stream_ns_vorticity


class CavityPlan(NamedTuple):
    spatial: StreamNavierStokes2DPlan
    lift_velocity: jax.Array
    lift_vorticity: jax.Array
    lift_mass: jax.Array
    lift_diffusion: jax.Array
    ramp_time: float


class CavityStepper(NamedTuple):
    dt: float
    cholesky: jax.Array


def cavity_lift(x, y):
    """Analytic psi, velocity, vorticity, Laplacian(velocity), on a tensor grid."""
    x, y = jnp.asarray(x)[:, None], jnp.asarray(y)[None, :]
    f = 16 * x**2 * (1 - x) ** 2
    fx = 32 * x - 96 * x**2 + 64 * x**3
    fxx = 32 - 192 * x + 192 * x**2
    fxxx = -192 + 384 * x
    h, hy, hyy = y**3 - y**2, 3 * y**2 - 2 * y, 6 * y - 2
    velocity = jnp.stack((f * hy, -fx * h), axis=-1)
    laplacian = jnp.stack((fxx * hy + 6 * f, -fxxx * h - fx * hyy), axis=-1)
    return f * h, velocity, -fxx * h - f * hyy, laplacian


def cavity_ramp(t, duration):
    """C2 quintic startup, with zero initial velocity and acceleration."""
    z = jnp.clip(t / duration, 0.0, 1.0)
    value = z**3 * (10 - 15 * z + 6 * z**2)
    rate = 30 * z**2 * (1 - z) ** 2 / duration
    return value, rate


def plan_cavity(*, n=41, reynolds=100.0, ramp_time=1.0, quadrature_order=None):
    """Unit length and peak lid speed; Re=1/nu, fixed walls on every side."""
    if not isinstance(n, int) or n <= 32:
        raise ValueError("n must be an integer >32")
    if not np.isfinite(reynolds) or reynolds <= 0:
        raise ValueError("reynolds must be finite and positive")
    if not np.isfinite(ramp_time) or ramp_time <= 0:
        raise ValueError("ramp_time must be finite and positive")
    nodes = np.linspace(0, 1, n)
    p = plan_stream_navier_stokes2d(
        nodes, nodes, viscosity=1 / reynolds, quadrature_order=quadrature_order
    )
    _, velocity, omega, laplacian = cavity_lift(p.x.points, p.y.points)
    return CavityPlan(
        p,
        velocity,
        omega,
        stream_ns_load(p, velocity),
        stream_ns_load(p, laplacian),
        ramp_time,
    )


def cavity_diffusion(p, a):
    """Positive weak viscous operator (the Hessian inner product)."""
    return (
        p.x.bending @ a
        + a @ p.y.bending.T
        + 2 * p.x.lam[:, None] * a * p.y.lam[None, :]
    )


def cavity_explicit_load(c, a, t):
    """Rotational convection plus viscous and inertial boundary lift terms."""
    p = c.spatial
    s, rate = cavity_ramp(t, c.ramp_time)
    velocity = stream_ns_velocity(p, a) + s * c.lift_velocity
    omega = stream_ns_vorticity(p, a) + s * c.lift_vorticity
    convection = stream_ns_load(
        p, jnp.stack((velocity[..., 1] * omega, -velocity[..., 0] * omega), axis=-1)
    )
    return convection + p.nu * s * c.lift_diffusion - rate * c.lift_mass


def cavity_rhs(c, a, t):
    p = c.spatial
    return (
        cavity_explicit_load(c, a, t) - p.nu * cavity_diffusion(p, a)
    ) / p.denominator


def plan_cavity_stepper(c, dt):
    """Factor M+dt*nu*D/2 once on the host for a fixed time step.

    Dense 2D Cholesky is deliberately limited to modest demonstration grids:
    storage O((n-4)^4). BSPF spatial assembly still uses the existing 1D factors.
    Convection remains explicit and imposes an advective time-step restriction.
    """
    import scipy.linalg as la

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    p = c.spatial
    nx, ny = p.denominator.shape
    d = (
        np.kron(np.asarray(p.x.bending), np.eye(ny))
        + np.kron(np.eye(nx), np.asarray(p.y.bending))
        + np.diag((2 * p.x.lam[:, None] * p.y.lam[None, :]).ravel())
    )
    matrix = np.diag(np.asarray(p.denominator).ravel()) + dt * float(p.nu) / 2 * d
    return CavityStepper(dt, jnp.asarray(la.cholesky(matrix, lower=True)))


def cavity_step(c, stepper, a, t, load=None):
    """Second-order IMEX midpoint with a backward-Euler half-step predictor.

    Optional load(t) is an integrated physical body force, used for verification.
    The full step uses Crank--Nicolson diffusion and midpoint convection/lifting.
    """
    p, dt = c.spatial, stepper.dt

    def explicit(state, time):
        value = cavity_explicit_load(c, state, time)
        return value if load is None else value + load(time)

    def solve(rhs):
        return jla.cho_solve((stepper.cholesky, True), rhs.ravel()).reshape(a.shape)

    return imex_midpoint(
        a,
        t,
        dt,
        lambda z: p.denominator * z,
        lambda z: p.nu * cavity_diffusion(p, z),
        explicit,
        solve,
    )


def cavity_fields(c, a, t, *, nodes=True):
    """Return full streamfunction, velocity, and vorticity including the lid lift."""
    p = c.spatial
    x, y = (p.x.x, p.y.x) if nodes else (p.x.points, p.y.points)
    bx, by = (p.x.bn, p.y.bn) if nodes else (p.x.b, p.y.b)
    psi, velocity, omega, _ = cavity_lift(x, y)
    s, _ = cavity_ramp(t, c.ramp_time)
    return (
        bx @ a @ by.T + s * psi,
        stream_ns_velocity(p, a, nodes=nodes) + s * velocity,
        stream_ns_vorticity(p, a, nodes=nodes) + s * omega,
    )
