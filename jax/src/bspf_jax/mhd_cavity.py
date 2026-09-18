"""2D incompressible resistive MHD in a conducting BSPF cavity (rho=mu=1).

u=curl(psi), B=curl(A), j=-Delta(A). A=0 on every wall, with free normal
derivative (tangential B). The weak current j_h=M_A^-1 K_A A is used in BOTH
Lorentz work and resistive dissipation, giving an exact semidiscrete exchange
identity. This is the mixed Galerkin current, not a second nodal derivative.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np

from .cavity import (
    CavityPlan,
    plan_cavity,
    cavity_explicit_load,
    cavity_diffusion,
    cavity_ramp,
)
from .stream_navier_stokes import (
    StreamLine,
    _stream_line,
    stream_ns_load,
    stream_ns_velocity,
)


class MHDCavityPlan(NamedTuple):
    fluid: CavityPlan
    magnetic: StreamLine
    magnetic_laplacian: jax.Array
    resistivity: float
    lid_speed: float
    time_offset: float


def plan_mhd_cavity(
    *,
    n=49,
    reynolds=100.0,
    magnetic_reynolds=200.0,
    lid_speed=1.0,
    time_offset=0.0,
    quadrature_order=None,
):
    """Rm uses unit length and reference speed 1 (not the initial Alfven speed)."""
    if not np.isfinite(magnetic_reynolds) or magnetic_reynolds <= 0:
        raise ValueError("magnetic_reynolds must be finite and positive")
    if (
        not np.isfinite(lid_speed)
        or lid_speed < 0
        or not np.isfinite(time_offset)
        or time_offset < 0
    ):
        raise ValueError("lid_speed and time_offset must be finite and nonnegative")
    fluid = plan_cavity(n=n, reynolds=reynolds, quadrature_order=quadrature_order)
    m = _stream_line(
        np.linspace(0, 1, n),
        quadrature_order=quadrature_order,
        clamped=False,
        dirichlet=True,
    )
    # The scaled lift and its loads retain the existing cavity RHS implementation.
    fluid = fluid._replace(
        lift_velocity=lid_speed * fluid.lift_velocity,
        lift_vorticity=lid_speed * fluid.lift_vorticity,
        lift_mass=lid_speed * fluid.lift_mass,
        lift_diffusion=lid_speed * fluid.lift_diffusion,
    )
    return MHDCavityPlan(
        fluid,
        m,
        m.lam[:, None] + m.lam[None, :],
        1 / magnetic_reynolds,
        lid_speed,
        time_offset,
    )


def mhd_velocity(p, a, t, *, nodes=False):
    c = p.fluid
    s, _ = cavity_ramp(t + p.time_offset, c.ramp_time)
    if nodes:
        from .cavity import cavity_lift

        lift = p.lid_speed * cavity_lift(c.spatial.x.x, c.spatial.y.x)[1]
    else:
        lift = c.lift_velocity
    return stream_ns_velocity(c.spatial, a, nodes=nodes) + s * lift


def magnetic_fields(p, b, *, nodes=False):
    m = p.magnetic
    basis, grad = (m.bn, m.gn) if nodes else (m.b, m.g)
    ax, ay = grad @ b @ basis.T, basis @ b @ grad.T
    flux = basis @ b @ basis.T
    current = basis @ (p.magnetic_laplacian * b) @ basis.T
    return flux, jnp.stack((ay, -ax), axis=-1), current


def mhd_exchange(p, a, b, t):
    """Return velocity Lorentz load and magnetic advection with paired work."""
    m = p.magnetic
    velocity = mhd_velocity(p, a, t)
    _, magnetic, current = magnetic_fields(p, b)
    grad_a = jnp.stack((-magnetic[..., 1], magnetic[..., 0]), axis=-1)
    lorentz = stream_ns_load(p.fluid.spatial, current[..., None] * grad_a)
    advected = jnp.sum(velocity * grad_a, axis=-1)
    weights = m.weights[:, None] * m.weights[None, :]
    induction = -m.b.T @ (weights * advected) @ m.b
    return lorentz, induction


def mhd_explicit(p, a, b, t):
    lorentz, induction = mhd_exchange(p, a, b, t)
    return cavity_explicit_load(p.fluid, a, t + p.time_offset) + lorentz, induction


def mhd_rhs(p, a, b, t):
    fa, fb = mhd_explicit(p, a, b, t)
    spatial = p.fluid.spatial
    return (
        (fa - spatial.nu * cavity_diffusion(spatial, a)) / spatial.denominator,
        fb - p.resistivity * p.magnetic_laplacian * b,
    )


def mhd_step(p, stepper, a, b, t):
    """Coupled second-order midpoint, implicit viscosity/resistivity.

    Explicit coupled terms require dt to resolve fluid AND Alfven propagation.
    """
    dt = stepper.dt
    spatial = p.fluid.spatial
    mass = spatial.denominator * a
    magnetic_factor = dt / 2 * p.resistivity * p.magnetic_laplacian

    def solve(rhs):
        return jla.cho_solve((stepper.cholesky, True), rhs.ravel()).reshape(a.shape)

    fa, fb = mhd_explicit(p, a, b, t)
    ah = solve(mass + dt / 2 * fa)
    bh = (b + dt / 2 * fb) / (1 + magnetic_factor)
    fa, fb = mhd_explicit(p, ah, bh, t + dt / 2)
    anew = solve(mass - dt / 2 * spatial.nu * cavity_diffusion(spatial, a) + dt * fa)
    bnew = ((1 - magnetic_factor) * b + dt * fb) / (1 + magnetic_factor)
    return anew, bnew


def island_pair(p, *, peak_field=6.0, separation=0.26, width_x=0.10, width_y=0.14):
    """Two initially non-equilibrium same-sign flux islands, centered in the box.

    Not the periodic Fadeev equilibrium benchmark. A broad polynomial envelope
    enforces exact zero flux at all walls. Normalize the *projected* field on
    quadrature points to peak_field (Alfven speed units).
    """
    if (
        not all(
            np.isfinite(v) and v > 0 for v in (peak_field, separation, width_x, width_y)
        )
        or separation >= 1
    ):
        raise ValueError("Require positive finite field/widths and 0<separation<1")
    m = p.magnetic
    x, y = m.points[:, None], m.points[None, :]
    flux = (
        256
        * x**2
        * (1 - x) ** 2
        * y**2
        * (1 - y) ** 2
        * sum(
            jnp.exp(-(((x - center) / width_x) ** 2) - ((y - 0.5) / width_y) ** 2)
            for center in (0.5 - separation / 2, 0.5 + separation / 2)
        )
    )
    weights = m.weights[:, None] * m.weights[None, :]
    b = m.b.T @ (weights * flux) @ m.b
    maximum = jnp.max(jnp.linalg.norm(magnetic_fields(p, b)[1], axis=-1))
    return b * peak_field / maximum


def mhd_budget(p, a, b, t):
    """Independent physical wall work, dissipation and modal energy derivative.

    The residual tests spatial boundary lifting/weak consistency. Temporal
    integration error must also be checked with accumulated wall work/losses.
    """
    c, m = p.fluid, p.magnetic
    f = c.spatial
    w = m.weights[:, None] * m.weights[None, :]
    velocity = mhd_velocity(p, a, t)
    _, magnetic, current = magnetic_fields(p, b)
    s, rate = cavity_ramp(t + p.time_offset, c.ramp_time)
    x, y = f.x.points[:, None], f.y.points[None, :]
    fx = 32 * x - 96 * x**2 + 64 * x**3
    fxx = 32 - 192 * x + 192 * x**2
    ff = 16 * x**2 * (1 - x) ** 2
    h, hy, hyy = y**3 - y**2, 3 * y**2 - 2 * y, 6 * y - 2
    xx = f.x.h @ a @ f.y.b.T + s * p.lid_speed * fxx * h
    xy = f.x.g @ a @ f.y.g.T + s * p.lid_speed * fx * hy
    yy = f.x.b @ a @ f.y.h.T + s * p.lid_speed * ff * hyy
    viscous = f.nu * jnp.sum(w * (xx**2 + 2 * xy**2 + yy**2))
    ohmic = p.resistivity * jnp.sum(w * current**2)
    lid = s * p.lid_speed * ff[:, 0]
    wall_uy = f.x.b @ a @ f.y.hn[-1] + 4 * lid
    power = f.nu * jnp.sum(f.x.weights * lid * wall_uy)
    lorentz, induction = mhd_exchange(p, a, b, t)
    transfer = -jnp.sum(p.magnetic_laplacian * b * induction)
    da, db = mhd_rhs(p, a, b, t)
    du = stream_ns_velocity(f, da) + rate * c.lift_velocity
    derivative = jnp.sum(w * jnp.sum(velocity * du, axis=-1)) + jnp.sum(
        p.magnetic_laplacian * b * db
    )
    return jnp.array(
        [
            0.5 * jnp.sum(w * jnp.sum(velocity**2, axis=-1)),
            0.5 * jnp.sum(p.magnetic_laplacian * b * b),
            power,
            viscous,
            ohmic,
            transfer,
            derivative - power + viscous + ohmic,
        ]
    )
