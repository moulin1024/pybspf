"""Incompressible 2D NS with fixed Dirichlet velocity and BSPF projection."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from pybspf.time_integration import rk4_stages

from pybspf.basis import basis_matrix
from pybspf.basis import open_knots
from bspf_models._numerics._energy_stable import EnergyClosure2D
from bspf_models._numerics._energy_stable import energy_closure2d
from bspf_models._numerics._energy_stable import energy_divergence
from bspf_models._numerics._energy_stable import energy_project
from bspf_models._numerics._energy_stable import sbp84_diffusion
from bspf_models.elliptic.pressure import PressurePoisson2DPlan
from bspf_models.elliptic.pressure import _differentiate
from bspf_models._numerics.trial_spaces import _fourier
from bspf_models.elliptic.pressure import project_pressure2d
from bspf_models.elliptic.pressure import pressure_divergence


class NavierStokes2DPlan(NamedTuple):
    pressure: PressurePoisson2DPlan
    dx: jax.Array
    dy: jax.Array
    dxx: jax.Array
    dyy: jax.Array
    viscosity: jax.Array
    energy: EnergyClosure2D | None = None


class NSStageDiagnostics(NamedTuple):
    converged: jax.Array
    schur_linf: jax.Array
    divergence_linf: jax.Array


def plan_navier_stokes2d(pressure_plan, *, degree=13, viscosity=0.002, closure="bspf"):
    """Build fixed-viscosity NS matrices; degree must match the pressure plan.

    Reuses its exact D1 and QR spline coefficient map. D2 is independently
    constructed as F2+(B2-F2 B)P, as in the source NS benchmark. No global
    2D matrix is formed. All arrays use (nx, ny, component).

    closure="sbp84" selects energy-compatible finite differences (interior
    order 8, boundary order 4), split advection, compatible dissipative diffusion, and an
    H-orthogonal tensor-direct projection with zero refinement. It uses the
    supplied pressure plan's grid only; use ns_divergence/ns_project_velocity
    for this option, not the original BSPF pressure operators. This option
    changes the spatial discretization and does not retain spectral accuracy.
    """
    if not np.isfinite(viscosity) or viscosity <= 0:
        raise ValueError("viscosity must be finite and positive")
    if closure not in ("bspf", "sbp84"):
        raise ValueError("closure must be 'bspf' or 'sbp84'")
    if closure == "sbp84":
        energy = energy_closure2d(pressure_plan.x.x, pressure_plan.y.x)
        dx, dy = energy.x.derivative, energy.y.derivative
        return NavierStokes2DPlan(
            pressure_plan,
            dx,
            dy,
            sbp84_diffusion(dx, energy.x.weights),
            sbp84_diffusion(dy, energy.y.weights),
            jnp.asarray(viscosity),
            energy,
        )
    operators = []
    for line in (pressure_plan.x, pressure_plan.y):
        n = line.x.size
        nb = line.projector.shape[0]
        if not isinstance(degree, int) or not 2 <= degree < nb:
            raise ValueError("Require 2 <= degree < number of splines")
        knots = open_knots(line.x[0], line.x[-1], degree=degree, n_basis=nb)
        B = basis_matrix(knots, line.x, degree=degree)
        B1 = basis_matrix(knots, line.x, degree=degree, derivative=1)
        if not bool(
            jnp.allclose(
                B1 - _fourier(B, line.multiplier), line.low, rtol=1e-10, atol=1e-10
            )
        ):
            raise ValueError("degree must match pressure-plan spline degree")
        B2 = basis_matrix(knots, line.x, degree=degree, derivative=2)
        mult2 = -((2 * jnp.pi * jnp.fft.fftfreq(n - 1, d=line.x[1] - line.x[0])) ** 2)
        d2 = _fourier(jnp.eye(n), mult2) + (B2 - _fourier(B, mult2)) @ line.projector
        operators.append((_differentiate(line, jnp.eye(n), 0), d2))
    return NavierStokes2DPlan(
        pressure_plan,
        operators[0][0],
        operators[1][0],
        operators[0][1],
        operators[1][1],
        jnp.asarray(viscosity),
    )


def ns_derivatives(plan, velocity):
    return (
        jnp.einsum("ij,jkc->ikc", plan.dx, velocity),
        jnp.einsum("ij,kjc->kic", plan.dy, velocity),
    )


def ns_vorticity(plan, velocity):
    """Out-of-plane vorticity dv/dx - du/dy."""
    dx, dy = ns_derivatives(plan, velocity)
    return dx[..., 1] - dy[..., 0]


def ns_raw_rhs(plan, velocity):
    """Advection plus viscosity using the selected closure, before projection."""
    dx, dy = ns_derivatives(plan, velocity)
    diffusion = jnp.einsum("ij,jkc->ikc", plan.dxx, velocity) + jnp.einsum(
        "ij,kjc->kic", plan.dyy, velocity
    )
    if plan.energy is not None:
        # Skew split is energy conservative even with a small divergence error.
        flux_x = velocity[..., 0, None] * velocity
        flux_y = velocity[..., 1, None] * velocity
        conservative = jnp.einsum("ij,jkc->ikc", plan.dx, flux_x) + jnp.einsum(
            "ij,kjc->kic", plan.dy, flux_y
        )
        return (
            -0.5
            * (velocity[..., 0, None] * dx + velocity[..., 1, None] * dy + conservative)
            + plan.viscosity * diffusion
        )
    return (
        -velocity[..., 0, None] * dx
        - velocity[..., 1, None] * dy
        + plan.viscosity * diffusion
    )


def ns_rhs(plan, velocity, force=None, *, reference=None, sponge=None):
    """Projected NS acceleration, with homogeneous boundary increments.

    Fixed nonzero boundary velocities are permitted when the initial field is
    divergence-free and net boundary flux is compatible. No boundary forcing
    is implicitly added. Optional explicit sponge damps toward reference;
    callers must supply both as finite arrays with nonnegative sponge rates.
    No subgrid model or filtering is used.
    """
    if velocity.shape != plan.pressure.mask.shape + (2,) or jnp.iscomplexobj(velocity):
        raise ValueError("velocity must be a real (nx, ny, 2) array")
    raw = ns_raw_rhs(plan, velocity)
    if force is not None:
        if force.shape != velocity.shape:
            raise ValueError("force must match velocity shape")
        raw = raw + force
    if (reference is None) != (sponge is None):
        raise ValueError("Provide both reference and sponge or neither")
    if sponge is not None:
        if reference.shape != velocity.shape or sponge.shape != velocity.shape[:2]:
            raise ValueError("reference/sponge must match vector/scalar grid shapes")
        raw -= sponge[..., None] * (velocity - reference)
    return ns_project_velocity(plan, raw)


def ns_divergence(plan, velocity):
    """Full-grid divergence in the selected NS spatial discretization."""
    if plan.energy is not None:
        return energy_divergence(plan.energy, velocity)
    return pressure_divergence(plan.pressure, velocity)


def ns_project_velocity(plan, raw):
    """Project a field onto divergence-free, homogeneous wall increments.

    With sbp84 this is H-orthogonal and direct, with no iterative refinement.
    Nonzero fixed boundary velocities must be supplied separately as a lift.
    """
    if raw.shape != plan.pressure.mask.shape + (2,) or jnp.iscomplexobj(raw):
        raise ValueError("raw must be a real (nx, ny, 2) array")
    if plan.energy is not None:
        acceleration = energy_project(plan.energy, raw)
        residual = ns_divergence(plan, acceleration)
        norm = jnp.linalg.norm(residual)
        rhs_norm = jnp.linalg.norm(
            ns_divergence(plan, plan.pressure.mask[..., None] * raw)
        )
        maximum = jnp.max(abs(residual))
        valid = (
            jnp.all(jnp.isfinite(raw))
            & jnp.all(jnp.isfinite(acceleration))
            & (norm <= 1e-9 + 1e-10 * rhs_norm)
        )
        return acceleration, NSStageDiagnostics(valid, maximum, maximum)
    acceleration, result = project_pressure2d(plan.pressure, raw, completion=False)
    return acceleration, NSStageDiagnostics(
        result.converged,
        result.schur_residual_linf,
        jnp.max(abs(pressure_divergence(plan.pressure, acceleration))),
    )


def ns_rk4_step(plan, velocity, dt, force=None, *, reference=None, sponge=None):
    """One RK4 step with optional fixed forcing and explicit boundary sponge."""

    def rhs(u):
        return ns_rhs(plan, u, force, reference=reference, sponge=sponge)

    updated, (da, db, dc, dd) = rk4_stages(velocity, dt, rhs)
    diagnostics = NSStageDiagnostics(
        da.converged & db.converged & dc.converged & dd.converged,
        jnp.max(jnp.stack([x.schur_linf for x in (da, db, dc, dd)])),
        jnp.max(jnp.stack([x.divergence_linf for x in (da, db, dc, dd)])),
    )
    return updated, diagnostics


def kh_initial_velocity(plan, *, thickness=0.12, perturbation=0.015, wavelength=1.5):
    """Finite-domain tanh shear with a compactly tapered streamfunction seed.

    Four boundaries retain (tanh(y/thickness), 0); x boundaries exchange equal
    opposing streams and y boundaries move tangentially. This is not a periodic
    benchmark. Project the seed once; the base has x-independent horizontal flow.
    """
    for name, value in (("thickness", thickness), ("wavelength", wavelength)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(perturbation):
        raise ValueError("perturbation must be finite")
    x, y = plan.pressure.x.x, plan.pressure.y.x
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    xc, yc = (x[0] + x[-1]) / 2, (y[0] + y[-1]) / 2
    sx, sy = 2 * (X - xc) / (x[-1] - x[0]), 2 * (Y - yc) / (y[-1] - y[0])
    envelope = (1 - sx * sx) ** 4 * (1 - sy * sy) ** 4
    psi = (
        perturbation
        * thickness
        * envelope
        * jnp.exp(-(((Y - yc) / (2 * thickness)) ** 2))
        * jnp.cos(2 * jnp.pi * (X - xc) / wavelength)
    )
    seed = jnp.stack([psi @ plan.dy.T, -plan.dx @ psi], axis=-1)
    if plan.energy is None:
        seed, diagnostic = project_pressure2d(plan.pressure, seed, completion=False)
    else:
        seed, diagnostic = ns_project_velocity(plan, seed)
    base = jnp.stack([jnp.tanh((Y - yc) / thickness), jnp.zeros_like(Y)], axis=-1)
    return base + seed, base, diagnostic
