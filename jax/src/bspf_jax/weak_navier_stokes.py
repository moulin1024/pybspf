"""Same-space BSPF weak momentum and mass-orthogonal direct 2D projection.

Only one-dimensional matrices are assembled. Runtime is pure JAX float64;
SciPy and MPFR are optional host-setup dependencies (install bspf-jax[weak-ns]).
The pressure constraint uses the full BSPF weak test space, with the
matching mass-adjoint gradient. Its Schur complement is diagonalized directly. Fixed boundary
values are supplied through a divergence-free, time-independent lift.
"""

from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .rectangle_poisson import tensor_elliptic_solve


class WeakNSLine(NamedTuple):
    x: jax.Array
    quadrature_x: jax.Array
    weights: jax.Array
    values: jax.Array
    gradients: jax.Array
    derivative: jax.Array
    mass_rows: jax.Array
    stiffness_rows: jax.Array
    inverse_mass: jax.Array
    root: jax.Array
    inverse_root: jax.Array
    pressure_mass: jax.Array
    pressure_derivative: jax.Array
    weak_derivative: jax.Array
    pressure_vectors: jax.Array
    pressure_eigenvalues: jax.Array
    diffusion_vectors: jax.Array
    diffusion_eigenvalues: jax.Array


class WeakNavierStokes2DPlan(NamedTuple):
    x: WeakNSLine
    y: WeakNSLine
    viscosity: jax.Array
    inverse_pressure_sum: jax.Array


def _line(
    coordinates, *, q, degree, n_basis, window, modes, assembly_bits, quadrature_order
):
    import scipy.linalg as la
    from scipy.interpolate import BSpline
    from scipy.special import roots_legendre

    from .pressure import _make_line
    from ._weak_basis import mp_trial_values

    x = np.asarray(coordinates, dtype=float)
    if (
        x.ndim != 1
        or x.size <= n_basis
        or not np.all(np.isfinite(x))
        or x[-1] <= x[0]
        or not np.allclose(
            np.diff(x), (x[-1] - x[0]) / (x.size - 1), rtol=1e-10, atol=1e-14
        )
    ):
        raise ValueError("Require a finite increasing uniform grid with n > n_basis")
    if not (
        1 <= q <= degree + 1 <= n_basis
        and 2 * q < n_basis
        and q <= modes <= window <= x.size
    ):
        raise ValueError("Incompatible spline or endpoint parameters")
    original = _make_line(x, q, n_basis, degree, window, "chebyshev", modes, 1e-12)
    line = SimpleNamespace(x=x, P=np.asarray(original.projector))
    breaks = np.linspace(x[0], x[-1], n_basis - degree + 1)
    knots = np.r_[
        np.repeat(x[0], degree + 1), breaks[1:-1], np.repeat(x[-1], degree + 1)
    ]
    spline = BSpline(knots, np.eye(n_basis), degree)
    if quadrature_order is None:
        # Resolve triple products used by nonlinear convection, not only mass.
        quadrature_order = max(
            24, int(np.ceil(1.5 * np.pi * (x.size - 1) / (len(breaks) - 1))) + 8
        )
    if not isinstance(quadrature_order, (int, np.integer)) or quadrature_order < 2:
        raise ValueError("quadrature_order must be an integer >= 2")
    z, w = roots_legendre(quadrature_order)
    points = np.concatenate(
        [(a + b) / 2 + (b - a) / 2 * z for a, b in zip(breaks[:-1], breaks[1:])]
    )
    weights = np.concatenate([(b - a) / 2 * w for a, b in zip(breaks[:-1], breaks[1:])])
    b, g = mp_trial_values(line, spline, points, bits=assembly_bits)
    cardinal, derivative = mp_trial_values(line, spline, x, bits=assembly_bits)
    if np.max(abs(cardinal - np.eye(x.size))) > 1e-7:
        raise ValueError("BSPF cardinal reconstruction lost accuracy")
    mass_rows = b[:, 1:-1].T @ (weights[:, None] * b)
    stiffness_rows = g[:, 1:-1].T @ (weights[:, None] * g)
    mass = mass_rows[:, 1:-1]
    stiffness = stiffness_rows[:, 1:-1]
    root = la.cholesky(mass, lower=False)
    inverse_root = la.solve_triangular(root, np.eye(x.size - 2))
    inverse_mass = inverse_root @ inverse_root.T

    # Full pressure test space, interior (zero-wall) velocity trial space.
    # S = <pressure, velocity>, C = <pressure, d(velocity)>.
    # Schur = Ax (x) By + Bx (x) Ay, A=C M^-1 C.T, B=S M^-1 S.T.
    # B is singular: diagonalize A against A+B, never invert B.
    full_mass = b.T @ (weights[:, None] * b)
    pressure_mass = full_mass[:, 1:-1] @ inverse_root
    pressure_derivative = (b.T @ (weights[:, None] * g[:, 1:-1])) @ inverse_root
    aa = pressure_derivative @ pressure_derivative.T
    bb = pressure_mass @ pressure_mass.T
    pressure_values, vectors = la.eigh(aa, aa + bb)
    if (
        np.max(abs(pressure_values[:2])) > 1e-8
        or np.max(abs(pressure_values[-2:] - 1)) > 1e-8
        or pressure_values[2] <= 1e-10
        or pressure_values[-3] >= 1 - 1e-10
    ):
        raise ValueError("Unexpected weak pressure pencil null space")
    pressure_values[:2], pressure_values[-2:] = 0.0, 1.0
    weak_derivative = la.cho_solve(
        la.cho_factor(full_mass), b.T @ (weights[:, None] * g)
    )
    diffusion_values, diffusion_vectors = la.eigh(stiffness, mass)
    return WeakNSLine(
        *map(
            jnp.asarray,
            (
                x,
                points,
                weights,
                b,
                g,
                derivative,
                mass_rows,
                stiffness_rows,
                inverse_mass,
                root,
                inverse_root,
                pressure_mass,
                pressure_derivative,
                weak_derivative,
                vectors,
                pressure_values,
                diffusion_vectors,
                diffusion_values,
            ),
        )
    )


def plan_weak_navier_stokes2d(
    x,
    y,
    *,
    viscosity=0.002,
    q=9,
    degree=13,
    n_basis=32,
    window=16,
    modes=12,
    assembly_bits=113,
    quadrature_order=None,
):
    """Host setup of a tensor-direct weak BSPF NS plan; no 2D matrix assembly.

    Requires JAX x64, SciPy and gmpy2. All runtime arrays are float64. Each axis
    uses the original QR BSPF reconstruction; only its setup is evaluated in
    extended precision to avoid spline/Fourier cancellation. Quadrature is
    overintegrated for nonlinear products. No sponge or extra viscosity.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 before building a weak NS plan")
    if not np.isfinite(viscosity) or viscosity < 0:
        raise ValueError("viscosity must be finite and nonnegative")
    if not isinstance(assembly_bits, int) or assembly_bits < 80:
        raise ValueError("assembly_bits must be an integer >= 80")
    options = dict(
        q=q,
        degree=degree,
        n_basis=n_basis,
        window=window,
        modes=modes,
        assembly_bits=assembly_bits,
        quadrature_order=quadrature_order,
    )
    lx, ly = _line(x, **options), _line(y, **options)
    ex = np.asarray(lx.pressure_eigenvalues)[:, None]
    ey = np.asarray(ly.pressure_eigenvalues)[None, :]
    sums = ex * (1 - ey) + (1 - ex) * ey
    inverse = np.zeros_like(sums)
    np.divide(1.0, sums, out=inverse, where=sums > 0)
    return WeakNavierStokes2DPlan(lx, ly, jnp.asarray(viscosity), jnp.asarray(inverse))


def _tensor(left, value, right):
    """left @ value @ right.T, with an optional component dimension."""
    return jnp.einsum("ai,ij...,bj->ab...", left, value, right, optimize=True)


def weak_ns_divergence(plan, velocity):
    """L2-projected divergence in the full BSPF pressure space (nodal values).

    This is a weak divergence, not the pointwise derivative. It includes
    contributions from the fixed nonzero boundary lift.
    """
    return (
        plan.x.weak_derivative @ velocity[..., 0]
        + velocity[..., 1] @ plan.y.weak_derivative.T
    )


def weak_ns_pointwise_divergence(plan, velocity):
    """Unprojected nodal divergence, an independent accuracy diagnostic."""
    return plan.x.derivative @ velocity[..., 0] + velocity[..., 1] @ plan.y.derivative.T


def weak_ns_vorticity(plan, velocity):
    return plan.x.derivative @ velocity[..., 1] - velocity[..., 0] @ plan.y.derivative.T


def weak_ns_project(plan, raw):
    """Mass-orthogonal weak-divergence projection, one direct Schur solve.

    Eight exact pressure gauge modes are pseudoinverted. The full pressure
    test space is used, including endpoint basis functions. Velocity increments
    are exactly zero on all walls. No global matrix or refinement is used.
    """
    if raw.shape != (plan.x.x.size, plan.y.x.size, 2):
        raise ValueError("raw must have shape (nx, ny, 2)")
    x, y = plan.x, plan.y
    w = _tensor(x.root, raw[1:-1, 1:-1], y.root)
    cx, cy = x.pressure_derivative, y.pressure_derivative
    sx, sy = x.pressure_mass, y.pressure_mass
    rhs = cx @ w[..., 0] @ sy.T + sx @ w[..., 1] @ cy.T
    modal = (
        x.pressure_vectors.T @ rhs @ y.pressure_vectors
    ) * plan.inverse_pressure_sum
    pressure = x.pressure_vectors @ modal @ y.pressure_vectors.T
    fx = w[..., 0] - cx.T @ pressure @ sy
    fy = w[..., 1] - sx.T @ pressure @ cy
    interior = _tensor(x.inverse_root, jnp.stack((fx, fy), axis=-1), y.inverse_root)
    return jnp.zeros_like(raw).at[1:-1, 1:-1].set(interior)


def weak_ns_load(plan, quadrature_force):
    """Integrate a physical force sampled at the tensor quadrature points."""
    x, y = plan.x, plan.y
    return _tensor(
        x.values[:, 1:-1].T * x.weights,
        quadrature_force,
        y.values[:, 1:-1].T * y.weights,
    )


def weak_ns_momentum_load(plan, velocity):
    """Skew Galerkin convection and weak diffusion; interior test functions.

    The identical quadrature/transpose pair makes the convective contribution
    energy-neutral for zero boundary velocity, even before dealiasing or
    projection. Inhomogeneous fixed boundaries exchange physical energy.
    """
    x, y = plan.x, plan.y
    u = _tensor(x.values, velocity, y.values)
    ux = _tensor(x.gradients, velocity, y.values)
    uy = _tensor(x.values, velocity, y.gradients)
    convection = -0.5 * weak_ns_load(plan, u[..., 0, None] * ux + u[..., 1, None] * uy)
    convection += 0.5 * _tensor(
        x.gradients[:, 1:-1].T * x.weights,
        u[..., 0, None] * u,
        y.values[:, 1:-1].T * y.weights,
    )
    convection += 0.5 * _tensor(
        x.values[:, 1:-1].T * x.weights,
        u[..., 1, None] * u,
        y.gradients[:, 1:-1].T * y.weights,
    )
    diffusion = _tensor(x.stiffness_rows, velocity, y.mass_rows) + _tensor(
        x.mass_rows, velocity, y.stiffness_rows
    )
    return convection - plan.viscosity * diffusion


def weak_ns_rhs(plan, velocity, force_load=None):
    """Acceleration with fixed wall values; force_load is an integrated load."""
    load = weak_ns_momentum_load(plan, velocity)
    if force_load is not None:
        load = load + force_load
    interior = _tensor(plan.x.inverse_mass, load, plan.y.inverse_mass)
    raw = jnp.zeros_like(velocity).at[1:-1, 1:-1].set(interior)
    return weak_ns_project(plan, raw)


def weak_ns_rk4_step(plan, velocity, dt, force_load=None):
    """RK4 of the unsplit projected weak system; diffusion is explicit here.

    Pressure is a direct projection at every stage. CFL/viscous step limits
    still apply. No pressure-correction time splitting error is introduced.
    """

    def rhs(u):
        return weak_ns_rhs(plan, u, force_load)

    a = rhs(velocity)
    b = rhs(velocity + dt / 2 * a)
    c = rhs(velocity + dt / 2 * b)
    d = rhs(velocity + dt * c)
    return velocity + dt / 6 * (a + 2 * b + 2 * c + d)


def weak_ns_energy(plan, velocity):
    """Quadrature kinetic energy, including nonzero boundary lifts."""
    value = _tensor(plan.x.values, velocity, plan.y.values)
    weights = plan.x.weights[:, None] * plan.y.weights[None, :]
    return 0.5 * jnp.sum(weights[..., None] * value * value)


def weak_ns_helmholtz(plan, load, *, alpha=1.0, beta=1.0):
    """Direct scalar (alpha M + beta K) solve with homogeneous Dirichlet data.

    This is a standalone diffusion/Helmholtz primitive, NOT a simultaneous
    incompressible Stokes solve. The NS RK4 integrator does not split with it.
    load has shape (nx-2, ny-2), and is an integrated weak load.
    """
    x, y = plan.x, plan.y
    denominator = alpha + beta * (
        x.diffusion_eigenvalues[:, None] + y.diffusion_eigenvalues[None, :]
    )
    return tensor_elliptic_solve(
        load, denominator, x.diffusion_vectors, y.diffusion_vectors
    )


def weak_kh_initial_velocity(
    plan, *, thickness=0.12, perturbation=0.03, wavelength=1.5
):
    """Same nonperiodic KH lift/seed as the original strong solver."""
    if (
        thickness <= 0
        or wavelength <= 0
        or not np.all(np.isfinite([thickness, perturbation, wavelength]))
    ):
        raise ValueError("Require finite parameters and positive thickness/wavelength")
    x, y = plan.x.x, plan.y.x
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    xc, yc = (x[0] + x[-1]) / 2, (y[0] + y[-1]) / 2
    sx, sy = 2 * (X - xc) / (x[-1] - x[0]), 2 * (Y - yc) / (y[-1] - y[0])
    psi = (
        perturbation
        * thickness
        * (1 - sx * sx) ** 4
        * (1 - sy * sy) ** 4
        * jnp.exp(-(((Y - yc) / (2 * thickness)) ** 2))
        * jnp.cos(2 * jnp.pi * (X - xc) / wavelength)
    )
    seed = jnp.stack((psi @ plan.y.derivative.T, -plan.x.derivative @ psi), axis=-1)
    base = jnp.stack((jnp.tanh((Y - yc) / thickness), jnp.zeros_like(Y)), axis=-1)
    return base + weak_ns_project(plan, seed), base
