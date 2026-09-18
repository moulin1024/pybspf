"""Axisymmetric BSPF Grad--Shafranov equilibrium with fixed filament coils.

mu0=1. A distant rectangular perfectly conducting computational wall fixes the
total boundary flux. The plasma boundary is the unknown psi=0 core contour.
There is no plasma current in the surrounding vacuum. Toroidal F=R*Bphi is
constant; p=alpha*max(psi,0)^3/3 inside the connected core only.
"""

from dataclasses import dataclass
import numpy as np
import scipy.linalg as la
from scipy.special import ellipk, ellipe
from scipy.ndimage import label
from .stream_navier_stokes import _stream_line
from ._flow_kernels import tensor_elliptic_solve


def coil_field(r, z, coil_r, coil_z, current=1.0):
    """Flux psi=R*Aphi, Br, Bz of a circular filament (mu0=1)."""
    r, z = np.broadcast_arrays(np.asarray(r, dtype=float), np.asarray(z, dtype=float))
    if np.any(r <= 0) or coil_r <= 0:
        raise ValueError("All major radii must be positive")
    dz = z - coil_z
    q = (r + coil_r) ** 2 + dz**2
    d = (r - coil_r) ** 2 + dz**2
    if np.any(d == 0):
        raise ValueError("Cannot evaluate on an ideal filament")
    m = 4 * r * coil_r / q
    k, e = ellipk(m), ellipe(m)
    psi = (
        current * np.sqrt(r * coil_r) / (2 * np.pi * np.sqrt(m)) * ((2 - m) * k - 2 * e)
    )
    br = (
        current
        * dz
        / (2 * np.pi * r * np.sqrt(q))
        * (-k + (coil_r**2 + r * r + dz * dz) / d * e)
    )
    bz = (
        current / (2 * np.pi * np.sqrt(q)) * (k + (coil_r**2 - r * r - dz * dz) / d * e)
    )
    return psi, br, bz


@dataclass
class AxisymmetricBSPF:
    radial: object
    vertical: object
    mass_r: np.ndarray
    stiffness_r: np.ndarray
    eigenvalues: np.ndarray
    transform: np.ndarray

    @property
    def shape(self):
        return (self.radial.b.shape[1], self.vertical.b.shape[1])

    def solve(self, load):
        return tensor_elliptic_solve(
            load,
            self.eigenvalues[:, None] + np.asarray(self.vertical.lam)[None, :],
            left=self.transform,
        )

    def action(self, a):
        return (
            self.stiffness_r @ a
            + (self.mass_r @ a) * np.asarray(self.vertical.lam)[None, :]
        )

    def load(self, j):
        r, z = self.radial, self.vertical
        return (
            np.asarray(r.b).T
            @ (np.asarray(r.weights)[:, None] * np.asarray(z.weights)[None, :] * j)
            @ np.asarray(z.b)
        )

    def evaluate(self, a, nodes=False):
        r, z = self.radial, self.vertical
        br, gr = (r.bn, r.gn) if nodes else (r.b, r.g)
        bz, gz = (z.bn, z.gn) if nodes else (z.b, z.g)
        rr = np.asarray(r.x if nodes else r.points)[:, None]
        return (
            np.asarray(br) @ a @ np.asarray(bz).T,
            -(np.asarray(br) @ a @ np.asarray(gz).T) / rr,
            (np.asarray(gr) @ a @ np.asarray(bz).T) / rr,
        )


def plan_axisymmetric_bspf(
    n=33, r_bounds=(1.0, 3.0), z_bounds=(-1.6, 1.6), quadrature_order=None
):
    import jax

    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 before BSPF setup")
    if r_bounds[0] <= 0:
        raise ValueError("The tokamak cross section must stay away from R=0")
    r = _stream_line(
        np.linspace(*r_bounds, n),
        clamped=False,
        dirichlet=True,
        quadrature_order=quadrature_order,
    )
    z = _stream_line(
        np.linspace(*z_bounds, n),
        clamped=False,
        dirichlet=True,
        quadrature_order=quadrature_order,
    )
    b, g, w = map(np.asarray, (r.b, r.g, r.weights))
    w = w / np.asarray(r.points)
    mass = b.T @ (w[:, None] * b)
    stiffness = g.T @ (w[:, None] * g)
    lam, transform = la.eigh(stiffness, mass)
    return AxisymmetricBSPF(r, z, mass, stiffness, lam, transform)


def external_field(r, z, coils, offset=0.0):
    shape = np.broadcast_shapes(np.shape(r), np.shape(z))
    values = [np.full(shape, offset), np.zeros(shape), np.zeros(shape)]
    for cr, cz, current in coils:
        for target, field in zip(values, coil_field(r, z, cr, cz, current)):
            target += field
    return tuple(values)


def fit_fixed_coils(*, quadrupole=-0.004, vertical=0.03, offset=-0.1):
    """Fit distant coil currents to a vacuum multipole, then freeze them.

    This constructs an illustrative device, not a reconstruction of a tokamak.
    Both target R² and R⁴-4R²Z² obey Delta*=0. The fit only defines the fixed
    coil configuration; subsequent equilibrium uses exact filament fields.
    """
    r, z = np.meshgrid(np.linspace(1, 3, 17), np.linspace(-1.6, 1.6, 21), indexing="ij")
    groups = [(0.7, 2.0), (1.5, 2.0), (2.5, 2.0), (3.3, 2.0), (0.65, 0.7), (3.5, 0.7)]
    columns = []
    for cr, cz in groups:
        columns.append(
            (coil_field(r, z, cr, cz)[0] + coil_field(r, z, cr, -cz)[0]).ravel()
        )
    columns.append(np.ones(r.size))
    target = (
        offset + vertical * (r * r - 4) + quadrupole * (r**4 - 4 * r * r * z * z - 16)
    )
    coefficients = la.lstsq(np.array(columns).T, target.ravel())[0]
    coils = [
        (cr, sign * cz, float(value))
        for (cr, cz), value in zip(groups, coefficients[:-1])
        for sign in (-1, 1)
    ]
    error = np.max(abs(np.array(columns).T @ coefficients - target.ravel()))
    return coils, float(coefficients[-1]), float(error)


def solve_equilibrium(
    p,
    coils,
    *,
    offset=0.0,
    plasma_current=1.0,
    max_iterations=400,
    tolerance=1e-10,
    relaxation=0.2,
):
    """Symmetric fixed-current Picard solve; no time evolution is performed.

    Up/down symmetry selects the centered equilibrium even if its physical
    vertical mode is unstable. Iteration convergence is NOT an MHD growth rate.
    """
    r, z = p.radial, p.vertical
    rr, zz = np.asarray(r.points)[:, None], np.asarray(z.points)[None, :]
    weights = np.asarray(r.weights)[:, None] * np.asarray(z.weights)[None, :]
    coil_psi, _, _ = external_field(rr, zz, coils, offset)
    br, bz = np.asarray(r.b), np.asarray(z.b)
    seed = np.exp(-(((rr - 2) / 0.4) ** 2) - (zz / 0.7) ** 2)
    a = p.solve(p.load(plasma_current * seed / np.sum(weights * seed)))
    history = []
    for iteration in range(max_iterations):
        psi = br @ a @ bz.T + coil_psi
        # Exact up/down symmetry in physical space removes roundoff drift in setup.
        psi = (psi + psi[:, ::-1]) / 2
        components, _ = label(psi > 0)
        center_r = np.argmin(abs(rr[:, 0] - 2))
        core_label = components[center_r, np.argmin(abs(zz[0]))]
        if core_label == 0:
            raise RuntimeError("No positive central plasma: adjust coil offset/current")
        core = components == core_label
        raw = rr * np.maximum(psi, 0) ** 2 * core
        alpha = plasma_current / np.sum(weights * raw)
        current = alpha * raw
        target = p.solve(p.load(current))
        error = la.norm(target - a) / max(la.norm(target), 1e-30)
        history.append(error)
        a = (1 - relaxation) * a + relaxation * target
        if error < tolerance:
            break
    else:
        raise RuntimeError(f"Equilibrium failed to converge, residual={error:g}")
    psi = br @ a @ bz.T + coil_psi
    if np.any(core[[0, -1]]) or np.any(core[:, [0, -1]]):
        raise RuntimeError("Plasma reaches the computational wall")
    return dict(
        a=a,
        alpha=alpha,
        psi=psi,
        current=current,
        core=core,
        pressure=alpha * np.maximum(psi, 0) ** 3 * core / 3,
        residual=error,
        iterations=iteration + 1,
        history=np.array(history),
    )
