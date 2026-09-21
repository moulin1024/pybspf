"""Independent checks of the plasma/vacuum interface and variational dynamics."""

import jax
import numpy as np
import scipy.linalg as la
import pytest
from scipy.interpolate import BSpline
from scipy.sparse.linalg import spsolve
from types import SimpleNamespace
from bspf_models._numerics._weak_basis import mp_trial_values
from bspf_models.plasma.tokamak_equilibrium import plan_axisymmetric_bspf
from bspf_models.plasma.tokamak_equilibrium import fit_fixed_coils
from bspf_models.plasma.tokamak_equilibrium import solve_equilibrium
from bspf_models.plasma.tokamak_vacuum import assemble_plasma_vacuum
from bspf_models.plasma.tokamak_vacuum import line_interpolant
from bspf_models.plasma.tokamak_vacuum import vacuum_response
from bspf_models.plasma.tokamak_vacuum import magnetic_displacement

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def setup():
    p = plan_axisymmetric_bspf(33)
    coils, offset, _ = fit_fixed_coils(quadrupole=-0.004, vertical=0.03, offset=-0.2)
    eq = solve_equilibrium(p, coils, offset=offset, max_iterations=700)
    m = assemble_plasma_vacuum(
        p,
        eq,
        coils,
        offset,
        modes=8,
        angles=64,
        radial_quadrature=24,
        vacuum_layers=12,
        vacuum_method="fem",
    )
    return p, m


def test_interpolation_against_multiprecision_bspf(setup):
    p, _ = setup
    for line in (p.radial, p.vertical):
        x = np.asarray(line.x)
        points = np.random.default_rng(61).uniform(x[0] + 0.02, x[-1] - 0.02, 11)
        knots = np.r_[
            np.repeat(x[0], 14),
            np.linspace(x[0], x[-1], 20)[1:-1],
            np.repeat(x[-1], 14),
        ]
        raw = mp_trial_values(
            SimpleNamespace(x=x, P=np.asarray(line.projector)),
            BSpline(knots, np.eye(32), 13),
            points,
            second=True,
            transform=np.asarray(line.transform),
        )
        interp = line_interpolant(line)
        for k, value in enumerate(raw):
            exact = value @ np.asarray(line.rotation)
            relative = la.norm(interp(points, k) - exact) / la.norm(exact)
            assert relative < (1e-7, 1e-6, 2e-4)[k]


def test_vacuum_manufactured_solution_converges():
    errors = []
    for n, layers in ((48, 8), (96, 16), (192, 32)):
        t = np.arange(n) * 2 * np.pi / n
        inner = np.column_stack((2 + 0.35 * np.cos(t), 0.55 * np.sin(t)))
        outer = np.column_stack((2 + 0.85 * np.cos(t), 1.3 * np.sin(t)))
        v = vacuum_response(inner, outer, layers)
        # psi=R^2 has Delta* psi=0; both boundaries prescribed independently.
        exact = v.points[:, 0] ** 2
        value = exact.copy()
        value[v.free] = 0
        value[v.free] = spsolve(
            v.stiffness[v.free][:, v.free], -(v.stiffness @ value)[v.free]
        )
        errors.append(np.max(abs(value - exact)))
        assert v.residual < 1e-12
        assert la.eigvalsh(v.boundary_energy)[0] > 0
    assert errors[1] < 0.4 * errors[0] and errors[2] < 0.4 * errors[1]


def test_divergence_and_induction_independent_difference(setup):
    _, m = setup
    points = np.array([[1.9, 0.2], [2.1, -0.2], [2.25, 0.1]])
    raw = np.random.default_rng(3).normal(size=m.transform.shape[0]) * 0.01
    f = m.basis.evaluate(points)
    div = (f["xrr"] + f["xr"] / points[:, 0, None] + f["xzz"]) @ raw
    assert np.max(abs(div)) < 1e-12
    _, qr, qz, qp = magnetic_displacement(
        f, m.evaluator.evaluate(points, True), points, m.toroidal_f
    )

    def cross(p):
        f = m.basis.evaluate(p)
        xr, xz, xp = [f[k] @ raw for k in ("xr", "xz", "xp")]
        _, pr, pz = m.evaluator.evaluate(p)
        br, bz, bp = -pz / p[:, 0], pr / p[:, 0], m.toroidal_f / p[:, 0]
        return np.column_stack(
            (xp * bz - xz * bp, xz * br - xr * bz, xr * bp - xp * br)
        )

    h = 1e-5
    dr = (cross(points + [h, 0]) - cross(points - [h, 0])) / (2 * h)
    dz = (cross(points + [0, h]) - cross(points - [0, h])) / (2 * h)
    c = cross(points)
    curl = np.column_stack(
        (-dz[:, 1], dr[:, 1] + c[:, 1] / points[:, 0], dz[:, 0] - dr[:, 2])
    )
    np.testing.assert_allclose(
        np.column_stack((qr @ raw, qz @ raw, qp @ raw)), curl, rtol=2e-6, atol=1e-8
    )


def test_interface_flux_and_vacuum_response(setup):
    _, m = setup
    q = np.random.default_rng(4).normal(size=len(m.stiffness)) * 1e-4
    xi = m.displacement(q, m.boundary)
    psi, pr, pz = m.evaluator.evaluate(m.boundary)
    flux = m.vacuum_flux(q)
    np.testing.assert_allclose(
        flux[m.vacuum.inner], -xi[:, 0] * pr - xi[:, 1] * pz, atol=1e-12
    )
    assert np.max(abs(flux[m.vacuum.outer])) == 0
    assert np.max(abs((m.vacuum.stiffness @ flux)[m.vacuum.free])) < 1e-12
    assert np.max(abs(psi)) < 1e-11
    np.testing.assert_allclose(
        q @ m.vacuum_stiffness @ q, flux @ (m.vacuum.stiffness @ flux), rtol=1e-10
    )
    # A moved interface remains a flux surface to first order.
    for epsilon in (1e-3, 5e-4):
        delta = (
            m.evaluator.evaluate(m.boundary + epsilon * xi[:, :2])[0]
            + epsilon * flux[m.vacuum.inner]
        )
        assert np.max(abs(delta)) < 1e-7 * epsilon


def test_energy_and_growth_without_exterior_inertia(setup):
    _, m = setup
    assert not hasattr(m, "density") and not hasattr(m, "resistivity")
    assert m.diagnostics["mass_identity_error"] < 1e-9
    values, vectors = m.modes()
    assert values[0] < 0
    # Free-boundary vacuum provides a nonzero restoring contribution.
    mode = vectors[:, 0]
    assert mode @ m.vacuum_stiffness @ mode > 0
    assert la.norm(m.displacement(mode, m.boundary)[:, :2]) > 0
    # General mixed-mode midpoint evolution conserves signed total energy.
    rng = np.random.default_rng(25)
    q = rng.normal(size=len(mode)) * 1e-4
    v = rng.normal(size=len(mode)) * 1e-4
    energy = lambda q, v: (v @ v + q @ m.stiffness @ q) / 2
    e0 = energy(q, v)
    dt = 0.02
    fac = la.cho_factor(np.eye(len(q)) + dt**2 / 4 * m.stiffness)
    for _ in range(40):
        qn = la.cho_solve(fac, q + dt * v - dt**2 / 4 * (m.stiffness @ q))
        v -= dt / 2 * (m.stiffness @ (q + qn))
        q = qn
    np.testing.assert_allclose(energy(q, v), e0, rtol=2e-11, atol=1e-16)


def test_close_conducting_wall_restores_vertical_stability(setup, monkeypatch):
    p, far = setup
    ev = far.evaluator
    # Rebuild the same equilibrium; only the vacuum's outer boundary changes.
    eq = solve_equilibrium(p, ev.coils, offset=ev.offset, max_iterations=700)

    def forbid_fem(*args, **kwargs):
        raise AssertionError("The BSPF backend must not call a finite-element solver")

    monkeypatch.setattr("bspf_models.plasma.tokamak_vacuum.vacuum_response", forbid_fem)
    close = assemble_plasma_vacuum(
        p,
        eq,
        ev.coils,
        ev.offset,
        modes=8,
        angles=64,
        radial_quadrature=24,
        vacuum_layers=12,
        vacuum_method="bspf",
        vacuum_radial_modes=12,
        vacuum_angular_modes=49,
        wall_scale=1.2,
    )
    np.testing.assert_allclose(close.plasma_stiffness, far.plasma_stiffness, atol=1e-12)
    assert far.modes()[0][0] < -0.01
    assert close.modes()[0][0] > -1e-8
    # The edge is still free to move and separated from the wall by a vacuum.
    wall = close.vacuum.points[close.vacuum.outer]
    np.testing.assert_allclose(
        wall - [2, 0], 1.2 * (close.boundary - [2, 0]), atol=1e-12
    )
