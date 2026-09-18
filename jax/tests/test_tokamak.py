"""Independent cylindrical identities, GS accuracy, and linear MHD work budget."""

import jax
import numpy as np
import pytest
import scipy.linalg as la
from numpy.polynomial import Polynomial
from bspf_jax.tokamak_equilibrium import (
    coil_field,
    plan_axisymmetric_bspf,
    fit_fixed_coils,
    solve_equilibrium,
    external_field,
)
from bspf_jax.tokamak_linear import (
    assemble_linear_tokamak,
    growing_modes,
    with_exterior,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def plan():
    pytest.importorskip("gmpy2")
    return plan_axisymmetric_bspf()


@pytest.fixture(scope="module")
def model(plan):
    coils, offset, _ = fit_fixed_coils(quadrupole=-0.004, vertical=0.03, offset=-0.2)
    eq = solve_equilibrium(
        plan, coils, offset=offset, max_iterations=700, tolerance=1e-10
    )
    return assemble_linear_tokamak(plan, eq, coils, offset)


def test_filament_field_and_vacuum_equation():
    r, z = np.array([1.2, 1.8, 2.6]), np.array([-0.4, 0.1, 0.8])
    h = 2e-4

    def flux(r, z):
        return coil_field(r, z, 3.5, 2.0, current=1.7)[0]

    f, br, bz = coil_field(r, z, 3.5, 2.0, current=1.7)
    dr = (flux(r + h, z) - flux(r - h, z)) / (2 * h)
    dz = (flux(r, z + h) - flux(r, z - h)) / (2 * h)
    np.testing.assert_allclose(dr / r, bz, rtol=2e-7, atol=1e-8)
    np.testing.assert_allclose(-dz / r, br, rtol=2e-7, atol=1e-8)
    star = (
        (flux(r + h, z) - 2 * f + flux(r - h, z)) / h**2
        - dr / r
        + (flux(r, z + h) - 2 * f + flux(r, z - h)) / h**2
    )
    assert np.max(abs(star)) < 2e-7


def test_continuous_manufactured_grad_shafranov(plan):
    p = plan
    r, z = p.radial, p.vertical
    rr, zz = np.asarray(r.points)[:, None], np.asarray(z.points)[None, :]
    f = Polynomial.fromroots([1, 1, 3, 3])
    g = Polynomial([1.6**4, 0, -2 * 1.6**2, 0, 1])
    source = (
        -(f.deriv(2)(rr) * g(zz) - f.deriv()(rr) * g(zz) / rr + f(rr) * g.deriv(2)(zz))
        / rr
    )
    a = p.solve(p.load(source))
    psi, br, bz = p.evaluate(a)
    np.testing.assert_allclose(psi, f(rr) * g(zz), atol=3e-10)
    np.testing.assert_allclose(br, -f(rr) * g.deriv()(zz) / rr, atol=3e-9)
    np.testing.assert_allclose(bz, f.deriv()(rr) * g(zz) / rr, atol=3e-9)


def test_free_boundary_equilibrium(model):
    p, e = model.equilibrium_plan, model.equilibrium
    w = np.asarray(p.radial.weights)[:, None] * np.asarray(p.vertical.weights)[None, :]
    assert not e["core"][[0, -1]].any() and not e["core"][:, [0, -1]].any()
    assert np.max(abs(e["psi"] - e["psi"][:, ::-1])) < 1e-10
    np.testing.assert_allclose(np.sum(w * e["current"]), 1, atol=1e-12)
    assert (
        la.norm(p.action(e["a"]) - p.load(e["current"])) / la.norm(p.load(e["current"]))
        < 1e-8
    )
    assert np.max(abs(e["current"][~e["core"]])) == 0
    assert np.max(abs(e["pressure"][~e["core"]])) == 0


def test_full_vector_linear_energy_exchange(model):
    q = model.matrices
    rng = np.random.default_rng(23)
    state = rng.normal(size=len(q["A"])) * 1e-6
    derivative = la.solve(q["M"], q["A"] @ state, assume_a="pos")
    a, c, b, d = [v.ravel() for v in model.split(state)]
    current = q["K"] @ b
    expected = (
        a @ q["current_force"] @ b
        - a @ q["viscous"] @ a
        - c @ q["toroidal_viscous"] @ c
        - current @ q["resistive_flux"] @ current
        - d @ q["resistive_toroidal"] @ d
    )
    np.testing.assert_allclose(
        state @ q["energy"] @ derivative, expected, rtol=2e-11, atol=2e-12
    )


def test_induction_from_independent_curl_of_u_cross_b(model):
    p = model.equilibrium_plan
    r, z = p.radial, p.vertical
    rr = np.asarray(r.points)[:, None]
    zz = np.asarray(z.points)[None, :]
    w = np.asarray(r.weights)[:, None] * np.asarray(z.weights)[None, :]
    rng = np.random.default_rng(31)
    state = np.zeros(len(model.matrices["A"]))
    nv, nt = [int(np.prod(s)) for s in model.shapes[:2]]
    state[: nv + nt] = rng.normal(size=nv + nt) * 1e-6
    _, u, _ = model.fields(state)
    _, br, bz = p.evaluate(model.equilibrium["a"])
    _, cr, cz = external_field(rr, zz, model.coils, model.offset)
    br += cr
    bz += cz
    bm, gm = np.asarray(r.b), np.asarray(r.g)
    zm, gzm = [np.asarray(a) @ model.parity_flux for a in (z.b, z.g)]
    cross_r = u[..., 2] * bz - u[..., 1] * model.toroidal_f / rr
    cross_z = u[..., 0] * model.toroidal_f / rr - u[..., 2] * br
    # Integrate curl_phi(u cross B) against R*test, transferring derivatives.
    expected = (
        -bm.T @ (w * rr * cross_r) @ gzm + (gm + bm / rr).T @ (w * rr * cross_z) @ zm
    )
    actual = (model.matrices["A"] @ state)[-expected.size :].reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-9)


def test_axisymmetric_divergence_and_wall_traces(model):
    state = np.random.default_rng(16).normal(size=len(model.matrices["A"])) * 1e-6
    a, _, b, _ = model.split(state)
    a = a @ model.parity_velocity.T
    b = b @ model.parity_flux.T
    vr, vz = model.velocity_r, model.velocity_z
    r, z = model.equilibrium_plan.radial, model.equilibrium_plan.vertical
    rr = np.asarray(vr.x)[:, None]
    # (1/R) d_R(R*u_R) + d_Z u_Z, using independently contracted derivatives.
    divu = (
        -np.asarray(vr.gn) @ (a @ np.asarray(vz.gn).T)
        + (np.asarray(vr.gn) @ a) @ np.asarray(vz.gn).T
    ) / rr
    divb = (
        -np.asarray(r.gn) @ (b @ np.asarray(z.gn).T)
        + (np.asarray(r.gn) @ b) @ np.asarray(z.gn).T
    ) / rr
    assert np.max(abs(divu)) < 1e-11 and np.max(abs(divb)) < 1e-11
    _, u, magnetic = model.fields(state, nodes=True)
    assert max(abs(u[[0, -1]]).max(), abs(u[:, [0, -1]]).max()) < 1e-11
    assert (
        max(abs(magnetic[[0, -1], :, 0]).max(), abs(magnetic[:, [0, -1], 1]).max())
        < 1e-11
    )


def test_verified_growing_mode_and_exterior_invariance(model):
    values, vectors, residuals = growing_modes(model, shift=0.15, count=4)
    assert values[0].real > 0 and abs(values[0].imag) < 1e-8
    assert residuals[0] < 1e-8
    altered = with_exterior(model, halo_density=0.005, vacuum_resistivity=2.0)
    np.testing.assert_array_equal(altered.coils, model.coils)
    np.testing.assert_array_equal(altered.equilibrium["psi"], model.equilibrium["psi"])
    # No prescribed forcing: zero perturbation remains exactly zero.
    np.testing.assert_array_equal(model.matrices["A"] @ np.zeros(len(vectors)), 0)
