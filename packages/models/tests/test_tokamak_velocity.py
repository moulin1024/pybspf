"""Shared elliptic/momentum kernels and independently evolved magnetic state."""

import jax
import numpy as np
import scipy.linalg as la
import pytest
from pybspf.tensor import tensor_elliptic_solve
from bspf_models._numerics._tensor_pcg import plan_tensor_preconditioner
from bspf_models._numerics._tensor_pcg import tensor_pcg
from bspf_models.plasma.tokamak_equilibrium import plan_axisymmetric_bspf
from bspf_models.plasma.tokamak_equilibrium import fit_fixed_coils
from bspf_models.plasma.tokamak_equilibrium import solve_equilibrium
from bspf_models.plasma.tokamak_vacuum import assemble_plasma_vacuum
from bspf_models.plasma.tokamak_velocity import plan_tokamak_velocity

jax.config.update("jax_enable_x64", True)


def test_shared_tensor_inverse_and_pcg_against_independent_direct_solve():
    rng = np.random.default_rng(45)

    def spd(n):
        a = rng.normal(size=(n, n))
        return a.T @ a + np.eye(n)

    mr, kr, mt, kt = spd(5), spd(5), spd(7), spd(7)
    a = np.kron(kr, mt) + np.kron(mr, kt)
    pre = plan_tensor_preconditioner(mr, kr, mt, kt, 1, 1)
    rhs = rng.normal(size=(35, 3))
    np.testing.assert_allclose(pre(rhs), la.solve(a, rhs), rtol=2e-12, atol=2e-13)
    direct = tensor_elliptic_solve(
        rhs[:, 0].reshape(5, 7), pre.denominator, pre.left, pre.right
    )
    np.testing.assert_allclose(
        direct.ravel(), la.solve(a, rhs[:, 0]), rtol=2e-12, atol=2e-13
    )
    # A nonseparable positive addition means the tensor solve alone is insufficient.
    correction = rng.normal(size=(35, 9))
    variable = a + correction @ correction.T
    result, diag = tensor_pcg(variable, rhs, pre)
    np.testing.assert_allclose(result, la.solve(variable, rhs), rtol=2e-10, atol=2e-12)
    assert diag["elliptic_max_iterations"] > 1
    assert diag["elliptic_max_relative_residual"] < 3e-12


@pytest.fixture(scope="module")
def flow():
    p = plan_axisymmetric_bspf(33)
    coils, offset, _ = fit_fixed_coils(quadrupole=-0.004, vertical=0.03, offset=-0.2)
    eq = solve_equilibrium(p, coils, offset=offset, max_iterations=700)
    spatial = assemble_plasma_vacuum(
        p,
        eq,
        coils,
        offset,
        modes=8,
        angles=64,
        radial_quadrature=24,
        vacuum_layers=12,
        wall_scale=1.2,
        vacuum_radial_modes=12,
        vacuum_angular_modes=49,
    )
    return plan_tokamak_velocity(spatial)


def test_velocity_induction_is_not_a_displacement_only_rhs(flow):
    n = flow.size
    state = np.zeros(3 * n)
    state[2 * n :] = np.random.default_rng(22).normal(size=n) * 1e-5
    rhs = flow.rhs(state)
    np.testing.assert_allclose(
        rhs[n : 2 * n], -flow.induction.T @ state[2 * n :], atol=1e-14
    )
    assert la.norm(rhs[n : 2 * n]) > 1e-7
    np.testing.assert_array_equal(rhs[:n], 0)
    assert flow.stiffness_reference_error < 2e-12
    assert flow.spatial.diagnostics["elliptic_solver"] == "tensor_pcg"
    assert flow.spatial.diagnostics["elliptic_max_iterations"] > 1


def test_general_state_energy_exchange(flow):
    state = np.random.default_rng(33).normal(size=3 * flow.size) * 1e-5
    q, v, b = flow.split(state)
    dq, dv, db = flow.split(flow.rhs(state))
    power = (
        v @ dv + b @ db - dq @ flow.equilibrium_drive @ q + dq @ flow.vacuum_force(q)
    )
    assert abs(power) < 2e-17
    np.testing.assert_allclose(dq, v, atol=0)
    np.testing.assert_allclose(db, flow.induction @ v, atol=1e-15)
    np.testing.assert_array_equal(flow.step(np.zeros_like(state), 0.001), 0)


def test_rk4_time_order_and_frozen_flux(flow):
    values, vectors = flow.spatial.modes()
    index = len(values) // 2
    omega = np.sqrt(values[index])
    q0 = vectors[:, index] * 1e-4
    exact_q = np.cos(omega) * q0
    exact_v = -omega * np.sin(omega) * q0
    errors = []
    for dt in (0.02, 0.01, 0.005):
        state = flow.initial_state(q0)
        for _ in range(round(1 / dt)):
            state = flow.step(state, dt)
        q, v, b = flow.split(state)
        errors.append(la.norm(q - exact_q) + la.norm(v - exact_v))
        np.testing.assert_allclose(b, flow.induction @ q, atol=2e-16, rtol=2e-10)
    assert errors[0] / errors[1] > 14, errors
    assert errors[1] / errors[2] > 14, errors


def test_state_and_alfven_step_validation(flow):
    with pytest.raises(ValueError):
        flow.rhs(np.zeros(flow.size))
    with pytest.raises(ValueError):
        flow.step(np.zeros(3 * flow.size), 3 / flow.maximum_frequency)
