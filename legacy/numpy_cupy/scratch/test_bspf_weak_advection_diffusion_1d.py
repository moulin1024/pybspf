"""Independent invariants for the 1D research prototype (not NS integration)."""

import numpy as np
import pytest
import scipy.linalg as la
from scipy.integrate import solve_ivp

from bspf_weak_advection_diffusion_1d import (
    assemble,
    trial_values,
    analytic_field,
    exact_semidiscrete_error,
)


@pytest.mark.parametrize("n", [48, 65])
def test_same_cardinal_space_and_derivative(n):
    p = assemble(n)
    values, gradients = trial_values(p.line, p.spline, p.x)
    np.testing.assert_allclose(values, np.eye(n), atol=3e-11)
    np.testing.assert_allclose(gradients, p.strong_d1, atol=3e-10)
    # Includes both odd/even periodic sample counts and the Nyquist cosine.
    np.testing.assert_allclose(values[[0, -1], 1:-1], 0, atol=3e-11)


@pytest.mark.parametrize("speed", [-3.0, 0.0, 1.0, 3.0])
def test_energy_law_and_spectrum(speed):
    p = assemble(48)
    a = p.generator(speed=speed)
    assert la.eigvalsh(p.mass)[0] > 0
    assert la.eigvals(a).real.max() < 0
    expected = -0.004 * p.stiffness
    np.testing.assert_allclose(
        p.mass @ a + a.T @ p.mass, expected, atol=1e-13, rtol=1e-12
    )
    u = np.random.default_rng(2).normal(size=46)
    rate = u @ p.mass @ a @ u
    physical = -0.002 * np.dot(p.quadrature_weights, (p.gradients @ u) ** 2)
    np.testing.assert_allclose(rate, physical, rtol=1e-12)


def test_polynomial_consistency_and_exponential_error_formula():
    p = assemble(48)
    x, q = p.x, p.quadrature_x
    u = 1 - (x / 3) ** 2
    phi = 1 - (q / 3) ** 2
    force = p.project_load(-phi - 2 * q / 9 + 0.004 / 9)
    a = p.generator()
    error = exact_semidiscrete_error(a, u[1:-1], force, 1.0)
    assert abs(error).max() < 1e-10
    # Verify the block-exponential formula with an independent time integrator.
    phi, px, pxx = analytic_field(q, "nonperiodic")
    initial = analytic_field(x, "nonperiodic")[0][1:-1]
    f = p.project_load(-phi + px - 0.002 * pxx)
    result = solve_ivp(
        lambda t, y: a @ y + np.exp(-t) * f,
        [0, 0.1],
        initial,
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
    )
    exact = np.exp(-0.1) * initial + exact_semidiscrete_error(a, initial, f, 0.1)
    np.testing.assert_allclose(result.y[:, -1], exact, atol=2e-12)


def test_high_precision_setup_preserves_the_same_space():
    pytest.importorskip("gmpy2")
    original = assemble(48)
    accurate = assemble(48, assembly_bits=113)
    np.testing.assert_array_equal(accurate.line.P, original.line.P)
    np.testing.assert_array_equal(accurate.strong_d1, original.strong_d1)
    np.testing.assert_allclose(accurate.values, original.values, atol=3e-12)
    np.testing.assert_allclose(accurate.gradients, original.gradients, atol=3e-11)
    assert accurate.raw_sbp_defect < 1e-12
