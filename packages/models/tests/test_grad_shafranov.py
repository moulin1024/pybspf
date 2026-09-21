"""GS signs, physical R, exact flux geometry and independent Solov'ev tests."""

import jax
import numpy as np
import pytest

from bspf_models.elliptic.convex_poisson import ArcLengthBoundary
from bspf_models.plasma.grad_shafranov import FixedBoundaryGSPlan
from bspf_models.plasma.solovev import SolovevEquilibrium
from bspf_models.plasma.solovev import SolovevFluxDomain

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("logarithmic", [0.0, 0.08])
def test_analytic_gs_identity_profiles_and_force_balance(logarithmic):
    eq = SolovevEquilibrium(logarithmic=logarithmic, mu0=0.7)
    points = np.random.default_rng(5).uniform([1.7, -0.2], [2.3, 0.2], (61, 2))
    psi, grad, h = eq.jets(points)
    r = points[:, 0]
    delta = h[:, 0, 0]+h[:, 1, 1]-grad[:, 0]/r
    np.testing.assert_allclose(-delta, eq.source(points), atol=5e-15)
    np.testing.assert_allclose(eq.source(points), eq.mu0*r*r*eq.p_prime+eq.ff_prime, atol=2e-15)
    f = eq.toroidal_function(psi)
    b = eq.magnetic_field(points)
    current = np.column_stack((-eq.ff_prime/f*grad[:, 1], -delta,
                               eq.ff_prime/f*grad[:, 0]))/(eq.mu0*r[:, None])
    pressure_gradient = np.column_stack((eq.p_prime*grad[:, 0], np.zeros(len(r)), eq.p_prime*grad[:, 1]))
    np.testing.assert_allclose(np.cross(current, b), pressure_gradient, atol=2e-15)
    for axis in (0, 1):
        shift = np.eye(2)[axis]*1e-5
        vp, gp, _ = eq.jets(points+shift)
        vm, gm, _ = eq.jets(points-shift)
        np.testing.assert_allclose((vp-vm)/2e-5, grad[:, axis], atol=2e-9)
        np.testing.assert_allclose((gp-gm)/2e-5, h[:, :, axis], atol=2e-9)


@pytest.mark.parametrize("logarithmic", [0.0, 0.08])
def test_exact_closed_boundary_geometry_and_quadrature(logarithmic):
    eq = SolovevEquilibrium(logarithmic=logarithmic)
    domain = SolovevFluxDomain(eq)
    t = np.linspace(0, domain.period, 131, endpoint=False)+0.0017
    edge = domain.curve(t)
    np.testing.assert_allclose(eq.jets(edge+[eq.major_radius, 0])[0], 0, atol=4e-15)
    np.testing.assert_allclose(domain.curve(t+domain.period), edge, atol=1e-14)
    step = 1e-5
    for nu in (0, 1):
        np.testing.assert_allclose((domain.curve(t+step, nu)-domain.curve(t-step, nu))/2/step,
                                   domain.curve(t, nu+1), atol=5e-9)
    tangent, curvature = domain.curve(t, 1), domain.curve(t, 2)
    assert np.min(tangent[:, 0]*curvature[:, 1]-tangent[:, 1]*curvature[:, 0]) > 0
    assert domain.geometry_checks["level_hessian_determinant_lower"] > 0
    p, w, _ = domain.volume_rule(12)
    p2, w2, _ = domain.volume_rule(20)
    assert np.min(eq.jets(p+[eq.major_radius, 0])[0]) > 0
    np.testing.assert_allclose(w.sum(), w2.sum(), rtol=2e-12)
    # Area from a completely different boundary integral (Green's theorem).
    q, weights = np.polynomial.legendre.leggauss(128)
    curve = domain.curve(2*(q+1))
    tangent = domain.curve(2*(q+1), 1)
    area = np.sum(weights*(curve[:, 0]*tangent[:, 1]-curve[:, 1]*tangent[:, 0]))
    np.testing.assert_allclose(w.sum(), area, rtol=2e-12)
    arc = ArcLengthBoundary(domain)
    edge, parameters = arc.sample(64, offset=0.371)
    assert arc.length > 0
    np.testing.assert_allclose(eq.jets(edge+[2, 0])[0], 0, atol=4e-15)


@pytest.fixture(scope="module")
def plan():
    eq = SolovevEquilibrium()
    return FixedBoundaryGSPlan(SolovevFluxDomain(eq), nodes=17, volume_order=10, boundary_count=128)


def test_solovev_solve_uses_only_source_and_constant_boundary(plan):
    eq = SolovevEquilibrium()
    def source(points):
        np.testing.assert_array_equal(points, plan.points)
        return eq.source(points)
    s = plan.solve(source, 0.0)
    q = np.random.default_rng(14).uniform([1.85, -0.2], [2.15, 0.2], (37, 2))
    value, gradient, delta = s.evaluate(q)
    exact, exact_gradient, _ = eq.jets(q)
    np.testing.assert_allclose(value, exact, atol=3e-8)
    np.testing.assert_allclose(gradient, exact_gradient, atol=3e-7)
    np.testing.assert_allclose(-delta, eq.source(q), atol=3e-6)
    np.testing.assert_allclose(s.magnetic_field(q, eq.toroidal_function), eq.magnetic_field(q), atol=3e-7)
    edge, _ = plan.arc.sample(256, offset=0.371)
    np.testing.assert_allclose(s.evaluate(edge+[2, 0])[0], 0, atol=3e-8)
    assert s.diagnostics["physical_data_only"]


def test_physical_radius_vacuum_mode_and_nonzero_boundary(plan):
    # Delta* R²=0, whereas the ordinary Cartesian Laplacian is 2. This
    # distinguishes the GS operator from Poisson and catches wrong R offsets.
    s = plan.solve(0, lambda points: points[:, 0]**2)
    q = np.array([[1.8, 0.12], [2.1, -0.11], [2.25, 0.1]])
    v, g, d = s.evaluate(q)
    np.testing.assert_allclose(v, q[:, 0]**2, atol=3e-8)
    np.testing.assert_allclose(g[:, 0], 2*q[:, 0], atol=3e-7)
    np.testing.assert_allclose(d, 0, atol=3e-6)


def test_profile_wrapper_reuse_and_independent_residual(plan):
    eq = SolovevEquilibrium()
    s = plan.solve_solovev(p_prime=eq.p_prime, ff_prime=eq.ff_prime)
    q = np.array([[1.9, 0.1], [2.2, -0.15]])
    twice = plan.solve_solovev(p_prime=2*eq.p_prime, ff_prime=2*eq.ff_prime)
    np.testing.assert_allclose(twice.evaluate(q)[0], 2*s.evaluate(q)[0], atol=1e-12)
    residual = s.validate(eq.source, volume_order=12, boundary_count=256)
    assert residual["gs_relative_residual_l2"] < 1e-5
    assert residual["boundary_linf"] < 3e-8


def test_invalid_geometry_and_input(plan):
    with pytest.raises(ValueError, match="R>0"):
        FixedBoundaryGSPlan(plan.domain, major_radius=1.0, half_width=1.2)
    with pytest.raises(ValueError, match="major_radius differs"):
        FixedBoundaryGSPlan(plan.domain, major_radius=3.0)
    with pytest.raises(ValueError, match="finite"):
        plan.solve(np.nan)
    with pytest.raises(ValueError, match="R>0"):
        plan.prepare(np.array([[0, 0]]))
    with pytest.raises(ValueError, match="F"):
        SolovevEquilibrium().toroidal_function(-100)


def test_compiled_response_multiple_rhs_and_boundary(plan):
    from bspf_models.plasma.grad_shafranov import evaluate_gs_factors
    fast = plan.compile_response()
    sources = [0, SolovevEquilibrium().source,
               lambda p: np.sin(7*p[:, 0])*np.cos(5*p[:, 1])]
    boundaries = [0, lambda p: p[:, 0]**2, lambda p: np.sin(p[:, 1])]
    for f, g in zip(sources, boundaries):
        reference = plan.solve(f, g)
        actual = fast.solve(f, g)
        expected = evaluate_gs_factors(plan.source_basis, reference.coefficients)[0]
        np.testing.assert_allclose(actual.flux, expected, atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(actual.as_solution().coefficients, reference.coefficients,
                                   atol=1e-13, rtol=1e-13)
        assert actual.diagnostics['training_relative_residual'] == reference.diagnostics['training_relative_residual']
    np.testing.assert_array_equal(fast.solve(0).flux, 0)
    with pytest.raises(ValueError, match='derivatives=True'):
        fast.solve(0).gradient
    with pytest.raises(ValueError, match='finite'):
        fast.solve(np.nan)
    with pytest.raises(ValueError, match='chunk_size'):
        plan.compile_response(chunk_size=0)


def test_compiled_response_independent_jets(plan):
    eq = SolovevEquilibrium()
    points = np.random.default_rng(76).uniform([1.9, -.15], [2.1, .15], (43, 2))
    fast = plan.compile_response(points, derivatives=True, chunk_size=13)
    actual = fast.solve(eq.source)
    reference = plan.solve(eq.source)
    v, g, h = reference.jets(points)
    np.testing.assert_allclose(actual.flux, v, atol=2e-12)
    np.testing.assert_allclose(actual.gradient, g, atol=2e-11)
    np.testing.assert_allclose(actual.hessian, h, atol=2e-10)
    np.testing.assert_allclose(actual.delta_star, -eq.source(points), atol=3e-6)
    np.testing.assert_allclose(actual.magnetic_field(eq.toroidal_function),
                               eq.magnetic_field(points), atol=3e-7)
