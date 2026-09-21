import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_legendre

from bspf_models.elliptic.embedded_poisson import benchmark_domains
from bspf_models.elliptic.panel_poisson import PanelPoissonPlan
from bspf_models.elliptic.panel_poisson import adaptive_solve
from bspf_models.elliptic.panel_poisson import log_moments


def harmonic(points):
    x, y = points.T
    return np.exp(3 * x) * np.cos(3 * y) + 0.1 * np.real((x + 1j * y) ** 7)


@pytest.mark.parametrize("x", [-1.0, -0.991, 0.13, 0.97, 1.0])
def test_logarithmic_moments_against_independent_adaptive_quadrature(x):
    expected = []
    for n in range(13):
        value = 0.0
        if x > -1:
            value += quad(
                lambda t: eval_legendre(n, t),
                -1,
                x,
                weight="alg-logb",
                wvar=(0, 0),
                epsabs=2e-13,
            )[0]
        if x < 1:
            value += quad(
                lambda t: eval_legendre(n, t),
                x,
                1,
                weight="alg-loga",
                wvar=(0, 0),
                epsabs=2e-13,
            )[0]
        expected.append(value)
    np.testing.assert_allclose(log_moments(x, 12), expected, atol=2e-13)


def test_requires_all_geometry_knots():
    domain = benchmark_domains()[0]
    with pytest.raises(ValueError, match="original spline knot"):
        PanelPoissonPlan(domain, breaks=[0, domain.period])


def test_constant_mode_and_nonzero_poisson_particular():
    domain = benchmark_domains()[0]
    plan = PanelPoissonPlan(domain, order=12)
    constant = plan.solve(np.full(plan.count, 2.5))
    np.testing.assert_allclose(constant.density, 0, atol=2e-11)
    np.testing.assert_allclose(constant.constant, 2.5, atol=1e-13)

    # -Delta(particular)=1; the boundary correction is nonzero and nonconstant.
    def particular(points):
        return -np.sum(points**2, axis=1) / 4

    def exact(points):
        return particular(points) + harmonic(points)

    solution = plan.solve(exact, particular)
    points = np.array([[0.0, 0.0], [0.31, -0.22], [-0.4, 0.2]])
    np.testing.assert_allclose(solution.interior(points), exact(points), atol=2e-9)
    assert abs(plan.weights_dt @ solution.density) < 1e-12
    assert np.linalg.norm(solution.density) > 1


def test_endpoint_adaptivity_reduces_independent_residual():
    domain = benchmark_domains()[0]
    initial = PanelPoissonPlan(domain, order=12).solve(harmonic)
    solution, history = adaptive_solve(
        domain, harmonic, order=12, tolerance=1e-9, max_refinements=4
    )
    # Probe much closer to the original knots than the adaptive marking probes.
    t = np.mod(
        np.arange(domain.period)[:, None] + np.array([0, 1e-6, -1e-6]), domain.period
    ).ravel()
    exact = harmonic(domain.curve(t))
    initial_error = np.max(abs(initial.boundary(t) - exact))
    final_error = np.max(abs(solution.boundary(t) - exact))
    assert history[-1]["converged"]
    assert final_error < initial_error / 10
    # A second, higher-order quadrature must agree at knot and near-knot probes.
    np.testing.assert_allclose(
        solution.boundary(t), solution.boundary(t, 80), atol=2e-12
    )
    assert np.unique(np.diff(solution.plan.breaks)).size > 1


def test_adaptive_limit_reports_failure():
    _, history = adaptive_solve(
        benchmark_domains()[0], harmonic, tolerance=1e-15, max_refinements=0
    )
    assert not history[-1]["converged"]
