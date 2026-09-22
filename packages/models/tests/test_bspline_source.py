"""Independent domain-only source audits and complete elliptic solves."""
import numpy as np
import pytest
pytest.importorskip('finufft')
from bspf_models.elliptic.embedded_poisson import SplineDomain
from bspf_models.elliptic.spline_annulus import SplineAnnulus, AnnulusPanelPlan
from bspf_models.elliptic.bspline_source import BSplineSourcePlan


@pytest.fixture(scope='module')
def domain():
    t = np.arange(8)*2*np.pi/8
    outer = SplineDomain(np.column_stack((np.cos(t), .85*np.sin(t))))
    inner = SplineDomain([.12, -.06]+np.column_stack((.25*np.cos(t), .2*np.sin(t))))
    return SplineAnnulus(outer.curve, inner.curve)


@pytest.fixture(scope='module', params=[5, 7])
def plan(domain, request):
    return BSplineSourcePlan(domain, degree=request.param, grid_size=1025)


@pytest.mark.parametrize('sigma', [0., 4., -4.])
def test_domain_only_source_and_elliptic_solution(plan, sigma):
    def exact(x):
        return np.exp(.3*x[:, 0]+.2*x[:, 1])+.1*np.sin(2*x[:, 0])*np.cos(1.5*x[:, 1])
    def source(x):
        assert np.all(plan.domain.contains(x))
        return ((sigma-.13)*np.exp(.3*x[:, 0]+.2*x[:, 1])
                +.1*(sigma+6.25)*np.sin(2*x[:, 0])*np.cos(1.5*x[:, 1]))
    panel = AnnulusPanelPlan(plan.domain, sigma, order=12, subdivisions=2)
    solution = panel.solve((exact, exact), source=source, source_plan=plan, source_tolerance=1e-7)
    points = [plan.domain.sample(16, shift=.271)]
    for component, boundary in enumerate(plan.domain.boundaries):
        t = boundary.a+(np.arange(6)+.271)/6*(boundary.b-boundary.a)
        for distance in (1e-4, 1e-6):
            points.append(boundary.curve(t)-distance*boundary.normal(t, 1 if component == 0 else -1))
    points = np.vstack(points)
    np.testing.assert_allclose(solution.interior(points), exact(points), atol=2e-7, rtol=2e-7)
    assert solution.particular.stats['validation_relative_max'] < 1e-7


def test_polynomial_reproduction_and_complex_rhs(plan):
    def polynomial(x):
        assert np.all(plan.domain.contains(x))
        return (1+2j)*(1+x[:, 0]+.3*x[:, 0]*x[:, 1]+.2*x[:, 1]**3)
    particular = plan.fit(polynomial, 0., tolerance=1e-8)
    points = plan.domain.sample(21, shift=.129)
    np.testing.assert_allclose(particular.source_values(points), polynomial(points), atol=3e-8)


def test_unresolved_source_rejected(plan):
    with pytest.raises(ValueError, match='source validation failed'):
        plan.fit(lambda x: np.sin(1000*x[:, 0]), 0., tolerance=1e-7)


def test_knot_refinement_reduces_source_error(domain):
    errors = []
    for spans in (3, 5):
        plan = BSplineSourcePlan(domain, spans=spans, grid_size=513)
        result = plan.fit(lambda x: np.sin(12*x[:, 0])*np.cos(9*x[:, 1]), 0., tolerance=np.inf)
        errors.append(result.stats['validation_relative_max'])
    assert errors[1] < errors[0]/5


@pytest.mark.parametrize('kwargs', [{'degree': 6}, {'spans': 1}, {'spans': 2.5},
                                   {'stability': 0}, {'stability': np.nan}])
def test_invalid_parameters(domain, kwargs):
    with pytest.raises(ValueError):
        BSplineSourcePlan(domain, **kwargs)


def test_invalid_fit_tolerance_and_operator(plan):
    for tolerance in (0., -1., np.nan):
        with pytest.raises(ValueError, match='positive tolerance'):
            plan.fit(lambda x: np.ones(len(x)), 0., tolerance=tolerance)
    with pytest.raises(ValueError, match='finite sigma'):
        plan.fit(lambda x: np.ones(len(x)), np.nan)
