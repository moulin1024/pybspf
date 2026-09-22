"""Full-section elliptic/GS regression including an interior magnetic O point."""
import numpy as np
import pytest
from bspf_models.elliptic.embedded_poisson import SplineDomain
from bspf_models.elliptic.spline_annulus import SplineSection, AnnulusPanelPlan
from bspf_models.plasma.spline_annulus_gs import SplineAnnulusGSPlan
from bspf_models.plasma.solovev import SolovevEquilibrium


@pytest.fixture(scope='module')
def section():
    t=np.arange(12)*2*np.pi/12
    return SplineSection(SplineDomain(np.column_stack((.8*np.cos(t),.7*np.sin(t)))).curve)


@pytest.mark.parametrize('sigma',[0.,4.,-4.])
def test_full_section_no_inner_boundary(section,sigma):
    assert section.contains(np.array([[0.,0.]]))[0]
    plan=AnnulusPanelPlan(section,sigma,order=12,subdivisions=2)
    exact=(lambda x: x[:,0]+.3*x[:,1]) if sigma == 0 else ((lambda x:np.exp(2*x[:,0])) if sigma > 0 else (lambda x:np.cos(2*x[:,0])))
    solution=plan.solve((exact,))
    points=np.array([[0.,0.],[.4,.1],[-.2,.3]])
    np.testing.assert_allclose(solution.interior(points),exact(points),atol=1e-8)
    with pytest.raises(ValueError,match='component'):
        solution.boundary(1,np.array([.2]))


def test_full_gs_axis_and_diagnostic_derivatives(section):
    plan=SplineAnnulusGSPlan(section,major_radius=3,modes=16,samples=72,padding=1.5,
                            order=12,subdivisions=2)
    exact=SolovevEquilibrium(major_radius=3,logarithmic=.03)
    solution=plan.solve(exact.source,(lambda x:exact.jets(x)[0],),tolerance=1e-8,source_tolerance=1e-8)
    axis=solution.magnetic_axis([3.02,.02])
    np.testing.assert_allclose(axis['position'],[3.,0.],atol=2e-6)
    assert abs(axis['flux']-exact.axis_flux) < 1e-7
    points=np.array([[3.,0.],[3.2,.1]])
    f,g,h=solution.jets(points)
    ef,eg,eh=exact.jets(points)
    np.testing.assert_allclose(f,ef,atol=1e-7)
    np.testing.assert_allclose(g,eg,atol=2e-6)
    np.testing.assert_allclose(h,eh,atol=2e-5)
    assert len(plan.domain.boundaries)==1
