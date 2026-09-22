"""Local domain-only extension, fast Fourier evaluation and operator checks."""
import numpy as np
import pytest
pytest.importorskip('finufft')
from bspf_models.elliptic.embedded_poisson import SplineDomain
from bspf_models.elliptic.spline_annulus import SplineAnnulus, AnnulusPanelPlan
from bspf_models.elliptic.local_source import LocalSourcePlan, LocalParticular


@pytest.fixture(scope='module')
def plan():
    t=np.arange(8)*2*np.pi/8
    outer=SplineDomain(np.column_stack((np.cos(t),.85*np.sin(t))))
    inner=SplineDomain([.12,-.06]+np.column_stack((.25*np.cos(t),.2*np.sin(t))))
    return LocalSourcePlan(SplineAnnulus(outer.curve,inner.curve),grid_size=1025,degree=16)


@pytest.mark.parametrize('sigma',[0.,4.,-4.])
def test_domain_only_source_and_end_to_end(plan,sigma):
    def exact(x): return np.exp(.3*x[:,0]+.2*x[:,1])+.1*np.sin(2*x[:,0])*np.cos(1.5*x[:,1])
    def source(x):
        assert np.all(plan.domain.contains(x))
        return (sigma-.13)*np.exp(.3*x[:,0]+.2*x[:,1])+.1*(sigma+6.25)*np.sin(2*x[:,0])*np.cos(1.5*x[:,1])
    boundary=AnnulusPanelPlan(plan.domain,sigma,order=12,subdivisions=2)
    solution=boundary.solve((exact,exact),source=source,source_plan=plan,source_tolerance=1e-7)
    points=np.array([[.6,.1],[-.5,.2],[.1,.5]])
    np.testing.assert_allclose(solution.interior(points),exact(points),rtol=2e-7,atol=2e-7)
    np.testing.assert_allclose(solution.particular.source_values(points),source(points),rtol=1e-7,atol=1e-7)


def test_nufft_ordering_phase_and_resonant_particular(plan):
    # A single exact Fourier mode tests the FFT convention independently of fitting.
    c=np.zeros((plan.n,plan.n),complex)
    j=plan.n//2
    c[j+1,j]=.7+.2j
    q=plan.frequencies[j+1,j]
    points=np.array([[.6,.1],[-.5,.2]])
    expected=c[j+1,j]*np.exp(1j*(points-plan.origin)@q)
    np.testing.assert_allclose(plan._evaluate(c,points),expected,atol=1e-11,rtol=1e-11)
    part=LocalParticular(plan,c,-np.dot(q,q),{})
    h=2e-4
    lap=np.zeros(len(points),complex)
    for axis in np.eye(2):
        lap+=(-part(points+2*h*axis)+16*part(points+h*axis)-30*part(points)
              +16*part(points-h*axis)-part(points-2*h*axis))/(12*h*h)
    np.testing.assert_allclose(-lap+part.sigma*part(points),expected,atol=2e-7,rtol=2e-7)


def test_unresolved_source_is_not_accepted(plan):
    with pytest.raises(ValueError,match='local source validation failed'):
        plan.fit(lambda x:np.sin(1000*x[:,0]),0,tolerance=1e-7)
