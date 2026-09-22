"""Independent analytic GS fields and explicit failure behavior."""
import numpy as np
import pytest
from bspf_models.elliptic.embedded_poisson import SplineDomain
from bspf_models.elliptic.spline_annulus import SplineAnnulus
from bspf_models.plasma.spline_annulus_gs import SplineAnnulusGSPlan


@pytest.fixture(scope='module')
def plan():
    t=np.arange(8)*2*np.pi/8
    outer=SplineDomain(np.column_stack((np.cos(t), .8*np.sin(t))))
    inner=SplineDomain(np.column_stack((.25*np.cos(t), .2*np.sin(t))))
    domain=SplineAnnulus(outer.curve,inner.curve)
    return SplineAnnulusGSPlan(domain,major_radius=3.,modes=16,samples=72,order=16,subdivisions=2)


def test_forced_gs_polynomial_and_physical_coordinates(plan):
    # -Delta*(R^4 + Z^2) = -8 R^2 - 2, unlike the Cartesian Laplacian.
    exact=lambda p:p[:,0]**4+p[:,1]**2
    def source(p):
        assert np.all(plan.domain.contains(p-[3.,0.]))
        return -8*p[:,0]**2-2
    result=plan.solve(source,(exact,exact),tolerance=1e-8,source_tolerance=1e-8)
    p=np.array([[3.6,.1],[2.6,.2],[3.1,.5]])
    np.testing.assert_allclose(result.flux(p),exact(p),rtol=1e-7,atol=1e-7)
    assert result.boundary_error<1e-7


def test_nonlinear_profile_fixed_point(plan):
    # psi=R^2 has Delta*psi=0; S=lambda*psi*(psi-R^2) is nonlinear.
    # Its cancellation only at the solution exercises profile iteration.
    exact=lambda p:p[:,0]**2
    result=plan.solve_profiles(lambda psi:-.001*psi,lambda psi:.001*psi**2,boundary_flux=(exact,exact),
                               tolerance=1e-7,source_tolerance=1e-8)
    p=np.array([[3.6,.1],[2.6,.2]])
    np.testing.assert_allclose(result.flux(p),exact(p),rtol=2e-7,atol=1e-7)


def test_failure_is_explicit(plan):
    with pytest.raises(RuntimeError,match='Picard iteration failed'):
        plan.solve(1.,max_iterations=1,tolerance=1e-12)
    with pytest.raises(ValueError,match='R>0'):
        SplineAnnulusGSPlan(plan.domain,major_radius=.1)
    with pytest.raises(ValueError,match='interior'):
        result=plan.solve(0.)
        result.flux(np.array([[3.,0.]]))  # excluded hole


def test_loose_iteration_tolerance_cannot_bypass_final_source_gate(plan):
    # A deliberately unresolved source must fail even when the PDE residual
    # tolerance alone would accept the iterate.
    with pytest.raises(RuntimeError,match='source_error'):
        plan.solve(lambda p: np.cos(120*p[:,0]), tolerance=1e6,
                   source_tolerance=1e-8, max_iterations=2)


def test_gs_iteration_never_builds_target_matrices(plan, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('GS must apply the layer without dense target operators')
    monkeypatch.setattr(plan.poisson, 'potential_matrix', forbidden)
    result = plan.solve(0., (0., 0.), tolerance=1e-8)
    np.testing.assert_allclose(result.flux(np.array([[3.6, .1], [2.6, .2]])), 0., atol=1e-13)
    assert '_potentials' not in vars(plan)
