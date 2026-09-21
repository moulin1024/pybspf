"""Independent linear ITG checks: spectra, FLR boundary operator and energy drive."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest
from bspf_jax.linear_itg import (plan_itg_radial,plan_linear_itg,linear_itg_fields,
    linear_itg_rhs,linear_itg_diagnostics,linear_itg_initial,integrate_linear_itg)
from bspf_jax.linear_itg_reference import (growing_root,velocity_generator,
    continuum_dispersion,dispersion)


@pytest.fixture(scope='module')
def radial():return plan_itg_radial(33)


def test_standard_bspf_dirichlet_spectrum_and_mass_adjoint(radial):
    expected=(np.arange(1,5)*np.pi/12)**2
    np.testing.assert_allclose(radial.eigenvalues[:4],expected,rtol=1e-9)
    np.testing.assert_allclose(radial.values.T@(radial.weights[:,None]*radial.values),np.eye(31),atol=2e-12)
    np.testing.assert_allclose(radial.samples[[0,-1],:],0,atol=1e-14)
    exact=np.sqrt(2/12)*np.sin(np.pi*np.asarray(radial.points)/12)
    np.testing.assert_allclose(radial.values[:,0],exact,atol=2e-7)
    p=plan_linear_itg(radial,n_v=12,n_mu=8)
    # A symmetric spectral function of the Galerkin Laplacian is mass-adjoint.
    u=np.random.default_rng(2).normal(size=31);v=np.random.default_rng(3).normal(size=31)
    j=np.asarray(p.b[:,0]/p.sqrt_weights[0])
    left=np.sum(np.asarray(radial.weights)*(radial.values@u)*(radial.values@(j*v)))
    right=np.sum(np.asarray(radial.weights)*(radial.values@(j*u))*(radial.values@v))
    np.testing.assert_allclose(left,right,atol=1e-12)


@pytest.mark.parametrize('a_n,a_t',[(0.,0.),(.7,4.)])
@pytest.mark.parametrize('rho',[1.,.6])
def test_gradient_free_energy_identity(radial,a_n,a_t,rho):
    p=plan_linear_itg(radial,n_v=12,n_mu=8,a_n=a_n,a_t=a_t,rho=rho)
    rng=np.random.default_rng(6)
    x=jnp.asarray(rng.normal(size=p.b.shape)+1j*rng.normal(size=p.b.shape))*.001
    diag=linear_itg_diagnostics(p,x)
    dw=jax.jvp(lambda u:linear_itg_diagnostics(p,u)[2],(x,),(linear_itg_rhs(p,x),))[1]
    np.testing.assert_allclose(dw,diag[3]+diag[4],atol=2e-17)
    np.testing.assert_allclose(diag[3:5],jnp.array([a_n*diag[5],a_t*diag[6]]),atol=2e-17)
    assert float(diag[2])>0


def test_reference_root_independent_matrix_and_time_evolution(radial):
    p=plan_linear_itg(radial,n_v=12,n_mu=8)
    omega=growing_root(n_v=12,n_mu=8)
    eig=np.linalg.eigvals(velocity_generator(n_v=12,n_mu=8))
    assert np.min(np.abs(eig+1j*omega))<1e-10
    x=linear_itg_initial(p,omega=omega)
    h,t,b=integrate_linear_itg(p,x,.002,steps=100,save_every=20)
    np.testing.assert_allclose(h[-1],x*np.exp(-1j*omega*float(t[-1])),rtol=1e-9,atol=1e-18)
    d=jax.vmap(lambda u:linear_itg_diagnostics(p,u))(h)
    np.testing.assert_allclose((d[:,2]-d[0,2]-jnp.sum(b,axis=1))/d[0,2],0,atol=1e-10)


@pytest.mark.parametrize('rho',[1.,.6])
def test_continuous_reference_against_resolved_velocity_quadrature(rho):
    z=.32+.2j
    np.testing.assert_allclose(continuum_dispersion(z,rho=rho),dispersion(z,n_v=192,n_mu=160,rho=rho),atol=2e-7)


def test_no_gradient_has_no_growing_velocity_eigenmode():
    eig=np.linalg.eigvals(velocity_generator(a_n=0.,a_t=0.,n_v=12,n_mu=8))
    assert np.max(np.abs(eig.real))<1e-12


def test_reject_zonal_and_overrefined_grid(radial):
    with pytest.raises(ValueError):plan_itg_radial(193)
    with pytest.raises(ValueError):plan_linear_itg(radial,ky=0.)
