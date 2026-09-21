import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from scipy.special import i0e
from scipy.linalg import expm
import bspf_jax.gyrokinetic_slab as gk


def test_flr_and_drift_limit():
    p=gk.plan_slab_gk(n_mu=40,rho=.7)
    b=(np.asarray(p.kx)[:,None]**2+np.asarray(p.ky)[None,:]**2)*.7**2
    np.testing.assert_allclose(p.polarization,2-i0e(b),atol=2e-10)
    q=gk.plan_slab_gk(rho=0)
    np.testing.assert_allclose(q.j0,1)
    np.testing.assert_allclose(q.polarization,1,atol=1e-14)


def test_rhs_invariants_and_nonzero_nonlinearity():
    p=gk.plan_slab_gk((8,8,8),n_v=8,n_mu=8)
    rng=np.random.default_rng(4)
    g=gk.slab_gk_project(p,jnp.asarray(rng.normal(size=(8,8,8,8,8))*.02))
    rhs=gk.slab_gk_rhs(p,g)
    _,psi=gk.slab_gk_fields(p,g)
    assert abs(float(jnp.mean(jnp.sum(rhs*p.weights,axis=(-2,-1)))))<1e-15
    assert abs(float(jnp.mean(jnp.sum((g+psi[:,:,:,None,:])*rhs*p.weights,axis=(-2,-1)))))<1e-15
    from dataclasses import replace
    assert float(jnp.linalg.norm(rhs-gk.slab_gk_rhs(replace(p,nonlinear=False),g)))>1e-5


def test_linear_mode_against_independent_matrix_exponential():
    p=gk.plan_slab_gk((6,6,6),n_v=8,n_mu=6,rho=.8,nonlinear=False)
    shape=(6,6,6,8,6)
    phase=p.x[:,None,None]+p.z[None,None,:]+jnp.zeros((1,6,1))
    init=jnp.broadcast_to(.01*jnp.cos(phase)[:,:,:,None,None],shape)
    hist,t=gk.integrate_slab_gk(p,init,.005,steps=40,save_every=40)
    J=np.broadcast_to(np.asarray(p.j0[1,0]),(8,6)).ravel()
    v=np.repeat(np.asarray(p.v),6)
    A=-1j*v[:,None]*(np.eye(48)+np.outer(J,J*np.asarray(p.weights).ravel())/float(p.polarization[1,0]))
    exact=expm(.2*A)@np.ones(48)*.01
    final=np.fft.fftn(np.asarray(hist[-1]),axes=(0,1,2))[1,0,1].ravel()*2/216
    np.testing.assert_allclose(final,exact,atol=2e-11)
    d=jax.vmap(lambda u:gk.slab_gk_diagnostics(p,u))(hist)
    assert abs(float(d[-1,3]/d[0,3]-1))<1e-10


def test_gyroaverage_ring_and_field_residual():
    p=gk.plan_slab_gk((8,8,8),n_v=6,n_mu=8,rho=.6)
    theta=np.arange(512)*2*np.pi/512
    arg=.6*np.sqrt(2*np.asarray(p.mu))
    ring=np.mean(np.cos(arg[:,None]*np.cos(theta)),axis=1)
    np.testing.assert_allclose(p.j0[1,0],ring,atol=2e-15)
    g=gk.slab_gk_project(p,jnp.asarray(np.random.default_rng(9).normal(size=(8,8,8,6,8))*.01))
    phi,_=gk.slab_gk_fields(p,g)
    gh=np.fft.fftn(g,axes=(0,1,2)); ph=np.fft.fftn(phi)
    charge=np.sum(gh*np.asarray(p.j0)[:,:,None,None,:]*np.asarray(p.weights),axis=(-2,-1))*np.asarray(p.mask)
    charge[0,0,0]=0
    np.testing.assert_allclose(ph*np.asarray(p.polarization)[:,:,None],charge,atol=2e-15)
    assert abs(ph[0,0,0])<1e-15
