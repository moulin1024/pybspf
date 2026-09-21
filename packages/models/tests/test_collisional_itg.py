"""BGK moments, Maxwellian nullspace, free-energy dissipation and work budget."""
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import pytest
from dataclasses import replace
from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_initial
from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_project
from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_diagnostics
from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_rhs
from bspf_models.kinetic.collisional_itg import plan_collisional_itg
from bspf_models.kinetic.collisional_itg import collisional_itg_collision
from bspf_models.kinetic.collisional_itg import collisional_itg_rhs
from bspf_models.kinetic.collisional_itg import collisional_itg_rates
from bspf_models.kinetic.collisional_itg import collisional_itg_transport
from bspf_models.kinetic.collisional_itg import integrate_collisional_itg


@pytest.fixture(scope='module')
def p():return plan_collisional_itg(n_x=17,n_v=4,n_mu=3,nu=.2,model="local",a_n=.3,a_t=4.)


def random_state(p):
    b=p.base;x=nonlinear_itg_initial(b,amplitude=.3)
    return x+nonlinear_itg_project(b,jnp.asarray(np.random.default_rng(11).normal(size=x.shape))*.02)


def test_collision_moments_and_frequency(p):
    x=random_state(p);c=collisional_itg_collision(p,x)
    moments=c.reshape(c.shape[:-2]+(-1,))@p.moments
    np.testing.assert_allclose(moments,0,atol=2e-16)
    np.testing.assert_allclose(collisional_itg_collision(replace(p,nu=2*p.nu),x),2*c,atol=1e-16)
    np.testing.assert_allclose(collisional_itg_rhs(replace(p,nu=jnp.asarray(0.)),x),nonlinear_itg_rhs(p.base,x),atol=1e-15)
    for nu in [-1,np.nan]:
        with pytest.raises(ValueError):plan_collisional_itg(p.base,nu=nu)


def test_maxwellian_moment_nullspace(p):
    b=p.base;x=random_state(p);flat=x.reshape(x.shape[:-2]+(-1,))
    h=((flat@p.moments)@p.moments.T).reshape(x.shape)
    hh=jnp.fft.fftn(h,axes=(1,2),norm='ortho')
    bb=b.gyro[:,:,None,None,:]*b.sqrt_weights
    # Invert H = X + b (b^T X)/D to prescribe a local Maxwellian H.
    phi=jnp.sum(bb*hh,axis=(-2,-1))/(b.polarization+jnp.sum(bb*bb,axis=(-2,-1)))
    x=jnp.fft.ifftn(hh-bb*phi[:,:,:,None,None],axes=(1,2),norm='ortho').real
    np.testing.assert_allclose(collisional_itg_collision(p,x),0,atol=2e-16)


def test_h_theorem_and_full_budget(p):
    x=random_state(p);c=collisional_itg_collision(p,x)
    dwc=jax.jvp(lambda u:nonlinear_itg_diagnostics(p.base,u)[2],(x,),(c,))[1]
    rates=collisional_itg_rates(p,x)
    assert rates[2]>0
    np.testing.assert_allclose(dwc,-rates[2],rtol=1e-12,atol=1e-15)
    rhs=collisional_itg_rhs(p,x)
    dw=jax.jvp(lambda u:nonlinear_itg_diagnostics(p.base,u)[2],(x,),(rhs,))[1]
    np.testing.assert_allclose(dw,rates[0]+rates[1]-rates[2],rtol=1e-12,atol=1e-15)
    flux=collisional_itg_transport(p,x)
    np.testing.assert_allclose(rates[:2],flux*jnp.array([p.base.a_n,p.base.a_t]),rtol=1e-12,atol=1e-15)


@pytest.mark.parametrize("model",["local","gyroaveraged"])
def test_same_stage_collision_work_convergence(p,model):
    p=plan_collisional_itg(p.base,nu=float(p.nu),model=model)
    x=random_state(p);errors=[]
    for dt,steps in [(.1,10),(.05,20)]:
        h,t,w=integrate_collisional_itg(p,x,dt,steps=steps,save_every=steps)
        d=jax.vmap(lambda u:nonlinear_itg_diagnostics(p.base,u))(h)
        errors.append(abs(float(d[-1,2]-d[0,2]-w[-1,0]-w[-1,1]+w[-1,2])))
        assert float(w[-1,2])>0
    assert errors[0]>10*errors[1]
    assert errors[1]/float(d[-1,2])<1e-6


def test_gyroaveraged_bgk_matches_particle_angle_projection(p):
    from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_fields
    g=plan_collisional_itg(p.base,nu=.2,model='gyroaveraged');b=g.base
    x=random_state(g);phi,_=nonlinear_itg_fields(b,x)
    hh=jnp.fft.fftn(x,axes=(1,2),norm='ortho')+b.gyro[:,:,None,None,:]*phi[:,:,:,None,None]*b.sqrt_weights
    actual=jnp.fft.fftn(collisional_itg_collision(g,x),axes=(1,2),norm='ortho')
    r,y,z=2,1,1;h=np.asarray(hh[r,y,z]).ravel()
    theta=2*np.pi*np.arange(256)/256
    s=np.asarray(b.sqrt_weights);mu=np.broadcast_to(np.asarray(b.mu),s.shape).ravel()
    alpha=float(b.rho)*np.sqrt(2*(float(b.radial.eigenvalues[r])+float(b.ky[y])**2)*mu)
    phase=np.exp(1j*alpha[:,None]*np.cos(theta))
    basis=np.concatenate((np.broadcast_to(np.asarray(g.moments)[:,None,:],(h.size,256,3)),
        (s.ravel()*np.sqrt(2*mu))[:,None,None]*np.stack((np.cos(theta),np.sin(theta)),axis=-1)[None,:,:]),axis=-1)
    particle=h[:,None]*phase.conj()
    moments=np.einsum('vak,va->k',basis,particle)/256
    relaxed=np.einsum('vak,k->va',basis,moments)
    np.testing.assert_allclose(np.einsum('vak,va->k',basis,particle-relaxed)/256,0,atol=2e-15)
    expected=-float(g.nu)*(h-np.mean(phase*relaxed,axis=1))
    np.testing.assert_allclose(np.asarray(actual[r,y,z]).ravel(),expected,atol=2e-15)
    eig=np.linalg.eigvalsh(np.einsum('ryvi,ryvj->ryij',np.asarray(g.gyro_moments),np.asarray(g.gyro_moments)))
    assert eig.max()<=1+1e-12
    rates=collisional_itg_rates(g,x)
    dw=jax.jvp(lambda u:nonlinear_itg_diagnostics(b,u)[2],(x,),(collisional_itg_rhs(g,x),))[1]
    np.testing.assert_allclose(dw,rates[0]+rates[1]-rates[2],atol=1e-14)
    assert rates[2]>0


def test_gyroaveraged_long_wavelength_limit(p):
    b=replace(p.base,rho=jnp.asarray(1e-6))
    g=plan_collisional_itg(b,model='gyroaveraged')
    # At k rho -> 0 the full operator tends to the local moment projection.
    B=np.asarray(g.gyro_moments)[0,0]
    np.testing.assert_allclose(B@B.T,np.asarray(g.moments)@np.asarray(g.moments).T,atol=1e-12)
