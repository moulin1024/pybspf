"""Independent multimode bracket, adiabatic closure, and conservation checks."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest
from bspf_models.kinetic.nonlinear_itg import plan_nonlinear_itg
from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_initial
from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_project
from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_fields
from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_bracket
from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_rhs
from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_diagnostics
from bspf_models.kinetic.nonlinear_itg import integrate_nonlinear_itg
from bspf_models.kinetic.linear_itg import plan_linear_itg
from bspf_models.kinetic.linear_itg import linear_itg_rhs


@pytest.fixture(scope='module')
def p():return plan_nonlinear_itg(n_x=17,n_v=4,n_mu=3)


def test_zonal_subtraction_only_for_surface_average(p):
    base=1-jnp.sum(p.gyro*p.gyro*(p.sqrt_weights**2).sum(axis=0),axis=-1)
    np.testing.assert_allclose(p.polarization[:,0,0],base[:,0],atol=1e-15)
    np.testing.assert_allclose(p.polarization[:,0,1],base[:,0]+1,atol=1e-15)
    np.testing.assert_allclose(p.polarization[:,1,0],base[:,1]+1,atol=1e-15)
    assert float(jnp.min(p.polarization))>0


def test_full_alternation_and_fourier_galerkin_support(p):
    rng=np.random.default_rng(71)
    fields=[]
    for _ in range(3):
        a=jnp.asarray(rng.normal(size=(p.radial.eigenvalues.size,p.ky.size,p.kz.size,1,1)))
        fields.append(nonlinear_itg_project(p,a)[...,0,0])
    a,b,c=fields
    ab=nonlinear_itg_bracket(p,a,b)
    ba=nonlinear_itg_bracket(p,b,a)
    np.testing.assert_allclose(ab,-ba,atol=2e-13)
    tabc=jnp.vdot(c,ab)
    tbca=jnp.vdot(a,nonlinear_itg_bracket(p,b,c))
    np.testing.assert_allclose(tabc,tbca,atol=2e-12)
    spectrum=jnp.fft.fftn(ab,axes=(1,2))
    assert float(jnp.max(jnp.abs(spectrum*(~p.mask)[None])))<2e-12


def test_bracket_matches_independent_analytic_derivatives():
    p=plan_nonlinear_itg(n_x=33,n_v=4,n_mu=3)
    nr=p.radial.eigenvalues.size
    y=2*jnp.pi*jnp.arange(p.ky.size)/p.ky.size
    z=2*jnp.pi*jnp.arange(p.kz.size)/p.kz.size
    y,z=y[:,None],z[None,:]
    a=jnp.zeros((nr,p.ky.size,p.kz.size)).at[0].set(jnp.cos(y+z))
    b=jnp.zeros_like(a).at[1].set(jnp.sin(y-z))
    x=p.radial.points[:,None,None];k=jnp.pi/12;s=jnp.sqrt(2/12)
    ax=s*k*jnp.cos(k*x)*jnp.cos(y+z)
    ay=-s*p.ky[1]*jnp.sin(k*x)*jnp.sin(y+z)
    bx=2*s*k*jnp.cos(2*k*x)*jnp.sin(y-z)
    by=s*p.ky[1]*jnp.sin(2*k*x)*jnp.cos(y-z)
    expected=jnp.tensordot(p.radial.values.T*p.radial.weights,(ax*by-ay*bx),axes=(1,0))
    actual=nonlinear_itg_bracket(p,a,b)
    np.testing.assert_allclose(actual,expected,atol=2e-10)
    assert float(jnp.linalg.norm(actual))>1e-2


@pytest.mark.parametrize('linear',[False,True])
def test_semidiscrete_energy_identity_without_repairs(p,linear):
    x=nonlinear_itg_initial(p,amplitude=.7)
    noise=jnp.asarray(np.random.default_rng(25).normal(size=x.shape))*.01
    x=x+nonlinear_itg_project(p,noise)
    rhs=nonlinear_itg_rhs(p,x,include_linear=linear)
    rates=jax.jvp(lambda u:nonlinear_itg_diagnostics(p,u),(x,),(rhs,))[1]
    scale=float(jnp.linalg.norm(x)*jnp.linalg.norm(rhs))
    assert abs(float(rates[2]))<1e-12*scale
    if not linear:
        assert float(jnp.max(jnp.abs(rates[:2])))<1e-12*scale
    else:
        assert abs(float(rates[0]))>1e-7
    assert float(jnp.linalg.norm(rhs))>1e-3


def test_recovers_undriven_single_harmonic_linear_model(p):
    x=nonlinear_itg_initial(p)
    xh=jnp.fft.fftn(x,axes=(1,2),norm='ortho')
    actual=jnp.fft.fftn(nonlinear_itg_rhs(p,x,include_nonlinear=False),axes=(1,2),norm='ortho')[:,1,1]
    linear=plan_linear_itg(p.radial,n_v=p.velocity.size,n_mu=p.mu.size,a_n=0.,a_t=0.)
    expected=linear_itg_rhs(linear,xh[:,1,1].reshape(linear.b.shape)).reshape(actual.shape)
    np.testing.assert_allclose(actual[:,...],expected,atol=2e-14)


def test_time_evolution_changes_modes_and_preserves_invariants(p):
    x=nonlinear_itg_initial(p,amplitude=.7)
    h,t=integrate_nonlinear_itg(p,x,.05,steps=10,save_every=5,include_linear=False)
    d=jax.vmap(lambda u:nonlinear_itg_diagnostics(p,u))(h)
    np.testing.assert_allclose((d[:,:3]-d[0,:3])/d[0,:3],0,atol=1e-8)
    assert float(jnp.linalg.norm(h[-1]-x)/jnp.linalg.norm(x))>1e-4
    assert float(d[-1,3]/d[0,2])>1e-10


def test_driven_budget_and_linear_limit(p):
    from dataclasses import replace
    from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_drive_power
    p=replace(p,a_n=jnp.asarray(.4),a_t=jnp.asarray(4.))
    x=nonlinear_itg_initial(p)+nonlinear_itg_project(p,jnp.asarray(np.random.default_rng(29).normal(size=nonlinear_itg_initial(p).shape))*.01)
    rhs=nonlinear_itg_rhs(p,x)
    dw=jax.jvp(lambda u:nonlinear_itg_diagnostics(p,u)[2],(x,),(rhs,))[1]
    power=nonlinear_itg_drive_power(p,x)
    np.testing.assert_allclose(dw,power.sum(),rtol=2e-12,atol=2e-14)
    assert abs(float(power.sum()))>1e-7
    linear=plan_linear_itg(p.radial,n_v=p.velocity.size,n_mu=p.mu.size,a_n=.4,a_t=4.)
    xh=jnp.fft.fftn(x,axes=(1,2),norm='ortho')
    actual=jnp.fft.fftn(nonlinear_itg_rhs(p,x,include_nonlinear=False),axes=(1,2),norm='ortho')
    expected=linear_itg_rhs(linear,xh[:,1,1].reshape(linear.b.shape)).reshape(actual[:,1,1].shape)
    np.testing.assert_allclose(actual[:,1,1],expected,atol=3e-14)
    np.testing.assert_allclose(actual[:,0,0],0,atol=3e-14)


def test_driven_same_stage_work_budget(p):
    from dataclasses import replace
    from bspf_models.kinetic.nonlinear_itg import integrate_driven_itg
    p=replace(p,a_t=jnp.asarray(4.))
    x=nonlinear_itg_initial(p,amplitude=.1)
    errors=[]
    for dt,steps in [(.1,10),(.05,20)]:
        h,t,work=integrate_driven_itg(p,x,dt,steps=steps,save_every=steps)
        d=jax.vmap(lambda u:nonlinear_itg_diagnostics(p,u))(h)
        errors.append(abs(float(d[-1,2]-d[0,2]-work[-1].sum())))
        assert float(work[-1,1])>0
        assert float(work[-1,0])==0
    assert errors[0]>10*errors[1]
    assert errors[1]/float(d[-1,2])<1e-6
