"""MMS checks use independent analytic derivatives and continuum field moments."""
from dataclasses import replace
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from bspf_models.kinetic.gyrokinetic_slab import plan_slab_gk
from bspf_models.kinetic.gyrokinetic_slab import slab_gk_rhs
from bspf_models.kinetic.gyrokinetic_slab import slab_gk_fields
from bspf_models.kinetic.gyrokinetic_slab import integrate_slab_gk
from bspf_models.kinetic.gyrokinetic_slab import slab_gk_diagnostics
from bspf_models.kinetic.gyrokinetic_mms import plan_slab_mms


def test_hand_derived_source_against_independent_jvp():
    p=plan_slab_gk((7,7,7),n_v=6,n_mu=8,rho=.8)
    m=plan_slab_mms(p,r=.3)
    t=.19
    g,phi,psi,gt,stream,bracket=m.components(t)
    auto_gt=jax.jvp(lambda s:m.exact(s)[0],(t,),(1.,))[1]
    np.testing.assert_allclose(gt,auto_gt,atol=2e-17)
    def shifted(s,axis):
        mm=replace(m,**{axis:getattr(m,axis)+s})
        c=mm.components(t)
        return c[0],c[2]
    g1,psi1=jax.jvp(lambda s:shifted(s,'theta1'),(0.,),(1.,))[1]
    g2,psi2=jax.jvp(lambda s:shifted(s,'theta2'),(0.,),(1.,))[1]
    np.testing.assert_allclose(stream,m.v*(g1-g2+(psi1-psi2)[:,:,:,None,:]),atol=2e-17)
    np.testing.assert_allclose(bracket,2*(psi1[:,:,:,None,:]*g2-psi2[:,:,:,None,:]*g1),atol=2e-18)
    assert float(jnp.max(jnp.abs(bracket)))>1e-4


def test_continuum_mms_field_and_forced_rhs():
    p=plan_slab_gk((7,7,7),n_v=40,n_mu=32,rho=.8)
    m=plan_slab_mms(p)
    g,phi,psi,gt,stream,bracket=m.components(.17)
    np.testing.assert_allclose(slab_gk_fields(p,g)[0],phi,atol=2e-14)
    residual=slab_gk_rhs(p,g)+m.source(.17)-gt
    np.testing.assert_allclose(residual,0,atol=3e-13)
    # The infinite-harmonic field reference itself must be converged.
    a=plan_slab_mms(p,r=.3,harmonics=16).exact(.17)[1]
    b=plan_slab_mms(p,r=.3,harmonics=32).exact(.17)[1]
    np.testing.assert_allclose(a,b,atol=1e-16)


def test_rk_stage_source_times_and_integrated_budget():
    p=plan_slab_gk((4,4,4),n_v=4,n_mu=4)
    init=jnp.zeros((4,4,4,4,4))
    def source(t): return jnp.ones_like(init)*.01*(1+t+t*t)
    history,t,budget=integrate_slab_gk(p,init,.05,steps=8,save_every=2,
                                    source=source,return_budget=True)
    exact=.01*(t+t*t/2+t**3/3)
    np.testing.assert_allclose(history, jnp.broadcast_to(exact[:,None,None,None,None,None],history.shape),atol=2e-17)
    diag=jax.vmap(lambda u:slab_gk_diagnostics(p,u))(history)
    np.testing.assert_allclose(diag[:,0],budget[:,0],atol=2e-17)
    # Power is degree five in time; RK4 quadrature has a small finite defect.
    assert float(jnp.max(jnp.abs(diag[:,3]-budget[:,1])))<1e-10


def test_analytic_free_energy_and_source_power():
    from bspf_models.kinetic.gyrokinetic_slab import slab_gk_source_rates
    p=plan_slab_gk((7,7,7),n_v=64,n_mu=40,rho=.8)
    m=plan_slab_mms(p)
    g,_=m.exact(.17)
    energy,power=m.exact_energy(.17)
    np.testing.assert_allclose(slab_gk_diagnostics(p,g)[3],energy,rtol=2e-12)
    rates=slab_gk_source_rates(p,g,m.source(.17))
    np.testing.assert_allclose(rates[1],power,rtol=2e-12)
    assert abs(float(rates[0]))<1e-16
