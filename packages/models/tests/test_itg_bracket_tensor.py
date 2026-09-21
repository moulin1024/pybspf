"""Equivalent BSPF weak-form contraction, not a new spatial discretization."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from bspf_models.kinetic.collisional_itg import plan_collisional_itg
from bspf_models.kinetic.collisional_itg import collisional_itg_rhs
from bspf_models.kinetic.collisional_itg import collisional_itg_rates
from bspf_models.kinetic.collisional_itg import integrate_collisional_itg
from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_initial
from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_project
from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_diagnostics
from bspf_models.kinetic.itg_bracket_tensor import plan_itg_bracket_tensor
from bspf_models.kinetic.itg_bracket_tensor import collisional_itg_tensor_rhs
from bspf_models.kinetic.itg_bracket_tensor import integrate_collisional_itg_tensor


def test_tensor_matches_quadrature_and_energy_identity():
    p=plan_collisional_itg(n_x=17,n_v=4,n_mu=3,a_t=4.)
    tensor=plan_itg_bracket_tensor(p.base.radial)
    x=nonlinear_itg_initial(p.base,amplitude=.7)
    x+=nonlinear_itg_project(p.base,jnp.asarray(np.random.default_rng(42).normal(size=x.shape))*.01)
    a=collisional_itg_rhs(p,x);b=collisional_itg_tensor_rhs(p,tensor,x)
    np.testing.assert_allclose(a,b,atol=1e-14,rtol=1e-12)
    dw=jax.jvp(lambda u:nonlinear_itg_diagnostics(p.base,u)[2],(x,),(b,))[1]
    rates=collisional_itg_rates(p,x)
    np.testing.assert_allclose(dw,rates[0]+rates[1]-rates[2],atol=2e-14,rtol=1e-12)
    h,t,w=integrate_collisional_itg(p,x,.05,steps=10,save_every=10)
    hh,tt,ww=integrate_collisional_itg_tensor(p,tensor,x,.05,steps=10,save_every=10)
    np.testing.assert_allclose(h,hh,atol=3e-14,rtol=1e-12)
    np.testing.assert_allclose(w,ww,atol=1e-14,rtol=1e-12)
