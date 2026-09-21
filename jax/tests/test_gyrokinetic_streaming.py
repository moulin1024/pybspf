"""Velocity streaming must preserve the global field and full nonlinear RHS."""
import importlib.util
from pathlib import Path
from dataclasses import replace
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from bspf_jax.gyrokinetic_slab import (plan_slab_gk,slab_gk_fields,slab_gk_rhs,
    slab_gk_project,slab_gk_rhs_with_field_hat)
from bspf_jax.gyrokinetic_mms import plan_slab_mms


def test_velocity_blocks_match_full_random_nonlinear_rhs():
    p=plan_slab_gk((7,7,7),n_v=6,n_mu=5,rho=.8)
    g=slab_gk_project(p,jnp.asarray(np.random.default_rng(14).normal(size=(7,7,7,6,5))*.02))
    phi,_=slab_gk_fields(p,g)
    ph=jnp.fft.fftn(phi)
    reference=slab_gk_rhs(p,g)
    blocks=[]
    for start in range(0,5,2):
        end=min(start+2,5)
        bp=replace(p,mu=p.mu[start:end],weights=p.weights[:,start:end],j0=p.j0[:,:,start:end])
        gh=jnp.fft.fftn(g[:,:,:,:,start:end],axes=(0,1,2))
        blocks.append(slab_gk_rhs_with_field_hat(bp,gh,ph))
    np.testing.assert_allclose(jnp.concatenate(blocks,axis=-1),reference,atol=2e-16)


def test_streamed_mms_metrics_match_full_arrays_and_partial_blocks():
    path=Path(__file__).resolve().parents[2]/'examples/pde/validate_slab_mms_large.py'
    spec=importlib.util.spec_from_file_location('mms_large',path)
    large=importlib.util.module_from_spec(spec);spec.loader.exec_module(large)
    p=plan_slab_gk((7,7,7),n_v=8,n_mu=8,rho=.8)
    m=plan_slab_mms(p,r=.3)
    g,phi,_,gt,st,br=m.components(.17)
    gp=slab_gk_project(p,g)
    defect=slab_gk_rhs(p,gp)+slab_gk_project(p,gt+st+br)-slab_gk_project(p,gt)
    norm=lambda u:float(jnp.sqrt(jnp.mean(jnp.sum(u*u*p.weights,axis=(-2,-1)))))
    ref=[norm(gp-g),norm(defect),float(jnp.sqrt(jnp.mean((slab_gk_fields(p,gp)[0]-phi)**2)))]
    for size in (1,3):
        row=large.run(7,n_v=8,n_mu=8,mu_block=size,v_block=3,verbose=False)
        actual=[row[k] for k in ('projection_error','forced_rhs_defect','phi_error')]
        np.testing.assert_allclose(actual,ref,rtol=2e-12,atol=1e-16)


def test_separable_mms_application_matches_direct_at_two_resolutions():
    path=Path(__file__).resolve().parents[2]/'examples/pde/validate_slab_mms_large.py'
    spec=importlib.util.spec_from_file_location('mms_large_equivalence',path)
    large=importlib.util.module_from_spec(spec);spec.loader.exec_module(large)
    for n in (7,17):
        direct=large.run(n,n_v=12,n_mu=16,v_block=4,method='direct',verbose=False)
        factored=large.run(n,n_v=12,n_mu=16,method='separable',verbose=False)
        for key in ('projection_error','forced_rhs_defect','phi_error'):
            np.testing.assert_allclose(factored[key],direct[key],rtol=3e-10,atol=1e-16)


def test_high_order_laguerre_rule_is_finite_and_normalized():
    p=plan_slab_gk((7,7,7),n_v=12,n_mu=224,rho=.8)
    assert np.all(np.isfinite(p.weights))
    np.testing.assert_allclose(np.sum(p.weights),1.,atol=3e-15)
    np.testing.assert_allclose(np.sum(p.weights*np.asarray(p.mu)[None,:]),1.,atol=3e-14)
