"""Nonperiodic BSPF, transparent reservoirs and independent characteristic reference."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest
from bspf_jax.open_slab_packet import (plan_open_slab_packet,packet_reference,
    packet_initial,packet_rhs,packet_diagnostics,packet_reference_moments)
from bspf_jax.fast_axis import FastAxis


def test_source_free_reference_and_zero_incoming_characteristics():
    p=plan_open_slab_packet(49,n_v=16)
    z=jnp.linspace(-3.,3.,37);t=.7
    xt=jax.jvp(lambda time:packet_reference(p,z,time),(t,),(1.,))[1]
    xz=jax.jvp(lambda shift:packet_reference(p,z+shift,t),(0.,),(1.,))[1]
    n=jnp.einsum('v,zvm->zm',p.sqrt_weights,xz)
    axz=p.velocity[None,:,None]*(xz+p.sqrt_weights[None,:,None]*p.coupling[None,None,:]*n[:,None,:])
    np.testing.assert_allclose(xt+axz,0,atol=1e-15)
    face=packet_reference(p,jnp.array([-3.,3.]),1.2)
    a=jnp.einsum('mij,bjm->bmi',p.inverse,face)
    np.testing.assert_allclose(jnp.where(p.speeds>0,a[0],0),0,atol=1e-17)
    np.testing.assert_allclose(jnp.where(p.speeds<0,a[1],0),0,atol=1e-17)
    assert float(jnp.max(jnp.abs(face[0]-face[1])))>1e-5


@pytest.mark.parametrize('endpoint_blend,endpoint_options',[
    (0.,{}),(.5,{}),(.5,dict(endpoint_method='chebyshev',boundary_points=24,chebyshev_modes=12))])
def test_bspf_boundary_particle_and_energy_identity(endpoint_blend,endpoint_options):
    p=plan_open_slab_packet(65,degree=5,n_basis=12,n_v=12,endpoint_blend=endpoint_blend,**endpoint_options)
    assert isinstance(p.axis,FastAxis)
    # Random data exercises incoming-penalty and outgoing parts independently.
    x=jnp.asarray(np.random.default_rng(18).normal(size=(65,12,2))*.003)
    rhs,rate=packet_rhs(p,x)
    derivative=jax.jvp(lambda u:packet_diagnostics(p,u),(x,),(rhs,))[1]
    np.testing.assert_allclose(derivative,jnp.array([-rate[0],-rate[1]-rate[2]]),atol=3e-11)
    assert float(rate[1])>0 and float(rate[2])>0


def test_analytic_particle_and_free_energy_integrals():
    p=plan_open_slab_packet(65,n_v=24)
    actual=packet_diagnostics(p,packet_initial(p))
    # Integrate the exact reference, not its finite-resolution BSPF interpolant.
    ref=packet_reference(p,p.axis.points,0.)
    n=jnp.einsum('v,zvm->zm',p.sqrt_weights,ref)
    reference_integral=jnp.array([jnp.sum(p.axis.weights*n[:,0]),
        .5*jnp.sum(p.axis.weights[:,None]*p.energy_factor[None,:]*
          (jnp.sum(ref*ref,axis=1)+p.coupling*n*n))])
    np.testing.assert_allclose(reference_integral,packet_reference_moments(p,0.),rtol=2e-12,atol=1e-15)
    end=packet_reference_moments(p,1.2)
    assert end[0]<float(actual[0]) and end[1]<float(actual[1])
