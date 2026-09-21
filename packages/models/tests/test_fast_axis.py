import bspf_models.kinetic.drift_kinetic as bspf_drift_kinetic
import bspf_models.kinetic.fast_drift_kinetic as bspf_fast_drift_kinetic
import pybspf.plans as bspf_plans
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pybspf.fast_axis import plan_fast_axis
from pybspf.fast_axis import axis_values
from pybspf.fast_axis import axis_adjoint
from pybspf.fast_axis import axis_mass
from pybspf.fast_axis import axis_solve
from pybspf.fast_axis import axis_transport
from pybspf.fast_axis import plan_axis_multiplier
from pybspf.fast_axis import axis_apply_multiplier
from pybspf.fast_axis import sample_aligned_knots
from pybspf.basis import basis_matrix
from pybspf.operators import decompose

def test_fast_log_mirror_equilibrium_characteristics_and_flux_balance():
    z,v=jnp.linspace(-2.,2.,33),jnp.linspace(-2.,2.,37)
    p=bspf_drift_kinetic.plan_drift_kinetic(bspf_plans.plan_1d(z,degree=5,knots=sample_aligned_knots(z,degree=5,n_basis=10),boundary_points=7),
        bspf_plans.plan_1d(v,degree=5,knots=sample_aligned_knots(v,degree=5,n_basis=10),boundary_points=7),magnetic_field=lambda z:1+z*z/2,
        magnetic_gradient=lambda z:z,mu_max=2.,n_mu=4)
    def exact(t,z,v,mu):
        w=jnp.sqrt(mu);c=jnp.cos(w*t);s=jnp.sin(w*t)
        return -.8*(z*c-v*s/w)**2-.6*(v*c+w*z*s-.5)**2-.2*mu
    z3,v3,mu=z[:,None,None],v[None,:,None],p.mu[None,None,:]
    times=jnp.linspace(0.,.2,3)
    g,tr=bspf_drift_kinetic.integrate_log_drift_kinetic(p,exact(0,z3,v3,mu),times,log_inflow=exact,substeps=40)
    reference=exact(times[:,None,None,None],z3[None],v3[None],mu[None])
    np.testing.assert_allclose(jnp.exp(g),jnp.exp(reference),atol=2e-8,rtol=0)
    moments,minimum,qminimum=bspf_drift_kinetic.log_drift_kinetic_diagnostics(p,g)
    assert np.all(np.asarray(minimum)>=0) and np.all(np.asarray(qminimum)>=0)
    nh=moments[:,jnp.array([0,3])]
    assert float(jnp.max(jnp.abs(nh-nh[0]-tr.sum(axis=-1))/nh[0]))<2e-8
    equilibrium=lambda t,z,v,mu:3+.02*(v*v/2+mu*(1+z*z/2))
    df,_=bspf_drift_kinetic.drift_kinetic_rhs(p,0.,equilibrium(0,z3,v3,mu),inflow=equilibrium)
    np.testing.assert_allclose(df,0,atol=2e-7)


@pytest.mark.parametrize("nodes", [64, 65])
def test_default_factory_never_calls_dense_axis_assembly(nodes, monkeypatch):
    import bspf_models.kinetic.drift_kinetic as module
    p = bspf_plans.plan_1d(jnp.linspace(-4.,4.,nodes), degree=7, n_basis=21, boundary_points=9)
    def forbidden(*args, **kwargs):
        raise AssertionError('dense axis assembly was called')
    for name in ('plan_parallel_kinetic','_quadrature_trial','_transport_axis'):
        monkeypatch.setattr(module,name,forbidden)
    plan=bspf_drift_kinetic.plan_drift_kinetic(p,p,magnetic_field=lambda x:1+x*x/2,
        magnetic_gradient=lambda x:x,mu_max=2.,n_mu=2)
    assert isinstance(plan,bspf_fast_drift_kinetic.FastDriftKineticPlan)
    assert not hasattr(plan,'z_values')
    assert not hasattr(plan,'force_matrix')
