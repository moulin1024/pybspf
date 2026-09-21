"""Independent dense references for FFT/low-rank axes; dense matrices test-only."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import bspf_jax as b
from bspf_jax.fast_axis import (plan_fast_axis, axis_values, axis_adjoint, axis_mass,
    axis_solve, axis_transport, plan_axis_multiplier, axis_apply_multiplier, sample_aligned_knots)
from bspf_jax.basis import basis_matrix
from bspf_jax.operators import decompose


def test_mild_knots_preserve_alignment_and_interior_coverage():
    x=jnp.linspace(-3.,3.,129)
    knots=np.asarray(sample_aligned_knots(x,degree=7,n_basis=16,endpoint_blend=.5))
    breaks=np.unique(knots)
    np.testing.assert_allclose((breaks+3)*128/6,np.rint((breaks+3)*128/6),atol=1e-13)
    np.testing.assert_allclose(breaks,-breaks[::-1],atol=1e-14)
    assert len(breaks)==10
    assert 0 < breaks[1]-breaks[0] < 6/9
    assert np.max(np.diff(breaks)) < 1.
    with pytest.raises(ValueError,match='distinct breaks'):
        sample_aligned_knots(jnp.linspace(0.,1.,10),degree=7,n_basis=16,endpoint_blend=1.)


@pytest.fixture(scope='module', params=[64,65])
def system(request):
    p=b.plan_1d(jnp.linspace(-4.,4.,request.param),degree=7,n_basis=21,boundary_points=9)
    a=plan_fast_axis(p)
    # Independent direct Fourier sums, not shifted FFT implementations.
    split=decompose(p,jnp.eye(p.x.size))
    spec=jnp.fft.fft(split.residual,axis=0)/p.x.size
    phase=jnp.exp(1j*(a.points[:,None]-p.x[0])*p.omega)
    q=basis_matrix(p.knots,a.points,degree=p.degree)@split.coefficients+(phase@spec).real
    g=basis_matrix(p.knots,a.points,degree=p.degree,derivative=1)@split.coefficients+(phase@(1j*p.omega[:,None]*spec)).real
    return p,a,q,g


def test_values_adjoint_mass_inverse_and_ibp(system):
    _,a,q,g=system
    f=jnp.asarray(np.random.default_rng(42).normal(size=(a.x.size,3)))
    h=jnp.asarray(np.random.default_rng(43).normal(size=(a.points.size,3)))
    np.testing.assert_allclose(axis_values(a,f),q@f,atol=1e-9,rtol=1e-10)
    np.testing.assert_allclose(axis_values(a,f,1),g@f,atol=2e-9,rtol=1e-10)
    for k in (0,1):
        np.testing.assert_allclose(jnp.sum(axis_values(a,f,k)*h),jnp.sum(f*axis_adjoint(a,h,k)),atol=2e-9)
    mass=q.T@(a.weights[:,None]*q)
    np.testing.assert_allclose(axis_mass(a,f),mass@f,atol=1e-9,rtol=1e-10)
    np.testing.assert_allclose(mass@axis_solve(a,f),f,atol=1e-9,rtol=1e-10)
    s=g.T@(a.weights[:,None]*q)
    e=jnp.zeros_like(s).at[0,0].set(-1).at[-1,-1].set(1)
    s=.5*(s-s.T+e)
    result=jax.jit(axis_transport)(a,f)
    np.testing.assert_allclose(result,jnp.linalg.solve(mass,s@f),atol=3e-8,rtol=1e-9)
    np.testing.assert_allclose(2*jnp.sum(f*(mass@result),axis=0),f[-1]**2-f[0]**2,atol=1e-8)


@pytest.mark.parametrize('kind',['constant','linear','nonpolynomial'])
def test_toeplitz_multiplication_matches_quadrature(system,kind):
    _,a,q,_=system
    value={'constant':jnp.ones_like(a.points),'linear':a.points,
           'nonpolynomial':jnp.sin(a.points)+.3*jnp.exp(-a.points**2)}[kind]
    op=plan_axis_multiplier(a,value)
    f=jnp.asarray(np.random.default_rng(5).normal(size=(a.x.size,4)))
    mass=q.T@(a.weights[:,None]*q)
    expected=jnp.linalg.solve(mass,q.T@((a.weights*value)[:,None]*(q@f)))
    result=jax.jit(axis_apply_multiplier)(a,op,f)
    np.testing.assert_allclose(result,expected,atol=2e-9,rtol=1e-9)


def test_axis_storage_scales_linearly_and_has_no_full_axis_operator(system):
    p,a,_,_=system
    larger=plan_fast_axis(b.plan_1d(jnp.linspace(-4.,4.,129),degree=7,n_basis=21,boundary_points=9))
    def storage(plan):
        leaves=jax.tree_util.tree_leaves(plan)
        for leaf in leaves:
            assert leaf.shape != (plan.x.size,plan.x.size)
            assert leaf.shape != (plan.points.size,plan.x.size)
        return sum(v.size*v.dtype.itemsize for v in leaves)
    assert storage(larger)<2.1*storage(a)
    assert a.mass_factor.shape==(50+(a.x.size % 2 == 0),)*2
    with pytest.raises(ValueError,match='2.n_basis'):
        plan_fast_axis(b.plan_1d(jnp.linspace(-1.,1.,25),degree=5,n_basis=10))


def test_fast_log_mirror_equilibrium_characteristics_and_flux_balance():
    z,v=jnp.linspace(-2.,2.,33),jnp.linspace(-2.,2.,37)
    p=b.plan_drift_kinetic(b.plan_1d(z,degree=5,knots=sample_aligned_knots(z,degree=5,n_basis=10),boundary_points=7),
        b.plan_1d(v,degree=5,knots=sample_aligned_knots(v,degree=5,n_basis=10),boundary_points=7),magnetic_field=lambda z:1+z*z/2,
        magnetic_gradient=lambda z:z,mu_max=2.,n_mu=4)
    def exact(t,z,v,mu):
        w=jnp.sqrt(mu);c=jnp.cos(w*t);s=jnp.sin(w*t)
        return -.8*(z*c-v*s/w)**2-.6*(v*c+w*z*s-.5)**2-.2*mu
    z3,v3,mu=z[:,None,None],v[None,:,None],p.mu[None,None,:]
    times=jnp.linspace(0.,.2,3)
    g,tr=b.integrate_log_drift_kinetic(p,exact(0,z3,v3,mu),times,log_inflow=exact,substeps=40)
    reference=exact(times[:,None,None,None],z3[None],v3[None],mu[None])
    np.testing.assert_allclose(jnp.exp(g),jnp.exp(reference),atol=2e-8,rtol=0)
    moments,minimum,qminimum=b.log_drift_kinetic_diagnostics(p,g)
    assert np.all(np.asarray(minimum)>=0) and np.all(np.asarray(qminimum)>=0)
    nh=moments[:,jnp.array([0,3])]
    assert float(jnp.max(jnp.abs(nh-nh[0]-tr.sum(axis=-1))/nh[0]))<2e-8
    equilibrium=lambda t,z,v,mu:3+.02*(v*v/2+mu*(1+z*z/2))
    df,_=b.drift_kinetic_rhs(p,0.,equilibrium(0,z3,v3,mu),inflow=equilibrium)
    np.testing.assert_allclose(df,0,atol=2e-7)


def test_default_factory_never_calls_dense_axis_assembly(system, monkeypatch):
    import bspf_jax.drift_kinetic as module
    p,_,_,_=system
    def forbidden(*args, **kwargs):
        raise AssertionError('dense axis assembly was called')
    for name in ('plan_parallel_kinetic','_quadrature_trial','_transport_axis'):
        monkeypatch.setattr(module,name,forbidden)
    plan=b.plan_drift_kinetic(p,p,magnetic_field=lambda x:1+x*x/2,
        magnetic_gradient=lambda x:x,mu_max=2.,n_mu=2)
    assert isinstance(plan,b.FastDriftKineticPlan)
    assert not hasattr(plan,'z_values')
    assert not hasattr(plan,'force_matrix')
