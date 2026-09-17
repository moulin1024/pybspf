from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import bspf_jax as b


def test_rk4_pytree_complex_jit_and_nonuniform_outputs():
    times=jnp.array([0.,.03,.2,.5])
    initial={"decay":jnp.array(1.),"phase":jnp.array(1.+0j)}
    def rhs(t,y):return {"decay":-y["decay"],"phase":1j*y["phase"]}
    result=jax.jit(partial(b.integrate_rk4,rhs,substeps=20))(initial,times)
    np.testing.assert_allclose(result['decay'],jnp.exp(-times),atol=1e-9)
    np.testing.assert_allclose(result['phase'],jnp.exp(1j*times),atol=1e-9)


def test_rk4_fourth_order_and_parameter_gradient():
    errors=[]
    for n in (5,9,17):
        out=b.integrate_rk4(lambda t,y:y,jnp.array(1.),jnp.linspace(0,1,n))
        errors.append(abs(float(out[-1])-np.e))
    assert 12 < errors[0]/errors[1] < 18
    assert 12 < errors[1]/errors[2] < 18
    def final(rate):
        return b.integrate_rk4(lambda t,y:rate*y,jnp.array(1.),jnp.linspace(0,1,101))[-1]
    np.testing.assert_allclose(jax.jit(jax.grad(final))(.3),np.exp(.3),rtol=1e-9)


def test_single_output():
    out=b.integrate_rk4(lambda t,y:-y,jnp.ones(3),jnp.array([0.]))
    np.testing.assert_array_equal(out,jnp.ones((1,3)))


@pytest.mark.parametrize('substeps',[0,1.5,True])
def test_bad_substeps(substeps):
    with pytest.raises(ValueError):b.integrate_rk4(lambda t,y:y,jnp.array(1.),jnp.array([0.,1.]),substeps=substeps)


def test_endpoint_jets_can_override_flux_under_jit():
    x=jnp.linspace(0,1,33)
    p=b.plan_1d(x,degree=3,n_basis=8,boundary_points=5)
    def fit(f):
        jets=b.endpoint_jets(p,f).at[:,1].set(0.)
        return b.decompose(p,f,boundary=jets)
    result=jax.jit(fit)(x*x)
    jets=p.constraint@result.coefficients
    np.testing.assert_allclose(jets[jnp.array([1,3])],0.,atol=1e-10)
