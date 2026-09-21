
import pybspf.basis as bspf_basis
import pybspf.calculus as bspf_calculus
import pybspf.operators as bspf_operators
import pybspf.plans as bspf_plans
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import BSpline

import pybspf as b


@pytest.fixture(scope="module")
def plan():
    return bspf_plans.plan_1d(jnp.linspace(-1, 1, 65), degree=5, n_basis=16,
                     constraint_order=4, boundary_points=7)


@pytest.mark.parametrize("degree", [1, 3, 5])
def test_basis_matches_scipy(degree):
    t = bspf_basis.open_knots(-1., 2., degree=degree, n_basis=degree+5, clustering=1.3)
    x = jnp.linspace(-1, 2, 41)
    ref = BSpline(np.asarray(t), np.eye(degree+5), degree)
    for k in range(degree+2):
        actual = bspf_basis.basis_matrix(t, x, degree=degree, derivative=k)
        expected = ref(x, nu=k) if k <= degree else np.zeros(actual.shape)
        np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=1e-11)
    np.testing.assert_allclose(bspf_basis.basis_matrix(t, x, degree=degree).sum(axis=1), 1, atol=1e-14)


def test_fit_kkt_and_reconstruction(plan):
    f = jnp.sin(3*plan.x) + .1*plan.x
    result = bspf_operators.decompose(plan, f)
    np.testing.assert_allclose(result.spline+result.residual, f, atol=1e-14)
    np.testing.assert_allclose(plan.constraint@result.coefficients, bspf_operators.endpoint_jets(plan, f).reshape(-1), atol=1e-9)


def test_polynomial_derivatives_and_complex_batch(plan):
    x = plan.x
    f = jnp.stack([x**3, 1j*x**3], axis=-1)
    result = jax.jit(partial(bspf_operators.derivatives, orders=(0,1,2,3,4)))(plan, f)
    expected = [x**3,3*x**2,6*x,6*jnp.ones_like(x),jnp.zeros_like(x)]
    for k in range(5):
        np.testing.assert_allclose(result[k], jnp.stack([expected[k],1j*expected[k]],axis=-1), atol=2e-6)


def test_vmap_and_grad(plan):
    f = jnp.sin(plan.x)
    batch = jnp.stack([f,2*f,3*f])
    fn = jax.jit(jax.vmap(lambda v: bspf_operators.differentiate(plan,v)))
    np.testing.assert_allclose(fn(batch), jnp.stack([bspf_operators.differentiate(plan,v) for v in batch]),atol=1e-10)
    energy = lambda v: jnp.sum(bspf_operators.differentiate(plan,v)**2)
    tangent = jnp.cos(2*plan.x)
    grad = jax.jit(jax.grad(energy))(f)
    eps=1e-5
    fd=(energy(f+eps*tangent)-energy(f-eps*tangent))/(2*eps)
    np.testing.assert_allclose(jnp.vdot(grad,tangent),fd,rtol=1e-6,atol=1e-6)


def test_regularization_is_differentiable(plan):
    f=jnp.sin(7*plan.x)
    fun=lambda lam:jnp.sum(bspf_operators.decompose(bspf_plans.with_regularization(plan,lam),f).spline**2)
    grad=jax.jit(jax.grad(fun))(.01)
    fd=(fun(.010001)-fun(.009999))/.000002
    np.testing.assert_allclose(grad,fd,rtol=2e-6)


def test_interpolation_and_primitives(plan):
    x=plan.x
    f=x**3+2*x+1j*x**2
    q=jnp.linspace(-1,1,101)
    np.testing.assert_allclose(jax.jit(bspf_calculus.interpolate)(plan,f,q),q**3+2*q+1j*q**2,atol=1e-11)
    primitive=x**4/4+x**2+1j*x**3/3
    expected=primitive-primitive[0]+2
    np.testing.assert_allclose(jax.jit(bspf_calculus.antiderivative)(plan,f,left_value=2),expected,atol=1e-11)
    integral=jax.jit(bspf_calculus.integrate)(plan,f,a=-.4,b=.7)
    F=lambda t:t**4/4+t**2+1j*t**3/3
    np.testing.assert_allclose(integral,F(.7)-F(-.4),atol=1e-11)


def test_integral_bounds_ad(plan):
    f=jnp.cos(plan.x)
    F=lambda t:bspf_calculus.integrate(plan,f,a=-.4,b=t)
    np.testing.assert_allclose(jax.grad(F)(.3),bspf_calculus.interpolate(plan,f,jnp.array([.3]))[0],atol=1e-9)
    np.testing.assert_allclose(bspf_calculus.integrate(plan,f,a=.4,b=-.2),-bspf_calculus.integrate(plan,f,a=-.2,b=.4),atol=1e-12)
    assert jnp.isnan(bspf_calculus.integrate(plan,f,a=-2,b=0))


@pytest.mark.parametrize("dimension", [2,3])
def test_tensor_calculus(dimension):
    coords=tuple(jnp.linspace(-1,1,n) for n in (17,19,21)[:dimension])
    p=bspf_plans.tensor_plan(*(bspf_plans.plan_1d(x,degree=3,n_basis=8,boundary_points=5) for x in coords))
    mesh=jnp.meshgrid(*coords,indexing='ij')
    f=sum(x**2 for x in mesh)
    np.testing.assert_allclose(jax.jit(bspf_operators.laplacian)(p,f),2*dimension,atol=1e-8)
    g=jax.jit(bspf_operators.gradient)(p,f)
    np.testing.assert_allclose(g,jnp.stack([2*x for x in mesh]),atol=1e-9)
    np.testing.assert_allclose(bspf_operators.divergence(p,g),2*dimension,atol=1e-8)
    np.testing.assert_allclose(bspf_operators.curl(p,g),0,atol=1e-8)
    H=jax.jit(bspf_operators.hessian)(p,f)
    expected=jnp.eye(dimension).reshape((dimension,dimension)+(1,)*dimension)*2
    np.testing.assert_allclose(H,jnp.broadcast_to(expected,H.shape),atol=1e-8)
    components=jax.jit(bspf_operators.tensor_decompose)(p,f)
    assert len(components)==2**dimension
    np.testing.assert_allclose(sum(components.values()),f,atol=1e-12)
    np.testing.assert_allclose(jax.jit(bspf_calculus.integrate_box)(p,f),dimension*2**dimension/3,atol=1e-10)
    new=tuple(jnp.linspace(-1,1,9+i) for i in range(dimension))
    exact=sum(x**2 for x in jnp.meshgrid(*new,indexing='ij'))
    np.testing.assert_allclose(jax.jit(bspf_calculus.interpolate_grid)(p,f,new),exact,atol=1e-10)


def test_boundary_jets(plan):
    f=plan.x**2
    bc=jnp.array([[1.,-2.,2.,0.],[1.,2.,2.,0.]])
    s=jax.jit(bspf_operators.decompose)(plan,f,boundary=bc)
    np.testing.assert_allclose(plan.constraint@s.coefficients,bc.reshape(-1),atol=1e-10)


def test_no_correction(plan):
    f=jnp.sin(6*plan.x)
    s=bspf_operators.decompose(plan,f)
    np.testing.assert_allclose(bspf_operators.differentiate(plan,f,correction=False),plan.basis[1]@s.coefficients)


@pytest.mark.parametrize("x", [[0,0,1],[0,2,1],[0,1,3],[0,np.nan,2],[[0,1],[2,3]]])
def test_bad_grid(x):
    with pytest.raises(ValueError):bspf_plans.plan_1d(x)


def test_shape_and_order_errors(plan):
    with pytest.raises(ValueError):bspf_operators.differentiate(plan,jnp.ones(64))
    with pytest.raises(ValueError):bspf_operators.differentiate(plan,jnp.ones(65),order=1.5)
    with pytest.raises(ValueError):bspf_operators.differentiate(plan,jnp.ones(65),axis=1)
    with pytest.raises(ValueError):bspf_plans.plan_1d(plan.x,lam=-1)


@pytest.mark.parametrize("n", [32,33])
@pytest.mark.parametrize("complex_input", [False,True])
def test_nonpolynomial_matches_independent_scipy_kkt(n,complex_input):
    from scipy.linalg import solve
    x=np.linspace(-.3,1.7,n)
    p=bspf_plans.plan_1d(x,degree=3,n_basis=10,constraint_order=2,boundary_points=5,lam=.002)
    ref=BSpline(np.asarray(p.knots),np.eye(10),3)
    B=ref(x)
    w=np.full(n,(x[1]-x[0]));w[[0,-1]]/=2
    C=np.stack([ref(x[0]),ref(x[0],nu=1),ref(x[-1]),ref(x[-1],nu=1)])
    # Independently estimate jets by fitting local polynomials at each endpoint.
    f=np.sin(5*x)+.17*np.cos(9*x)
    if complex_input:f=f+1j*np.exp(-x**2)
    jets=[]
    for ids,at in [(slice(0,5),x[0]),(slice(-5,None),x[-1])]:
        coeff=np.polynomial.polynomial.polyfit(x[ids]-at,f[ids],4)
        jets.extend(coeff[:2])
    Q=B.T@(w[:,None]*B)
    K=np.block([[2*(Q+.002*np.eye(10)),-C.T],[C,np.zeros((4,4))]])
    coeff=solve(K,np.concatenate([2*B.T@(w*f),jets]))[:10]
    residual=f-B@coeff
    for k in (1,2,3,4):
        spline=ref(x,nu=k)@coeff if k<=3 else np.zeros_like(f)
        tail=np.fft.ifft(np.fft.fft(residual)*(2j*np.pi*np.fft.fftfreq(n,d=x[1]-x[0]))**k)
        expected=spline+(tail if complex_input else tail.real)
        np.testing.assert_allclose(bspf_operators.differentiate(p,f,order=k),expected,atol=2e-7,rtol=2e-8)


def test_zero_constraints_and_empty_batches():
    p=bspf_plans.plan_1d(jnp.linspace(0,1,17),degree=3,n_basis=6,constraint_order=0)
    f=jnp.ones((17,0))
    assert bspf_operators.differentiate(p,f).shape == f.shape
    np.testing.assert_allclose(bspf_operators.differentiate(p,jnp.ones(17)),0,atol=1e-11)


def test_second_primitive_and_nonzero_constants(plan):
    x=plan.x
    f=6*x
    expected=x**3-x[0]**3-3*x[0]**2*(x-x[0])+2+4*(x-x[0])
    np.testing.assert_allclose(jax.jit(partial(bspf_calculus.antiderivative,order=2))(plan,f,left_value=2,left_slope=4),expected,atol=1e-10)


def test_nonpolynomial_interpolant_and_integral_consistency(plan):
    f=jnp.sin(13*plan.x)+.2*plan.x
    np.testing.assert_allclose(bspf_calculus.interpolate(plan,f,plan.x),f,atol=1e-12)
    # Fundamental theorem for the SAME interpolant, including a nontrivial FFT residual.
    q=.271
    derivative=jax.jacfwd(lambda t:bspf_calculus.interpolate(plan,f,jnp.array([t]))[0])(q)
    eps=1e-5
    fd=(bspf_calculus.interpolate(plan,f,jnp.array([q+eps]))[0]-bspf_calculus.interpolate(plan,f,jnp.array([q-eps]))[0])/(2*eps)
    np.testing.assert_allclose(derivative,fd,atol=2e-6)
    primitive=bspf_calculus.antiderivative(plan,f)
    np.testing.assert_allclose(primitive[-1],bspf_calculus.integrate(plan,f),atol=1e-11)


def test_slice_dependent_boundary_jets(plan):
    x=plan.x
    f=jnp.stack([x**2,2*x**2],axis=1)
    jets=jnp.array([[1.,-2.,2.,0.],[1.,2.,2.,0.]])
    bc=jnp.stack([jets,2*jets],axis=-1)
    result=jax.jit(bspf_operators.decompose)(plan,f,boundary=bc)
    np.testing.assert_allclose(plan.constraint@result.coefficients,bc.reshape(8,2),atol=1e-10)


def test_tensor_mixed_nonzero_derivative_and_batches():
    x,y,z=(jnp.linspace(-1,1,n) for n in (17,19,21))
    p=bspf_plans.plan_3d(x,y,z,degree=3,n_basis=8,boundary_points=5)
    X,Y,Z=jnp.meshgrid(x,y,z,indexing='ij')
    f=X**2*Y*Z
    batch=jnp.stack([f,2j*f],axis=-1)
    out=jax.jit(partial(bspf_operators.mixed_partial,orders=(1,1,1)))(p,batch)
    exact=jnp.stack([2*X,4j*X],axis=-1)
    np.testing.assert_allclose(out,exact,atol=1e-8)


def test_plan_is_immutable(plan):
    from dataclasses import FrozenInstanceError
    with pytest.raises(FrozenInstanceError):plan.degree=3
    leaves,_=jax.tree_util.tree_flatten(plan)
    assert all(isinstance(a,jax.Array) for a in leaves)


def test_smooth_function_refinement():
    errors=[]
    for n in (33,65,129):
        x=jnp.linspace(0,1,n)
        p=bspf_plans.plan_1d(x,degree=5,n_basis=16,boundary_points=7)
        exact=jnp.exp(x)
        result=bspf_operators.derivatives(p,exact,orders=(1,2))
        errors.append([float(jnp.max(jnp.abs(result[k]-exact))) for k in (1,2)])
    errors=np.asarray(errors)
    assert np.all(errors[1:] < errors[:-1])
    assert errors[-1,0] < 1e-9
    assert errors[-1,1] < 1e-7
