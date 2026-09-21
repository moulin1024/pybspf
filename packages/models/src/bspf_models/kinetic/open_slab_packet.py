"""Source-free ion-acoustic packet escaping a nonperiodic BSPF interval.

Uniform-B electrostatic delta-f ions with adiabatic electrons, ky=0. Modes
kperp=0 and kperp>0 are independent; the latter retains full Bessel FLR.
g_k(z,v,mu)=J0(k rho sqrt(2 mu))*u_k(z,v) is an invariant subspace of this
linear problem. Evolve x=sqrt(w_v)*u for well-conditioned velocity norms.
The open ends admit no incoming plasma characteristics. No volume forcing.
"""
from dataclasses import dataclass
from functools import partial
import numpy as np
from scipy.special import roots_laguerre,roots_hermitenorm,j0
import jax
import jax.numpy as jnp
from pybspf.plans import plan_1d
from pybspf.fast_axis import plan_fast_axis
from pybspf.fast_axis import sample_aligned_knots
from pybspf.fast_axis import axis_transport
from pybspf.fast_axis import axis_lift
from pybspf.fast_axis import axis_values


@partial(jax.tree_util.register_dataclass,
    data_fields=['axis','velocity','sqrt_weights','mu','mu_weights','gyro',
                 'gamma','coupling','energy_factor','speeds','right','inverse'],
    meta_fields=[])
@dataclass(frozen=True)
class OpenSlabPacket:
    axis: object
    velocity: object
    sqrt_weights: object
    mu: object
    mu_weights: object
    gyro: object
    gamma: object
    coupling: object
    energy_factor: object
    speeds: object
    right: object
    inverse: object


def plan_open_slab_packet(n_z=65,*,degree=7,n_basis=16,n_v=128,n_mu=32,
                         quadrature_order=12,length=6.,rho=.8,tau=1.,
                         velocity_rule="legendre",v_max=8.,endpoint_blend=0.,
                         endpoint_method="finite_difference",boundary_points=None,
                         chebyshev_modes=None):
    """Build the physical packet model with independently configurable endpoint fits.

    The historical default is degree+2 finite-difference samples. Chebyshev
    mode/window validation and defaults are delegated to plan_1d. Knot grading
    and endpoint estimation are separate choices; neither changes the FFT grid.
    """
    if n_v<2 or n_mu<2 or length<=0 or rho<0 or tau<=0:
        raise ValueError('invalid grid or physical parameter')
    z=jnp.linspace(-length/2,length/2,n_z)
    zp=plan_1d(z,degree=degree,knots=sample_aligned_knots(z,degree=degree,n_basis=n_basis,endpoint_blend=endpoint_blend),
               boundary_points=(degree+2 if boundary_points is None and endpoint_method=="finite_difference" else boundary_points),
               endpoint_method=endpoint_method,chebyshev_modes=chebyshev_modes)
    axis=plan_fast_axis(zp,quadrature_order=quadrature_order)
    if velocity_rule=="hermite":
        v,w=roots_hermitenorm(n_v);w/=np.sqrt(2*np.pi)
    elif velocity_rule=="legendre":
        if v_max<=0: raise ValueError('v_max must be positive')
        v,w=np.polynomial.legendre.leggauss(n_v)
        v*=v_max;w*=v_max*np.exp(-v*v/2)/np.sqrt(2*np.pi)
    else:
        raise ValueError('velocity_rule must be legendre or hermite')
    s=np.sqrt(w);squared_norm=w.sum()
    mu,wm=roots_laguerre(n_mu)
    gyro=j0(np.array([0.,1.])[:,None]*rho*np.sqrt(2*mu)[None,:])
    gamma=np.sum(gyro**2*wm,axis=1)
    coupling=gamma/(tau+1-gamma)
    speeds=[];right=[];inverse=[]
    for lam in coupling:
        P=np.eye(n_v)+((np.sqrt(1+lam*squared_norm)-1)/squared_norm)*np.outer(s,s)
        Pi=np.eye(n_v)+((1/np.sqrt(1+lam*squared_norm)-1)/squared_norm)*np.outer(s,s)
        C,U=np.linalg.eigh((P*v[None,:])@P)
        speeds.append(C);right.append(Pi@U);inverse.append(U.T@P)
    return OpenSlabPacket(axis,*map(jnp.asarray,[v,s,mu,wm,gyro,gamma,coupling,
        gamma*np.array([1.,.5]),np.array(speeds),np.array(right),np.array(inverse)]))


def packet_shape(z):
    """C9 compact density packet inside (-2.2,1.4), zero outside."""
    r=(z+.4)/1.8
    return jnp.maximum(1-r*r,0.)**10


def packet_initial(p):
    velocity=jnp.exp(-.25*(p.velocity-1.1)**2)*p.sqrt_weights
    return packet_shape(p.axis.x)[:,None,None]*velocity[None,:,None]*jnp.array([.01,.003])[None,None,:]


def packet_reference(p,z,t,*,periodic=False):
    """Exact characteristics of the velocity-quadrature system, independent of BSPF.

    This is NOT advertised as an exact continuum-velocity solution. Refining
    the quadrature is a separate check. Initial compact support ensures all
    incoming boundary characteristics are exactly zero for t>=0.
    """
    x0=jnp.exp(-.25*(p.velocity-1.1)**2)*p.sqrt_weights
    amplitudes=jnp.einsum('mij,j->mi',p.inverse,x0)*jnp.array([.01,.003])[:,None]
    foot=jnp.asarray(z)[...,None,None]-t*p.speeds
    if periodic:
        length=p.axis.x[-1]-p.axis.x[0]
        foot=(foot-p.axis.x[0])%length+p.axis.x[0]
    waves=packet_shape(foot)*amplitudes
    return jnp.einsum('mij,...mj->...im',p.right,waves)


def packet_fields(p,x):
    density=jnp.einsum('v,...vm->...m',p.sqrt_weights,x)
    return density*p.coupling


def packet_rhs(p,x):
    density=jnp.einsum('v,zvm->zm',p.sqrt_weights,x)
    flux=p.velocity[None,:,None]*(x+p.sqrt_weights[None,:,None]*p.coupling[None,None,:]*density[:,None,:])
    a_left=jnp.einsum('mij,jm->mi',p.inverse,x[0])
    a_right=jnp.einsum('mij,jm->mi',p.inverse,x[-1])
    left=jnp.einsum('mij,mj->im',p.right,jnp.minimum(p.speeds,0)*a_left)
    right=jnp.einsum('mij,mj->im',p.right,jnp.maximum(p.speeds,0)*a_right)
    result=axis_transport(p.axis,flux)+axis_lift(p.axis,left,right)
    # Total physical particle perturbation comes from kperp=0; cosine averages out.
    particle_out=jnp.dot(p.sqrt_weights,right[:,0]-left[:,0])
    outgoing=.5*jnp.sum(p.energy_factor[:,None]*(jnp.maximum(-p.speeds,0)*a_left**2+jnp.maximum(p.speeds,0)*a_right**2))
    penalty=.5*jnp.sum(p.energy_factor[:,None]*(jnp.maximum(p.speeds,0)*a_left**2+jnp.maximum(-p.speeds,0)*a_right**2))
    return result,jnp.stack((particle_out,outgoing,penalty))


def packet_diagnostics(p,x):
    xq=axis_values(p.axis,x)
    n=jnp.einsum('v,zvm->zm',p.sqrt_weights,xq)
    energy=.5*jnp.sum(p.axis.weights[:,None]*p.energy_factor[None,:]*(jnp.sum(xq*xq,axis=1)+p.coupling*n*n))
    particle=jnp.sum(p.axis.weights*n[:,0])
    return jnp.stack((particle,energy))


@partial(jax.jit,static_argnames=['steps','save_every'])
def integrate_open_packet(p,x,dt,*,steps,save_every=1):
    if steps<1 or save_every<1 or steps%save_every:
        raise ValueError('positive steps must be divisible by save_every')
    zero=jnp.zeros(3,dtype=x.dtype)
    def block(state,_):
        def step(_,state):
            u,budget=state
            a,fa=packet_rhs(p,u);b,fb=packet_rhs(p,u+dt*a/2)
            c,fc=packet_rhs(p,u+dt*b/2);d,fd=packet_rhs(p,u+dt*c)
            return u+dt*(a+2*b+2*c+d)/6,budget+dt*(fa+2*fb+2*fc+fd)/6
        state=jax.lax.fori_loop(0,save_every,step,state)
        return state,state
    _,(history,budgets)=jax.lax.scan(block,(x,zero),None,length=steps//save_every)
    return jnp.concatenate((x[None],history)),jnp.arange(steps//save_every+1)*dt*save_every,jnp.concatenate((zero[None],budgets))


def packet_reference_moments(p,t):
    """Analytic spatial integrals, using incomplete beta functions (no BSPF)."""
    from scipy.special import betainc,beta
    s=np.asarray(p.sqrt_weights);v=np.asarray(p.velocity)
    x0=s*np.exp(-.25*(v-1.1)**2)
    amplitudes=np.einsum('mij,j->mi',np.asarray(p.inverse),x0)*np.array([.01,.003])[:,None]
    speeds=np.asarray(p.speeds)
    def integral(power):
        lo=(float(p.axis.x[0])-t*speeds+.4)/1.8
        hi=(float(p.axis.x[-1])-t*speeds+.4)/1.8
        def primitive(x):
            x=np.clip(x,-1,1)
            return np.sign(x)*.5*beta(.5,power+1)*betainc(.5,power+1,x*x)
        return 1.8*(primitive(hi)-primitive(lo))
    energy=.5*np.sum(np.asarray(p.energy_factor)[:,None]*amplitudes**2*integral(20))
    particle=np.sum((s@np.asarray(p.right)[0])*amplitudes[0]*integral(10)[0])
    return np.array([particle,energy])
