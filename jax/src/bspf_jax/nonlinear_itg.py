"""Collisionless multimode ITG with optional fixed-gradient drive.

Standard BSPF radial Galerkin basis, dealiased periodic y/z, full Bessel FLR,
Flux-surface-subtracted adiabatic electrons; zero gradients by default.
No collisions, artificial dissipation, or invariant corrections.
"""
from dataclasses import dataclass
from functools import partial
import numpy as np
from scipy.special import roots_hermitenorm, roots_laguerre, j0
import jax
import jax.numpy as jnp
from .linear_itg import plan_itg_radial


@partial(jax.tree_util.register_dataclass,
    data_fields=['radial','ky','kz','mask','gyro','polarization','sqrt_weights',
                 'velocity','mu','rho','curvature','a_n','a_t'],meta_fields=[])
@dataclass(frozen=True)
class NonlinearITG:
    radial: object
    ky: object
    kz: object
    mask: object
    gyro: object
    polarization: object
    sqrt_weights: object
    velocity: object
    mu: object
    rho: object
    curvature: object
    a_n: object
    a_t: object


def plan_nonlinear_itg(radial=None, *, n_x=33,n_y=9,n_z=7,n_v=8,n_mu=6,
                       ky_min=.3,kz_min=.1,rho=1.,tau=1.,curvature=.2,a_n=0.,a_t=0.):
    """State X=sqrt(w)*g has shape (Nradial,Ny,Nz,Nv,Nmu), real in y/z.

    All radial degrees of freedom are retained. Strict 2/3 Fourier truncation
    defines the periodic Galerkin space, including (ky,kz)=(0,0) zonal modes.
    No Fourier truncation is applied as an energy-removing after-step filter.
    """
    for name,n,minimum in [('n_y',n_y,5),('n_z',n_z,5),('n_v',n_v,2),('n_mu',n_mu,2)]:
        if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if not all(np.isfinite(t) for t in [ky_min,kz_min,rho,tau,curvature,a_n,a_t]) or min(ky_min,kz_min,rho,tau)<=0:
        raise ValueError('positive finite wavenumber spacings, rho and tau required')
    if radial is None:radial=plan_itg_radial(n_x)
    my=np.fft.fftfreq(n_y)*n_y;mz=np.fft.fftfreq(n_z)*n_z
    ky=ky_min*my;kz=kz_min*mz
    mask=(abs(my)[:,None]<n_y/3)&(abs(mz)[None,:]<n_z/3)
    v,wv=roots_hermitenorm(n_v);wv/=np.sqrt(2*np.pi)
    mu,wm=roots_laguerre(n_mu)
    s=np.sqrt(wv[:,None]*wm[None,:])
    gyro=j0(rho*np.sqrt(2*(np.asarray(radial.eigenvalues)[:,None,None]+ky[None,:,None]**2)*mu[None,None,:]))
    gamma=np.sum(gyro*gyro*wm[None,None,:],axis=-1)
    electron=tau*np.ones((n_y,n_z));electron[0,0]=0.
    d=1-gamma[:,:,None]+electron[None,:,:]
    if np.any(d<=0):raise ValueError('Dirichlet polarization must be positive, including zonal modes')
    return NonlinearITG(radial,*map(jnp.asarray,[ky,kz,mask,gyro,d,s,v,mu,rho,curvature,a_n,a_t]))


def _fft(x):return jnp.fft.fftn(x,axes=(1,2),norm='ortho')
def _ifft(x):return jnp.fft.ifftn(x,axes=(1,2),norm='ortho').real


def nonlinear_itg_project(p,x):
    return _ifft(_fft(x)*p.mask[None,:,:,None,None])


def nonlinear_itg_fields(p,x):
    """Fourier modal phi and unweighted gyro(phi), real in periodic coordinates."""
    xh=_fft(x)*p.mask[None,:,:,None,None]
    charge=jnp.sum(xh*p.gyro[:,:,None,None,:]*p.sqrt_weights,axis=(-2,-1))
    phi=charge/p.polarization
    psi=_ifft(p.gyro[:,:,None,:]*phi[:,:,:,None])
    return phi,psi


def _dy(p,x):
    shape=(1,p.ky.size)+(1,)*(x.ndim-2)
    return _ifft(1j*p.ky.reshape(shape)*_fft(x))


def _radial(matrix,x):return jnp.tensordot(matrix,x,axes=(1,0))


def nonlinear_itg_bracket(p,a,b):
    """Fully alternating weak bracket, equivalent to {a,b} when resolved.

    a and b have equal shapes (radial,Ny,Nz,...). Return B(a,b) satisfying
    <c,B(a,b)> = cyclic permutations and antisymmetry in ANY pair. This
    preserves both entropy and field energy for the E x B term, without
    subtracting a computed energy defect. FFT derivatives and weighted
    radial adjoints use exactly the same trial/quadrature operators.
    """
    q=p.radial.values;g=p.radial.derivative_values
    weighted_q=q.T*p.radial.weights[None,:]
    weighted_g=g.T*p.radial.weights[None,:]
    aq=_radial(q,a);ax=_radial(g,a);ay=_radial(q,_dy(p,a))
    bq=_radial(q,b);bx=_radial(g,b);by=_radial(q,_dy(p,b))
    adv=_radial(weighted_q,ax*by-ay*bx)
    radial_adjoint=_radial(weighted_g,bq*ay-aq*by)
    periodic_adjoint=-_dy(p,_radial(weighted_q,aq*bx-bq*ax))
    result=(adv+radial_adjoint+periodic_adjoint)/3
    shape=(1,)+p.mask.shape+(1,)*(result.ndim-3)
    return _ifft(_fft(result)*p.mask.reshape(shape))


def nonlinear_itg_rhs(p,x, *, include_linear=True,include_nonlinear=True):
    """Streaming/drift and configured gradient drive, plus nonlinear advection."""
    phi,psi=nonlinear_itg_fields(p,x)
    result=jnp.zeros_like(x)
    if include_nonlinear:
        # Broadcast over v only after evaluating gyro(phi); there is no 1/sqrt(w).
        result=p.rho*nonlinear_itg_bracket(p,psi[:,:,:,None,:],x)
    if include_linear:
        xh=_fft(x)*p.mask[None,:,:,None,None]
        weighted_psi=p.gyro[:,:,None,None,:]*phi[:,:,:,None,None]*p.sqrt_weights
        motion=p.kz[None,None,:,None,None]*p.velocity[None,None,None,:,None]
        motion=motion+p.rho*p.ky[None,:,None,None,None]*p.curvature*(p.velocity[None,None,None,:,None]**2+p.mu[None,None,None,None,:])
        star=p.rho*p.ky[None,:,None,None,None]*(p.a_n+p.a_t*(p.velocity[None,None,None,:,None]**2/2+p.mu[None,None,None,None,:]-1.5))
        result=result+_ifft(-1j*motion*(xh+weighted_psi)+1j*star*weighted_psi)
    return result


def nonlinear_itg_diagnostics(p,x):
    """Entropy S, field energy E, W=S+E, zonal field E, nonzonal field E.

    Average over y,z, integrate over x and Maxwell-weighted velocity. The
    field-energy split is diagnostic, not an imposed zonal-flow source.
    """
    phi,_=nonlinear_itg_fields(p,x)
    size=p.ky.size*p.kz.size
    entropy=.5*jnp.sum(x*x)/size
    energy_modes=.5*p.polarization*jnp.abs(phi)**2/size
    field=jnp.sum(energy_modes);zonal=jnp.sum(energy_modes[:,0,0])
    return jnp.stack((entropy,field,entropy+field,zonal,field-zonal))


def nonlinear_itg_initial(p, *, amplitude=.3):
    """Three radial modes, multiple y/z modes and velocity dependence; no zonal seed."""
    nr=p.radial.eigenvalues.size
    yy=2*jnp.pi*jnp.arange(p.ky.size)/p.ky.size
    zz=2*jnp.pi*jnp.arange(p.kz.size)/p.kz.size
    y,z=yy[:,None],zz[None,:]
    v=p.velocity[:,None];energy=v*v/2+p.mu[None,:]
    x=jnp.zeros((nr,p.ky.size,p.kz.size,p.velocity.size,p.mu.size))
    x=x.at[0].set(amplitude*jnp.cos(y+z)[:,:,None,None]*p.sqrt_weights*(1+.15*v))
    x=x.at[1].set(.8*amplitude*jnp.sin(y+z)[:,:,None,None]*p.sqrt_weights*(1+.3*(energy-1.5)))
    x=x.at[2].set(.6*amplitude*jnp.cos(2*y-z+.2)[:,:,None,None]*p.sqrt_weights*(1-.2*v))
    return nonlinear_itg_project(p,x)


@partial(jax.jit,static_argnames=['steps','save_every','include_linear','include_nonlinear'])
def integrate_nonlinear_itg(p,x,dt,*,steps,save_every=10,include_linear=True,include_nonlinear=True):
    """RK4 with no clipping, filtering, invariant projection or correction."""
    if steps<1 or save_every<1 or steps%save_every:
        raise ValueError('steps must be positive and divisible by save_every')
    def rhs(u):return nonlinear_itg_rhs(p,u,include_linear=include_linear,include_nonlinear=include_nonlinear)
    def block(u,_):
        def step(_,u):
            a=rhs(u);b=rhs(u+dt*a/2);c=rhs(u+dt*b/2);d=rhs(u+dt*c)
            return u+dt*(a+2*b+2*c+d)/6
        u=jax.lax.fori_loop(0,save_every,step,u)
        return u,u
    _,h=jax.lax.scan(block,x,None,length=steps//save_every)
    return jnp.concatenate((x[None],h)),jnp.arange(steps//save_every+1)*dt*save_every


def nonlinear_itg_drive_power(p,x):
    """Density and temperature gradient work; same normalization as W.

    ky=0 modes receive no direct drive. This diagnostic does not alter X.
    """
    phi,_=nonlinear_itg_fields(p,x)
    xh=_fft(x)*p.mask[None,:,:,None,None]
    psi=p.gyro[:,:,None,None,:]*phi[:,:,:,None,None]*p.sqrt_weights
    flux=1j*p.rho*p.ky[None,:,None,None,None]*psi
    thermal=p.velocity[None,None,None,:,None]**2/2+p.mu[None,None,None,None,:]-1.5
    h=xh+psi
    size=p.ky.size*p.kz.size
    return jnp.stack((p.a_n*jnp.real(jnp.vdot(h,flux)),
                      p.a_t*jnp.real(jnp.vdot(h,thermal*flux))))/size


@partial(jax.jit,static_argnames=['steps','save_every','include_nonlinear'])
def integrate_driven_itg(p,x,dt,*,steps,save_every=10,include_nonlinear=True):
    """RK4 state and same-stage integrated drive work, without energy repair."""
    if steps<1 or save_every<1 or steps%save_every:
        raise ValueError('steps must be positive and divisible by save_every')
    def rhs(u):
        return nonlinear_itg_rhs(p,u,include_nonlinear=include_nonlinear),nonlinear_itg_drive_power(p,u)
    def block(carry,_):
        def step(_,carry):
            u,work=carry
            a,pa=rhs(u);b,pb=rhs(u+dt*a/2);c,pc=rhs(u+dt*b/2);d,pd=rhs(u+dt*c)
            return u+dt*(a+2*b+2*c+d)/6,work+dt*(pa+2*pb+2*pc+pd)/6
        carry=jax.lax.fori_loop(0,save_every,step,carry)
        return carry,carry
    zero=jnp.zeros(2,dtype=x.dtype)
    _,(h,w)=jax.lax.scan(block,(x,zero),None,length=steps//save_every)
    return (jnp.concatenate((x[None],h)),jnp.arange(steps//save_every+1)*dt*save_every,
            jnp.concatenate((zero[None],w)))
