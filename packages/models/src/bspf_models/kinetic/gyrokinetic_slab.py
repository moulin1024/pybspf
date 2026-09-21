"""Periodic electrostatic delta-f slab gyrokinetics, full Bessel FLR.

Dimensionless g=delta F/F0, phi=e*Phi/Ti, v in sqrt(Ti/mi),
mu=mu_physical*B/Ti. F0 velocity measure is normal(v)*exp(-mu).
Adiabatic electrons respond to all nonconstant modes (no zonal subtraction).
"""
from dataclasses import dataclass
from functools import partial
import numpy as np
from scipy.special import j0, roots_laguerre
import jax
import jax.numpy as jnp


@partial(jax.tree_util.register_dataclass,
         data_fields=['x','y','z','v','mu','weights','kx','ky','kz','mask','j0','polarization'],
         meta_fields=['nonlinear'])
@dataclass(frozen=True)
class SlabGKPlan:
    x: object
    y: object
    z: object
    v: object
    mu: object
    weights: object
    kx: object
    ky: object
    kz: object
    mask: object
    j0: object
    polarization: object
    nonlinear: bool


def plan_slab_gk(shape=(12,12,12), *, lengths=(2*np.pi,)*3,
                 n_v=24, n_mu=16, rho=1., tau=1., nonlinear=True):
    """Plan a matrix-free 3x2v slab; tau=Ti/Te, rho=thermal gyroradius.

    rho=0 is the drift-kinetic comparison. Strict 2/3 spectral truncation
    eliminates quadratic aliases, including even-grid Nyquist modes.
    """
    if len(shape)!=3 or any(isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<4 for n in shape):
        raise ValueError('shape must contain three integers >=4')
    if len(lengths)!=3 or not np.all(np.isfinite(lengths)) or min(lengths)<=0:
        raise ValueError('lengths must be positive and finite')
    if any(isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<2 for n in (n_v,n_mu)):
        raise ValueError('velocity quadrature sizes must be integers >=2')
    if not np.isfinite(rho) or rho<0 or not np.isfinite(tau) or tau<=0:
        raise ValueError('rho must be nonnegative and tau positive, both finite')
    coords=[jnp.arange(n)*L/n for n,L in zip(shape,lengths)]
    modes=[np.fft.fftfreq(n)*n for n in shape]
    ks=[jnp.asarray(2*np.pi*m/L) for m,L in zip(modes,lengths)]
    mask=jnp.asarray((np.abs(modes[0])[:,None,None]<shape[0]/3)&
        (np.abs(modes[1])[None,:,None]<shape[1]/3)&(np.abs(modes[2])[None,None,:]<shape[2]/3))
    v,wv=np.polynomial.hermite_e.hermegauss(n_v)
    # NumPy laggauss overflows at high orders (e.g. 192); use stable roots.
    mu,wm=roots_laguerre(n_mu)
    weights=jnp.asarray(wv[:,None]/np.sqrt(2*np.pi)*wm[None,:])
    kp2=np.asarray(ks[0])[:,None]**2+np.asarray(ks[1])[None,:]**2
    gyro=jnp.asarray(j0(rho*np.sqrt(2*kp2[:,:,None]*mu[None,None,:])))
    gamma=jnp.sum(gyro**2*jnp.asarray(wm),axis=-1)
    # Quadrature-consistent Gamma0; approaches exp(-b)*I0(b), b=kperp^2*rho^2.
    d=tau+1-gamma
    return SlabGKPlan(*coords,jnp.asarray(v),jnp.asarray(mu),weights,*ks,mask,gyro,d,bool(nonlinear))


def _fft(a): return jnp.fft.fftn(a,axes=(0,1,2))
def _ifft(a): return jnp.fft.ifftn(a,axes=(0,1,2)).real


def slab_gk_project(p,g):
    """Project a real (nx,ny,nz,nv,nmu) perturbation into retained modes."""
    if g.shape!=(p.x.size,p.y.size,p.z.size,p.v.size,p.mu.size) or jnp.iscomplexobj(g):
        raise ValueError('g must be real with shape (nx,ny,nz,nv,nmu)')
    return _ifft(_fft(g)*p.mask[:,:,:,None,None])


def _fields_hat(p,gh):
    charge=jnp.sum(gh*p.j0[:,:,None,None,:]*p.weights,axis=(-2,-1))
    return slab_gk_solve_charge(p,charge)


def slab_gk_solve_charge(p,charge):
    """Solve for Fourier phi from a fully accumulated gyroaveraged charge.

    For velocity-blocked operation, sum charge over ALL velocity blocks first;
    p.polarization must also come from the full velocity quadrature plan.
    Fourier arrays use the unnormalized forward FFT convention.
    """
    phi=charge/p.polarization[:,:,None]*p.mask
    # Fixed gauge and neutral background: spatially uniform perturbation is
    # conserved but generates no potential, consistent with the energy below.
    return phi.at[0,0,0].set(0)


def slab_gk_fields(p,g):
    """Return phi and gyroaveraged phi (nx,ny,nz,nmu)."""
    phi=_fields_hat(p,_fft(g))
    return _ifft(phi),_ifft(phi[:,:,:,None]*p.j0[:,:,None,:])


def slab_gk_rhs(p,g):
    gh=_fft(g)*p.mask[:,:,:,None,None]
    return slab_gk_rhs_with_field_hat(p,gh,_fields_hat(p,gh))


def slab_gk_rhs_with_field_hat(p,gh,phi_hat):
    """Apply the production RHS to a velocity block with the global field.

    gh is the block's Fourier distribution; phi_hat must be solved using all
    blocks. p may be sliced in v and mu, retaining full spatial axes. Returns
    the real-space RHS. This is also the kernel used by slab_gk_rhs.
    """
    gh=gh*p.mask[:,:,:,None,None]
    psi=phi_hat[:,:,:,None,None]*p.j0[:,:,None,None,:]
    rhs=-1j*p.kz[None,None,:,None,None]*p.v[None,None,None,:,None]*(gh+psi)
    if p.nonlinear:
        dx=1j*p.kx[:,None,None,None,None]
        dy=1j*p.ky[None,:,None,None,None]
        bracket=_ifft(dx*psi)*_ifft(dy*gh)-_ifft(dy*psi)*_ifft(dx*gh)
        rhs-= _fft(bracket)*p.mask[:,:,:,None,None]
    return _ifft(rhs)


def slab_gk_diagnostics(p,g):
    """Volume-mean delta density, entropy, field energy, total free energy,
    minimum sampled F/F0. Delta-f g itself may be negative.
    """
    phi,_=slab_gk_fields(p,g)
    dphi=_ifft(_fft(phi)*p.polarization[:,:,None])
    entropy=.5*jnp.mean(jnp.sum(g*g*p.weights,axis=(-2,-1)))
    field=.5*jnp.mean(phi*dphi)
    number=jnp.mean(jnp.sum(g*p.weights,axis=(-2,-1)))
    return jnp.stack((number,entropy,field,entropy+field,jnp.min(1+g)))


def slab_gk_source_rates(p,g,source):
    """Volume-mean particle injection and free-energy power of a source."""
    _,psi=slab_gk_fields(p,g)
    particle=jnp.mean(jnp.sum(source*p.weights,axis=(-2,-1)))
    power=jnp.mean(jnp.sum((g+psi[:,:,:,None,:])*source*p.weights,axis=(-2,-1)))
    return jnp.stack((particle,power))


@partial(jax.jit,static_argnames=['steps','save_every','source','return_budget'])
def integrate_slab_gk(p,g,dt,*,steps,save_every=1,source=None,return_budget=False):
    """RK4 with spectral projection; optional prescribed analytic source(t).

    Source is evaluated at each RK stage time and projected to retained modes.
    If return_budget=True, also return accumulated (particle, free-energy)
    source transfers, integrated with the same RK stages. No clipping/repair.
    Select dt to resolve retained |kz*v| and nonlinear advection frequencies.
    Returned states include t=0; steps must be divisible by save_every.
    """
    if steps<1 or save_every<1 or steps%save_every:
        raise ValueError('positive steps must be divisible by save_every')
    g=slab_gk_project(p,g)
    def rhs(t,u):
        r=slab_gk_rhs(p,u)
        if source is None:
            return r,jnp.zeros(2,dtype=u.dtype)
        s=slab_gk_project(p,source(t))
        rates=slab_gk_source_rates(p,u,s) if return_budget else jnp.zeros(2,dtype=u.dtype)
        return r+s,rates
    def block(state,block_index):
        def step(i,state):
            u,budget=state
            t=(block_index*save_every+i)*dt
            a,pa=rhs(t,u); b,pb=rhs(t+dt/2,u+dt*a/2)
            c,pc=rhs(t+dt/2,u+dt*b/2); d,pd=rhs(t+dt,u+dt*c)
            return u+dt*(a+2*b+2*c+d)/6,budget+dt*(pa+2*pb+2*pc+pd)/6
        state=jax.lax.fori_loop(0,save_every,step,state)
        return state,state
    zero=jnp.zeros(2,dtype=g.dtype)
    _,(history,budgets)=jax.lax.scan(block,(g,zero),jnp.arange(steps//save_every))
    history=jnp.concatenate((g[None],history))
    times=jnp.arange(steps//save_every+1)*dt*save_every
    if return_budget:
        return history,times,jnp.concatenate((zero[None],budgets))
    return history,times
