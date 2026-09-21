"""Linearized BGK closures for the finite-domain ITG prototype.

local: C=-nu(I-P)H preserves three local gyrocenter moments.
gyroaveraged: C=-nu(I-B B*)H is the gyroaverage of a particle-space BGK
projector on density, three momenta and energy. B contains J0 and J1;
finite-k gyrocenter moments have collisional transport. Both dissipate W.
H=X+sqrt(w)J phi. Neither model is a Coulomb/Landau collision operator.
"""
from dataclasses import dataclass
from scipy.special import j0,j1
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from .nonlinear_itg import (plan_nonlinear_itg,nonlinear_itg_fields,
    nonlinear_itg_rhs,nonlinear_itg_drive_power,nonlinear_itg_diagnostics)


@partial(jax.tree_util.register_dataclass,data_fields=['base','nu','moments','gyro_moments'],meta_fields=['model'])
@dataclass(frozen=True)
class CollisionalITG:
    base: object
    nu: object
    moments: object
    gyro_moments: object
    model: str


def plan_collisional_itg(base=None, *, nu=.15, model="gyroaveraged", **kwargs):
    if not np.isfinite(nu) or nu<0:raise ValueError('nu must be finite and nonnegative')
    if model not in ['local','gyroaveraged']:raise ValueError('model must be local or gyroaveraged')
    if base is None:base=plan_nonlinear_itg(**kwargs)
    elif kwargs:raise ValueError('base and base-plan keyword arguments are mutually exclusive')
    s=np.asarray(base.sqrt_weights)
    v=np.broadcast_to(np.asarray(base.velocity)[:,None],s.shape)
    energy=v*v/2+np.asarray(base.mu)[None,:]-1.5
    moment=np.stack([s,s*v,s*energy],axis=-1).reshape(-1,3)
    q,_=np.linalg.qr(moment)
    arg=float(base.rho)*np.sqrt(2*(np.asarray(base.radial.eigenvalues)[:,None,None]+np.asarray(base.ky)[None,:,None]**2)*np.asarray(base.mu)[None,None,:])
    J=np.broadcast_to(j0(arg)[:,:,None,:],arg.shape[:2]+s.shape).reshape(arg.shape[:2]+(-1,))
    perp=(s[None,None,:,:]*np.sqrt(2*np.asarray(base.mu))[None,None,None,:]*j1(arg)[:,:,None,:]).reshape(arg.shape[:2]+(-1,))
    B=np.concatenate((J[:,:,:,None]*q[None,None,:,:],perp[:,:,:,None]),axis=-1)
    # Gyroaverage of an orthogonal particle-space Maxwellian projector.
    if np.max(np.linalg.eigvalsh(np.einsum('ryvi,ryvj->ryij',B,B)))>1+1e-12:
        raise ValueError('gyroaveraged BGK must be contractive in velocity norm')
    return CollisionalITG(base,jnp.asarray(nu),jnp.asarray(q),jnp.asarray(B),model)


def collision_remainder(p,x):
    """Return (I-P)H in weighted velocity coordinates, real in y/z."""
    _,psi=nonlinear_itg_fields(p.base,x)
    h=x+psi[:,:,:,None,:]*p.base.sqrt_weights
    hflat=h.reshape(h.shape[:-2]+(-1,))
    return (hflat-(hflat@p.moments)@p.moments.T).reshape(h.shape)


def _gyro_collision_parts(p,x):
    b=p.base;phi,_=nonlinear_itg_fields(b,x)
    h=jnp.fft.fftn(x,axes=(1,2),norm='ortho')+b.gyro[:,:,None,None,:]*phi[:,:,:,None,None]*b.sqrt_weights
    h=h.reshape(h.shape[:-2]+(-1,))
    moments=jnp.einsum('ryvk,ryzv->ryzk',p.gyro_moments,h)
    rem=h-jnp.einsum('ryvk,ryzk->ryzv',p.gyro_moments,moments)
    return h,rem


def collisional_itg_collision(p,x):
    if p.model=='local':return -p.nu*collision_remainder(p,x)
    _,rem=_gyro_collision_parts(p,x)
    return -p.nu*jnp.fft.ifftn(rem.reshape(x.shape),axes=(1,2),norm='ortho').real


def collisional_itg_rhs(p,x, *, include_nonlinear=True):
    return nonlinear_itg_rhs(p.base,x,include_nonlinear=include_nonlinear)+collisional_itg_collision(p,x)


def collisional_itg_rates(p,x):
    """Density drive, temperature drive, positive collisional dissipation.

    dW/dt = P_n + P_T - D. All terms use radial integral and y/z mean.
    """
    drive=nonlinear_itg_drive_power(p.base,x)
    if p.model=='local':
        r=collision_remainder(p,x);loss=p.nu*jnp.sum(r*r)
    else:
        h,r=_gyro_collision_parts(p,x);loss=p.nu*jnp.real(jnp.vdot(h,r))
    loss=loss/(p.base.ky.size*p.base.kz.size)
    return jnp.concatenate((drive,loss[None]))


def collisional_itg_transport(p,x):
    """Particle flux and temperature-gradient-conjugate heat flux.

    These are radial integrals of y/z-averaged correlations, not SI fluxes.
    Q = <g v_Ex (epsilon-3/2)>; thus P_T = a_T Q, including when a_T=0.
    """
    b=p.base
    phi,_=nonlinear_itg_fields(b,x)
    xh=jnp.fft.fftn(x,axes=(1,2),norm='ortho')
    flux=1j*b.rho*b.ky[None,:,None,None,None]*b.gyro[:,:,None,None,:]*phi[:,:,:,None,None]*b.sqrt_weights
    thermal=b.velocity[None,None,None,:,None]**2/2+b.mu[None,None,None,None,:]-1.5
    size=b.ky.size*b.kz.size
    return jnp.stack((jnp.real(jnp.vdot(xh,flux)),jnp.real(jnp.vdot(xh,thermal*flux))))/size


@partial(jax.jit,static_argnames=['steps','save_every','include_nonlinear'])
def integrate_collisional_itg(p,x,dt,*,steps,save_every=10,include_nonlinear=True):
    """RK4 with same-stage drive and collision work; no state correction."""
    if steps<1 or save_every<1 or steps%save_every:
        raise ValueError('steps must be positive and divisible by save_every')
    def rhs(u):return collisional_itg_rhs(p,u,include_nonlinear=include_nonlinear),collisional_itg_rates(p,u)
    def block(carry,_):
        def step(_,carry):
            u,work=carry
            a,pa=rhs(u);b,pb=rhs(u+dt*a/2);c,pc=rhs(u+dt*b/2);d,pd=rhs(u+dt*c)
            return u+dt*(a+2*b+2*c+d)/6,work+dt*(pa+2*pb+2*pc+pd)/6
        carry=jax.lax.fori_loop(0,save_every,step,carry)
        return carry,carry
    zero=jnp.zeros(3,dtype=x.dtype)
    _,(h,w)=jax.lax.scan(block,(x,zero),None,length=steps//save_every)
    return (jnp.concatenate((x[None],h)),jnp.arange(steps//save_every+1)*dt*save_every,
            jnp.concatenate((zero[None],w)))
