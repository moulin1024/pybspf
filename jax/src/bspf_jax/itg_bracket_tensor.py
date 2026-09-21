"""Precontract the same standard-BSPF quadrature for repeated ITG E x B calls.

No approximation or truncation is introduced: the weak trilinear form is
identical to nonlinear_itg_bracket. A small radial tensor replaces repeated
transforms of every velocity column to radial quadrature points.
"""
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from .nonlinear_itg import nonlinear_itg_fields,nonlinear_itg_rhs
from .collisional_itg import collisional_itg_collision,collisional_itg_rates


def plan_itg_bracket_tensor(radial):
    q=np.asarray(radial.values);g=np.asarray(radial.derivative_values);w=np.asarray(radial.weights)
    # T_ijk = integral Q_i G_j Q_k. Enforce its exact i,k symmetry against roundoff.
    t=np.einsum('qi,qj,qk,q->ijk',q,g,q,w,optimize=True)
    t=(t+t.transpose(2,1,0))/2
    return jnp.asarray(t-t.transpose(1,0,2))


def _dy(p,x):
    h=jnp.fft.fftn(x,axes=(1,2),norm='ortho')
    return jnp.fft.ifftn(1j*p.ky[None,:,None,None]*h,axes=(1,2),norm='ortho').real


def itg_bracket_tensor(p,tensor,psi,x):
    """Galerkin psi is (radial,y,z,mu); bandlimited X is (radial,y,z,v,mu)."""
    ma=jnp.tensordot(tensor,psi,axes=(1,0))  # (i,k,y,z,mu)
    may=jnp.tensordot(tensor,_dy(p,psi),axes=(1,0))
    xh=jnp.fft.fftn(x,axes=(1,2),norm='ortho')
    dyx=jnp.fft.ifftn(1j*p.ky[None,:,None,None,None]*xh,axes=(1,2),norm='ortho').real
    # The strict 2/3 Galerkin projection makes the product-rule identity exact
    # on retained Fourier modes; use it before projection to save one large FFT.
    one=jnp.einsum('ikyzm,kyzvm->iyzvm',ma+ma.swapaxes(0,1),dyx)
    two=jnp.einsum('ikyzm,kyzvm->iyzvm',2*may.swapaxes(0,1)-may,x)
    out=jnp.fft.fftn((one+two)/3,axes=(1,2),norm='ortho')
    return jnp.fft.ifftn(out*p.mask[None,:,:,None,None],axes=(1,2),norm='ortho').real


def collisional_itg_tensor_rhs(p,tensor,x):
    _,psi=nonlinear_itg_fields(p.base,x)
    return (nonlinear_itg_rhs(p.base,x,include_nonlinear=False)+collisional_itg_collision(p,x)
            +p.base.rho*itg_bracket_tensor(p.base,tensor,psi,x))


@partial(jax.jit,static_argnames=['steps','save_every'])
def integrate_collisional_itg_tensor(p,tensor,x,dt,*,steps,save_every=10):
    """Same RK4 stages and work accounting as integrate_collisional_itg."""
    if steps<1 or save_every<1 or steps%save_every:raise ValueError('invalid step counts')
    def rhs(u):return collisional_itg_tensor_rhs(p,tensor,u),collisional_itg_rates(p,u)
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
