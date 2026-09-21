"""Independent analytic, nonlinear manufactured solution for slab GK.

No calls to the discrete field solver, RHS, FFT, automatic differentiation,
velocity quadrature or finite differences are used to construct the source.
The domain is (2*pi)^3. r=0 is a two-mode solution; 0<r<1 gives analytic
rational spatial profiles with infinitely many Fourier harmonics.
"""
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
from scipy.special import j0, i0e


@dataclass(frozen=True,eq=False)
class SlabMMS:
    theta1: object
    theta2: object
    h1: object
    h2: object
    v: object
    j1: object
    j2: object
    coefficient1: object
    coefficient2: object
    harmonics: object
    r: float
    energy_coefficient1: float
    energy_coefficient2: float

    def exact_energy(self,t):
        """Continuum free energy and its analytic source power (volume means)."""
        A=.03*(1+.2*jnp.sin(.8*t)); Ap=.0048*jnp.cos(.8*t)
        B=.025*(1+.15*jnp.cos(.6*t)); Bp=-.00225*jnp.sin(.6*t)
        energy=(A*A*self.energy_coefficient1+B*B*self.energy_coefficient2)/4
        power=(A*Ap*self.energy_coefficient1+B*Bp*self.energy_coefficient2)/2
        return energy,power

    def components(self,t):
        # Fundamental wavevectors (1,0,1), (0,2,-1), frequencies 0.9,-0.7.
        th1=self.theta1-.9*t; th2=self.theta2+.7*t
        q1=jnp.exp(1j*th1); q2=jnp.exp(1j*th2)
        C=(q1/(1-self.r*q1)).real; S=(q2/(1-self.r*q2)).imag
        Cp=(1j*q1/(1-self.r*q1)**2).real
        Sp=(1j*q2/(1-self.r*q2)**2).imag
        A=.03*(1+.2*jnp.sin(.8*t)); Ap=.0048*jnp.cos(.8*t)
        B=.025*(1+.15*jnp.cos(.6*t)); Bp=-.00225*jnp.sin(.6*t)
        n=self.harmonics[:,None,None,None]
        co1=jnp.cos(n*th1); si1=jnp.sin(n*th1)
        co2=jnp.cos(n*th2); si2=jnp.sin(n*th2)
        phi=A*jnp.einsum('n,nxyz->xyz',self.coefficient1,co1)+B*jnp.einsum('n,nxyz->xyz',self.coefficient2,si2)
        psi=A*jnp.einsum('n,nm,nxyz->xyzm',self.coefficient1,self.j1,co1)+B*jnp.einsum('n,nm,nxyz->xyzm',self.coefficient2,self.j2,si2)
        psi1p=-A*jnp.einsum('n,nm,nxyz->xyzm',self.coefficient1*self.harmonics,self.j1,si1)
        psi2p=B*jnp.einsum('n,nm,nxyz->xyzm',self.coefficient2*self.harmonics,self.j2,co2)
        lift=lambda u:u[:,:,:,None,None]
        g=A*lift(C)*self.h1+B*lift(S)*self.h2
        gt=(Ap*lift(C)-.9*A*lift(Cp))*self.h1+(Bp*lift(S)+.7*B*lift(Sp))*self.h2
        gz=A*lift(Cp)*self.h1-B*lift(Sp)*self.h2
        bracket=2*(psi1p[:,:,:,None,:]*B*lift(Sp)*self.h2-psi2p[:,:,:,None,:]*A*lift(Cp)*self.h1)
        streaming=self.v*(gz+(psi1p-psi2p)[:,:,:,None,:])
        return g,phi,psi,gt,streaming,bracket

    def exact(self,t):
        g,phi,*_=self.components(t)
        return g,phi

    def source(self,t):
        _,_,_,gt,streaming,bracket=self.components(t)
        return gt+streaming+bracket


def plan_slab_mms(p,*,rho=.8,tau=1.,r=0.,harmonics=32):
    """Build a continuum MMS sampled on p's nodes, with analytic velocity moments.

    rho/tau must match the solver plan. A nonzero r uses a rapidly convergent
    analytic series for phi (the distribution itself is an exact rational
    function). Increase harmonics to verify the field reference truncation.
    """
    if not 0<=r<1 or not np.isfinite(rho) or rho<=0 or not np.isfinite(tau) or tau<=0:
        raise ValueError('require 0<=r<1, rho>0 and tau>0')
    if not isinstance(harmonics,int) or harmonics<1:
        raise ValueError('harmonics must be a positive integer')
    for x in (p.x,p.y,p.z):
        if not np.isclose(float(x[1]-x[0])*len(x),2*np.pi):
            raise ValueError('MMS requires a (2*pi)^3 periodic domain')
    n=np.arange(1,(1 if r==0 else harmonics)+1)
    v=p.v[None,None,None,:,None]; mu=p.mu[None,None,None,None,:]
    # Nonpolynomial in both velocity coordinates; all moments are analytic.
    h1=jnp.exp(-.4*v*v-.5*mu)*(1+.2*v)
    h2=jnp.exp(-.7*v*v-1.2*mu)*(1-.15*v)
    def field_coeff(k,a,s):
        b=(k*n*rho)**2
        moment=np.exp(-b/(2*(1+s)))/((1+s)*np.sqrt(1+2*a))
        return jnp.asarray(r**(n-1)*moment/(tau+1-i0e(b)))
    x=p.x[:,None,None]; y=p.y[None,:,None]; z=p.z[None,None,:]
    th1=jnp.broadcast_to(x+z,(p.x.size,p.y.size,p.z.size))
    th2=jnp.broadcast_to(2*y-z,th1.shape)
    J1=jnp.asarray(j0(n[:,None]*rho*np.sqrt(2*np.asarray(p.mu))[None,:]))
    J2=jnp.asarray(j0(2*n[:,None]*rho*np.sqrt(2*np.asarray(p.mu))[None,:]))
    c1=field_coeff(1,.4,.5); c2=field_coeff(2,.7,1.2)
    def energy_coefficient(k,a,s,d,c):
        h_squared=(1/np.sqrt(1+4*a)+d*d/(1+4*a)**1.5)/(1+2*s)
        D=tau+1-i0e((k*n*rho)**2)
        return float(h_squared/(1-r*r)+np.sum(D*np.asarray(c)**2))
    return SlabMMS(th1,th2,h1,h2,v,J1,J2,c1,c2,jnp.asarray(n),float(r),
        energy_coefficient(1,.4,.5,.2,c1),energy_coefficient(2,.7,1.2,-.15,c2))
