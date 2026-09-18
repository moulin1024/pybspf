"""Experimental pointwise-divergence-free BSPF streamfunction Galerkin NS.

Only 1D factors are assembled. Fixed Dirichlet velocity, clamped streamfunction
increments. Pressure eliminated exactly by divergence-free test functions.
"""
from types import SimpleNamespace
from typing import NamedTuple
import numpy as np
import scipy.linalg as la
from scipy.interpolate import BSpline
from scipy.special import roots_legendre
import jax
import jax.numpy as jnp
from bspf_jax.pressure import _make_line
from bspf_jax._weak_basis import mp_trial_values


class StreamLine(NamedTuple):
    x: jax.Array
    points: jax.Array
    weights: jax.Array
    b: jax.Array
    g: jax.Array
    h: jax.Array
    bn: jax.Array
    gn: jax.Array
    hn: jax.Array
    lam: jax.Array
    bending: jax.Array


class StreamPlan(NamedTuple):
    x: StreamLine
    y: StreamLine
    denominator: jax.Array
    nu: jax.Array


def line(x, *, quadrature_order=None):
    x=np.asarray(x)
    original=_make_line(x,9,32,13,16,'chebyshev',12,1e-12)
    host=SimpleNamespace(x=x,P=np.asarray(original.projector))
    breaks=np.linspace(x[0],x[-1],20)
    knots=np.r_[np.repeat(x[0],14),breaks[1:-1],np.repeat(x[-1],14)]
    spline=BSpline(knots,np.eye(32),13)
    order=quadrature_order or max(24,int(np.ceil(1.5*np.pi*(len(x)-1)/19))+8)
    q,w=roots_legendre(order)
    points=np.concatenate([(a+b)/2+(b-a)/2*q for a,b in zip(breaks[:-1],breaks[1:])])
    weights=np.concatenate([(b-a)/2*w for a,b in zip(breaks[:-1],breaks[1:])])
    bn,gn,hn=mp_trial_values(host,spline,x,second=True)
    b,g,h=mp_trial_values(host,spline,points,second=True)
    z=np.zeros((len(x),len(x)-4))
    z[1:-1]=la.null_space(gn[[0,-1],1:-1])
    bc,gc,hc=b@z,g@z,h@z
    mass=bc.T@(weights[:,None]*bc)
    stiff=gc.T@(weights[:,None]*gc)
    lam,v=la.eigh(stiff,mass)
    transform=z@v
    bc,gc,hc=b@transform,g@transform,h@transform
    bending=hc.T@(weights[:,None]*hc)
    return StreamLine(*map(jnp.asarray,(x,points,weights,bc,gc,hc,bn@transform,gn@transform,hn@transform,lam,bending)))


def plan(nx=64,ny=64,domain=((-3,3),(-1,1)),nu=.002):
    x=line(np.linspace(*domain[0],nx)); y=line(np.linspace(*domain[1],ny))
    return StreamPlan(x,y,x.lam[:,None]+y.lam[None,:],jnp.asarray(nu))


def velocity(p,a,nodes=False,thickness=None):
    x,y=p.x,p.y
    bx,gx=(x.bn,x.gn) if nodes else (x.b,x.g)
    by,gy=(y.bn,y.gn) if nodes else (y.b,y.g)
    u=bx@a@gy.T
    v=-gx@a@by.T
    if thickness is not None:
        yy=y.x if nodes else y.points
        u=u+jnp.tanh(yy[None,:]/thickness)
    return jnp.stack((u,v),axis=-1)


def vorticity(p,a,nodes=False,thickness=None):
    x,y=p.x,p.y
    bx,hx=(x.bn,x.hn) if nodes else (x.b,x.h)
    by,hy=(y.bn,y.hn) if nodes else (y.b,y.h)
    omega=-hx@a@by.T-bx@a@hy.T
    if thickness is not None:
        yy=y.x if nodes else y.points
        omega=omega-(1-jnp.tanh(yy[None,:]/thickness)**2)/thickness
    return omega


def load(p,force):
    x,y=p.x,p.y
    w=x.weights[:,None]*y.weights[None,:]
    return x.b.T@(w*force[...,0])@y.g-x.g.T@(w*force[...,1])@y.b


def rhs(p,a,force=None,thickness=None):
    u=velocity(p,a,thickness=thickness)
    omega=vorticity(p,a,thickness=thickness)
    conv=load(p,jnp.stack((u[...,1]*omega,-u[...,0]*omega),axis=-1))
    diff= p.x.bending@a + a@p.y.bending.T+2*p.x.lam[:,None]*a*p.y.lam[None,:]
    result=conv-p.nu*diff
    if thickness is not None:
        yy=p.y.points[None,:]
        th=jnp.tanh(yy/thickness)
        upp=-2*th*(1-th*th)/thickness**2
        result=result+p.nu*(p.x.b.T@(p.x.weights[:,None]*jnp.ones((len(p.x.weights),1)))) @ ((p.y.weights[None,:]*upp)@p.y.g)
    if force is not None:
        result=result+force
    return result/p.denominator


def rk4(p,a,dt,force=None,thickness=None):
    f=lambda v:rhs(p,v,force,thickness)
    k1=f(a); k2=f(a+dt/2*k1); k3=f(a+dt/2*k2); k4=f(a+dt*k3)
    return a+dt/6*(k1+2*k2+2*k3+k4)


def kh_seed(p,thickness=.12,amplitude=.03,wavelength=1.5):
    x=p.x.points[:,None]; y=p.y.points[None,:]
    k=2*jnp.pi/wavelength
    ex=(1-(x/3)**2)**4
    dex=-8*x/9*(1-(x/3)**2)**3
    ey=(1-y*y)**4*jnp.exp(-(y/(2*thickness))**2)
    dey=jnp.exp(-(y/(2*thickness))**2)*(-8*y*(1-y*y)**3-(1-y*y)**4*y/(2*thickness**2))
    u=amplitude*thickness*ex*jnp.cos(k*x)*dey
    v=-amplitude*thickness*(dex*jnp.cos(k*x)-k*ex*jnp.sin(k*x))*ey
    return load(p,jnp.stack((u,v),axis=-1))/p.denominator
