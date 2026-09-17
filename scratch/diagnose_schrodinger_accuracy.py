"""Spatial error of the BSPF Schrodinger weak form with exact modal evolution."""
import argparse
from diagnose_pde_weak import resolved
import jax
import jax.numpy as j
import numpy as np
from scipy.linalg import eigh
import bspf_jax as b


def run(n, degree, endpoint, strong=False, n_basis=24, points=None):
 x=j.linspace(0,20,n);t=np.linspace(0,2.5,51)
 opts=dict(degree=degree,n_basis=n_basis,boundary_points=points or degree+2)
 if endpoint=='chebyshev':opts.update(endpoint_method=endpoint,boundary_points=2*(degree+1),chebyshev_modes=degree+1,chebyshev_alpha=0.)
 p=b.plan_1d(x,**opts)
 weak=b.galerkin_1d(p,constraints=((0,1),(1,1)) if strong else ())
 psi=np.array(j.exp(5j*x-(x-10)**2)/(j.pi/2)**.25)
 k=np.arange(128)*np.pi/20
 c=np.sqrt(np.pi)/20*(np.exp(-(5+k)**2/4+1j*(5+k)*10)+np.exp(-(5-k)**2/4+1j*(5-k)*10))/(np.pi/2)**.25;c[0]*=.5
 exact=(np.exp(-1j*t[:,None]*k**2)*c)@np.cos(k[:,None]*np.array(x))
 M,K,_=resolved(p,weak,1)
 lam,V=eigh(np.array(K),np.array(M))
 modal=V.T@np.array(M)@psi[np.array(weak.free)]
 sol=((np.exp(-1j*t[:,None]*lam)*modal)@V.T)@np.array(weak.extension).T
 error=np.abs(sol-exact);idx=np.unravel_index(error.argmax(),error.shape)
 # Interpolation of a single important continuum mode checks trial-space approximation.
 z=j.linspace(0,20,4*n+1);mode=j.cos(32*j.pi*x/20)
 interp=float(j.max(j.abs(b.interpolate(p,mode,z)-j.cos(32*j.pi*z/20))))
 print(dict(n=n,n_basis=n_basis,degree=degree,endpoint=endpoint,strong=strong,error=error.max(),initial_error=error[0].max(),at_t=t[idx[0]],at_x=float(x[idx[1]]),mass_condition=np.linalg.cond(np.array(M)),eigenvalue32_error=lam[32]-(32*np.pi/20)**2,mode32_interpolation_error=interp),flush=True)
 jax.clear_caches()

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--n',type=int,default=257);parser.add_argument('--degree',type=int,default=7);parser.add_argument('--endpoint',default='finite_difference');parser.add_argument('--points',type=int);parser.add_argument('--strong',action='store_true');parser.add_argument('--n-basis',type=int,default=24);args=parser.parse_args();run(args.n,args.degree,args.endpoint,args.strong,args.n_basis,args.points)
