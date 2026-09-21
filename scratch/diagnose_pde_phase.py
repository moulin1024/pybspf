"""Separate PDE quadrature error from implicit-midpoint dispersion."""

import pybspf.galerkin as bspf_galerkin
import pybspf.plans as bspf_plans
from diagnose_pde_weak import resolved
import jax
import jax.numpy as j
import numpy as np
from scipy.linalg import eigh


def schrodinger(n=257):
 x=j.linspace(0,20,n);t=np.linspace(0,2.5,51)
 p=bspf_plans.plan_1d(x,degree=7,n_basis=24,boundary_points=9);weak=bspf_galerkin.galerkin_1d(p)
 psi=np.array(j.exp(5j*x-(x-10)**2)/(j.pi/2)**.25)
 k=np.arange(128)*np.pi/20
 c=np.sqrt(np.pi)/20*(np.exp(-(5+k)**2/4+1j*(5+k)*10)+np.exp(-(5-k)**2/4+1j*(5-k)*10))/(np.pi/2)**.25
 c[0]*=.5
 exact=(np.exp(-1j*t[:,None]*k**2)*c)@np.cos(k[:,None]*np.array(x))
 M,K,_=resolved(p,weak,1)
 for name,mass,stiff in [('trapezoid',weak.mass,weak.stiffness),('gauss',M,K)]:
  lam,V=eigh(np.array(stiff),np.array(mass));c=V.T@np.array(mass)@psi
  for dt in [0,.001,.0005,.0001]:
   phase=lam if dt==0 else 2*np.arctan(dt*lam/2)/dt
   sol=(np.exp(-1j*t[:,None]*phase)*c)@V.T
   print(f'schrodinger N={n} {name} dt={dt:g} max_error={np.max(np.abs(sol-exact)):.6e}',flush=True)


def beam(n=33):
 x=j.linspace(0,1,n);t=np.linspace(0,3,101)
 p=bspf_plans.plan_1d(x,degree=5,n_basis=16,boundary_points=7)
 weak=bspf_galerkin.galerkin_1d(p,derivative_order=2,constraints=((0,0),(0,1)))
 roots=np.concatenate(([1.875104068711961,4.694091132974175,7.854757438237613,10.995540734875467,14.13716839104647,17.27875965739948],(np.arange(6,10)+.5)*np.pi))
 def modes(z):
  r=roots[:,None];s=(np.cosh(r)+np.cos(r))/(np.sinh(r)+np.sin(r))
  # Stable equivalent to cosh(rz)-s*sinh(rz)-cos(rz)+s*sin(rz).
  sinh_r=np.sinh(r);cosh_r=np.cosh(r)
  one_minus_s=(np.sin(r)-np.cos(r)-np.exp(-r))/(sinh_r+np.sin(r))
  return .5*(one_minus_s*np.exp(r*z)+(1+s)*np.exp(-r*z))-np.cos(r*z)+s*np.sin(r*z)
 def static(z):return z**2*(z**2-4*z+6)/24
 g,w=np.polynomial.legendre.leggauss(256);z=(g+1)/2;w=w/2
 phi=modes(z);c=(-static(z)*phi)@w/((phi**2)@w)
 exact=static(np.array(x))+(np.cos(t[:,None]*roots**2)*c)@modes(np.array(x))
 M,K,F=resolved(p,weak,2)
 for name,mass,stiff,force in [('trapezoid',weak.mass,weak.stiffness,weak.extension.T@p.weights),('gauss',M,K,F)]:
  lam,V=eigh(np.array(stiff),np.array(mass));omega=np.sqrt(lam)
  static_coeff=V.T@np.array(force)/lam
  for dt in [0,.0005,.00025,.0001]:
   phase=omega if dt==0 else 2*np.arctan(dt*omega/2)/dt
   sol=((1-np.cos(t[:,None]*phase))*static_coeff)@V.T@np.array(weak.extension).T
   print(f'beam N={n} {name} dt={dt:g} max_error={np.max(np.abs(sol-exact)):.6e}',flush=True)

if __name__=='__main__':
 beam()
 schrodinger()
