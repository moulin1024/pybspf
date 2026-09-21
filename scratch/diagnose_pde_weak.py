"""Compare nodal trapezoid and resolved quadrature for the BSPF trial space."""

import pybspf.basis as bspf_basis
import pybspf.galerkin as bspf_galerkin
import pybspf.operators as bspf_operators
import pybspf.plans as bspf_plans
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as j
import numpy as np
from scipy.linalg import eigh


def resolved(plan, weak, order, gauss_order=8):
    # Split at every knot and data node to resolve polynomial pieces and Fourier modes.
    edges=np.unique(np.concatenate((np.array(plan.x),np.array(plan.knots))))
    a,h=edges[:-1],np.diff(edges)
    g,w=np.polynomial.legendre.leggauss(gauss_order)
    z=j.array((a[:,None]+h[:,None]*(g+1)/2).ravel())
    weights=j.array((h[:,None]*w/2).ravel())
    split=bspf_operators.decompose(plan,weak.extension)
    spectrum=j.fft.fft(split.residual,axis=0)/plan.x.size
    phase=j.exp(1j*(z[:,None]-plan.x[0])*plan.omega)
    Q=bspf_basis.basis_matrix(plan.knots,z,degree=plan.degree)@split.coefficients+(phase@spectrum).real
    G=bspf_basis.basis_matrix(plan.knots,z,degree=plan.degree,derivative=order)@split.coefficients+(phase@((1j*plan.omega[:,None])**order*spectrum)).real
    return Q.T@(weights[:,None]*Q),G.T@(weights[:,None]*G),Q.T@weights


if __name__=='__main__':
 for model in ('beam','schrodinger'):
  for n in (33,65,129):
   length=1 if model=='beam' else 20
   p=bspf_plans.plan_1d(j.linspace(0,length,n),degree=5,n_basis=16,boundary_points=7)
   k=2 if model=='beam' else 1
   weak=bspf_galerkin.galerkin_1d(p,derivative_order=k,constraints=((0,0),(0,1)) if k==2 else ())
   M,K,F=resolved(p,weak,k)
   target=1.875104068711961**4 if k==2 else (10*j.pi/length)**2
   index=0 if k==2 else 10
   for name,mass,stiff in [('trapezoid',weak.mass,weak.stiffness),('resolved',M,K)]:
    lam=eigh(np.array(stiff),np.array(mass),eigvals_only=True)[index]
    line=f'{model} n={n} {name}: eigenvalue relative error={float(lam/target-1):.5e}'
    if k==2:
     force=weak.extension.T@p.weights if name=='trapezoid' else F
     static=weak.extension@j.linalg.solve(stiff,force)
     exact=p.x**2*(p.x**2-4*p.x+6)/24
     line+=f', static max error={float(j.max(j.abs(static-exact))):.5e}'
    print(line,flush=True)
