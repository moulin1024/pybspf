"""Locate BSPF mirror negativity and isolate reconstruction/transport errors."""

import bspf_models.kinetic.drift_kinetic as bspf_drift_kinetic
import pybspf.plans as bspf_plans
import argparse
import json
import runpy
from pathlib import Path
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

exact=runpy.run_path('examples/pde/drift_kinetic_mirror.py')['exact']
parser=argparse.ArgumentParser()
parser.add_argument('--n',type=int,default=65)
parser.add_argument('--basis',type=int,default=21)
parser.add_argument('--degree',type=int,default=7)
parser.add_argument('--q',type=int,default=6)
parser.add_argument('--points',type=int,default=9)
parser.add_argument('--out',default='build/mirror_negativity')
args=parser.parse_args()
n=args.n
z=jnp.linspace(-4.,4.,n); v=jnp.linspace(-3.,3.,n)
kw=dict(degree=args.degree,n_basis=args.basis,constraint_order=args.q,boundary_points=args.points)
zp=bspf_plans.plan_1d(z,**kw); vp=bspf_plans.plan_1d(v,**kw)
p=bspf_drift_kinetic.plan_drift_kinetic(zp,vp,magnetic_field=lambda z:1+z*z/2,magnetic_gradient=lambda z:z,mu_max=2.,n_mu=12,backend="dense")
f=exact(0,z[:,None,None],v[None,:,None],p.mu[None,None,:])
fq=jnp.einsum('ai,ijm,bj->abm',p.z_values,f,p.v_values)
refq=exact(0,p.transport.z_points[:,None,None],p.transport.v_points[None,:,None],p.mu[None,None,:])
rhs,_=bspf_drift_kinetic.drift_kinetic_rhs(p,0.,f,inflow=exact)
# Analytic differential equation derivative at the initial time.
z3=z[:,None,None]; v3=v[None,:,None]; mu3=p.mu[None,None,:]
ref_rhs=(8*z3*v3-12*mu3*z3*(v3-.8))*f
pre=dict(initial_reconstruction_error=float(jnp.max(jnp.abs(fq-refq))),
 initial_reconstruction_min=float(jnp.min(fq)),initial_rhs_error=float(jnp.max(jnp.abs(rhs-ref_rhs))))
print(json.dumps(dict(config=vars(args),**pre)),flush=True)
t=jnp.linspace(0.,4.,81)
h,tr=bspf_drift_kinetic.integrate_drift_kinetic(p,f,t,inflow=exact,substeps=40)
ref=exact(t[:,None,None,None],z[None,:,None,None],v[None,None,:,None],p.mu[None,None,None,:])
mom=bspf_drift_kinetic.drift_kinetic_moments(p,h); nh=mom[:,jnp.array([0,3])]
res=nh-nh[0]-tr.sum(axis=-1)
mins=[]
# Independent off-grid check at quadrature points; stream time slices.
for i in range(0,81,5):
 vals=jnp.einsum('ai,ijm,bj->abm',p.z_values,h[i],p.v_values)
 mins.append(float(jnp.min(vals)))
idx=np.unravel_index(np.asarray(h).argmin(),h.shape)
report=dict(config=vars(args),**pre,min_distribution=float(jnp.min(h)),
 max_error=float(jnp.max(jnp.abs(h-ref))),min_quadrature=min(mins),
 min_coordinates=[float(a[k]) for a,k in zip((t,z,v,p.mu),idx)],
 balances=np.asarray(jnp.max(jnp.abs(res),axis=0)/nh[0]).tolist())
print(json.dumps(report),flush=True)
out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
tag=f'n{n}_b{args.basis}_d{args.degree}_q{args.q}_p{args.points}'
(out/(tag+'.json')).write_text(json.dumps(report,indent=2)+'\n')
