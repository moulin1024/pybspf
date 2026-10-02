"""Independent volume and split-boundary polynomial reproduction audit."""
import json,time
from pathlib import Path
import numpy as np
from bspf_models.fluids.embedded_navier_stokes import plan_embedded_navier_stokes2d,geometry_prior_options
from bspf_models.fluids._embedded.geometry import ObstacleGrid
records=[]
for degree,enriched in [(3,False),(5,False),(5,True)]:
 start=time.perf_counter();nu=.38;k=degree
 def exact(x):
  a,b=x.T
  return np.column_stack((.01*(a*a+b**k),-.02*a*b,.3+a+b+a*b,.02*a,.01*k*b**(k-1),-.02*b,-.02*a))
 def bc(x,tag):return exact(x)[:,:2]
 def traction(x,n):
  v=exact(x);g=v[:,3:].reshape(-1,2,2);return nu*np.einsum('qij,qj->qi',g,n)-v[:,2,None]*n
 p=plan_embedded_navier_stokes2d(bc,dt=.001,viscosity=nu,cells=3,degree=degree,order=56,convection_order=48,outflow=True,linear_backend='host_sparse',constraint_backend='implicit_qr',constraint_rank_tolerance=1e-9,**(geometry_prior_options(degree) if enriched else {}))
 try:
  a,b=p.force_points.T;force=np.column_stack((1+b-nu*(.02+.01*k*(k-1)*b**(k-2)),1+a))
  state=p.stokes_initial_state(force,traction(p.traction_points,p.traction_normals))
  grid=ObstacleGrid(3,63,p.spatial.grid.center,p.spatial.grid.axes)
  x=np.vstack([v[0] for v in grid.volume.values()]);w=np.concatenate([v[1] for v in grid.volume.values()]);diff=p.evaluate(x,state)-exact(x)
  norms={label:float(np.sqrt(np.sum(w[:,None]*diff[:,sl]**2))) for label,sl in [('velocity_L2',slice(0,2)),('pressure_L2',slice(2,3)),('gradient_L2',slice(3,7))]}
  bn=bt=0.
  for entries in grid.boundary.values():
   for x,w,n,tag in entries:
    if tag=='outer' and np.all(n[:,0]>.5):continue
    e=p.evaluate(x,state)[:,:2]-bc(x,tag);t=np.column_stack((-n[:,1],n[:,0]));bn+=w@np.sum(e*n,axis=1)**2;bt+=w@np.sum(e*t,axis=1)**2
  result=dict(degree=degree,enriched=enriched,**norms,boundary_normal_L2=float(np.sqrt(bn)),boundary_tangent_L2=float(np.sqrt(bt)),rank_history=p.info['constraint_rank_history'],seconds=time.perf_counter()-start)
  records.append(result);print(result,flush=True)
 finally:p.close()
Path('build/obstacle_stokes/ns/polynomial_reproduction_fixed.json').write_text(json.dumps(records,indent=2))
assert all(r['velocity_L2']<1e-7 and r['pressure_L2']<1e-6 and r['gradient_L2']<1e-6 for r in records)
