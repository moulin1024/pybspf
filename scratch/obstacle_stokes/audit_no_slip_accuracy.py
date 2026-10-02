"""Check that exact wall constraints do not destroy smooth Stokes approximation.

An analytic potential defines MMS data only; the numerical unknown is velocity.
"""
import json
from pathlib import Path
import numpy as np
import sympy as sy
from bspf_models.fluids.embedded_navier_stokes import plan_embedded_navier_stokes2d, geometry_prior_options
from bspf_models.fluids._embedded.geometry import ObstacleGrid

x, y = sy.symbols('x y')
phi = ((x-sy.Rational(13,100))/sy.Rational(27,100))**2 + ((y+sy.Rational(7,100))/sy.Rational(19,100))**2 - 1
potential = sy.Rational(1,10000)*phi**2*(1-x*x)**2*(1-y*y)**2*sy.exp(x/5)
u, v = sy.diff(potential,y), -sy.diff(potential,x)
expressions = [u,v,sy.diff(u,x),sy.diff(u,y),sy.diff(v,x),sy.diff(v,y)]
exact_fn = sy.lambdify((x,y), expressions, 'numpy', cse=True)
force_fn = sy.lambdify((x,y),[-sy.diff(z,x,2)-sy.diff(z,y,2) for z in (u,v)],'numpy',cse=True)
def exact(points): return np.asarray(exact_fn(*points.T)).T
def boundary(points,tag): return np.zeros_like(points)
records=[]
for degree in (5,7):
 for enforcement in ('nitsche','constraint'):
  plan=plan_embedded_navier_stokes2d(boundary,dt=.001,viscosity=1.,cells=3,degree=degree,
      order=64,convection_order=48,linear_backend='host_sparse',constraint_backend='implicit_qr',
      constraint_rank_tolerance=1e-9,wall_enforcement=enforcement,**geometry_prior_options(degree))
  try:
   state=plan.stokes_initial_state(np.asarray(force_fn(*plan.force_points.T)).T)
   grid=ObstacleGrid(3,77,plan.spatial.grid.center,plan.spatial.grid.axes,full_order=32)
   points=np.vstack([p for p,w in grid.volume.values()]);weights=np.concatenate([w for p,w in grid.volume.values()])
   base=plan.spatial.base;base.coefficients=np.r_[state.coefficients[:2*plan.nv],np.zeros(base.np)]
   got=base.evaluate(points)[:,[0,1,3,4,5,6]];want=exact(points)
   def norm(a): return float(np.sqrt(np.sum(weights[:,None]*a*a)))
   record=dict(degree=degree,enforcement=enforcement,relative_velocity_L2=norm(got[:,:2]-want[:,:2])/norm(want[:,:2]),
       relative_gradient_L2=norm(got[:,2:]-want[:,2:])/norm(want[:,2:]),wall_slip_L2=plan.wall_slip_error(state.coefficients),
       free_velocity_dofs=plan.projector.free,iterations=plan.last_iterations)
   records.append(record);print(record,flush=True)
  finally:plan.close()
Path('build/obstacle_stokes/ns/no_slip_accuracy.json').write_text(json.dumps(records,indent=2))
assert records[3]['relative_velocity_L2'] < records[1]['relative_velocity_L2']/3
assert records[3]['relative_gradient_L2'] < records[1]['relative_gradient_L2']/3
