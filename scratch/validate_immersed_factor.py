"""Independent continuous forcing check for the geometric wall representation."""
import json
from pathlib import Path
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la
from bspf_jax.immersed_flow import ImmersedFlowPlan, elliptic_wall_factor

p=ImmersedFlowPlan(nx=33,ny=25,wall_method='factor')
left,right,h=p.bounds
(cx,cy),(a,b)=p.hole.center,p.hole.axes
constant=.61

def psi(z):
 x,y=z
 f=((x-cx)/a)**2+((y-cy)/b)**2-1
 g=(x-left)*(right-x)*(1-(y/h)**2)/((cx-left)*(right-cx)*(1-(cy/h)**2))
 q=f*f/(f*f+p.wall_width**2*g*g)
 return q*p.peak*(y-y**3/(3*h*h)+2*h/3)+(1-q)*constant

grad=jax.grad(psi);hess=jax.jacfwd(grad);third=jax.jacfwd(hess)
@jax.jit
def exact(points):
 d=jax.vmap(grad)(points);dd=jax.vmap(hess)(points);ddd=jax.vmap(third)(points)
 u,v=d[:,1],-d[:,0]
 ux,uy,vx=dd[:,0,1],dd[:,1,1],-dd[:,0,0]
 lapu=ddd[:,1,0,0]+ddd[:,1,1,1]
 lapv=-ddd[:,0,0,0]-ddd[:,0,1,1]
 return u,v,ux,uy,vx,lapu,lapv

u,v,ux,uy,vx,lapu,lapv=[np.array(a) for a in exact(p.points)]
force=np.column_stack((u*ux+v*uy-p.nu*lapu-2*p.nu*p.peak/h**2+p.sigma*(u-p.peak*(1-(p.points[:,1]/h)**2)),u*vx-v*ux-p.nu*lapv+p.sigma*v))
load=p.force_load(force)
ou,ov,ox,oy,ovx,*_=[np.array(a) for a in exact(p.out_points)]
load+=p.nu*(p.out_ops[0].T@(p.out_weights*ox)+p.out_ops[1].T@(p.out_weights*ovx))
# This representation must be recovered without supplying exact derivatives to the solver.
state=la.cho_solve(p.mass_factor,p.force_load(np.column_stack((u-p.lift_fields[1],v-p.lift_fields[2]))))
r=p.explicit(state)+load-p.linear@state
boundary,_=p.arc.sample(194,offset=.31)
fields=p.evaluate(state,boundary)
record=dict(ndofs=p.ndofs,dofs=p.dofs,setup=p.setup_seconds,wall_max=float(np.max(np.hypot(fields[1],fields[2]))),velocity_representation_max=float(np.max(np.hypot(p.operators_fluid[0]@state+p.lift_fields[1]-u,p.operators_fluid[1]@state+p.lift_fields[2]-v))),continuous_weak_residual=float(la.norm(r)),mass_dual_residual=float(np.sqrt(r@la.cho_solve(p.mass_factor,r))))
print(json.dumps(record,indent=2),flush=True)
Path('build/immersed_flow/artifact_diagnosis/factor_mms.json').write_text(json.dumps(record,indent=2))
