"""Diagnose ripple generation without modifying spatial or time operators.

Compare the discrete vorticity time derivative against the pressure-free local
vorticity PDE in an upstream fluid patch. Numerical differentiation is only an
independent diagnostic; it is never used to change the evolved solution.
"""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np
from bspf_jax.immersed_flow import ImmersedFlowPlan,channel_lift
from bspf_jax.immersed_flow_gpu import _explicit
jax.config.update('jax_enable_x64',True)
device=jax.devices('gpu')[0]
root=Path('build/immersed_flow/re200_ripple_study')
p=ImmersedFlowPlan(assembly_device=device,basis_precision='float64',nx=73,ny=33,reynolds=200,wall_method='rational',quadrature_factor=2.5)
step=p.stepper(.01,device=device)
@jax.jit
def derivative(d,a):
 return jl.cho_solve((d['mass_factor'],True),_explicit(d,a)-d['linear']@a)
@jax.jit
def bspf_fields(bx,by,c):
 def pair(i,j):return jnp.sum((bx[i]@c)*by[j],axis=1)
 return pair(0,0),pair(0,1),-pair(1,0),pair(1,1),pair(0,2),-pair(2,0)
def evaluate_parts(coeff,points,affine=True):
 factors=[]
 for axis in range(2):
  coords,idx=np.unique(points[:,axis],return_inverse=True)
  factors.append(tuple(v[idx] for v in p._line_values(axis,coords)))
 bf=jax.device_get(bspf_fields(*jax.device_put((*factors,coeff.reshape(p.shape)),device)))
 rc=p.rational_modes@(p.rational_map@coeff)
 if affine:
  bf=tuple(v+b for v,b in zip(bf,channel_lift(points,p.bounds[2],p.peak)))
  rc=rc+p.rational_lift
 rf=jax.device_get(p.rational.evaluate(points,rc,device=device,return_device=True))
 return np.stack(bf),np.stack(rf)
x=np.linspace(-.85,-.35,61);y=np.linspace(-.8,.8,161)
xx,yy=np.meshgrid(x,y);points=np.column_stack((xx.ravel(),yy.ravel()));shape=xx.shape
states=[(0.,step.initial_state)]
a=step.initial_state
for k in range(50):
 a=step.step(a,k*.01)
 if k+1 in (1,5,10,50):states.append(((k+1)*.01,a))
saved=np.load('build/immersed_flow/re200_gpu_dt001/fields.npz')
# State coordinates can change under an equivalent setup normalization.
a=jnp.linalg.solve(jax.device_put(p.transform,device),jax.device_put(saved['coefficients']-p.lift_coefficients,device))
states.append((20.,a))
report=[];arrays=dict(x=x,y=y);rms=lambda z:float(np.sqrt(np.mean(z*z)))
for t,a in states:
 coeff=p.coefficients(jax.device_get(a))
 b,r=evaluate_parts(coeff,points);f=b+r
 wb=(b[5]-b[4]).reshape(shape);wr=(r[5]-r[4]).reshape(shape);w=wb+wr
 bd=np.diff(wb,n=4,axis=0);rd=np.diff(wr,n=4,axis=0);wd=bd+rd
 record=dict(t=t,omega_rms=rms(w),bspf_fourth_difference_rms=rms(bd),rational_fourth_difference_rms=rms(rd),total_fourth_difference_rms=rms(wd),rational_to_bspf_fourth_difference=rms(rd)/rms(bd))
 arrays[f'omega_{t}']=w;arrays[f'bspf_{t}']=wb;arrays[f'rational_{t}']=wr
 if t in (0.,.5,20.):
  da=jax.device_get(derivative(step.data,a));dc=p.transform@da
  db,dr=evaluate_parts(dc,points,affine=False);dw=((db[5]-db[4])+(dr[5]-dr[4])).reshape(shape)
  arrays[f'omega_rate_{t}']=dw;record['discrete_rate_rms']=rms(dw);record['discrete_rate_fourth_difference_rms']=rms(np.diff(dw,n=4,axis=0))
  record['local_vorticity_equation']=[]
  for h in (.002,.001):
   shifts=[(0,0),(-2*h,0),(-h,0),(h,0),(2*h,0),(0,-2*h),(0,-h),(0,h),(0,2*h)]
   probe=np.vstack([points+shift for shift in shifts]);bb,rr=evaluate_parts(coeff,probe)
   ww=((bb[5]-bb[4])+(rr[5]-rr[4])).reshape(9,*shape)
   wx=(ww[1]-8*ww[2]+8*ww[3]-ww[4])/(12*h)
   wy=(ww[5]-8*ww[6]+8*ww[7]-ww[8])/(12*h)
   lap=(-ww[1]+16*ww[2]-30*ww[0]+16*ww[3]-ww[4]-ww[5]+16*ww[6]-30*ww[0]+16*ww[7]-ww[8])/(12*h*h)
   rhs=-f[1].reshape(shape)*wx-f[2].reshape(shape)*wy+p.nu*lap
   residual=dw-rhs
   arrays[f'pde_rhs_{t}_{h}']=rhs;arrays[f'pde_residual_{t}_{h}']=residual
   record['local_vorticity_equation'].append(dict(h=h,rhs_rms=rms(rhs),residual_rms=rms(residual),rhs_fourth_difference_rms=rms(np.diff(rhs,n=4,axis=0)),residual_fourth_difference_rms=rms(np.diff(residual,n=4,axis=0))))
 report.append(record);print(json.dumps(record),flush=True)
np.savez_compressed(root/'source_audit_fields.npz',**arrays)
(root/'source_audit.json').write_text(json.dumps(report,indent=2)+'\n')
# Check the algebraic solve and direct reconstruction independently.
a0=states[0][1];da0=derivative(step.data,a0)
rhs=_explicit(step.data,a0)-step.data['linear']@a0
solve_residual=step.data['mass']@da0-rhs
linear_residual=step.data['linear']@a0+step.data['linear_lift']
checks=dict(initial_mass_solve_relative_residual=float(jnp.linalg.norm(solve_residual)/jnp.linalg.norm(rhs)),initial_linear_balance_relative_residual=float(jnp.linalg.norm(linear_residual)/jnp.linalg.norm(rhs)))
a=states[-1][1];coeff=p.coefficients(jax.device_get(a));idx=np.arange(0,len(p.points),max(1,len(p.points)//151))
b,r=evaluate_parts(coeff,p.points[idx]);direct=(b+r)[1:];resident=jax.device_get(jnp.stack([o@a+lift for o,lift in zip(step.data['ops'],step.data['lift'])]))[:,idx]
checks['reconstruction_at_quadrature_max_abs_error']=float(np.max(abs(direct-resident)))
checks['reconstruction_at_quadrature_relative_error']=float(np.linalg.norm(direct-resident)/np.linalg.norm(resident))
# The initial Stokes balance cancels viscous+sponge loads. The remaining
# derivative is the mass projection of the nonlinear force, before any timestep.
checks['energy_condition']=p.energy_condition
(root/'source_audit_checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks),flush=True)
