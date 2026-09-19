"""Causal diagnostic: smooth convection startup, then unchanged Re200 NS."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np
from bspf_jax.immersed_flow import ImmersedFlowPlan
from bspf_jax.immersed_flow_gpu import _explicit
from bspf_jax._flow_kernels import imex_midpoint
jax.config.update('jax_enable_x64',True)
device=jax.devices('gpu')[0];p=ImmersedFlowPlan(assembly_device=device,basis_precision='float64',nx=73,ny=33,reynolds=200,wall_method='rational')
step=p.stepper(.01,device=device)
@jax.jit
def ramp_step(d,a,t,dt):
 def explicit(a,t):
  z=t/2
  # Flat to every order at t=0 and t=2; exactly one thereafter.
  left=jnp.exp(-1/jnp.maximum(z,1e-15));right=jnp.exp(-1/jnp.maximum(1-z,1e-15))
  g=jnp.where(z<=0,0.,jnp.where(z>=1,1.,left/(left+right)))
  return g*(_explicit(d,a)+d['linear_lift'])-d['linear_lift']
 return imex_midpoint(a,t,dt,lambda a:d['mass']@a,lambda a:d['linear']@a,explicit,lambda b:jl.cho_solve((d['factor'],True),b))
# Verify exactly the original discrete evolution after the startup interval.
a=step.initial_state;ordinary=step.step(a,3.);control=ramp_step(step.data,a,jax.device_put(3.,device),step.dt)
parity=float(jnp.max(jnp.abs(ordinary-control)));assert parity<1e-12
states=[];history=[]
for k in range(2001):
 if k%50==0:
  states.append(jax.device_get(a));diag={key:float(value) for key,value in jax.device_get(step.diagnostics(a)).items()};diag['full_ns_acceleration_l2']=diag.pop('acceleration_l2');history.append(dict(t=k*.01,**diag));print(json.dumps(history[-1]),flush=True)
 if k<2000:a=ramp_step(step.data,a,jax.device_put(k*.01,device),step.dt)
x=np.linspace(-1,5,401);y=np.linspace(-1,1,161)
fields=np.stack([np.stack([f[key] for key in ('u','v','vorticity','psi')]) for f in p.grid_many(np.asarray(states),x,y)])
root=Path('build/immersed_flow/re200_ripple_study');np.savez_compressed(root/'startup_control_fields.npz',x=x,y=y,t=np.arange(41)*.5,fields=fields)
ix=(x>-.85)&(x<-.45);iy=(y>-.8)&(y<.8)
def measure(f):return [float(np.sqrt(np.mean(np.diff(frame[2][np.ix_(iy,ix)],n=4,axis=0)**2))) for frame in f]
reference=np.load('build/immersed_flow/re200_gpu_dt001/fields.npz')['fields'];ramp=measure(fields);original=measure(reference)
report=dict(ramp_interval=[0,2],post_ramp_step_parity_max_abs_error=parity,metric='Upstream raw y-fourth-difference RMS on common output grid',ramp=ramp,original=original,ramp_late_mean=float(np.mean(ramp[20:])),original_late_mean=float(np.mean(original[20:])),history=history)
(root/'startup_control.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('history','ramp','original')}),flush=True)
