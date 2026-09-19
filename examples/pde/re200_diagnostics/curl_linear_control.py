"""GPU homogeneous linear control for the experimental curl-residual operator."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np
from bspf_jax.immersed_flow import ImmersedFlowPlan
from curl_residual import assemble
jax.config.update('jax_enable_x64',True);device=jax.devices('gpu')[0]
p=ImmersedFlowPlan(assembly_device=device,basis_precision='float64',nx=73,ny=33,reynolds=200,wall_method='rational')
s=p.stepper(.01,device=device);d,meta=assemble(p,s)
rng=np.random.default_rng(201)
a=jax.device_put(rng.normal(size=p.dofs),device)
a=a/jnp.sqrt(a @ s.data['mass'] @ a)
@jax.jit
def power(m,l,state,dt):
 fac=jl.lu_factor(m+dt/2*l)
 right=m-dt/2*l
 def body(_,x):return jl.lu_solve(fac,right @ x)
 return jax.lax.fori_loop(0,100,body,state)
rows=[]
for dt in (.01,.005,.0025):
 for tag,m,l in [('original',s.data['mass'],s.data['linear']),('curl',d['augmented_mass'],d['augmented_linear'])]:
  end=power(m,l,a,jax.device_put(dt,device))
  norm=float(jnp.sqrt(end @ s.data['mass'] @ end))
  row=dict(dt=dt,time=100*dt,form=tag,final_velocity_norm=norm);rows.append(row);print(json.dumps(row),flush=True)
Path('build/immersed_flow/re200_ripple_study/curl_residual/linear_control.json').write_text(json.dumps(rows,indent=2)+'\n')
