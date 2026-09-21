"""Check integration-by-parts and kinetic energy on the actual Re200 state."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from bspf_models.fluids.immersed_flow import ImmersedFlowPlan
jax.config.update('jax_enable_x64',True)
p=ImmersedFlowPlan(assembly_device=jax.devices('gpu')[0],basis_precision='float64',nx=73,ny=33,reynolds=200,wall_method='rational',quadrature_factor=4)
step=p.stepper(.01,device=jax.devices('gpu')[0]);d=step.data
@jax.jit
def audit(d,a):
 u,v,ux,uy,vx=[o@a+b for o,b in zip(d['ops'],d['lift'])]
 bu,bv=d['ops'][:2];w=d['weights'];outu,outv=[o@a+b for o,b in zip(d['out_ops'],d['out_lift'])]
 adv=bu.T@(w*(u*ux+v*uy))+bv.T@(w*(u*vx-v*ux))
 om=vx-uy;rot=bu.T@(-w*v*om)+bv.T@(w*u*om)+d['out_ops'][0].T@(d['out_weights']*.5*(outu**2+outv**2))
 return dict(adv=adv,defect=adv-rot,power=jnp.sum(w*(u*(u*ux+v*uy)+v*(u*vx-v*ux))),out_power=.5*jnp.sum(d['out_weights']*outu*(outu**2+outv**2)))
# Reconstruct from original coefficients to allow eigenvector sign changes.
import scipy.linalg as la
coeff=np.load('build/immersed_flow/re200_gpu_q4/fields.npz')['coefficients']
state=la.lstsq(p.transform,coeff-p.lift_coefficients)[0]
a=jax.device_put(state,step.device)
r=jax.device_get(audit(d,a))
energy=lambda b:float(np.sqrt(max(b@la.cho_solve(p.mass_factor,b),0)))
report=dict(advective_rotational_relative_dual_defect=energy(r['defect'])/energy(r['adv']),advective_rotational_dual_defect=energy(r['defect']),bulk_power=float(r['power']),boundary_power=float(r['out_power']-16/35*p.bounds[2]*p.peak**3))
report['power_balance_defect']=report['bulk_power']-report['boundary_power']
print(json.dumps(report,indent=2),flush=True)
Path('build/immersed_flow/re200_ripple_study/weak_form_audit.json').write_text(json.dumps(report,indent=2)+'\n')
