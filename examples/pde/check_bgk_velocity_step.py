"""Short fine-velocity time-step check from a common projected saturated state."""
import numpy as np,time,json
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from scipy.special import eval_hermitenorm,eval_laguerre,gammaln
from bspf_jax.collisional_itg import *
from bspf_jax.nonlinear_itg import nonlinear_itg_diagnostics
from bspf_jax.itg_bracket_tensor import *
def basis(b):
 s=np.asarray(b.sqrt_weights);sv=np.sqrt((s*s).sum(axis=1));sm=np.sqrt((s*s).sum(axis=0))
 return np.stack([sv*eval_hermitenorm(k,np.asarray(b.velocity))*np.exp(-.5*gammaln(k+1)) for k in range(len(sv))],1),np.stack([sm*eval_laguerre(k,np.asarray(b.mu)) for k in range(len(sm))],1)
p0=plan_collisional_itg(n_x=33,n_v=8,n_mu=6,a_t=4.);p=plan_collisional_itg(n_x=33,n_v=24,n_mu=16,a_t=4.)
x=np.load('build/bgk_itg_gyro_n33/history.npz')['final_state'];h,l=basis(p0.base);hh,ll=basis(p.base)
c=np.einsum('...vm,vi,mj->...ij',x,h,l,optimize=True)
x=jnp.asarray(np.einsum('...ij,vi,mj->...vm',c,hh[:,:8],ll[:,:6],optimize=True))
tensor=plan_itg_bracket_tensor(p.base.radial)
outputs=[]
for dt in [.1,.05]:
 start=time.perf_counter();a,t,w=integrate_collisional_itg_tensor(p,tensor,x,dt,steps=round(5/dt),save_every=round(1/dt));a.block_until_ready()
 ds=np.asarray(jax.vmap(lambda u:nonlinear_itg_diagnostics(p.base,u))(a));q=np.asarray(jax.vmap(lambda u:collisional_itg_transport(p,u))(a))[:,1]
 err=ds[:,2]-ds[0,2]-np.asarray(w)[:,0]-np.asarray(w)[:,1]+np.asarray(w)[:,2]
 outputs.append((np.asarray(a[-1]),q,float(max(abs(err))/max(ds[:,2]))))
 print(dt,'seconds',time.perf_counter()-start,'budget',outputs[-1][2],flush=True)
r=dict(dt_coarse=.1,dt_fine=.05,velocity=[24,16],duration=5,
 budget_coarse=outputs[0][2],budget_fine=outputs[1][2],state_difference=float(np.linalg.norm(outputs[0][0]-outputs[1][0])/np.linalg.norm(outputs[1][0])),heat_difference=float(np.linalg.norm(outputs[0][1]-outputs[1][1])/np.linalg.norm(outputs[1][1])))
print(json.dumps(r,indent=2),flush=True)
# Observable-focused step budget for a 5% transport study: 0.01% Q,
# 0.1% full-state difference, and 1e-5 work residual; record actual errors.
assert r['budget_coarse']<1e-5 and r['heat_difference']<1e-4 and r['state_difference']<1e-3
open('build/bgk_velocity_scan/time_step_check.json','w').write(json.dumps(r,indent=2)+'\n')
