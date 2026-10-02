"""Replay a fixed checkpoint; separate setup from synchronized step timing."""
import argparse,cProfile,pstats,json,time
from pathlib import Path
import numpy as np
from thin_airfoil_structural import boundary
from bspf_models.fluids.embedded_navier_stokes import plan_embedded_navier_stokes2d,geometry_prior_options
ap=argparse.ArgumentParser();ap.add_argument('--degree',type=int,default=5);ap.add_argument('--label',default='before');ap.add_argument('--projector-backend',choices=['implicit_qr','array'],default='implicit_qr');ap.add_argument('--steps',type=int,default=20);a=ap.parse_args()
start=time.perf_counter()
p=plan_embedded_navier_stokes2d(boundary,dt=1e-4,viscosity=.002,cells=7,degree=a.degree,order=64,convection_order=48,center=(0.,0.),axes=(1.,.1),edges=[np.array([-60,-3,-1.2,-.4,.4,1.2,3,60]),np.array([-40,-2,-.2,-.05,.05,.2,2,40])],outflow=False,constraint_backend='implicit_qr',linear_backend='host_sparse',constraint_rank_tolerance=1e-9,wall_enforcement='constraint',projector_backend=a.projector_backend,**geometry_prior_options(a.degree))
print('SETUP',time.perf_counter()-start,flush=True)
try:
 with np.load(f'build/obstacle_stokes/ns/thin_airfoil_structural_p{a.degree}_noslip.npz') as z:state=p._State(**{k:z[k].item() if z[k].ndim==0 else z[k] for k in p._State._fields})
 p.initialize(state.coefficients,time=state.time)
 for _ in range(2):p.step(state)
 prof=cProfile.Profile();prof.enable();times=[];iters=[]
 for _ in range(a.steps):
  t=time.perf_counter();state,d=p.step(state);times.append(time.perf_counter()-t);iters.append(p.last_iterations)
  assert d.valid
 prof.disable()
 out=Path(f'build/obstacle_stokes/ns/runtime_p{a.degree}_{a.label}')
 with out.with_suffix('.profile.txt').open('w') as f:pstats.Stats(prof,stream=f).sort_stats('cumtime').print_stats(45)
 r=dict(median_seconds=float(np.median(times)),times=times,iterations=iters,info=p.info,final_diagnostics={k:bool(v) if k=='valid' else float(v) for k,v in d._asdict().items()},wall_slip=p.wall_slip_error(state.coefficients))
 out.with_suffix('.json').write_text(json.dumps(r,indent=2));np.savez_compressed(out.with_suffix('.npz'),**state._asdict());print(r['median_seconds'],iters,flush=True)
finally:p.close()
