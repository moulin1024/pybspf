"""Fixed-space Stokes trace rank experiment (no filtering or added modes)."""
import json
from pathlib import Path
import jax
import numpy as np
import scipy.linalg as la
from bspf_models.fluids.immersed_flow import ImmersedFlowPlan
from bspf_models.fluids.immersed_flow import channel_lift
jax.config.update('jax_enable_x64', True)
out = Path('build/immersed_flow/artifact_diagnosis')
p = ImmersedFlowPlan(nx=73, ny=33, wall_rcond=1e-3)
print('SETUP', p.setup_seconds, p.constraint_rank, flush=True)
boundary, _ = p.arc.sample(p.boundary_count)
op = p.operators(boundary)
c = np.vstack(op[1:3]) * p.scale
target = -np.concatenate(channel_lift(boundary)[1:3])
u, s, vh = la.svd(c, full_matrices=False)
b, _ = p.arc.sample(512, offset=.371)
x,y=np.linspace(-1,5,401),np.linspace(-1,1,161)
records=[]
for threshold in (1e-3, 1e-5, 1e-7, 1e-10):
    rank=int(np.sum(s>threshold*s[0]))
    v=vh[p.constraint_rank:rank]
    if len(v):
        constraints=v @ (p.transform / p.scale[:,None])
        rhs=(u[:,p.constraint_rank:rank].T@target)/s[p.constraint_rank:rank] - v@(p.lift_coefficients/p.scale)
        q, r = la.qr(constraints.T, mode='full')
        a0=q[:,:len(v)]@la.solve_triangular(r[:len(v)].T, rhs, lower=True)
        z=q[:,len(v):]
        a=a0+z@la.solve(z.T@p.linear@z,-z.T@(p.linear@a0+p.linear_lift), assume_a='pos')
    else:
        a=p.stokes_state
    f=p.grid(a,x,y)
    _,ub,vb,*_=p.evaluate(a,b)
    row=dict(threshold=threshold,rank=rank,wall_max=float(np.max(np.hypot(ub,vb))),**p.diagnostics(a))
    row.pop('acceleration_l2') # original loose-space residual differs by constraints
    records.append(row)
    np.savez(out/f'stokes_{threshold:.0e}.npz',x=x,y=y,**f)
    print(json.dumps(row),flush=True)
np.savez(out/'trace_spectrum.npz',singular_values=s)
(out/'stokes_rank.json').write_text(json.dumps(records,indent=2))
