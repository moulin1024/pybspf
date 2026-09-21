import jax
jax.config.update('jax_enable_x64',True)
import numpy as np,pickle
from pathlib import Path
from bspf_models.plasma.tokamak_equilibrium import plan_axisymmetric_bspf
from bspf_models.plasma.tokamak_equilibrium import fit_fixed_coils
from bspf_models.plasma.tokamak_equilibrium import solve_equilibrium
cache=Path('build/tokamak_plan33.pkl')
if cache.exists():p=pickle.loads(cache.read_bytes())
else:
 p=plan_axisymmetric_bspf();cache.write_bytes(pickle.dumps(p))
for q,v,offset in [(-.001,-.005,-.15),(-.001,-.005,-.2),(-.002,.005,-.18),(-.002,.005,-.2),(-.004,.03,-.2),(-.004,.03,-.23)]:
 coils,c0,err=fit_fixed_coils(quadrupole=q,vertical=v,offset=offset)
 try:
  eq=solve_equilibrium(p,coils,offset=c0,max_iterations=700,tolerance=1e-9)
  i,j=np.where(eq['core']);r=np.asarray(p.radial.points);z=np.asarray(p.vertical.points)
  print(q,v,offset,eq['iterations'],eq['residual'],(r[i].min(),r[i].max(),z[j].min(),z[j].max()),'kappa',(z[j].max()-z[j].min())/(r[i].max()-r[i].min()),flush=True)
  np.savez_compressed(f'build/tokamak_eq_{abs(q):g}_{abs(offset):g}.npz',**eq,coils=coils,offset=c0)
 except Exception as e:print(q,v,offset,str(e),flush=True)
