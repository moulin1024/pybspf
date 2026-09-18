import pickle,numpy as np
import jax
jax.config.update('jax_enable_x64',True)
from pathlib import Path
from bspf_jax.tokamak_linear import assemble_linear_tokamak,growing_modes
p=pickle.loads(Path('build/tokamak_plan33.pkl').read_bytes())
path=Path('build/tokamak_eq_0.002_0.18.npz')
if not path.exists():path=Path('build/tokamak_eq_0.001_0.15.npz')
eq=dict(np.load(path));print('equilibrium',path,flush=True)
model=assemble_linear_tokamak(p,eq,eq['coils'],float(eq['offset']))
Path('build/tokamak_linear33.pkl').write_bytes(pickle.dumps(model))
for shift in (.1,.5,1.):
 g,v,r=growing_modes(model,shift=shift)
 print('shift',shift,'gamma',g,'residual',r,flush=True)
 np.savez_compressed(f'build/tokamak_modes_{shift:g}.npz',gamma=g,vectors=v,residual=r)
