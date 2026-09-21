import pickle,numpy as np,jax
from pathlib import Path
jax.config.update('jax_enable_x64',True)
from bspf_models.plasma.tokamak_linear import assemble_linear_tokamak
from bspf_models.plasma.tokamak_linear import growing_modes
p=pickle.loads(Path('build/tokamak_plan33.pkl').read_bytes())
eq=dict(np.load('build/tokamak_eq_0.004_0.2.npz'))
model=assemble_linear_tokamak(p,eq,eq['coils'],float(eq['offset']))
Path('build/tokamak_selected33.pkl').write_bytes(pickle.dumps(model))
g,v,r=growing_modes(model,shift=.15)
print('gamma',g,'residual',r,flush=True)
np.savez_compressed('build/tokamak_selected_modes33.npz',gamma=g,vectors=v,residual=r)
