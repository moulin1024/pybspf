from pathlib import Path
import json, platform
import numpy as np
from pybspf import PressurePoisson2D, ClosedBSPFLine
import argparse
parser=argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
out=parser.parse_args().out; out.mkdir(parents=True, exist_ok=True)
data={}
for method in ('taylor','chebyshev'):
    options=dict(q=2,n_basis=5,degree=4,baseline_points=4)
    if method=='chebyshev': options.update(endpoint_method=method,baseline_points=6,chebyshev_modes=5)
    x,y=np.linspace(-1,2,7),np.linspace(0,2,8)
    raw=np.random.default_rng(5).normal(size=(7,8,2))
    v,r=PressurePoisson2D(x,y,**options).project(raw.swapaxes(0,1))
    data[method+'_velocity']=v.swapaxes(0,1); data[method+'_pressure']=r.pressure.T
for method in ('taylor','chebyshev'):
    x,y=np.linspace(0,1,64),np.linspace(0,1,65); xx,yy=np.meshgrid(x,y,indexing='ij')
    a,c=5.3*np.pi,3.7*np.pi
    raw=np.stack([a*np.cos(a*xx+.2)*np.cos(c*yy-.1),-c*np.sin(a*xx+.2)*np.sin(c*yy-.1)],axis=-1)
    options={} if method=='taylor' else dict(endpoint_method=method,chebyshev_modes=14,baseline_points=18)
    _,r=PressurePoisson2D(x,y,**options).project(raw.swapaxes(0,1))
    data[method+'_production']=r.pressure.T
np.savez_compressed(out/'pressure.npz',**data)
line=ClosedBSPFLine(33); points=np.linspace(0,1,41)
np.savez_compressed(out/'closed_line.npz',points=points,scalar=line.scalar_values(points,2),tangent=line.tangent_values(points,2),normal=line.values(points,2),derivative_map=line.derivative_map)
(out/'manifest.json').write_text(json.dumps(dict(source_commit='361a39593de3ee6656b00f52cdd0b778a58280c5',python=platform.python_version(),numpy=np.__version__,seed=5,description='Legacy pressure projection and derivative-closed space migration references; generator preserved alongside archive.'),indent=2)+'\n')
