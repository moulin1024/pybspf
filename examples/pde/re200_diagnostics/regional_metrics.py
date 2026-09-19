"""Report unsmoothed fourth differences on a common physical output grid."""
import json
from pathlib import Path
import numpy as np
root=Path('build/immersed_flow')
regions=dict(upstream=(-.85,-.45,-.8,.8),upper=(0,1.4,.5,.8),lower=(0,1.4,-.85,-.6))
report=[]
for name in ('re200_gpu_dt001','re200_gpu_q4','re200_gpu_mpfr','re200_gpu_73x65','re200_gpu_145x33','re200_gpu_97x65'):
 d=np.load(root/name/'fields.npz');f=d['fields'];x,y=d['x'],d['y'];s=json.loads((root/name/'summary.json').read_text())
 item=dict(case=name,regions={})
 for region,(xa,xb,ya,yb) in regions.items():
  ix=(x>xa)&(x<xb);iy=(y>ya)&(y<yb)
  patch=f[:,2][:,iy][:,:,ix]
  m=np.sqrt(np.mean(np.diff(patch,n=4,axis=1)**2,axis=(1,2)))
  item['regions'][region]=dict(t2=float(m[4]),late_mean=float(np.mean(m[20:])),final=float(m[-1]))
 item['boundary']={k:s['checks'][k] for k in ('hole_wall_max_speed','outer_dirichlet_max_error','max_relative_flux_error')}
 report.append(item)
(root/'re200_ripple_study/regional_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
for item in report:print(item['case'], {k:float(f'{v["late_mean"]:.5g}') for k,v in item['regions'].items()})
