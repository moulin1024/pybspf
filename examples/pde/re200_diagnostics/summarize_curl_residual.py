"""Retain the rejected curl-test prototype evidence and raw-field comparison."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from curl_residual_control import json_safe

root=Path('build/immersed_flow/re200_ripple_study/curl_residual')
out=Path('docs/data/re200_ripple/curl_residual');out.mkdir(parents=True,exist_ok=True)
rows=[]
for name in ('zero','strength1','unweighted_dt0025','boundary_compatible','boundary_cell',
             'boundary_cell_dt0025','boundary_cell_dt0025_q4'):
    report=json.loads((root/name/'report.json').read_text())
    (out/(name+'.json')).write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n')
    row=dict(case=name,initial_residual_rms=report['initial_local_pde_residual']['rms'],
             initial_residual_d4y=report['initial_local_pde_residual']['d4y'],
             final_time=report['history'][-1]['time'],final=report['history'][-1],
             failed=report.get('failed'),dt=report['configuration']['dt'])
    rows.append(row)
for name in ('derivative_checks','linear_control'):
    report=json.loads((root/(name+'.json')).read_text())
    (out/(name+'.json')).write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n')

baseline=np.load(root/'zero/fields.npz')
weighted=np.load(root/'boundary_compatible/fields.npz')
x,y=baseline['x'],baseline['y']
# Compare unfiltered fields at exactly the same time and on the same probe.
a,b=baseline['final_omega'],weighted['final_omega']
metrics=dict(t2_baseline_d4y=float(np.sqrt(np.mean(np.diff(a,n=4,axis=0)**2))),
             t2_weighted_d4y=float(np.sqrt(np.mean(np.diff(b,n=4,axis=0)**2))),
             t2_relative_vorticity_difference=float(np.linalg.norm(b-a)/np.linalg.norm(a)))
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
limit=max(np.max(abs(a)),np.max(abs(b)))
for ax,field,title in zip(axes[:2],(a,b),('Original, t=2','Boundary-compatible curl test, t=2')):
    im=ax.pcolormesh(x,y,field,cmap='RdBu_r',vmin=-limit,vmax=limit,shading='auto')
    fig.colorbar(im,ax=ax);ax.set(title=title,xlabel='x',ylabel='y')
i=np.argmin(abs(x+.6))
axes[2].plot(a[:,i],y,label='Original')
axes[2].plot(b[:,i],y,label='Curl test')
axes[2].set(title=f'Raw vorticity at x={x[i]:.3f}',xlabel='Vorticity',ylabel='y')
axes[2].legend();axes[2].grid(alpha=.2)
fig.savefig(out/'t2_comparison.png',dpi=150)
(out/'comparison.json').write_text(json.dumps(json_safe(dict(cases=rows,metrics=metrics)),indent=2,allow_nan=False)+'\n')
print(json.dumps(json_safe(dict(cases=rows,metrics=metrics)),indent=2,allow_nan=False))
