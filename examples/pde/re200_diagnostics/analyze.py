import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
root=Path('build/immersed_flow')
cases=[('73×33, dt=.02, q=2.5','re200_gpu'),('73×33, dt=.01, q=2.5','re200_gpu_dt001'),('73×33, dt=.01, q=4','re200_gpu_q4'),('73×65, dt=.01, q=2.5','re200_gpu_73x65')]
loaded=[];report=[]
for label,name in cases:
 d=np.load(root/name/'fields.npz'); s=json.loads((root/name/'summary.json').read_text()); f=d['fields']; x,y=d['x'],d['y']
 xx,yy=np.meshgrid(x,y);fluid=np.isfinite(f[0,0]);assert np.all(np.isfinite(f[:,:,fluid]))
 ix=(x>-.85)&(x<-.45); iy=(y>-.8)&(y<.8)
 measures=[]
 for snapshot in f:
  patch=snapshot[2][np.ix_(iy,ix)]
  measures.append(float(np.sqrt(np.mean(np.diff(patch,n=4,axis=0)**2))))
 rec=dict(case=name,label=label,setup_seconds=s['setup_seconds'],evolution_seconds=s['evolution_and_diagnostics_seconds'],
  reconstruction_seconds=s['reconstruction_seconds'],dofs=s['retained_dofs'],quadrature_points=s['quadrature_points'],
  final_upstream_fourth_difference_rms=measures[-1],late_mean_upstream_fourth_difference_rms=float(np.mean(measures[20:])),
  checks={k:s['checks'][k] for k in ('hole_wall_max_speed','outer_dirichlet_max_error','max_relative_flux_error')},
  final=s['final'])
 report.append(rec); loaded.append((label,d,f,s))
base=loaded[1][2];mask=(xx<=3)&np.isfinite(base[-1,0])
for rec,(_,d,f,s) in zip(report,loaded):
 du=base[-1,:2][:,mask]-f[-1,:2][:,mask]
 rec['final_velocity_relative_l2_vs_dt001']=float(np.linalg.norm(du)/np.linalg.norm(base[-1,:2][:,mask]))
 dw=base[-1,2][mask]-f[-1,2][mask]
 rec['final_vorticity_relative_l2_vs_dt001']=float(np.linalg.norm(dw)/np.linalg.norm(base[-1,2][mask]))
# The raw fourth-difference diagnostic cancels smooth cubic trends; it is not a
# filter applied to the solution, and is not an error against an exact solution.
result=dict(metric='RMS of raw fourth differences in y of vorticity on the common output grid',
 patch=dict(x=[-.85,-.45],y=[-.8,.8]),cases=report,
 spatial_final_reduction=report[1]['final_upstream_fourth_difference_rms']/report[-1]['final_upstream_fourth_difference_rms'],
 spatial_late_mean_reduction=report[1]['late_mean_upstream_fourth_difference_rms']/report[-1]['late_mean_upstream_fourth_difference_rms'])
(root/'re200_ripple_study/results.json').write_text(json.dumps(result,indent=2)+'\n')
fig,axes=plt.subplots(3,2,figsize=(12,9),layout='constrained',width_ratios=(2.5,1))
for row,idx in enumerate((1,2,3)):
 label,d,f,s=loaded[idx]; x,y=d['x'],d['y']; omega=f[-1,2]-2*y[:,None]
 ax=axes[row,0];im=ax.pcolormesh(x,y,omega,cmap='RdBu_r',vmin=-22,vmax=22,shading='auto',rasterized=True)
 ax.add_patch(Ellipse(d['center'],*(2*d['axes']),facecolor='.65',edgecolor='black'))
 ax.set(xlim=(-1,3),ylim=(-1,1),aspect='equal',title=label,ylabel='y',xlabel='x')
 ax=axes[row,1];jm=ax.pcolormesh(x,y,omega,cmap='RdBu_r',vmin=-.5,vmax=.5,shading='auto',rasterized=True)
 ax.set(xlim=(-.85,-.45),ylim=(-.8,.8),title='Upstream: same enlarged color scale',xlabel='x',ylabel='y')
fig.colorbar(im,ax=axes[:,0],label=r'$\omega-2y$',shrink=.75)
fig.colorbar(jm,ax=axes[:,1],label=r'$\omega-2y$',shrink=.75)
fig.suptitle('Re = 200, t = 20 | Raw solver fields; no smoothing')
fig.savefig(root/'re200_ripple_study/comparison.png',dpi=160)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for idx in (1,2,3):
 label,d,f,s=loaded[idx]; x,y=d['x'],d['y']; k=np.argmin(abs(x+.65))
 axes[0].plot(f[-1,2,:,k]-2*y,y,label=label)
 axes[1].plot(d['t'],[np.sqrt(np.mean(np.diff(frame[2][np.ix_(iy,ix)],n=4,axis=0)**2)) for frame in f],label=label)
axes[0].set(xlabel=r'$\omega-2y$',ylabel='y',ylim=(-.8,.8),title='Upstream profile at x ≈ -0.65')
axes[1].set(xlabel='t',ylabel='Fourth-difference RMS',yscale='log',title='Upstream ripple diagnostic')
for ax in axes: ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.savefig(root/'re200_ripple_study/profiles.png',dpi=160)
plt.close(fig)
print(json.dumps(result,indent=2))
