import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('build/immersed_flow/re200_ripple_study');d=np.load(root/'source_audit_fields.npz');x,y=d['x'],d['y'];k=np.argmin(abs(x+.65));initial=d['omega_rate_0.0'];physical=d['pde_rhs_0.0_0.001'];error=initial-physical
fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
ax[0,0].plot(y,physical[:,k],label='Local vorticity PDE');ax[0,0].plot(y,initial[:,k],label='Discrete spatial derivative',alpha=.8);ax[0,0].set(title='Before the first time step: x ≈ −0.65',xlabel='y',ylabel='Vorticity rate');ax[0,0].legend()
im=ax[0,1].pcolormesh(x,y,error,cmap='RdBu_r',shading='auto',vmin=-1.5,vmax=1.5);fig.colorbar(im,ax=ax[0,1],label='Discrete rate − local PDE rate');ax[0,1].set(title='Initial spatial residual, upstream of obstacle',xlabel='x',ylabel='y')
early=(d['omega_0.01']-d['omega_0.0'])/.01-physical
ax[1,0].plot(y,error[:,k],label='Initial spatial residual');ax[1,0].plot(y,early[:,k],label='After first step: Δω/Δt − initial PDE rate',linestyle='--');ax[1,0].set(title='The first step inherits the oscillatory pattern',xlabel='y',ylabel='Vorticity rate error');ax[1,0].legend(fontsize=8)
for label,path in [('73×33',root),('145×33',root/'initial_145x33'),('73×65',root/'initial_73x65')]:
 data=np.load(path/'source_audit_fields.npz')
 residual=data['omega_rate_0.0']-data['pde_rhs_0.0_0.001']
 fourth=np.diff(residual,n=4,axis=0);window=np.hanning(len(fourth))[:,None]
 freq=np.fft.rfftfreq(len(fourth),y[1]-y[0]);power=np.mean(abs(np.fft.rfft(fourth*window,axis=0))**2,axis=1);power/=power.sum()
 ax[1,1].plot(freq,power,'o-',markersize=3,label=label)
for cutoff in (8,16):ax[1,1].axvline(cutoff,color='.5',linestyle='--',linewidth=1)
ax[1,1].set(xlim=(0,20),title='Ripple frequency follows the y basis cutoff',xlabel='Cycles per unit y (cutoffs: 8 and 16)',ylabel='Normalized fourth-difference spectral power');ax[1,1].legend(fontsize=8)
for a in (ax[0,0],ax[1,0],ax[1,1]):a.grid(alpha=.2)
fig.suptitle('Re=200, 73×33 | Diagnostics of unchanged GPU solver; no smoothing')
fig.savefig(root/'source_diagnosis.png',dpi=160)
print(root/'source_diagnosis.png')
