"""Render the saved no-sponge Re=20 initial-RHS and time-step audits."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--baseline',type=Path,
                    default=Path('build/immersed_flow/rational_ns_nosponge_20260922'))
    args=ap.parse_args()
    d=np.load(args.out/'fields.npz')
    report=json.loads((args.out/'report.json').read_text())
    x,y=d['x'],d['y'];k=np.argmin(abs(x+.65))
    physical=d['physical_rate_0.001'];rate=d['discrete_rate'];error=rate-physical
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    ax=axes[0,0]
    ax.plot(y,physical[:,k],label='Local continuous vorticity equation')
    ax.plot(y,rate[:,k],label='Discrete projected acceleration',lw=1.2)
    ax.set(title=f'Before any time step: x={x[k]:.3f}',xlabel='y',ylabel='Vorticity rate')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[0,1]
    limit=np.max(abs(error))
    im=ax.pcolormesh(x,y,error,cmap='RdBu_r',vmin=-limit,vmax=limit,shading='nearest')
    fig.colorbar(im,ax=ax,label='Discrete rate minus local PDE rate')
    ax.set(title='Initial spatial residual in the upstream fluid',xlabel='x',ylabel='y')
    ax=axes[1,0]
    ax.plot(y[2:-2],np.diff(error,n=4,axis=0)[:,k],label='Initial spatial residual',lw=2)
    for dt in (.02,.005,.001):
        early=d[f'increment_rate_{dt}']-physical
        ax.plot(y[2:-2],np.diff(early,n=4,axis=0)[:,k],label=f'First step, dt={dt}',ls='--',lw=1)
    ax.set(title='First-step pattern tends to the spatial residual',xlabel='y',
           ylabel='Fourth difference of rate residual')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,1]
    b=np.load(args.baseline/'fields.npz');fine=np.load(args.out/'half_dt_final.npz')
    ix=np.argmin(abs(b['x']+.7));yc=b['y'][2:-2];sel=(yc>-.8)&(yc<.8)
    for label,w in [('dt=0.02',b['fields'][-1,2]),('dt=0.01',fine['vorticity'])]:
        ax.plot(yc[sel],np.diff(w,n=4,axis=0)[sel,ix],label=label,lw=1.2)
    ax.set(title='Same final time t=1: ripple survives halving dt',xlabel='y',
           ylabel='Fourth difference of vorticity')
    ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Rational/BSPF 73 x 33, Re=20, quadrature=4, sponge OFF\n'
                 'Production operators unchanged; fourth differences are diagnostic, not exact error norms')
    fig.savefig(args.out/'source_diagnosis.png',dpi=170)
    print(json.dumps(report['same_time_step_control'],indent=2))


if __name__=='__main__':
    main()
