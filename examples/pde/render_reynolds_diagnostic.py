"""Compare Re=20/200 and two time steps without smoothing the saved fields."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,default=Path('build/immersed_flow/broad_response_q6_20260922'))
    args=parser.parse_args()
    report=json.loads((args.out/'report.json').read_text())
    with np.load(args.out/'fields.npz') as d:
        x,y,t,fields,center,axes=[d[k] for k in ('x','y','t','fields','center','axes')]
    with np.load(args.out/'time_step_check.npz') as d:
        np.testing.assert_array_equal(x,d['x']);np.testing.assert_array_equal(y,d['y'])
        np.testing.assert_allclose(d['time'],t[-1],atol=0,rtol=0)
        coarse,fine=d['fields'];dt=d['dt']
        np.testing.assert_allclose(coarse,fields[-1],rtol=1e-11,atol=1e-11,equal_nan=True)
    with np.load(args.baseline/'fields.npz') as d:
        np.testing.assert_array_equal(x,d['x']);np.testing.assert_array_equal(y,d['y'])
        k=int(np.argmin(abs(d['t']-t[-1])))
        np.testing.assert_allclose(d['t'][k],t[-1],atol=1e-14,rtol=0)
        base=d['fields'][k]
    sx=(x>-.85)&(x<-.45);sy=(y[2:-2]>-.8)&(y[2:-2]<.8)
    def rough(f):return np.diff(f[2],n=4,axis=0)[sy][:,sx]
    def rms(a):return float(np.sqrt(np.mean(a*a)))
    maps=[rough(f) for f in (base,coarse,fine)]
    if not all(np.all(np.isfinite(m)) for m in maps):
        raise RuntimeError('Non-finite upstream diagnostic')
    finite=np.isfinite(fine[:2])
    metrics=dict(reynolds=report['reynolds'],time=float(t[-1]),dt=dt.tolist(),
        baseline_roughness=rms(maps[0]),coarse_roughness=rms(maps[1]),fine_roughness=rms(maps[2]),
        time_step_roughness_difference_rms=rms(maps[1]-maps[2]),
        time_step_velocity_relative_grid_l2=float(np.linalg.norm((coarse[:2]-fine[:2])[finite])/np.linalg.norm(fine[:2][finite])))
    finite=np.isfinite(fine[2])
    metrics['time_step_vorticity_relative_grid_l2']=float(np.linalg.norm((coarse[2]-fine[2])[finite])/np.linalg.norm(fine[2][finite]))
    metrics['fine_to_re20_roughness_ratio']=metrics['fine_roughness']/metrics['baseline_roughness']
    labels=['Re=20, dt=0.02',f'Re={report["reynolds"]:g}, dt={dt[0]:g}',f'Re={report["reynolds"]:g}, dt={dt[1]:g}']
    rmax=max(float(np.max(abs(m))) for m in maps)
    perturbations=[f[2]-2*y[:,None] for f in (base,coarse,fine)]
    wmax=max(float(np.nanpercentile(abs(a),99)) for a in perturbations)
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for k,label in enumerate(labels):
        im=axs[0,k].pcolormesh(x,y,perturbations[k],cmap='RdBu_r',vmin=-wmax,vmax=wmax,shading='nearest')
        axs[0,k].add_patch(Ellipse(center,*(2*axes),facecolor='.7',edgecolor='k'))
        axs[0,k].set(xlim=(-1,3),ylim=(-1,1),xlabel='x',ylabel='y',title=label,aspect='equal')
        rm=axs[1,k].pcolormesh(x[sx],y[2:-2][sy],maps[k],cmap='RdBu_r',vmin=-rmax,vmax=rmax,shading='nearest')
        axs[1,k].set(xlabel='x',ylabel='y',title=f'Upstream fourth-difference RMS: {rms(maps[k]):.3e}')
    fig.colorbar(im,ax=axs[0],label='Vorticity minus inlet shear (clipped)',extend='both',shrink=.7)
    fig.colorbar(rm,ax=axs[1],label='Unscaled fourth y-difference',shrink=.9)
    fig.suptitle(f'Broad-exponential response spaces, t={t[-1]:g}; raw fields, common color scales\n'
                 'Re-dependent thin scales; Re=20 quadrature 6, Re=200 quadrature 8')
    fig.savefig(args.out/'reynolds_diagnostic.png',dpi=150);plt.close(fig)
    (args.out/'diagnostic.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
