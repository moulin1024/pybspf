"""Unsmoothed NS and frozen-response comparisons for geometric enrichment."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--baseline',type=Path,default=Path('build/immersed_flow/rational_ns_nosponge_20260922'))
    ap.add_argument('--baseline-label',default='Original space')
    args=ap.parse_args();out=args.out
    report=json.loads((out/'report.json').read_text())
    new=np.load(out/'fields.npz');old=np.load(args.baseline/'fields.npz')
    np.testing.assert_array_equal(old['x'],new['x']);np.testing.assert_array_equal(old['y'],new['y'])
    x,y=new['x'],new['y'];xx,yy=np.meshgrid(x,y)
    def decorate(ax):
        ax.add_patch(Ellipse(new['center'],*(2*new['axes']),facecolor='.7',edgecolor='k'))
        ax.set(xlim=(-1,3),ylim=(-1,1),xlabel='x',ylabel='y')
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for row,(label,data) in enumerate([(args.baseline_label,old),(report.get('family','Enriched')+' space',new)]):
        for ax,t in zip(axes[row],[0,.5,1.]):
            k=np.argmin(abs(data['t']-t))
            im=ax.pcolormesh(x,y,data['fields'][k,2]-2*yy,shading='nearest',cmap='RdBu_r',vmin=-.25,vmax=.25)
            decorate(ax);ax.set_title(f'{label}, t={data["t"][k]:g}')
    fig.colorbar(im,ax=axes,label='Vorticity minus inlet shear; clipped at +/-0.25')
    fig.suptitle('Re=20, dt=0.02, same rational Stokes lift; no sponge or filtering')
    fig.savefig(out/'ns_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
    for xpos,ax in zip([-.7,-.4],axes[:2]):
        ix=np.argmin(abs(x-xpos))
        ax.plot(y,old['fields'][0,2,:,ix],color='.5',ls=':',label='Initial Stokes')
        ax.plot(y,old['fields'][-1,2,:,ix],label=args.baseline_label+', t=1',lw=1.1)
        ax.plot(y,new['fields'][-1,2,:,ix],label='Enriched, t=1',lw=1.1)
        ax.set(xlim=(-.8,.8),xlabel='y',ylabel='Raw vorticity',title=f'x={x[ix]:.3f}')
        ax.grid(alpha=.2);ax.legend()
    metrics={}
    for label,data in [('original',old),('enriched',new)]:
        d4=np.diff(data['fields'][:,2],n=4,axis=1);sx=(x>-.85)&(x<-.45);sy=(y[2:-2]>-.8)&(y[2:-2]<.8)
        rms=np.sqrt(np.mean(d4[:,sy][:,:,sx]**2,axis=(1,2)))
        axes[2].semilogy(data['t'],rms,'o-',label=label)
        metrics[label]=dict(times=data['t'].tolist(),roughness=rms.tolist())
    axes[2].set(xlabel='t',ylabel='Unscaled fourth y-difference RMS',title='Fixed-grid upstream roughness')
    axes[2].grid(alpha=.2);axes[2].legend()
    fig.savefig(out/'ns_sections.png',dpi=170);plt.close(fig)
    metrics['roughness_reduction_factor']=metrics['original']['roughness'][-1]/metrics['enriched']['roughness'][-1]
    f=np.load(out/f'frozen_{report["levels"][-1].get("level", len(report["levels"][-1]["lengths"]))}.npz')
    base=np.load(Path(report['reference'])/'helmholtz_comparison.npz')
    ref=f['reference'][4]-f['reference'][3]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,title,omega in zip(axes.ravel(),['Original actual','Enriched actual','Original best H1','Enriched best H1'],
            [base['actual_vorticity'],f['actual_omega'],base['best_h1_vorticity'],f['best_omega']]):
        im=ax.pcolormesh(x,y,omega-ref,shading='nearest',cmap='RdBu_r',vmin=-1,vmax=1)
        decorate(ax);ax.set_title(title)
    fig.colorbar(im,ax=axes,label='Frozen Helmholtz vorticity error; clipped at +/-1')
    fig.savefig(out/'frozen_comparison.png',dpi=170);plt.close(fig)
    (out/'comparison_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
