"""Raw, fixed-grid comparison of local, paired and broad response dictionaries."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--paired',type=Path,required=True)
    ap.add_argument('--broad',type=Path)
    ap.add_argument('--local',type=Path,default=Path('build/immersed_flow/enriched_response_q6_20260922'))
    args=ap.parse_args()
    datasets=[('Four-scale',args.local),('Thin + outer pair',args.paired)]
    if args.broad:datasets.append(('Broad exponentials',args.broad))
    loaded=[(name,path,np.load(path/'fields.npz')) for name,path in datasets]
    x,y=loaded[0][2]['x'],loaded[0][2]['y'];yc=y[2:-2]
    sx=(x>-.85)&(x<-.45);sy=(yc>-.8)&(yc<.8)
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    d4s=[np.diff(data['fields'][:,2],n=4,axis=1) for _,_,data in loaded]
    limit=float(np.quantile(abs(d4s[0][-1][sy][:,sx]),.995))
    metrics={};ix=np.argmin(abs(x+.7))
    for j,((name,path,data),d4) in enumerate(zip(loaded,d4s)):
        np.testing.assert_array_equal(data['x'],x);np.testing.assert_array_equal(data['y'],y)
        np.testing.assert_array_equal(data['fields'][0],loaded[0][2]['fields'][0])
        rms=np.sqrt(np.mean(d4[:,sy][:,:,sx]**2,axis=(1,2)))
        metric=dict(path=str(path),times=data['t'].tolist(),roughness=rms.tolist())
        ref=loaded[0][2]['fields'][-1]
        for key,a,b in [('velocity',data['fields'][-1,:2],ref[:2]),('vorticity',data['fields'][-1,2],ref[2])]:
            mask=np.isfinite(a)&np.isfinite(b)
            metric[key+'_relative_grid_difference_from_local']=float(np.linalg.norm((a-b)[mask])/np.linalg.norm(b[mask]))
        metrics[name]=metric
        im=axes[0,j].pcolormesh(x[sx],yc[sy],d4[-1][sy][:,sx],shading='nearest',cmap='RdBu_r',vmin=-limit,vmax=limit)
        axes[0,j].set(title=f'{name}, t=1',xlabel='x',ylabel='y')
        axes[1,0].plot(y,data['fields'][-1,2,:,ix],label=name,lw=1.1)
        axes[1,1].plot(yc,d4[-1,:,ix],label=name,lw=1.1)
        axes[1,2].semilogy(data['t'],rms,'o-',label=name)
    for j in range(len(loaded),3):axes[0,j].set_visible(False)
    fig.colorbar(im,ax=axes[0,:len(loaded)],label='Unscaled fourth y-difference of vorticity')
    axes[1,0].set(xlim=(-.8,.8),xlabel='y',ylabel='Raw vorticity',title=f'Unfiltered section, x={x[ix]:.3f}')
    section_limit=1.1*max(float(np.max(abs(d[-1,sy,ix]))) for d in d4s)
    axes[1,1].set(xlim=(-.8,.8),ylim=(-section_limit,section_limit),xlabel='y',ylabel='Fourth y-difference',title='Same unfiltered section: roughness')
    axes[1,2].set(xlabel='t',ylabel='Upstream roughness RMS',title='Same sampling and physical parameters')
    for ax in axes[1]:ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Re=20, dt=0.02, same Stokes initial field; no smoothing or added dissipation\nFourth differences are a roughness diagnostic, not a nonlinear error norm')
    fig.savefig(args.paired/'paired_comparison.png',dpi=170);plt.close(fig)
    metrics['pair_reduction_from_local']=metrics['Four-scale']['roughness'][-1]/metrics['Thin + outer pair']['roughness'][-1]
    if args.broad:metrics['broad_reduction_from_local']=metrics['Four-scale']['roughness'][-1]/metrics['Broad exponentials']['roughness'][-1]
    (args.paired/'paired_comparison.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
