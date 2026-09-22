"""Plot actual and best-approximation frozen-force responses against refined FEM."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();report=json.loads((args.out/'report.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    fig2,sections=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    details={}
    for row,case in enumerate(['stokes','helmholtz']):
        d=np.load(args.out/f'{case}_comparison.npz');x,y=d['x'],d['y']
        ref=d['reference'][4]-d['reference'][3]
        actual=d['actual_vorticity'];best=d['best_h1_vorticity']
        err=actual-ref;berr=best-ref
        limit=np.nanmax(abs(np.stack([err,berr])))
        for col,(field,title) in enumerate([(ref,'FEM reference'),(err,'Actual minus reference'),
                                           (berr,'Best H1 minus reference')]):
            ax=axes[row,col];lim=np.nanmax(abs(ref)) if col==0 else limit
            im=ax.pcolormesh(x,y,field,cmap='RdBu_r',vmin=-lim,vmax=lim,shading='nearest')
            ax.add_patch(Ellipse((.19,-.13),.62,.46,facecolor='.7',edgecolor='k'))
            ax.set(xlim=(-1,2),ylim=(-1,1),xlabel='x',ylabel='y',title=f'{case}: {title}')
            fig.colorbar(im,ax=ax,label='Vorticity' if col==0 else 'Vorticity error')
        ix=np.argmin(abs(x+.65))
        for name,label in [('actual','Actual'),('best_h1','Best H1'),('best_energy','Best energy'),('best_l2','Best L2')]:
            sections[row,0].plot(y,d[name+'_vorticity'][:,ix]-ref[:,ix],label=label,lw=1.1)
        sections[row,0].set(xlabel='y',ylabel='Vorticity error',title=f'{case}, x={x[ix]:.3f}')
        sections[row,0].legend(fontsize=8);sections[row,0].grid(alpha=.2)
        record=next(a for a in report['comparisons'] if a['case']==case)
        names=['actual','best_h1','best_energy','best_l2']
        sections[row,1].bar(np.arange(4),[100*record['metrics'][n]['velocity_relative_h1'] for n in names])
        sections[row,1].set(xticks=np.arange(4),xticklabels=['Actual','Best H1','Best energy','Best L2'],
                            ylabel='Velocity H1 error (%)',title=f'{case}: gradient-sensitive accuracy')
        ok=np.isfinite(err)&np.isfinite(berr);a,b=err[ok],berr[ok]
        details[case]=dict(actual_best_h1_vorticity_error_cosine=float(a@b/np.linalg.norm(a)/np.linalg.norm(b)),
                          actual_best_h1_vorticity_gap_over_actual_error=float(np.linalg.norm(a-b)/np.linalg.norm(a)))
    degree=report['reference_levels'][-1].get('degree',3)
    fig.suptitle('Same frozen nonlinear force, homogeneous response boundary conditions\n'
                 f'Independent curved P{degree}/P{degree-1} FEM; each error-map pair shares its color scale')
    fig.savefig(args.out/'response_errors.png',dpi=160)
    fig2.suptitle('Same BSPF+rational trial space: actual solve versus three best projections')
    fig2.savefig(args.out/'projection_comparison.png',dpi=160)
    (args.out/'plot_metrics.json').write_text(json.dumps(details,indent=2)+'\n')
    print(json.dumps(details,indent=2))


if __name__=='__main__':main()
