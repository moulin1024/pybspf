"""Compare mild endpoint spline grading with the uniform-break baseline."""
from pathlib import Path
import json,gc
import numpy as np
from open_slab_packet import run,jax,plt

def main():
    out=Path('build/open_slab_packet');out.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((out/'spatial.json').read_text())
    rows=[]
    for n in [49,65,97,129,193,257]:
        row,_=run(n,endpoint_blend=.5)
        if rows:
            row['observed_order']=float(np.log(rows[-1]['relative_g_error']/row['relative_g_error'])/np.log((n-1)/(rows[-1]['n']-1)))
        rows.append(row)
        (out/'mild_knots.json').write_text(json.dumps(rows,indent=2)+'\n')
        jax.clear_caches();gc.collect()
    row,_=run(257)
    baseline.append(row)
    (out/'uniform_knots.json').write_text(json.dumps(baseline,indent=2)+'\n')
    fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for data,label in [(baseline,'Uniform spline breaks'),(rows,'50% cosine blend')]:
        ns=np.array([r['n'] for r in data]);err=[r['relative_g_error'] for r in data]
        ax[0].loglog(ns-1,err,'o-',label=label)
        ax[1].loglog(ns-1,[r['max_particle_balance'] for r in data],'o-',label=label)
    ax[0].set(xlabel='N - 1',ylabel='Relative distribution L2 error',title='Source-free open GK packet, t=1.2')
    ax[1].set(xlabel='N - 1',ylabel='Relative particle budget defect',title='Same spline degree/core and time step')
    for a in ax:a.grid(True,which='both',alpha=.25);a.legend()
    fig.savefig(out/'knot_comparison.png',dpi=180);plt.close(fig)
    assert all(r['observed_order']>7 for r in rows[1:4])
    assert all(r['max_particle_balance']<1e-8 for r in rows)
    assert all(r['max_free_energy_balance']<1e-8 for r in rows)

if __name__=='__main__':main()
