"""Full fixed-window Chebyshev endpoint convergence sweep; preserve all plateaus."""
from pathlib import Path
import json,gc
import numpy as np
from open_slab_packet import run,jax,plt


def plot(out,rows):
    baseline=json.loads((out/'mild_knots.json').read_text())
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    fields=['relative_g_error','max_phi_error','max_particle_balance','max_free_energy_balance']
    titles=['Distribution error','Potential error','Particle budget defect','Free-energy budget defect']
    for data,label,color in [(baseline,'FD9 baseline','#87939f'),(rows,'Chebyshev: 12 modes, 16 points','#1768ac')]:
        for a,key,title in zip(axs.flat,fields,titles):
            a.loglog([r['n']-1 for r in data],[r[key] for r in data],'o-',label=label,color=color)
            a.set(xlabel='N - 1',title=title,ylabel='Max absolute error' if key=='max_phi_error' else 'Relative error')
    for a in axs.flat:a.grid(which='both',alpha=.2);a.legend(fontsize=9)
    axs[0,0].annotate(f"N=257: {rows[-1]['relative_g_error']:.2e}",
        (rows[-1]['n']-1,rows[-1]['relative_g_error']),xytext=(-105,18),textcoords='offset points',fontsize=10)
    fig.suptitle('Open GK packet: complete conservative-window sweep\nDegree 7, 16 spline functions, 50% knot grading; dt=0.0005, t=1.2',fontsize=14)
    fig.savefig(out/'conservative_window.png',dpi=180)
    fig.savefig(out/'conservative_window.pdf');plt.close(fig)


def main():
    out=Path('build/open_slab_packet');out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for n in [49,65,97,129,193,257]:
        row,arrays=run(n,endpoint_blend=.5,endpoint_method='chebyshev',boundary_points=16,chebyshev_modes=12)
        if rows:
            row['observed_order']=float(np.log(rows[-1]['relative_g_error']/row['relative_g_error'])/np.log((n-1)/(rows[-1]['n']-1)))
        rows.append(row)
        (out/'conservative_window.json').write_text(json.dumps(rows,indent=2)+'\n')
        if n==257:np.savez_compressed(out/'conservative_window_257.npz',**arrays)
        del arrays;jax.clear_caches();gc.collect()
    plot(out,rows)
    assert all(np.isfinite(r['relative_g_error']) for r in rows)
    assert all(r['max_particle_balance']<1e-8 and r['max_free_energy_balance']<1e-8 for r in rows)
    print('Complete sweep and budget checks passed.',flush=True)

if __name__=='__main__':main()
