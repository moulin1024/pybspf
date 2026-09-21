"""Undriven, nondissipative multimode conservation and transfer validation."""
from pathlib import Path
import json,time,gc
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bspf_models.kinetic.nonlinear_itg import plan_nonlinear_itg
from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_initial
from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_rhs
from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_fields
from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_diagnostics
from bspf_models.kinetic.nonlinear_itg import integrate_nonlinear_itg


def run(p,dt,*,end=20.,linear=False):
    x=nonlinear_itg_initial(p,amplitude=.7)
    steps=round(end/dt);save=max(1,steps//40)
    start=time.perf_counter()
    hist,t=integrate_nonlinear_itg(p,x,dt,steps=steps,save_every=save,include_linear=linear)
    d=np.asarray(jax.vmap(lambda u:nonlinear_itg_diagnostics(p,u))(hist))
    phi=np.asarray(jax.vmap(lambda u:nonlinear_itg_fields(p,u)[0])(hist))
    norm0=float(jnp.linalg.norm(x))
    fluct=hist[len(hist)//2]
    nl=nonlinear_itg_rhs(p,fluct,include_linear=False)
    full=nonlinear_itg_rhs(p,fluct,include_linear=True)
    nl_rates=np.asarray(jax.jvp(lambda u:nonlinear_itg_diagnostics(p,u),(fluct,),(nl,))[1])
    full_rates=np.asarray(jax.jvp(lambda u:nonlinear_itg_diagnostics(p,u),(fluct,),(full,))[1])
    scale=float(jnp.linalg.norm(fluct)*jnp.linalg.norm(nl))
    scale_full=float(jnp.linalg.norm(fluct)*jnp.linalg.norm(full))
    row=dict(dt=dt,end=end,include_linear=linear,
        max_entropy_relative_drift=float(np.max(abs(d[:,0]/d[0,0]-1))),
        max_field_relative_drift=float(np.max(abs(d[:,1]/d[0,1]-1))),
        max_free_energy_relative_drift=float(np.max(abs(d[:,2]/d[0,2]-1))),
        final_state_relative_change=float(jnp.linalg.norm(hist[-1]-x)/norm0),
        final_zonal_fraction=float(d[-1,3]/d[-1,1]),
        initial_zonal_fraction=float(d[0,3]/d[0,1]),
        nl_entropy_rate=float(nl_rates[0]),nl_field_rate=float(nl_rates[1]),
        nl_max_normalized_rate=float(np.max(abs(nl_rates[:3]))/scale),
        full_entropy_rate=float(full_rates[0]),full_field_rate=float(full_rates[1]),
        full_free_energy_rate=float(full_rates[2]),
        full_normalized_free_energy_rate=float(abs(full_rates[2])/scale_full),
        seconds=time.perf_counter()-start)
    print(json.dumps(row),flush=True)
    return row,dict(t=np.asarray(t),diagnostics=d,phi_hat=phi,final_state=np.asarray(hist[-1]))


def plot(out,report,nonlinear,full):
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for label,index in [('Entropy',0),('Field energy',1),('Total free energy',2)]:
        d=nonlinear['diagnostics']
        ax[0,0].plot(nonlinear['t'],(d[:,index]-d[0,index])/d[0,index],label=label)
    ax[0,0].set(title='Nonlinear advection alone: invariant drift',xlabel='t',ylabel='Relative change');ax[0,0].legend()
    d=nonlinear['diagnostics'];ax[0,1].plot(nonlinear['t'],d[:,3]/d[:,1],label='Zonal fraction of field energy')
    ax[0,1].set(title='Nontrivial transfer: initially no zonal field',xlabel='t',ylabel='Zonal / total field energy');ax[0,1].legend()
    d=full['diagnostics']
    for label,index in [('Entropy',0),('Field energy',1),('Total',2)]:
        ax[1,0].plot(full['t'],d[:,index]/d[0,2],label=label)
    ax[1,0].set(title='Streaming + magnetic drift + nonlinear advection',xlabel='t',ylabel='Energy / initial total');ax[1,0].legend()
    for key,label in [('nonlinear','Nonlinear only'),('full','Full undriven system')]:
        rows=report[key]
        ax[1,1].loglog([r['dt'] for r in rows],[max(r['max_free_energy_relative_drift'],1e-17) for r in rows],'o-',label=label)
    ax[1,1].set(title='Time-step refinement (fixed spatial discretization)',xlabel='dt',ylabel='Max relative free-energy drift');ax[1,1].legend()
    for a in ax.flat:a.grid(alpha=.2)
    fig.suptitle('Multimode BSPF / FLR: no gradient drive, no dissipation, no energy correction',fontsize=14)
    fig.savefig(out/'conservation.png',dpi=180);fig.savefig(out/'conservation.pdf');plt.close(fig)


def main():
    out=Path('build/nonlinear_itg_conservation');out.mkdir(parents=True,exist_ok=True)
    p=plan_nonlinear_itg(n_x=33,n_y=9,n_z=7,n_v=6,n_mu=4)
    report=dict(parameters=dict(n_x=33,radial_modes=31,n_y=9,n_z=7,n_v=6,n_mu=4,
        ky_min=.3,kz_min=.1,rho=1.,tau=1.,curvature=.2,amplitude=.7,end=20.,
        gradients=0,collisions=0,radial_quadrature=12,periodic_truncation='strict 2/3'),nonlinear=[],full=[])
    for linear,key in [(False,'nonlinear'),(True,'full')]:
        finals=[]
        for dt in [.2,.1,.05]:
            row,arrays=run(p,dt,linear=linear)
            report[key].append(row);finals.append(arrays['final_state'])
            if dt==.05:
                np.savez_compressed(out/f'{key}.npz',**arrays)
                if linear:full=arrays
                else:nonlinear=arrays
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        differences=[float(np.linalg.norm(finals[i]-finals[i+1])) for i in range(2)]
        report[key+'_state_self_convergence']=dict(differences=differences,ratio=differences[0]/differences[1])
    plot(out,report,nonlinear,full)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    assert report['nonlinear'][-1]['max_entropy_relative_drift']<1e-8
    assert report['nonlinear'][-1]['max_field_relative_drift']<1e-8
    assert report['full'][-1]['max_free_energy_relative_drift']<1e-6
    assert report['nonlinear'][-1]['final_zonal_fraction']>1e-4
    assert report['nonlinear'][-1]['final_state_relative_change']>.01
    assert all(r['nl_max_normalized_rate']<1e-12 and r['full_normalized_free_energy_rate']<1e-12 for key in ['nonlinear','full'] for r in report[key])
    print('All conservation and nontrivial-transfer checks passed.',flush=True)

if __name__=='__main__':main()
