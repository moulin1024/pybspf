"""Continuum analytic MMS: RK4 evolution, spatial consistency, velocity quadrature."""
from pathlib import Path
import json
import gc
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bspf_models.kinetic.gyrokinetic_slab import plan_slab_gk
from bspf_models.kinetic.gyrokinetic_slab import slab_gk_fields
from bspf_models.kinetic.gyrokinetic_slab import slab_gk_rhs
from bspf_models.kinetic.gyrokinetic_slab import slab_gk_project
from bspf_models.kinetic.gyrokinetic_slab import slab_gk_diagnostics
from bspf_models.kinetic.gyrokinetic_slab import integrate_slab_gk
from bspf_models.kinetic.gyrokinetic_mms import plan_slab_mms


def norm(p,a):
    return jnp.sqrt(jnp.mean(jnp.sum(a*a*p.weights,axis=(-2,-1))))


def temporal():
    p=plan_slab_gk((7,7,7),n_v=64,n_mu=40,rho=.8)
    m=plan_slab_mms(p)
    g0,_=m.exact(0.); exact,phi=m.exact(.4)
    rows=[]
    for dt,steps in [(.04,10),(.02,20),(.01,40)]:
        h,t,budget=integrate_slab_gk(p,g0,dt,steps=steps,save_every=steps,
                                   source=m.source,return_budget=True)
        d0=slab_gk_diagnostics(p,h[0]); df=slab_gk_diagnostics(p,h[-1])
        rows.append(dict(dt=dt,g_error=float(norm(p,h[-1]-exact)),
            phi_error=float(jnp.sqrt(jnp.mean((slab_gk_fields(p,h[-1])[0]-phi)**2))),
            particle_balance=float(abs(df[0]-d0[0]-budget[-1,0])),
            relative_analytic_energy_error=float(abs(df[3]-m.exact_energy(.4)[0])/m.exact_energy(.4)[0]),
            analytic_source_transfer_error=float(abs(budget[-1,1]-(m.exact_energy(.4)[0]-m.exact_energy(0.)[0]))),
            relative_free_energy_balance=float(abs(df[3]-d0[3]-budget[-1,1])/d0[3])))
        print('time',rows[-1],flush=True)
    rows[1]['order']=float(np.log2(rows[0]['g_error']/rows[1]['g_error']))
    rows[2]['order']=float(np.log2(rows[1]['g_error']/rows[2]['g_error']))
    _,_,_,gt,stream,bracket=m.components(.17)
    checks=dict(streaming_norm=float(norm(p,stream)),bracket_norm=float(norm(p,bracket)),
                time_derivative_norm=float(norm(p,gt)))
    return rows,checks


def velocity():
    results={}
    for axis,sizes in [('v',[4,8,12,16,24,32,40]),('mu',[2,4,8,12,16,24,32])]:
        rows=[]
        for n in sizes:
            p=plan_slab_gk((7,7,7),n_v=n if axis=='v' else 40,
                           n_mu=n if axis=='mu' else 32,rho=.8)
            m=plan_slab_mms(p); g,phi=m.exact(.17)
            error=float(jnp.sqrt(jnp.mean((slab_gk_fields(p,g)[0]-phi)**2)))
            rows.append({'n':n,'phi_error':error})
        results[axis]=rows
        print(axis,rows,flush=True)
    return results


def spatial():
    rows=[]
    for n in [7,13,19,25]:
        # r>0 is genuinely non-bandlimited; a two-mode MMS would not suffice.
        p=plan_slab_gk((n,n,n),n_v=24,n_mu=40,rho=.8)
        m=plan_slab_mms(p,r=.3)
        g,phi,_,gt,stream,bracket=m.components(.17)
        gp=slab_gk_project(p,g)
        defect=slab_gk_rhs(p,gp)+slab_gk_project(p,gt+stream+bracket)-slab_gk_project(p,gt)
        rows.append(dict(n=n,projection_error=float(norm(p,gp-g)),
                        forced_rhs_defect=float(norm(p,defect)),
                        phi_error=float(jnp.sqrt(jnp.mean((slab_gk_fields(p,gp)[0]-phi)**2)))))
        print('space',rows[-1],flush=True)
        del p,m,g,gp,phi,gt,stream,bracket,defect
        jax.clear_caches(); gc.collect()
    return rows


def main():
    out=Path('build/gyrokinetic_mms'); out.mkdir(parents=True,exist_ok=True)
    time,checks=temporal()
    jax.clear_caches(); gc.collect()
    vel=velocity()
    jax.clear_caches(); gc.collect()
    space=spatial()
    report=dict(time=time,velocity=vel,spatial=space,nontrivial_terms=checks,
        spatial_scope='Semi-discrete projected manufactured PDE defect, not a time-evolution mesh study',
        velocity_scope='Continuum exact field versus numerical velocity quadrature')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,ax=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    ax[0,0].loglog([r['dt'] for r in time],[r['g_error'] for r in time],'o-',label='weighted g error')
    dt=np.array([r['dt'] for r in time]); ax[0,0].loglog(dt,time[0]['g_error']*(dt/dt[0])**4,'--',label='dt^4')
    ax[0,0].set(title='Forced evolution, T=0.4',xlabel='dt'); ax[0,0].legend()
    for key in ['projection_error','forced_rhs_defect']:
        ax[0,1].semilogy([r['n'] for r in space],[r[key] for r in space],'o-',label=key)
    ax[0,1].set(title='Spatial semi-discrete convergence',xlabel='N per spatial axis'); ax[0,1].legend()
    for axes,key,label in [(ax[1,0],'v','Hermite nodes'),(ax[1,1],'mu','Laguerre nodes')]:
        axes.semilogy([r['n'] for r in vel[key]],[r['phi_error'] for r in vel[key]],'o-')
        axes.set(title='Field error against continuum MMS',xlabel=label,ylabel='RMS phi error')
    fig.savefig(out/'convergence.png',dpi=160); plt.close(fig)
    assert all(3.7<r['order']<4.3 for r in time[1:])
    assert checks['bracket_norm']>1e-7 and checks['streaming_norm']>1e-4
    assert time[-1]['relative_free_energy_balance']<1e-7
    assert time[-1]['relative_analytic_energy_error']<1e-9
    assert space[-1]['forced_rhs_defect']<space[0]['forced_rhs_defect']/20
    assert space[-1]['projection_error']<space[0]['projection_error']/20
    for rows in vel.values():
        assert rows[-1]['phi_error']<1e-12
        assert rows[-1]['phi_error']<rows[0]['phi_error']/1e5

if __name__=='__main__': main()
