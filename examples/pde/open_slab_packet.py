"""Physical source-free nonperiodic GK pulse, BSPF convergence and flux budgets."""
from pathlib import Path
import json,time,gc
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bspf_jax.open_slab_packet import (plan_open_slab_packet,packet_initial,
    packet_reference,packet_reference_moments,packet_fields,packet_diagnostics,
    integrate_open_packet)
from bspf_jax.fast_axis import axis_values


def run(n,*,steps=2400,degree=7,q=12,n_v=192,endpoint_blend=0.,
        endpoint_method="finite_difference",boundary_points=None,chebyshev_modes=None):
    st=time.perf_counter()
    p=plan_open_slab_packet(n,degree=degree,n_basis=2*degree+2,n_v=n_v,quadrature_order=q,endpoint_blend=endpoint_blend,
        endpoint_method=endpoint_method,boundary_points=boundary_points,chebyshev_modes=chebyshev_modes)
    history,t,budget=integrate_open_packet(p,packet_initial(p),1.2/steps,steps=steps,save_every=steps//12)
    ref=packet_reference(p,p.axis.points,1.2)
    vals=axis_values(p.axis,history[-1]);ref0=packet_reference(p,p.axis.points,0.)
    norm=lambda u:jnp.sqrt(jnp.sum(p.axis.weights[:,None,None]*u*u*p.gamma[None,None,:]))
    norm0=norm(ref0)
    diag=np.asarray(jax.vmap(lambda x:packet_diagnostics(p,x))(history))
    moments_ref=np.stack([packet_reference_moments(p,float(tt)) for tt in t])
    budgets=np.asarray(budget)
    defectN=diag[:,0]-diag[0,0]+budgets[:,0]
    defectW=diag[:,1]-diag[0,1]+budgets[:,1]+budgets[:,2]
    periodic=packet_reference(p,p.axis.points,1.2,periodic=True)
    u=np.asarray(history)/np.asarray(p.sqrt_weights)[None,None,:,None]
    row=dict(n=n,degree=degree,n_basis=2*degree+2,n_v=n_v,n_mu=32,v_max=8.,q=q,dt=1.2/steps,endpoint_blend=endpoint_blend,
        endpoint_method=endpoint_method,boundary_points=(degree+2 if boundary_points is None and endpoint_method=="finite_difference" else boundary_points),chebyshev_modes=chebyshev_modes,
        relative_g_error=float(norm(vals-ref)/norm0),
        max_phi_error=float(jnp.max(jnp.abs(packet_fields(p,vals)-packet_fields(p,ref)))),
        max_particle_balance=float(np.max(np.abs(defectN))/abs(diag[0,0])),
        max_free_energy_balance=float(np.max(np.abs(defectW))/diag[0,1]),
        relative_particle_reference_error=float(np.max(np.abs(diag[:,0]-moments_ref[:,0]))/moments_ref[0,0]),
        relative_energy_reference_error=float(np.max(np.abs(diag[:,1]-moments_ref[:,1]))/moments_ref[0,1]),
        escaped_particle_fraction=float(budgets[-1,0]/diag[0,0]),
        escaped_energy_fraction=float(budgets[-1,1]/diag[0,1]),
        boundary_penalty_fraction=float(budgets[-1,2]/diag[0,1]),
        periodic_wrong_boundary_relative_error=float(norm(periodic-ref)/norm0),
        sampled_F_over_F0_lower_bound=float(np.min(1+u[:,:,:,0]-np.abs(u[:,:,:,1]))),
        seconds=time.perf_counter()-st)
    print(json.dumps(row),flush=True)
    arrays=dict(z=np.asarray(p.axis.x),zq=np.asarray(p.axis.points),times=np.asarray(t),weighted_distribution=np.asarray(history),
        phi=np.asarray(jax.vmap(lambda x:packet_fields(p,x))(history)),
        phi_exact=np.asarray(jax.vmap(lambda tt:packet_fields(p,packet_reference(p,p.axis.x,tt)))(t)),
        phi_periodic=np.asarray(packet_fields(p,packet_reference(p,p.axis.x,1.2,periodic=True))),
        diagnostics=diag,reference_moments=moments_ref,budgets=budgets,
        velocity=np.asarray(p.velocity),sqrt_weights=np.asarray(p.sqrt_weights),gamma=np.asarray(p.gamma))
    return row,arrays


def velocity_scan():
    z=jnp.linspace(-3.,3.,1001)
    records=[];fields=[]
    for nv in [32,64,96,128,192,256,384]:
        p=plan_open_slab_packet(49,n_v=nv)
        fields.append(np.asarray(packet_fields(p,packet_reference(p,z,1.2))))
        records.append(dict(n_v=nv))
    for row,phi in zip(records,fields):
        row['max_phi_difference_to_384']=float(np.max(np.abs(phi-fields[-1])))
    p=plan_open_slab_packet(49,n_v=384,v_max=10.)
    cutoff=float(np.max(np.abs(np.asarray(packet_fields(p,packet_reference(p,z,1.2)))-fields[-1])))
    print('velocity',json.dumps(records),'cutoff',cutoff,flush=True)
    return records,cutoff


def main():
    out=Path('build/open_slab_packet');out.mkdir(parents=True,exist_ok=True)
    rows=[];saved=None
    for n in [49,65,97,129,193]:
        row,arrays=run(n)
        if rows: row['observed_order']=float(np.log(rows[-1]['relative_g_error']/row['relative_g_error'])/np.log((n-1)/(rows[-1]['n']-1)))
        rows.append(row)
        (out/'spatial.json').write_text(json.dumps(rows,indent=2)+'\n')
        if n==129:
            saved=arrays;np.savez_compressed(out/'solution.npz',**arrays)
        del arrays;jax.clear_caches();gc.collect()
    half,half_arrays=run(129,steps=4800)
    time_diff=float(np.max(np.abs(half_arrays['phi']-saved['phi'])))
    del half_arrays;jax.clear_caches();gc.collect()
    lower,_=run(129,degree=5)
    jax.clear_caches();gc.collect()
    more_quad,_=run(129,q=16)
    jax.clear_caches();gc.collect()
    velocity,cutoff=velocity_scan()
    report=dict(spatial=rows,half_dt=half,half_dt_max_phi_difference=time_diff,
        degree5=lower,quadrature16=more_quad,velocity=velocity,velocity_cutoff_difference=cutoff,
        scope='Source-free linear ion-acoustic pulse with FLR; BSPF nonperiodic z, transparent characteristic reservoirs; reference exact after velocity quadrature')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for index,tt in [(0,0),(4,.4),(8,.8),(12,1.2)]:
        ax[0,0].plot(saved['z'],saved['phi'][index,:,0],label=f't={tt:.1f}')
    ax[0,0].set(title='Open-field ion-acoustic pulse: phi(k=0)',xlabel='z');ax[0,0].legend()
    ax[0,1].plot(saved['z'],saved['phi_exact'][-1,:,0],label='open characteristic reference')
    ax[0,1].plot(saved['z'][::4],saved['phi'][-1,::4,0],'o',ms=3,label='BSPF, N=129')
    ax[0,1].plot(saved['z'],saved['phi_periodic'][:,0],'--',label='wrong periodic boundary')
    ax[0,1].set(title='Nonperiodic boundary matters, t=1.2',xlabel='z');ax[0,1].legend()
    ns=np.array([a['n'] for a in rows]);err=np.array([a['relative_g_error'] for a in rows])
    ax[1,0].loglog(ns-1,err,'o-',label='degree 7 BSPF')
    ax[1,0].loglog(ns-1,err[0]*((ns-1)/(ns[0]-1))**(-8.),'--',label='h^8 guide')
    ax[1,0].loglog([128],[lower['relative_g_error']],'s',label='degree 5 at N=129')
    ax[1,0].set(title='Spatial convergence (including fine-grid plateau)',xlabel='N-1',ylabel='relative weighted L2 error');ax[1,0].legend()
    diag=saved['diagnostics'];bud=saved['budgets'];W0=diag[0,1]
    ax[1,1].plot(saved['times'],diag[:,1]/W0,label='energy in domain')
    ax[1,1].plot(saved['times'],bud[:,1]/W0,label='energy escaped')
    ax[1,1].plot(saved['times'],(diag[:,1]+bud[:,1]+bud[:,2])/W0,'--',label='domain + escaped + penalty')
    ax[1,1].set(title='Physical open-boundary free-energy budget',xlabel='t');ax[1,1].legend()
    fig.savefig(out/'validation.png',dpi=160);plt.close(fig)
    # Separate velocity accuracy from BSPF spatial convergence; do not conflate.
    assert all(r['observed_order']>7 for r in rows[1:4])
    assert rows[3]['relative_g_error']<3e-9
    assert rows[3]['periodic_wrong_boundary_relative_error']>.05
    assert rows[3]['escaped_energy_fraction']>.01
    assert rows[3]['max_particle_balance']<1e-8
    assert rows[3]['max_free_energy_balance']<1e-8
    assert rows[3]['sampled_F_over_F0_lower_bound']>0
    assert lower['relative_g_error']>10*rows[3]['relative_g_error']
    assert time_diff<1e-10
    assert velocity[4]['max_phi_difference_to_384']<1e-11
    assert cutoff<1e-11

if __name__=='__main__': main()
