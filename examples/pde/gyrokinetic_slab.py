"""Run a nonlinear FLR slab, time refinement and drift-limit comparison."""
from pathlib import Path
import json
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bspf_models.kinetic.gyrokinetic_slab import plan_slab_gk
from bspf_models.kinetic.gyrokinetic_slab import slab_gk_project
from bspf_models.kinetic.gyrokinetic_slab import integrate_slab_gk
from bspf_models.kinetic.gyrokinetic_slab import slab_gk_diagnostics
from bspf_models.kinetic.gyrokinetic_slab import slab_gk_fields
from bspf_models.kinetic.gyrokinetic_slab import slab_gk_rhs
from dataclasses import replace


def run():
    out=Path('build/gyrokinetic_slab'); out.mkdir(parents=True,exist_ok=True)
    p=plan_slab_gk((9,9,9),n_v=16,n_mu=16,rho=.8)
    x=p.x[:,None,None,None,None]; y=p.y[None,:,None,None,None]
    z=p.z[None,None,:,None,None]; v=p.v[None,None,None,:,None]
    mu=p.mu[None,None,None,None,:]
    initial=.03*jnp.cos(x+z)*(1+.2*jnp.tanh(v))+ .02*jnp.cos(2*y-z)*(1+.3*mu/(1+mu))
    initial=slab_gk_project(p,initial)
    h,t=integrate_slab_gk(p,initial,.005,steps=400,save_every=10)
    fine,_=integrate_slab_gk(p,initial,.0025,steps=800,save_every=20)
    coarse,_=integrate_slab_gk(p,initial,.01,steps=200,save_every=200)
    drift=plan_slab_gk((9,9,9),n_v=16,n_mu=16,rho=0)
    hd,_=integrate_slab_gk(drift,initial,.005,steps=400,save_every=10)
    diag=np.asarray(jax.vmap(lambda g:slab_gk_diagnostics(p,g))(h))
    fd=np.asarray(jax.vmap(lambda g:slab_gk_fields(p,g)[0])(h))
    dd=np.asarray(jax.vmap(lambda g:slab_gk_fields(drift,g)[0])(hd))
    finer=np.asarray(jax.vmap(lambda g:slab_gk_diagnostics(p,g))(fine))
    err=float(jnp.max(jnp.abs(h[-1]-fine[-1])))
    coarse_error=float(jnp.max(jnp.abs(coarse[-1]-h[-1])))
    report={'shape':list(initial.shape),'rho':.8,'tau':1.,'t_final':2.,'dt':.005,
       'max_particle_drift':float(np.max(np.abs(diag[:,0]-diag[0,0]))),
       'max_relative_free_energy_drift':float(np.max(np.abs(diag[:,3]/diag[0,3]-1))),
       'half_dt_relative_free_energy_drift':float(np.max(np.abs(finer[:,3]/finer[0,3]-1))),
       'min_sampled_F_over_F0':float(diag[:,4].min()),
       'dt_halving_max_distribution_difference':err,
       'time_refinement_ratio':coarse_error/err,
       'nonlinear_rhs_norm':float(jnp.linalg.norm(slab_gk_rhs(p,initial)-slab_gk_rhs(replace(p,nonlinear=False),initial))),
       'flr_vs_drift_final_phi_difference':float(np.max(np.abs(fd[-1]-dd[-1])))}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(out/'solution.npz',times=t,g=h,phi=fd,phi_drift=dd,
                        diagnostics=diag,x=p.x,y=p.y,z=p.z,v=p.v,mu=p.mu)
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    axes[0,0].plot(t,np.sqrt(np.mean(fd**2,axis=(1,2,3))),label='full FLR, rho=0.8')
    axes[0,0].plot(t,np.sqrt(np.mean(dd**2,axis=(1,2,3))),label='rho=0')
    axes[0,0].set(title='Self-consistent potential',xlabel='t',ylabel='RMS phi'); axes[0,0].legend()
    axes[0,1].plot(t,diag[:,1],label='distribution entropy'); axes[0,1].plot(t,diag[:,2],label='field + polarization')
    axes[0,1].plot(t,diag[:,3],label='total free energy'); axes[0,1].legend(); axes[0,1].set(xlabel='t')
    axes[1,0].semilogy(t,np.maximum(np.abs(diag[:,3]/diag[0,3]-1),1e-17),label='dt=0.005')
    axes[1,0].semilogy(t,np.maximum(np.abs(finer[:,3]/finer[0,3]-1),1e-17),label='dt=0.0025')
    axes[1,0].set(title='Relative free-energy drift',xlabel='t'); axes[1,0].legend()
    im=axes[1,1].pcolormesh(p.x,p.y,fd[-1,:,:,0].T,shading='nearest',cmap='RdBu_r')
    axes[1,1].set(title='phi(x,y,z=0), t=2',xlabel='x',ylabel='y'); fig.colorbar(im,ax=axes[1,1])
    fig.savefig(out/'validation.png',dpi=160); plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)
    assert report['max_particle_drift']<1e-12
    assert report['max_relative_free_energy_drift']<1e-7
    assert report['min_sampled_F_over_F0']>0
    assert 10<report['time_refinement_ratio']<22
    assert report['nonlinear_rhs_norm']>1e-4
    assert err<1e-7

if __name__=='__main__': run()
