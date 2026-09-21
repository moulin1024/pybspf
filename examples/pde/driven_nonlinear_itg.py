"""Fixed-gradient ITG growth into nonlinear transfer, with self-generated zonal flow."""
from pathlib import Path
import argparse,json
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bspf_jax.nonlinear_itg import (plan_nonlinear_itg,nonlinear_itg_initial,
    nonlinear_itg_rhs,nonlinear_itg_fields,nonlinear_itg_diagnostics,
    nonlinear_itg_drive_power,integrate_driven_itg)


def run(p,dt,end,amplitude,nonlinear):
    x=nonlinear_itg_initial(p,amplitude=amplitude)
    diag=jax.jit(lambda u:nonlinear_itg_diagnostics(p,u))
    field=jax.jit(lambda u:nonlinear_itg_fields(p,u)[0])
    power=jax.jit(lambda u:nonlinear_itg_drive_power(p,u))
    ratio=jax.jit(lambda u:jnp.linalg.norm(nonlinear_itg_rhs(p,u,include_linear=False))/jnp.linalg.norm(nonlinear_itg_rhs(p,u,include_nonlinear=False)))
    times=[0.];ds=[np.asarray(diag(x))];phis=[np.asarray(field(x))];works=[np.zeros(2)];ps=[np.asarray(power(x))];ratios=[float(ratio(x))]
    sample=.5; block=5.;steps=round(block/dt);save=round(sample/dt)
    if not np.isclose(steps*dt,block) or not np.isclose(save*dt,sample) or not np.isclose(round(end/block)*block,end):
        raise ValueError('dt must divide 0.5 and end must be a multiple of 5')
    for k in range(round(end/block)):
        h,t,w=integrate_driven_itg(p,x,dt,steps=steps,save_every=save,include_nonlinear=nonlinear)
        h.block_until_ready()
        for j in range(1,len(h)):
            times.append(k*block+float(t[j]));ds.append(np.asarray(diag(h[j])));phis.append(np.asarray(field(h[j])))
            ps.append(np.asarray(power(h[j])));ratios.append(float(ratio(h[j])))
        works.extend(np.asarray(w[1:])+works[-1]);x=h[-1]
        if not np.isfinite(np.asarray(x)).all():raise RuntimeError(f'nonfinite state at {times[-1]}')
        print(f'nonlinear={nonlinear} t={times[-1]:g} W={ds[-1][2]:.5g} zonal/E={ds[-1][3]/ds[-1][1]:.4g} NL/L={ratios[-1]:.4g}',flush=True)
    return dict(t=np.array(times),diagnostics=np.array(ds),phi_hat=np.array(phis),work=np.array(works),power=np.array(ps),nl_ratio=np.array(ratios),final_state=np.asarray(x))


def plot(out,p,n,l):
    fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    t=n['t'];d=n['diagnostics'];dl=l['diagnostics']
    for data,label in [(n,'Nonlinear + zonal'),(l,'Linear control')]:
        ax[0,0].semilogy(t,data['diagnostics'][:,1],label=label)
    ax[0,0].semilogy(t,np.where(d[:,3]>1e-14,d[:,3],np.nan),label='Zonal field energy')
    ax[0,0].set(title='ITG growth and departure from linear control',ylabel='Field energy')
    ax[0,1].plot(t,d[:,3]/d[:,1],label='Zonal / total field energy')
    ax[0,1].set(title='Self-generated zonal component',ylabel='Fraction')
    ax[0,2].semilogy(t,n['nl_ratio'],label='Nonlinear / linear RHS norm')
    ax[0,2].axhline(.1,color='gray',ls=':',label='0.1 diagnostic threshold')
    ax[0,2].set(title='Nonlinear strength',ylabel='Norm ratio')
    for data,label in [(n,'Nonlinear'),(l,'Linear control')]:
        a=data['diagnostics'];res=a[:,2]-a[0,2]-data['work'].sum(axis=1)
        ax[1,0].plot(t,res/max(a[:,2]),label=label)
    ax[1,0].set(title='W - W(0) - integrated drive work',ylabel='Residual / max W')
    gamma=.5*np.gradient(np.log(d[:,1]),t);gammal=.5*np.gradient(np.log(dl[:,1]),t)
    ax[1,1].plot(t,gamma,label='Nonlinear');ax[1,1].plot(t,gammal,label='Linear control')
    ax[1,1].set(title='Instantaneous amplitude growth rate',ylabel='0.5 d log(E) / dt')
    # Physical zonal velocity: v_Ey = -rho d_x <phi>_yz; orthonormal FFT.
    zonal=np.real(n['phi_hat'][:, :,0,0])/np.sqrt(p.ky.size*p.kz.size)
    vel=-float(p.rho)*zonal@np.asarray(p.radial.derivative_values).T
    im=ax[1,2].pcolormesh(np.asarray(p.radial.points),t,vel,shading='auto',cmap='RdBu_r')
    fig.colorbar(im,ax=ax[1,2],label='Zonal v_Ey')
    ax[1,2].set(title='Zonal flow in finite radial interval',xlabel='x',ylabel='t')
    for a in ax.flat:
        if a is not ax[1,2]:a.set_xlabel('t');a.legend(fontsize=8);a.grid(alpha=.2)
    fig.suptitle('Driven multimode ITG: standard BSPF, full FLR, no collisions or artificial damping')
    fig.savefig(out/'growth.png',dpi=160);fig.savefig(out/'growth.pdf');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for data,label,style in [(n,'Nonlinear','-'),(l,'Linear control','--')]:
        energy=.5*np.asarray(p.polarization)*np.abs(data['phi_hat'][-1])**2/(p.ky.size*p.kz.size)
        ky=np.asarray(p.ky);order=np.argsort(ky)
        ax[0].semilogy(ky[order],np.where(energy.sum(axis=(0,2))[order]>1e-20,energy.sum(axis=(0,2))[order],np.nan),'o'+style,label=label)
        ax[1].semilogy(np.arange(1,energy.shape[0]+1),np.where(energy.sum(axis=(1,2))>1e-20,energy.sum(axis=(1,2)),np.nan),'o'+style,label=label)
    ax[0].set(xlabel='ky',ylabel='Field energy',title='Final periodic mode spectrum')
    ax[1].set(xlabel='Radial eigenmode',ylabel='Field energy',title='Final radial mode spectrum')
    for a in ax:a.legend();a.grid(alpha=.2)
    fig.savefig(out/'spectra.png',dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dt',type=float,default=.05);ap.add_argument('--end',type=float,default=40.)
    ap.add_argument('--nv',type=int,default=12);ap.add_argument('--nmu',type=int,default=8)
    ap.add_argument('--ny',type=int,default=9);ap.add_argument('--nz',type=int,default=7)
    ap.add_argument('--amplitude',type=float,default=.05);ap.add_argument('--out',default='build/driven_nonlinear_itg')
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    p=plan_nonlinear_itg(n_x=33,n_y=a.ny,n_z=a.nz,n_v=a.nv,n_mu=a.nmu,a_t=4.)
    n=run(p,a.dt,a.end,a.amplitude,True);np.savez_compressed(out/'nonlinear.npz',**n)
    l=run(p,a.dt,a.end,a.amplitude,False);np.savez_compressed(out/'linear.npz',**l)
    report=dict(parameters=vars(a),n_x=33,a_n=0.,a_t=4.,curvature=.2,collisions=0,artificial_damping=0)
    for data,key in [(n,'nonlinear'),(l,'linear')]:
        d=data['diagnostics'];res=d[:,2]-d[0,2]-data['work'].sum(axis=1)
        fit=(data['t']>=5)&(data['t']<=15)
        report[key]=dict(budget_relative_max=float(max(abs(res))/max(d[:,2])),
            fitted_gamma_5_15=float(np.polyfit(data['t'][fit],np.log(d[fit,1]),1)[0]/2),
            final_zonal_fraction=float(d[-1,3]/d[-1,1]),final_potential_nl_ratio=float(data['nl_ratio'][-1]),
            final_field_energy=float(d[-1,1]),final_free_energy=float(d[-1,2]))
    reached=np.where(n['nl_ratio']>=.1)[0]
    report['nonlinear_threshold_time']=float(n['t'][reached[0]]) if len(reached) else None
    report['final_field_to_linear']=float(n['diagnostics'][-1,1]/l['diagnostics'][-1,1])
    # Regression guards test a real transition and a resolved work budget.
    assert report['nonlinear']['budget_relative_max']<1e-5
    if a.end>=40 and a.amplitude==.05:
        assert report['nonlinear_threshold_time'] is not None
        assert report['nonlinear']['final_zonal_fraction']>.05
        assert report['final_field_to_linear']<.8
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');plot(out,p,n,l)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
