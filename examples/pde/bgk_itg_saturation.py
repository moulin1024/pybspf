"""BGK-driven multimode ITG: restartable runs and windowed saturation diagnostics."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.integrate import trapezoid
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_initial
from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_fields
from bspf_models.kinetic.nonlinear_itg import nonlinear_itg_diagnostics
from bspf_models.kinetic.collisional_itg import plan_collisional_itg
from bspf_models.kinetic.collisional_itg import collisional_itg_rates
from bspf_models.kinetic.collisional_itg import collisional_itg_transport
from bspf_models.kinetic.collisional_itg import integrate_collisional_itg


def analyze(data,start,at):
    t=data['t'];d=data['diagnostics'];w=data['work'];r=data['rates'];q=data['transport'][:,1]
    res=d[:,2]-d[0,2]-w[:,0]-w[:,1]+w[:,2]
    indices=np.flatnonzero(t>=start)
    if len(indices)<9:raise ValueError('at least 9 tail samples required')
    ids=indices
    def avg(y,ix):return float(trapezoid(y[ix],t[ix])/(t[ix[-1]]-t[ix[0]]))
    # Four disjoint equal-duration windows, sharing boundary samples only.
    edges=t[np.rint(np.linspace(ids[0],len(t)-1,5)).astype(int)]
    blocks=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        ix=np.flatnonzero((t>=lo-1e-9)&(t<=hi+1e-9));duration=t[ix[-1]]-t[ix[0]]
        powers=(w[ix[-1]]-w[ix[0]])/duration
        blocks.append(dict(start=float(t[ix[0]]),end=float(t[ix[-1]]),Q=float(powers[1]/at) if at else avg(q,ix),W=avg(d[:,2],ix),
            field=avg(d[:,1],ix),zonal_fraction=avg(d[:,3]/d[:,1],ix),
            drive=float(powers[0]+powers[1]),dissipation=float(powers[2]),dWdt=float((d[ix[-1],2]-d[ix[0],2])/duration)))
    duration=t[-1]-t[ids[0]];powers=(w[-1]-w[ids[0]])/duration
    sampled_meanq=avg(q,ids);meanq=float(powers[1]/at) if at else sampled_meanq;qblocks=np.array([b['Q'] for b in blocks]);wb=np.array([b['W'] for b in blocks])
    durations=np.array([b['end']-b['start'] for b in blocks])
    half=float(abs(np.average(qblocks[:2],weights=durations[:2])-np.average(qblocks[2:],weights=durations[2:]))/max(abs(meanq),1e-30))
    balance=float(abs(powers[0]+powers[1]-powers[2])/max(abs(powers[0]+powers[1]),abs(powers[2]),1e-30))
    wchange=float(abs(np.average(wb[:2],weights=durations[:2])-np.average(wb[2:],weights=durations[2:]))/max(np.average(wb,weights=durations),1e-30))
    # Compare integrated free-energy spectra, not just potential energy.
    spectra={};distances={}
    for name in ['spectrum_ky','spectrum_radial','spectrum_velocity']:
        spectra[name]=trapezoid(data[name][ids],t[ids],axis=0)/duration
        first=np.flatnonzero((t>=start)&(t<=edges[2]));second=np.flatnonzero(t>=edges[2])
        a=trapezoid(data[name][first],t[first],axis=0)/(t[first[-1]]-t[first[0]])
        b=trapezoid(data[name][second],t[second],axis=0)/(t[second[-1]]-t[second[0]])
        distances[name]=float(np.sum(abs(a-b))/max(np.sum((a+b)/2),1e-30))
    # Conservative screening criteria, not proof of ergodicity or grid convergence.
    criteria=dict(work_budget_below_1e_minus5=bool(max(abs(res))/max(d[:,2])<1e-5),nonzero_transport=bool(meanq>1e-5),heat_half_change_below_20pct=bool(half<.2),
        W_half_change_below_20pct=bool(wchange<.2),mean_power_imbalance_below_5pct=bool(balance<.05),
        spectral_half_changes_below_25pct=bool(max(distances.values())<.25))
    fluct=q[ids]-sampled_meanq
    variance=float(np.mean(fluct*fluct));sample_dt=float(np.median(np.diff(t[ids])))
    if variance>0:
        ac=np.correlate(fluct,fluct,mode='full')[len(fluct)-1:]/(len(fluct)*variance)
        zeros=np.flatnonzero(ac[1:]<=0);cut=int(zeros[0]+1) if len(zeros) else len(ac)
        tau_int=sample_dt*(.5+float(ac[1:cut].sum()))
    else:tau_int=sample_dt/2
    report=dict(heat_tail_std=float(np.sqrt(variance)),heat_autocorrelation_integral_time=tau_int,
        approximate_independent_heat_samples=duration/(2*tau_int),
        averaging_start=float(t[ids[0]]),averaging_end=float(t[-1]),blocks=blocks,
        mean_heat_flux=meanq,heat_block_standard_error=float(qblocks.std(ddof=1)/2),
        heat_half_relative_change=half,W_half_relative_change=wchange,
        mean_drive=float(powers[0]+powers[1]),mean_dissipation=float(powers[2]),mean_power_relative_imbalance=balance,
        budget_relative_max=float(max(abs(res))/max(d[:,2])),
        sampled_heat_work_relative_discrepancy=float(abs(sampled_meanq-meanq)/max(abs(meanq),1e-30)),
        velocity_last_order_energy_fraction=float((spectra['spectrum_velocity'][-1,:].sum()+spectra['spectrum_velocity'][:,-1].sum()-spectra['spectrum_velocity'][-1,-1])/spectra['spectrum_velocity'].sum()),
        radial_last_three_energy_fraction=float(spectra['spectrum_radial'][-3:].sum()/spectra['spectrum_radial'].sum()),
        spectrum_parseval_relative_max=float(max(np.max(abs(data['spectrum_ky'].sum(axis=1)-d[:,2])),np.max(abs(data['spectrum_radial'].sum(axis=1)-d[:,2])),np.max(abs(data['spectrum_velocity'].sum(axis=(1,2))-d[:,0])))/max(d[:,2])),
        spectral_half_relative_changes=distances,criteria=criteria,stationarity_screen_pass=all(criteria.values()))
    return report,spectra


def plot(out,data,report,spectra,ky):
    t=data['t'];d=data['diagnostics'];r=data['rates'];w=data['work'];q=data['transport'][:,1]
    fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for index,label in [(2,'Total W'),(1,'Field'),(3,'Zonal field')]:ax[0,0].semilogy(t,np.where(d[:,index]>1e-12,d[:,index],np.nan),label=label)
    ax[0,0].set(title='Driven BGK evolution',ylabel='Energy')
    ax[0,1].plot(t,q,label='Heat flux Q');ax[0,1].axhline(report['mean_heat_flux'],ls='--',color='k',label='Tail mean')
    ax[0,1].set(title='Temperature-gradient-conjugate heat flux',ylabel='Q (radial integral)')
    ax[0,2].plot(t,r[:,:2].sum(axis=1),label='Gradient injection');ax[0,2].plot(t,r[:,2],label='BGK dissipation')
    ax[0,2].set(title='Tail: instantaneous power',ylabel='Power')
    tail=t>=report['averaging_start']
    for a,values in [(ax[0,1],q[tail]),(ax[0,2],r[tail,1:].ravel())]:
        lo=float(values.min());hi=float(values.max());pad=max((hi-lo)*.1,1e-10)
        a.set_xlim(report['averaging_start'],t[-1]);a.set_ylim(lo-pad,hi+pad)
    ax[0,1].set_title('Tail: temperature-gradient-conjugate heat flux')
    res=d[:,2]-d[0,2]-w[:,0]-w[:,1]+w[:,2]
    ax[1,0].plot(t,res/max(d[:,2]),label='Work budget residual')
    ax[1,0].set(title='W-W0-integral(P-D) dt',ylabel='Residual / max W')
    for b in report['blocks']:ax[1,1].plot([b['start'],b['end']],[b['Q'],b['Q']],lw=3)
    ax[1,1].set(title='Four tail-window mean heat fluxes',ylabel='Mean Q')
    ax[1,2].plot(t,d[:,3]/d[:,1],label='Zonal / field energy');ax[1,2].set(title='Self-generated zonal component',ylabel='Fraction')
    for a in ax.flat:
        a.set_xlabel('t');a.axvspan(report['averaging_start'],t[-1],alpha=.08,color='gray');a.grid(alpha=.2)
        if a.get_legend_handles_labels()[0]:a.legend(fontsize=8)
    params=report.get('parameters',{})
    fig.suptitle(f"BGK {params.get('model','')} | nu={params.get('nu','')} | Nx={params.get('nx','')} | tail stationarity screen: {report['stationarity_screen_pass']}")
    fig.savefig(out/'saturation.png',dpi=150);fig.savefig(out/'saturation.pdf');plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    order=np.argsort(ky)
    ax[0].semilogy(ky[order],np.where(spectra['spectrum_ky'][order]>1e-20,spectra['spectrum_ky'][order],np.nan),'o-');ax[0].set(xlabel='ky',title='Time-averaged free energy')
    ax[1].semilogy(np.arange(1,len(spectra['spectrum_radial'])+1),spectra['spectrum_radial'],'o-');ax[1].set(xlabel='Radial eigenmode',title='Time-averaged free energy')
    im=ax[2].imshow(np.log10(np.maximum(spectra['spectrum_velocity'],1e-30)).T,origin='lower',aspect='auto');fig.colorbar(im,ax=ax[2],label='log10 entropy spectrum')
    ax[2].set(xlabel='Hermite order',ylabel='Laguerre order',title='Velocity-space entropy spectrum')
    mid=(report['averaging_start']+t[-1])/2
    for lo,hi,label in [(report['averaging_start'],mid,'First half'),(mid,t[-1],'Second half')]:
        ix=(t>=lo)&(t<=hi)
        for axis,name in [(ax[0],'spectrum_ky'),(ax[1],'spectrum_radial')]:
            spec=trapezoid(data[name][ix],t[ix],axis=0)/(t[ix][-1]-t[ix][0])
            xx=ky[order] if name=='spectrum_ky' else np.arange(1,len(spec)+1)
            yy=spec[order] if name=='spectrum_ky' else spec
            axis.semilogy(xx,np.where(yy>1e-20,yy,np.nan),'--',alpha=.7,label=label)
    for a in ax[:2]:a.set_ylabel('Energy');a.grid(alpha=.2);a.legend(fontsize=8)
    fig.savefig(out/'spectra.png',dpi=150);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--radial-kernel',choices=['quadrature','tensor'],default='quadrature');ap.add_argument('--model',choices=['local','gyroaveraged'],default='gyroaveraged');ap.add_argument('--nu',type=float,default=.15);ap.add_argument('--dt',type=float,default=.05)
    ap.add_argument('--end',type=float,default=600.);ap.add_argument('--average-start',type=float,default=400.)
    ap.add_argument('--nx',type=int,default=33);ap.add_argument('--ny',type=int,default=9);ap.add_argument('--nz',type=int,default=7)
    ap.add_argument('--nv',type=int,default=8);ap.add_argument('--nmu',type=int,default=6);ap.add_argument('--at',type=float,default=4.)
    ap.add_argument('--out',default='build/bgk_itg');ap.add_argument('--resume',action='store_true');ap.add_argument('--restart-from')
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.resume and a.restart_from:raise ValueError('choose resume or restart-from')
    old=json.loads((out/'parameters.json').read_text()) if a.resume else None
    if old:
        if old.get('model','local')!=a.model:raise ValueError('resume collision model mismatch')
        for key in ['nu','dt','nx','ny','nz','nv','nmu','at']:
            if old[key]!=vars(a)[key]:raise ValueError(f'resume parameter mismatch: {key}')
    (out/'parameters.json').write_text(json.dumps(vars(a),indent=2)+'\n')
    p=plan_collisional_itg(nu=a.nu,model=a.model,n_x=a.nx,n_y=a.ny,n_z=a.nz,n_v=a.nv,n_mu=a.nmu,a_t=a.at);b=p.base
    if a.radial_kernel=='tensor':
        from bspf_models.kinetic.itg_bracket_tensor import plan_itg_bracket_tensor
        from bspf_models.kinetic.itg_bracket_tensor import integrate_collisional_itg_tensor
        tensor=plan_itg_bracket_tensor(b.radial)
        advance=lambda u:integrate_collisional_itg_tensor(p,tensor,u,a.dt,steps=steps,save_every=save)
    else:advance=lambda u:integrate_collisional_itg(p,u,a.dt,steps=steps,save_every=save)
    # Orthonormal Hermite/Laguerre transforms, using the same Gaussian quadrature.
    from scipy.special import eval_hermitenorm,eval_laguerre,gammaln
    sv=np.sqrt(np.asarray(b.sqrt_weights**2).sum(axis=1));sm=np.sqrt(np.asarray(b.sqrt_weights**2).sum(axis=0))
    hv=np.stack([sv*eval_hermitenorm(k,np.asarray(b.velocity))*np.exp(-.5*gammaln(k+1)) for k in range(a.nv)],axis=1)
    lm=np.stack([sm*eval_laguerre(k,np.asarray(b.mu)) for k in range(a.nmu)],axis=1)
    hv=jnp.asarray(hv);lm=jnp.asarray(lm)
    @jax.jit
    def observe(x):
        d=nonlinear_itg_diagnostics(b,x);r=collisional_itg_rates(p,x);q=collisional_itg_transport(p,x)
        phi,_=nonlinear_itg_fields(b,x);xh=jnp.fft.fftn(x,axes=(1,2),norm='ortho')
        modes=(.5*jnp.sum(abs(xh)**2,axis=(-2,-1))+.5*b.polarization*abs(phi)**2)/(a.ny*a.nz)
        vel=jnp.einsum('...vm,vi,mj->...ij',x,hv,lm)
        vs=.5*jnp.sum(vel*vel,axis=(0,1,2))/(a.ny*a.nz)
        return d,r,q,jnp.sum(modes,axis=(0,2)),jnp.sum(modes,axis=(1,2)),vs
    names=['diagnostics','rates','transport','spectrum_ky','spectrum_radial','spectrum_velocity']
    if a.resume:
        loaded=dict(np.load(out/'history.npz'));x=jnp.asarray(loaded.pop('final_state'));data={k:list(v) for k,v in loaded.items()};start=float(data['t'][-1])
    else:
        if a.restart_from:
            loaded=np.load(a.restart_from);x=jnp.asarray(loaded['final_state']);start=float(loaded['t'][-1])
            expected=(a.nx-2,a.ny,a.nz,a.nv,a.nmu)
            if x.shape!=expected:raise ValueError(f'restart state shape {x.shape} differs from {expected}')
        else:x=nonlinear_itg_initial(b,amplitude=.05);start=0.
        values=observe(x)
        data={k:[np.asarray(v)] for k,v in zip(names,values)};data.update(t=[start],work=[np.zeros(3)])
    block=10.;sample=1.;steps=round(block/a.dt);save=round(sample/a.dt)
    if not np.isclose(steps*a.dt,block) or not np.isclose(save*a.dt,sample) or not np.isclose((a.end-start)/block,round((a.end-start)/block)):raise ValueError('dt must divide 1 and duration must be a multiple of 10')
    tick=time.perf_counter()
    for k in range(round((a.end-start)/block)):
        h,t,w=advance(x)
        h.block_until_ready()
        if not np.isfinite(np.asarray(h[-1])).all():raise RuntimeError(f'Nonfinite state after t={data["t"][-1]}: reduce dt, last good checkpoint retained')
        for j in range(1,len(h)):
            for name,val in zip(names,observe(h[j])):data[name].append(np.asarray(val))
            data['t'].append(start+k*block+float(t[j]))
        data['work'].extend(np.asarray(w[1:])+data['work'][-1]);x=h[-1]
        arrays={k:np.array(v) for k,v in data.items()};arrays['final_state']=np.asarray(x)
        np.savez_compressed(out/'history.tmp.npz',**arrays)
        (out/'history.tmp.npz').replace(out/'history.npz')
        d=data['diagnostics'][-1];r=data['rates'][-1]
        print(f't={data["t"][-1]:g} W={d[2]:.5g} Q={data["transport"][-1][1]:.5g} P={r[:2].sum():.5g} D={r[2]:.5g} zonal/E={d[3]/d[1]:.3f} wall={time.perf_counter()-tick:.0f}s',flush=True)
    arrays={k:np.array(v) for k,v in data.items()};report,spectra=analyze(arrays,a.average_start,a.at)
    report['parameters']=vars(a);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(out/'mean_spectra.npz',**spectra)
    plot(out,arrays,report,spectra,np.asarray(b.ky));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
