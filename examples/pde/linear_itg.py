"""Reproducible linear ITG validation, standard BSPF radial grids through 129."""
from pathlib import Path
import json,time,gc
import numpy as np
from scipy.special import j0
from scipy.optimize import root
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bspf_jax.linear_itg import (plan_itg_radial,plan_linear_itg,linear_itg_initial,
    linear_itg_fields,linear_itg_diagnostics,integrate_linear_itg)
from bspf_jax.linear_itg_reference import growing_root,continuum_growing_root,continuum_dispersion


def fit_omega(t,phi):
    growth=np.polyfit(t,np.log(np.abs(phi)),1)[0]
    frequency=-np.polyfit(t,np.unwrap(np.angle(phi)),1)[0]
    return complex(frequency,growth)


def evolve(n=65,*,n_v=48,n_mu=32,dt=.01,end=12.,generic=False,**parameters):
    start=time.perf_counter()
    radial=plan_itg_radial(n)
    p=plan_linear_itg(radial,n_v=n_v,n_mu=n_mu,**parameters)
    ref=growing_root(n_v=n_v,n_mu=n_mu,guess=parameters.get('ky',.3)/.3*(.32+.16j),**parameters)
    initial=linear_itg_initial(p,omega=None if generic else ref,amplitude=1e-9 if generic else 1e-7)
    steps=round(end/dt);save_every=max(1,steps//60)
    history,t,budgets=integrate_linear_itg(p,initial,dt,steps=steps,save_every=save_every)
    phis=np.asarray(jax.vmap(lambda x:linear_itg_fields(p,x)[0])(history))
    diagnostics=np.asarray(jax.vmap(lambda x:linear_itg_diagnostics(p,x))(history))
    t=np.asarray(t);budgets=np.asarray(budgets)
    fit_slice=slice(len(t)*2//3,None) if generic else slice(None)
    measured=fit_omega(t[fit_slice],phis[fit_slice,0])
    balance=diagnostics[:,2]-diagnostics[0,2]-budgets.sum(axis=1)
    eigen_shape=np.sqrt(2/12)*np.sin(np.pi*np.asarray(radial.points)/12)
    shape_error=float(jnp.sqrt(jnp.sum(radial.weights*(radial.values[:,0]-eigen_shape)**2)))
    # Independent, nontrivial radial FLR action, using a higher Dirichlet mode.
    mu=.4;ky=parameters.get('ky',.3);rho=parameters.get('rho',1.)
    probe=np.sqrt(2/12)*np.sin(7*np.pi*np.asarray(radial.points)/12)
    coeff=np.asarray(radial.values.T@(radial.weights*probe))
    multiplier=j0(rho*np.sqrt(2*mu*(np.asarray(radial.eigenvalues)+ky*ky)))
    result=np.asarray(radial.values)@(multiplier*coeff)
    exact=j0(rho*np.sqrt(2*mu*((7*np.pi/12)**2+ky*ky)))*probe
    flr_error=float(np.sqrt(np.sum(np.asarray(radial.weights)*np.abs(result-exact)**2))/np.sqrt(np.sum(np.asarray(radial.weights)*exact**2)))
    row=dict(n=n,n_v=n_v,n_mu=n_mu,dt=dt,end=end,generic_seed=generic,parameters=parameters,
        measured_frequency=measured.real,measured_growth=measured.imag,
        reference_frequency=ref.real,reference_growth=ref.imag,
        frequency_error=abs(measured.real-ref.real),growth_error=abs(measured.imag-ref.imag),
        radial_eigenfunction_l2_error=shape_error,radial_flr_mode7_relative_error=flr_error,
        max_relative_energy_balance=float(np.max(np.abs(balance))/np.max(diagnostics[:,2])),
        relative_balance_to_initial=float(np.max(np.abs(balance))/diagnostics[0,2]),
        seconds=time.perf_counter()-start)
    print(json.dumps(row),flush=True)
    arrays=dict(t=t,phi=phis[:,0],diagnostics=diagnostics,budgets=budgets,
        radial_points=np.asarray(radial.points),radial_eigenfunction=np.asarray(radial.values[:,0]),
        radial_exact=eigen_shape)
    return row,arrays


def scans():
    continuous=continuum_growing_root()
    velocity=[]
    for nv,nm in [(16,12),(24,16),(32,24),(48,32),(64,48),(80,64),(128,96),(192,160),(256,192)]:
        z=growing_root(n_v=nv,n_mu=nm)
        velocity.append(dict(n_v=nv,n_mu=nm,frequency=z.real,growth=z.imag,error=abs(z-continuous)))
    gradient=[]
    for at in [1.7,1.8,2.,2.5,3.,4.,6.,8.]:
        z=continuum_growing_root(a_t=at,guess=.25+.02j if at<2 else .32+.16j)
        gradient.append(dict(a_t=at,frequency=z.real,growth=z.imag))
    ky_scan=[]
    for ky in [.15,.2,.3,.4,.5,.7,1.]:
        try:
            z=continuum_growing_root(ky=ky,guess=ky/.3*(.32+.16j))
            ky_scan.append(dict(ky=ky,frequency=z.real,growth=z.imag,resolved=True))
        except ValueError:
            ky_scan.append(dict(ky=ky,resolved=False))
    # Follow the selected ITG branch to marginality from positive growth.
    marginal=[]
    for gamma in [.003,.001,.0003,.0001]:
        def fun(y):
            d=continuum_dispersion(y[0]+1j*gamma,a_t=y[1]);return [d.real,d.imag]
        sol=root(fun,[.235,1.65],tol=1e-9)
        if not sol.success or np.linalg.norm(fun(sol.x))>1e-8:raise RuntimeError('marginal branch solve failed')
        marginal.append(dict(growth=gamma,a_t=float(sol.x[1]),frequency=float(sol.x[0])))
    critical=float(np.polyfit([r['growth'] for r in marginal[-3:]],[r['a_t'] for r in marginal[-3:]],1)[1])
    return dict(model=dict(length=12.,ky=.3,k_parallel=.1,curvature=.2,a_n=0.,a_t=4.,rho=1.,tau=1.,
                radial_mode=1,geometry='constant-curvature shearless local model',boundary='spectral Dirichlet radial closure'),
                continuous=dict(frequency=continuous.real,growth=continuous.imag),velocity=velocity,
                gradient=gradient,ky=ky_scan,marginal=marginal,critical_gradient_extrapolation=critical)


def plot(out,report,saved,generic):
    fig,ax=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    t=saved['t'];phi=saved['phi'];r=report['spatial'][-1]
    ax[0,0].semilogy(t,abs(phi/phi[0]),'o',ms=3,label='BSPF evolution')
    ax[0,0].semilogy(t,np.exp(r['reference_growth']*t),'-',label='Independent velocity root')
    ax[0,0].set(title='Linear ITG growth, N=129',xlabel='t',ylabel='Potential amplitude / initial');ax[0,0].legend()
    for a,key,xkey,title in [(ax[0,1],'gradient','a_t','Continuous dispersion: gradient scan'),(ax[0,2],'ky','ky','Continuous dispersion: wavenumber scan')]:
        rows=[r for r in report[key] if r.get('resolved',True)]
        a.plot([r[xkey] for r in rows],[r['growth'] for r in rows],'o-',label='growth rate')
        a.plot([r[xkey] for r in rows],[r['frequency'] for r in rows],'s--',label='frequency')
        numerical=[r for r in report.get('parameter_validation',[]) if xkey in r['parameters']]
        if numerical:
            a.plot([r['parameters'][xkey] for r in numerical],[r['measured_growth'] for r in numerical],
                'kx',ms=7,label='BSPF growth (48 x 32 velocity)')
        a.set(title=title,xlabel=xkey);a.legend()
    ax[0,1].axvline(report['critical_gradient_extrapolation'],color='gray',ls=':',label='marginal branch')
    rows=report['spatial'];n=[r['n']-1 for r in rows]
    ax[1,0].loglog(n,[r['radial_flr_mode7_relative_error'] for r in rows],'o-',label='FLR action, radial mode 7')
    ax[1,0].loglog(n,[r['radial_eigenfunction_l2_error'] for r in rows],'s-',label='Eigenfunction, mode 1')
    ax[1,0].set(title='Standard BSPF radial accuracy',xlabel='N - 1',ylabel='L2 error');ax[1,0].legend(fontsize=9)
    vel=report['velocity'];ax[1,1].loglog([r['n_v'] for r in vel],[r['error'] for r in vel],'o-')
    ax[1,1].set(title='Velocity convergence to continuous integral',xlabel='Nv (Nmu also refined)',ylabel='Complex frequency error')
    energy=saved['diagnostics'][:,2];budget=saved['budgets'].sum(axis=1)
    ax[1,2].plot(t,energy/energy[0],label='Free energy / initial')
    ax[1,2].plot(t,1+budget/energy[0],'--',label='1 + gradient work / initial')
    ax[1,2].set(title='Energy comes from the fixed gradient',xlabel='t');ax[1,2].legend(fontsize=9)
    for a in ax.flat:a.grid(alpha=.2)
    fig.suptitle('Linear electrostatic ITG: constant curvature, shearless; finite radial BSPF interval',fontsize=15)
    fig.savefig(out/'validation.png',dpi=170);fig.savefig(out/'validation.pdf');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    ax[0].semilogy(generic['t'],abs(generic['phi']),label='Generic density seed')
    ax[0].set(xlabel='t',ylabel='Potential amplitude',title='Unforced growth after initial transient')
    for rr in report['marginal']:ax[1].plot(rr['growth'],rr['a_t'],'o',color='#2463a5')
    ax[1].axhline(report['critical_gradient_extrapolation'],ls='--',color='gray')
    ax[1].set(xlabel='Positive growth approaching zero',ylabel='a_T',title='Selected-branch onset extrapolation')
    for a in ax:a.grid(alpha=.2)
    fig.savefig(out/'onset_and_seed.png',dpi=170);plt.close(fig)


def main():
    out=Path('build/linear_itg');out.mkdir(parents=True,exist_ok=True)
    report=scans();print('continuous reference',report['continuous'],'critical',report['critical_gradient_extrapolation'],flush=True)
    (out/'reference_scans.json').write_text(json.dumps(report,indent=2)+'\n')
    report['spatial']=[]
    for n in [49,65,97,129]:
        row,arrays=evolve(n)
        report['spatial'].append(row)
        if n==129:saved=arrays;np.savez_compressed(out/'solution_129.npz',**arrays)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        jax.clear_caches();gc.collect()
    generic,generic_arrays=evolve(49,n_v=32,n_mu=24,dt=.02,end=60.,generic=True)
    report['generic_seed']=generic
    np.savez_compressed(out/'generic_seed.npz',**generic_arrays)
    jax.clear_caches();gc.collect()
    report['parameter_validation']=[]
    for options in [dict(a_t=2.5),dict(a_t=6.),dict(a_t=8.),dict(ky=.2),dict(ky=.5),dict(ky=.7)]:
        row,_=evolve(49,**options)
        report['parameter_validation'].append(row)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    jax.clear_caches();gc.collect()
    report['temporal']=[]
    for dt in [.2,.1,.05]:
        row,_=evolve(33,n_v=12,n_mu=8,dt=dt,end=4.)
        report['temporal'].append(row);jax.clear_caches();gc.collect()
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    plot(out,report,saved,generic_arrays)
    assert all(r['growth_error']<1e-7 and r['frequency_error']<1e-7 for r in report['spatial'])
    assert all(r['growth_error']<1e-7 and r['frequency_error']<1e-7 for r in report['parameter_validation'])
    assert all(r['max_relative_energy_balance']<1e-7 for r in report['spatial'])
    assert abs(generic['measured_growth']-generic['reference_growth'])<.01
    assert report['velocity'][-1]['error']<1e-7
    temporal=report['temporal']
    assert all(12<temporal[i]['growth_error']/temporal[i+1]['growth_error']<20 for i in range(2))
    print('Linear ITG benchmark checks passed.',flush=True)

if __name__=='__main__':main()
