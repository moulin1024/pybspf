"""Compare a fixed-Nx BGK velocity-resolution sweep with batch uncertainty."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bgk_itg_saturation import analyze
from bspf_jax.itg_statistics import heat_uncertainty,compare_heat_means


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='build/bgk_velocity_scan');ap.add_argument('--start',type=float,default=400.);ap.add_argument('--end',type=float,default=1200.)
    a=ap.parse_args();root=Path(a.root);rows=[];histories=[];specs=[]
    for nv,nmu in [(8,6),(12,8),(16,12),(24,16)]:
        directory=root/f'v{nv}_m{nmu}_dt01';params=json.loads((directory/'parameters.json').read_text());data=dict(np.load(directory/'history.npz'))
        if data['t'][-1]<a.end:raise ValueError(f'{directory} is only at t={data["t"][-1]}')
        data={k:(v[data['t']<=a.end] if k!='final_state' else v) for k,v in data.items()}
        diag,spec=analyze(data,a.start,params['at']);u=heat_uncertainty(data['t'],data['work'][:,1],params['at'],start=a.start,end=a.end)
        row=dict(velocity=[nv,nmu],parameters=params,diagnostics=diag,uncertainty=u,
            hermite_edge=float(spec['spectrum_velocity'][-1,:].sum()/spec['spectrum_velocity'].sum()),
            laguerre_edge=float(spec['spectrum_velocity'][:,-1].sum()/spec['spectrum_velocity'].sum()))
        row['window_extension']=[]
        for duration in (200.,400.,800.):
            if a.start+duration<=a.end:
                prefix={k:(v[data['t']<=a.start+duration] if k!='final_state' else v) for k,v in data.items()}
                prefix_diag,_=analyze(prefix,a.start,params['at'])
                row['window_extension'].append(dict(duration=duration,diagnostics=prefix_diag,
                    uncertainty=heat_uncertainty(prefix['t'],prefix['work'][:,1],params['at'],start=a.start,end=a.start+duration)))
        rows.append(row);histories.append(data);specs.append(spec)
    controls=['nx','ny','nz','nu','at','model','dt','radial_kernel']
    for row in rows:
        if row['parameters']['nx']!=33:raise ValueError('this scan requires Nx=33')
        for key in controls:
            if row['parameters'][key]!=rows[0]['parameters'][key]:raise ValueError(f'varying control: {key}')
    pairs=[]
    for first,second in zip(rows[:-1],rows[1:]):
        comparison=compare_heat_means(first['uncertainty'],second['uncertainty'])
        comparison.update(coarse=first['velocity'],fine=second['velocity']);pairs.append(comparison)
    report=dict(start=a.start,end=a.end,cases=rows,adjacent_comparisons=pairs,
        finest_two_pairs_within_5pct=all(p['equivalent_within_tolerance'] for p in pairs[-2:]),
        all_stationary=all(r['diagnostics']['stationarity_screen_pass'] for r in rows))
    (root/'velocity_scan.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained');labels=[]
    for index,(row,data,spec) in enumerate(zip(rows,histories,specs)):
        label=f"{row['velocity'][0]} x {row['velocity'][1]}";labels.append(label)
        t=data['t'];ix=t>=a.start;tt=t[ix];ww=data['work'][ix,1];mean=(ww[1:]-ww[0])/(row['parameters']['at']*(tt[1:]-tt[0]))
        ax[0,0].plot(tt[1:],mean,label=label)
        sel=row['uncertainty']['selected'];ax[0,1].errorbar(index,sel['mean'],yerr=sel['half_width'],fmt='o',capsize=5)
        bs=row['uncertainty']['blocks'];ax[0,2].plot([v['width'] for v in bs],[v['standard_error'] for v in bs],'o-',label=label)
        vs=spec['spectrum_velocity'];normal=vs.sum()
        ax[1,0].semilogy(np.arange(vs.shape[0]),vs.sum(axis=1)/normal,'o-',label=label)
        ax[1,1].semilogy(np.arange(vs.shape[1]),vs.sum(axis=0)/normal,'o-',label=label)
        d=row['diagnostics'];ax[1,2].plot(index,d['mean_drive'],'o',color='tab:blue');ax[1,2].plot(index,d['mean_dissipation'],'x',color='tab:orange')
    ax[0,0].set(title='Cumulative mean heat flux after burn-in',xlabel='t',ylabel='Mean Q');ax[0,0].legend()
    ax[0,1].set(title='Mean Q and conditional 95% batch intervals',xticks=range(4),xticklabels=labels,ylabel='Mean Q')
    ax[0,2].set(title='Sensitivity to batch duration',xlabel='Batch duration',ylabel='Estimated standard error');ax[0,2].legend()
    ax[1,0].set(title='Time-averaged Hermite entropy spectrum',xlabel='Hermite order',ylabel='Fraction of S');ax[1,0].legend()
    ax[1,1].set(title='Time-averaged Laguerre entropy spectrum',xlabel='Laguerre order',ylabel='Fraction of S');ax[1,1].legend()
    ax[1,2].plot([],[],'o',color='tab:blue',label='Mean injection');ax[1,2].plot([],[],'x',color='tab:orange',label='Mean dissipation');ax[1,2].legend()
    ax[1,2].set(title='Mean power budget',xticks=range(4),xticklabels=labels,ylabel='Power')
    for x in ax.flat:x.grid(alpha=.2)
    fig.suptitle(f'BGK velocity scan: Nx=33, Ny=9, Nz=7, nu=0.15; averaging [{a.start:g}, {a.end:g}]')
    fig.savefig(root/'velocity_scan.png',dpi=160);fig.savefig(root/'velocity_scan.pdf');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for row in rows:
        label=f"{row['velocity'][0]} x {row['velocity'][1]}"
        windows=row['window_extension'];estimates=[v['uncertainty']['selected'] for v in windows]
        ax[0].errorbar([v['duration'] for v in windows],[v['mean'] for v in estimates],
            yerr=[v['half_width'] for v in estimates],fmt='o-',capsize=4,label=label)
        batches=next(v for v in row['uncertainty']['blocks'] if v['width']==100.)
        centers=batches['start']+(np.arange(batches['count'])+.5)*batches['width']
        ax[1].plot(centers,batches['values'],'o-',label=label)
    ax[0].set(title='Extending the averaging window from t=400',xlabel='Window duration',ylabel='Mean Q with conditional 95% interval')
    ax[1].set(title='Consecutive 100-unit batch means',xlabel='Batch midpoint',ylabel='Mean Q')
    for axis in ax:axis.grid(alpha=.2);axis.legend()
    fig.savefig(root/'window_extension.png',dpi=160);fig.savefig(root/'window_extension.pdf');plt.close(fig)
    compact=[]
    for r in rows:
        d=r['diagnostics'];u=r['uncertainty']['selected']
        compact.append(dict(velocity=r['velocity'],Q=d['mean_heat_flux'],ci95=u['ci95'],batch_width=u['width'],batch_count=u['count'],
            stationary=d['stationarity_screen_pass'],velocity_edge=d['velocity_last_order_energy_fraction'],budget=d['budget_relative_max'],power_imbalance=d['mean_power_relative_imbalance']))
    print(json.dumps(dict(cases=compact,comparisons=pairs),indent=2))

if __name__=='__main__':main()
