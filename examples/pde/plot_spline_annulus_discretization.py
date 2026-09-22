"""Plot actual Q=48 source samples, Fourier extension and Poisson boundary nodes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import numpy as np
from numpy.polynomial.legendre import leggauss

from spline_annulus_turbulent_mms import RandomWaveMMS, geometry, FourierSourcePlan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('build/spline_annulus_turbulent_resolved/results.json'))
    parser.add_argument('--out', type=Path, default=Path('build/spline_annulus_discretization'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    record = next(r for r in json.loads(args.results.read_text())
                  if r['status'] == 'solved' and r['cutoff'] == 48 and r['sigma'] == 0)
    domain, _ = geometry()
    mms = RandomWaveMMS(48)
    plan = FourierSourcePlan(domain, modes=record['source_modes'], samples=4*record['source_modes']+8,
                             padding=record['source']['padding'])
    fit = plan.fit(lambda x: mms.forcing(x, 0), 0)
    print('source fitted', fit.stats, flush=True)
    curves = [b.curve(np.linspace(b.a, b.b, 1200)) for b in domain.boundaries]
    def boundaries(ax):
        for c in curves:
            ax.plot(*c.T, color='#172733', lw=1)
        ax.set_aspect('equal')
        ax.set_xlabel('x'); ax.set_ylabel('y')
    # Replay saved adaptive marks. Poisson has no wave-induced extra subdivisions.
    breaks = [b.knots.copy() for b in domain.boundaries]
    for row in record['history'][:-1]:
        indicators = np.array(row['panel_indicators'])
        marked = indicators > max(row['tolerance'], .3*indicators.max())
        new, index = [], 0
        for edges in breaks:
            extra = []
            for a, b in zip(edges[:-1], edges[1:]):
                if marked[index]: extra.append((a+b)/2)
                index += 1
            new.append(np.unique(np.r_[edges, extra]))
        assert index == len(marked)
        breaks = new
    nodes, ends = [], []
    for boundary, edges in zip(domain.boundaries, breaks):
        t = np.concatenate([(a+b)/2+(b-a)/2*leggauss(10)[0] for a,b in zip(edges[:-1], edges[1:])])
        nodes.append(boundary.curve(t)); ends.append(boundary.curve(edges[:-1]))
    assert sum(map(len,nodes)) == record['history'][-1]['unknowns']
    nb = plan.stats['bulk_samples']
    bulk, collar = plan.points[:nb], plan.points[nb:]
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'figure.facecolor': 'white'})
    fig, axs = plt.subplots(2,2,figsize=(13,11),layout='constrained')
    for ax in axs[0]:
        ax.scatter(*bulk.T, s=1.5, color='#498bb0', label=f'Interior grid: {len(bulk)}')
        ax.scatter(*collar.T, s=3, color='#e78238', label=f'Two interior collars: {len(collar)}')
        boundaries(ax)
    axs[0,0].set_title('A  Source fitting points (not a fitted volume mesh)')
    axs[0,0].legend(loc='lower left',fontsize=8)
    peak = curves[0][np.argmax(curves[0][:,1])]
    axs[0,1].set(xlim=(peak[0]-.12,peak[0]+.12),ylim=(peak[1]-.055,peak[1]+.015),
                 title='B  Outer-wall zoom: grid gap and interior collars')
    for x in nodes: axs[1,0].scatter(*x.T,s=5,color='#7255aa')
    for x in ends: axs[1,0].scatter(*x.T,s=20,marker='|',color='#de6930')
    boundaries(axs[1,0])
    axs[1,0].set_title(f'C  Final Poisson boundary nodes: {sum(map(len,nodes))}\n10 Gauss nodes/panel; orange marks = panel ends')
    coeff = abs(fit.coefficients)
    im = axs[1,1].scatter(*plan.frequencies.T, c=np.maximum(coeff,1e-12),s=9,
                         norm=LogNorm(vmin=max(coeff.max()*1e-8,1e-12),vmax=coeff.max()),cmap='viridis')
    theta=np.linspace(0,2*np.pi,300)
    axs[1,1].plot(48*np.cos(theta),48*np.sin(theta),'--',color='#e78238',lw=1,label='MMS cutoff |q| = 48')
    axs[1,1].set(xlabel='angular wavenumber q_x',ylabel='angular wavenumber q_y',aspect='equal',
                 title='D  Fourier modes: 49 x 49 = 2401\nColor = fitted source coefficient magnitude')
    axs[1,1].legend(fontsize=8)
    fig.colorbar(im,ax=axs[1,1],label='|c_mn|')
    fig.suptitle('B-spline annulus | Q = 48 | Poisson | padding = 1.5',fontsize=16)
    for ext in ('png','pdf'): fig.savefig(args.out/f'points_and_modes.{ext}',dpi=190)
    plt.close(fig)
    # Evaluate the fitted SOURCE extension, not the manufactured solution.
    edges=np.column_stack((plan.center-plan.lengths/2,plan.center+plan.lengths/2))
    xx,yy=np.meshgrid(np.linspace(*edges[0],240),np.linspace(*edges[1],240))
    x=np.column_stack((xx.ravel(),yy.ravel()))
    inside=domain.contains(x)
    extended=np.concatenate([plan.basis(chunk)@fit.coefficients for chunk in np.array_split(x,120)])
    exact=mms.forcing(x[inside],0)
    scale=np.max(abs(exact))
    reference=np.full(len(x),np.nan); reference[inside]=exact/scale
    error=np.full(len(x),np.nan);error[inside]=abs(extended[inside]-exact)/scale
    outside_ratio=float(np.max(abs(extended[~inside]))/scale)
    fig,axs=plt.subplots(1,3,figsize=(17,5.4),layout='constrained')
    vmax=max(1.,np.max(abs(extended.real))/scale)
    im=axs[0].pcolormesh(xx,yy,(extended.real/scale).reshape(xx.shape),shading='auto',
                        cmap='RdBu_r',norm=SymLogNorm(linthresh=1,vmin=-vmax,vmax=vmax))
    fig.colorbar(im,ax=axs[0],label='Re(Fourier extension) / max_Omega |f|; symlog')
    axs[0].set_title('E  Fitted source on the entire periodic box\nExterior / hole values are unconstrained')
    im=axs[1].pcolormesh(xx,yy,reference.reshape(xx.shape),shading='auto',cmap='RdBu_r',vmin=-1,vmax=1)
    fig.colorbar(im,ax=axs[1],label='Exact f / max_Omega |f|')
    axs[1].set_title('F  Manufactured source in the physical domain\nWhite = outside physical domain')
    im=axs[2].pcolormesh(xx,yy,np.maximum(error,1e-14).reshape(xx.shape),shading='auto',cmap='magma',
                        norm=LogNorm(vmin=1e-14,vmax=max(1e-8,np.nanmax(error))))
    fig.colorbar(im,ax=axs[2],label='|extension - f| / max_Omega |f|')
    axs[2].set_title('G  Independent interior source error\n240 x 240 display grid, masked to domain')
    for ax in axs:
        boundaries(ax)
        ax.set_xlim(edges[0]);ax.set_ylim(edges[1])
    fig.suptitle('Source extension: accurate inside, nonphysical continuation outside',fontsize=15)
    for ext in ('png','pdf'): fig.savefig(args.out/f'fourier_extension.{ext}',dpi=190)
    plt.close(fig)
    stats=dict(source=fit.stats, boundary_nodes=sum(map(len,nodes)),
               display_interior_relative_max=float(np.nanmax(error)),
               display_exterior_amplitude_ratio=outside_ratio,
               display_imaginary_amplitude_ratio=float(np.max(abs(extended.imag))/scale))
    (args.out/'plot_stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    np.savez_compressed(args.out/'discretization.npz',training=plan.points,validation=plan.validation,
                        frequencies=plan.frequencies,coefficients=fit.coefficients,
                        boundary_nodes=np.vstack(nodes), panel_ends=np.vstack(ends))
    print(json.dumps(stats),flush=True)

if __name__=='__main__': main()
