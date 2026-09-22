"""Synthetic-coil, limited free-boundary GS equilibrium on a full spline section.

All coordinates/currents are nondimensional and mu0=1 in this example. The
computational curve is a vacuum truncation, not a prescribed plasma surface.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from bspf_models.elliptic.embedded_poisson import SplineDomain
from bspf_models.elliptic.spline_annulus import SplineSection
from bspf_models.elliptic.bspline_source import BSplineSourcePlan
from bspf_models.plasma.spline_annulus_gs import SplineAnnulusGSPlan
from bspf_models.plasma.spline_free_boundary import FreeBoundaryGSPlan,FilamentCoils


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fft-grid',type=int,default=513)
    parser.add_argument('--sample-grid',type=int,default=129)
    parser.add_argument('--volume-order',type=int,default=48)
    parser.add_argument('--tolerance',type=float,default=1e-3)
    parser.add_argument('--inner-tolerance',type=float,default=1e-4)
    parser.add_argument('--coil-current',type=float,default=-.72)
    parser.add_argument('--plasma-current',type=float,default=1.)
    parser.add_argument('--power',type=int,default=4)
    parser.add_argument('--limiter-radius',type=float,default=3.6)
    parser.add_argument('--max-iterations',type=int,default=40)
    parser.add_argument('--relaxation',type=float,default=.3)
    parser.add_argument('--anderson-depth',type=int,default=4)
    parser.add_argument('--out',type=Path,default=Path('build/spline_free_boundary_gs'))
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    report=dict(passed=False,parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},history=[])
    def save(): (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    def progress(row):
        report['history'].append(row.copy()); print(row,flush=True);save()
    start=perf_counter()
    t=np.arange(12)*2*np.pi/12
    domain=SplineSection(SplineDomain(np.column_stack((1.4*np.cos(t),1.4*np.sin(t)))).curve)
    source=BSplineSourcePlan(domain,grid_size=args.fft_grid,sample_grid=args.sample_grid)
    gs=SplineAnnulusGSPlan(domain,major_radius=3.,source_plan=source,order=16,subdivisions=4)
    coils=FilamentCoils(np.array([[5.,2.],[5.,-2.]]),np.full(2,args.coil_current),mu0=1.)
    plan=FreeBoundaryGSPlan(gs,coils,[args.limiter_radius,0.],plasma_current=args.plasma_current,
                            power=args.power,volume_order=args.volume_order)
    report.update(source_plan=source.stats,coils=coils.positions.tolist(),coil_currents=coils.currents.tolist(),
                  limiter=plan.limiter.tolist(),setup_seconds=perf_counter()-start)
    save()
    def seed(x): return .6*np.exp(-((x[:,0]-3)**2+x[:,1]**2)/.6)
    try:
        solution=plan.solve(seed,tolerance=args.tolerance,pde_tolerance=args.inner_tolerance,
                            source_tolerance=args.inner_tolerance,relaxation=args.relaxation,
                            anderson_depth=args.anderson_depth,max_iterations=args.max_iterations,callback=progress)
        axis=solution.magnetic_axis()
        boundary=solution.plasma_boundary(axis['position'],count=64)
        audit=solution.current_audit()
        local=domain.sample(48,.371); points=gs.physical(local)
        flux=solution.flux(points).real; current=solution.current_density(points)
        report.update(passed=True,axis=axis,current_audit=audit,history=solution.history,
                      edge_flux=solution.edge_flux,elapsed_seconds=perf_counter()-start)
        np.savez_compressed(args.out/'fields.npz',points=points,flux=flux,current=current,
                            plasma_boundary=boundary,axis=axis['position'])
        save()
    except (ValueError,RuntimeError) as exc:
        report.update(error=str(exc),elapsed_seconds=perf_counter()-start);save();raise
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,values,title in zip(axes,(flux,current),('Poloidal flux','Toroidal plasma current')):
        image=ax.scatter(points[:,0],points[:,1],c=values,s=8,cmap='viridis')
        edge=np.vstack((boundary,boundary[0]))
        ax.plot(edge[:,0],edge[:,1],'r-',label='Computed plasma edge')
        b=domain.boundaries[0]; wall=gs.physical(b.curve(np.linspace(b.a,b.b,400)))
        ax.plot(wall[:,0],wall[:,1],'k--',label='Vacuum computational boundary')
        ax.plot(*axis['position'],'kx');ax.plot(*plan.limiter,'ro')
        ax.set(xlabel='R',ylabel='Z',title=title,aspect='equal');fig.colorbar(image,ax=ax)
    axes[0].legend(fontsize=8)
    fig.savefig(args.out/'equilibrium.png',dpi=160);plt.close(fig)
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
