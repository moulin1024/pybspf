"""Double-Dirichlet GS: Solov'ev regression and nonlinear profile equilibrium."""
import argparse
import json
from pathlib import Path
import numpy as np
from spline_annulus_convergence import geometry
from bspf_models.plasma.spline_annulus_gs import SplineAnnulusGSPlan
from bspf_models.plasma.solovev import SolovevEquilibrium


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--degree', type=int, choices=(3,5), default=5)
    parser.add_argument('--modes', type=int, default=32)
    parser.add_argument('--source-backend', choices=('fourier','local','bspline'), default='fourier')
    parser.add_argument('--fft-grid', type=int, default=1025)
    parser.add_argument('--local-degree', type=int, default=20)
    parser.add_argument('--spline-degree', type=int, choices=(5,7), default=7)
    parser.add_argument('--spline-spans', type=int, default=7)
    parser.add_argument('--spline-stability', type=float, default=1e-6)
    parser.add_argument('--sample-grid', type=int, default=129)
    parser.add_argument('--patch-radius', type=float)
    parser.add_argument('--patch-samples', type=int)
    parser.add_argument('--out', type=Path, default=Path('build/spline_annulus_gs'))
    args = parser.parse_args()
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    report = dict(degree=args.degree, source_modes=args.modes, source_backend=args.source_backend, passed=False)
    def save_report():
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    def checked_solve(name, solve):
        try:
            return solve()
        except (ValueError, RuntimeError) as exc:
            report[name] = dict(accepted=False, error=str(exc))
            report['solves_and_checks_seconds'] = perf_counter()-solve_start
            save_report()
            raise
    domain, _ = geometry()
    if args.degree == 5:
        from scipy.interpolate import BSpline
        from bspf_models.elliptic.spline_annulus import SplineAnnulus
        curves = []
        for b in domain.boundaries:
            n = len(b.knots)-1
            c = b.curve.c[:n]
            curves.append(BSpline(np.arange(-5,n+6), np.vstack((c,c[:5])), 5, extrapolate='periodic'))
        domain = SplineAnnulus(*curves)
    print('geometry degree',args.degree,'source mode radius',args.modes,flush=True)
    from time import perf_counter
    start = perf_counter()
    source_plan = None
    if args.source_backend == 'local':
        from bspf_models.elliptic.local_source import LocalSourcePlan
        source_plan = LocalSourcePlan(domain, grid_size=args.fft_grid, degree=args.local_degree,
                                     sample_grid=args.sample_grid, radius=args.patch_radius,
                                     patch_samples=args.patch_samples)
    elif args.source_backend == 'bspline':
        from bspf_models.elliptic.bspline_source import BSplineSourcePlan
        source_plan = BSplineSourcePlan(domain, grid_size=args.fft_grid, degree=args.spline_degree,
                                       spans=args.spline_spans, stability=args.spline_stability,
                                       sample_grid=args.sample_grid, radius=args.patch_radius,
                                       patch_samples=args.patch_samples)
    plan = SplineAnnulusGSPlan(domain, major_radius=3., modes=args.modes, samples=4*args.modes+8, padding=1.5,
                              order=18, subdivisions=2, source_plan=source_plan)
    report['setup_seconds'] = perf_counter()-start
    report['source_plan'] = plan.source_plan.stats
    print('setup',report['setup_seconds'],plan.source_plan.stats,flush=True)
    save_report()
    solve_start = perf_counter()
    local = domain.sample(24, .271)
    points = plan.physical(local)
    eq = SolovevEquilibrium(major_radius=3., logarithmic=.03)
    exact = lambda p: eq.jets(p)[0]
    sol = checked_solve('solovev', lambda: plan.solve_profiles(eq.p_prime, eq.ff_prime, boundary_flux=(exact,exact),
                              tolerance=1e-8, source_tolerance=1e-8, boundary_tolerance=1e-7, callback=lambda row: print(row,flush=True)))
    computed = sol.flux(points)
    truth = exact(points)
    regression = dict(relative_l2=float(np.linalg.norm(computed-truth)/np.linalg.norm(truth)),
                      relative_max=float(np.max(abs(computed-truth))/np.max(abs(truth))),
                      boundary_error=sol.boundary_error, history=sol.history)
    print('Solovev',json.dumps(regression),flush=True)
    report['solovev'] = regression
    save_report()
    # Manufactured nonconstant source exercises spatial oscillations in addition
    # to the variable-R GS term. No interior exact values enter the solve.
    def mms(p):
        r,z=p.T; return .1*np.sin(5*(r-3))*np.cos(4*z)
    def force(p):
        r,z=p.T
        return 41*mms(p)+.5*np.cos(5*(r-3))*np.cos(4*z)/r
    wave = checked_solve('wave_mms', lambda: plan.solve(
        force, (mms,mms), tolerance=1e-7, boundary_tolerance=1e-7))
    wavetruth=mms(points); wavecomputed=wave.flux(points)
    wavecheck=dict(relative_l2=float(np.linalg.norm(wavecomputed-wavetruth)/np.linalg.norm(wavetruth)),
                   relative_max=float(np.max(abs(wavecomputed-wavetruth))/np.max(abs(wavetruth))),
                   boundary_error=wave.boundary_error,history=wave.history)
    print('wave MMS',json.dumps(wavecheck),flush=True)
    report['wave_mms'] = wavecheck
    save_report()
    p_prime = lambda psi: .01+.002*psi**2
    ff_prime = lambda psi: .02+.01*psi**2
    nonlinear_history = []
    def progress(row):
        nonlinear_history.append(row)
        print('nonlinear iteration', row, flush=True)
    try:
        nonlinear = plan.solve_profiles(p_prime, ff_prime, boundary_flux=(0., .02),
                                       tolerance=1e-8, source_tolerance=1e-8, max_iterations=12,
                                       callback=progress)
    except (ValueError, RuntimeError) as exc:
        report['nonlinear'] = dict(accepted=False, error=str(exc), history=nonlinear_history)
        report['solves_and_checks_seconds'] = perf_counter()-solve_start
        save_report()
        raise
    flux = nonlinear.flux(points)
    # Check the ORIGINAL GS operator with an independent fourth-order FD stencil.
    h=3e-4
    safe=np.ones(len(local),dtype=bool)
    for axis in np.eye(2):
        for step in (-2,-1,1,2): safe &= domain.contains(local+step*h*axis)
    probes=points[safe][::max(1,sum(safe)//24)][:24]
    center=nonlinear.flux(probes)
    lap=np.zeros(len(probes),complex); radial=None
    for d,axis in enumerate(np.eye(2)):
        mm,m,p,pp=[nonlinear.flux(probes+step*h*axis) for step in (-2,-1,1,2)]
        lap+=(-pp+16*p-30*center+16*m-mm)/(12*h*h)
        if d==0: radial=(mm-8*m+8*p-pp)/(12*h)
    rhs=probes[:,0]**2*p_prime(center)+ff_prime(center)
    residual=-lap+radial/probes[:,0]-rhs
    check=dict(history=nonlinear.history,boundary_error=nonlinear.boundary_error,
               independent_fd_relative_residual=float(np.max(abs(residual))/np.max(abs(rhs))),
               flux_min=float(flux.real.min()),flux_max=float(flux.real.max()),
               fd_step=h,fd_points=len(probes))
    print('nonlinear',json.dumps(check),flush=True)
    passed = (regression['relative_l2']<1e-7 and wavecheck['relative_l2']<1e-6
              and check['independent_fd_relative_residual']<1e-5)
    report.update(nonlinear=check, passed=passed, solves_and_checks_seconds=perf_counter()-solve_start)
    save_report()
    np.savez_compressed(out/'fields.npz',points=points,solovev=computed,exact=truth,nonlinear=flux)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Triangulation is used only for display; mask triangles crossing the hole.
    import matplotlib.tri as tri
    triang=tri.Triangulation(points[:,0],points[:,1])
    vertices=local[triang.triangles]
    mask=~domain.contains(vertices.mean(axis=1))
    for a,b in ((0,1),(1,2),(2,0)): mask |= ~domain.contains((vertices[:,a]+vertices[:,b])/2)
    triang.set_mask(mask)
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,values,title in zip(axs,(abs(computed-truth),flux.real),('Solovev flux absolute error','Nonlinear GS flux: outer=0, inner=0.02')):
        im=ax.tripcolor(triang,values,shading='gouraud',cmap='viridis')
        for boundary in domain.boundaries:
            curve=plan.physical(boundary.curve(np.linspace(boundary.a,boundary.b,600)))
            ax.plot(*curve.T,'k',lw=.8)
        ax.set(aspect='equal',xlabel='R',ylabel='Z',title=title)
        fig.colorbar(im,ax=ax)
    fig.savefig(out/'gs_solution.png',dpi=180)
    assert passed, 'independent GS checks failed'

if __name__=='__main__': main()
