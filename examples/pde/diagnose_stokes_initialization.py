"""Compare Stokes solves and best H1 projections without changing the solver.

All spaces use the same 73x33 background grid, physical domain, quadrature,
Dirichlet/traction data, and zero sponge strength. Reference coefficients are
computed independently by the boundary-only lightning solver at two orders.
No filtering or time evolution is performed. Output directories must be new.
"""
import argparse
import gc
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import jax
import numpy as np
import scipy
import scipy.linalg as la
from bspf_models.fluids.immersed_flow import ImmersedFlowPlan
from lightning_stokes_reference import LightningStokes


def render(out, x, y, reference, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    fig, axes = plt.subplots(len(results), 3, figsize=(15, 3.4 * len(results)),
                             squeeze=False, layout='constrained')
    for row, result in zip(axes, results):
        method = result['method']
        a = np.load(out / f'{method}.npz')
        errors = [a['solve_omega'] - reference[3],
                  a['projection_omega'] - reference[3]]
        for ax, error, label in zip(row[:2], errors, ['Stokes solve', 'Best H1 projection']):
            im = ax.pcolormesh(x, y, error, shading='nearest', cmap='RdBu_r',
                               vmin=-.3, vmax=.3)
            ax.add_patch(Ellipse((.19, -.13), .62, .46, facecolor='.7', edgecolor='k'))
            ax.set(xlim=(-1, 2), ylim=(-1, 1), xlabel='x', ylabel='y',
                   title=f'{method}: {label} minus reference')
            fig.colorbar(im, ax=ax, label='Vorticity error; clipped at +/-0.3')
        iy = np.argmin(abs(y - .5))
        row[2].plot(x, errors[0][iy], label='Stokes solve', lw=1.6)
        row[2].plot(x, errors[1][iy], '--', label='Best H1 projection', lw=1)
        row[2].set(xlabel='x', ylabel='Vorticity error', title=f'Raw section y={y[iy]:g}')
        row[2].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        row[2].legend(); row[2].grid(alpha=.2)
    fig.suptitle('Unforced Stokes, sponge OFF: identical grid and quadrature; no smoothing\n'
                 'Maps share a color scale; section axes autoscale to expose residual errors')
    fig.savefig(out / 'projection_ripple.png', dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--quadrature-factor', type=float, default=2.5)
    ap.add_argument('--methods', nargs='+', choices=['svd', 'factor', 'rational'],
                    default=['svd', 'factor', 'rational'])
    ap.add_argument('--basis-workers', type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    jax.config.update('jax_enable_x64', True)
    x, y = np.linspace(-1, 5, 401), np.linspace(-1, 1, 161)
    xx, yy = np.meshgrid(x, y)
    fluid = ((xx - .19) / .31)**2 + ((yy + .13) / .23)**2 > 1
    fluid[[0, 0, -1, -1], [0, -1, 0, -1]] = False
    z = xx + 1j * yy
    reference = np.full((6,) + xx.shape, np.nan)
    checks = []
    previous = None
    for degree, poles, laurent, samples in [(96, 32, 48, 800), (120, 40, 60, 1000)]:
        ref = LightningStokes(degree, poles, laurent, samples)
        reference[:, fluid] = ref.evaluate(z[fluid])
        check = dict(degree=degree, corner_poles=poles, laurent=laurent,
                     samples=samples, boundary=ref.verify())
        if previous is not None:
            check['grid_velocity_max_change'] = float(np.nanmax(np.hypot(
                reference[0]-previous[0], reference[1]-previous[1])))
            check['grid_vorticity_max_change'] = float(np.nanmax(abs(reference[3]-previous[3])))
        checks.append(check)
        previous = reference.copy()
        print('REFERENCE', json.dumps(check), flush=True)
    np.savez_compressed(args.out / 'reference.npz', x=x, y=y, fields=reference)
    report = dict(command=sys.argv, git_head=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), python=platform.python_version(),
        numpy=np.__version__, scipy=scipy.__version__, jax=jax.__version__,
        parameters=dict(nx=73, ny=33, reynolds=20, buffer_strength=0,
                        quadrature_factor=args.quadrature_factor, wall_rcond=1e-10,
                        basis_precision='mpfr'), reference_checks=checks, results=[])
    for method in args.methods:
        start = perf_counter()
        print('START', method, flush=True)
        p = ImmersedFlowPlan(nx=73, ny=33, buffer_strength=0, wall_method=method,
            quadrature_factor=args.quadrature_factor, basis_workers=args.basis_workers)
        # Reference rows: u,v,p/nu,omega,ux,vx; omega=vx-uy, vy=-ux.
        r = ref.evaluate(p.points[:, 0] + 1j*p.points[:, 1])
        targets = (r[0], r[1], r[4], r[5]-r[3], r[5])
        weights = (1, 1, 2, 1, 1)
        differences = [a-b for a,b in zip(targets, p.lift_fields[1:])]
        rhs = sum(o.T @ (p.weights * d * weight)
                  for o,d,weight in zip(p.operators_fluid, differences, weights))
        gram = p.mass + p.stiffness
        best = la.solve(gram, rhs, assume_a='pos')
        def h1norm(fields):
            return np.sqrt(sum(p.weights @ (a*a*w) for a,w in zip(fields, weights)))
        row = dict(method=method, dofs=p.dofs, constraint_rank=p.constraint_rank,
                   quadrature_points=len(p.weights), energy_condition=p.energy_condition,
                   discarded_volume_modes=p.discarded_volume_modes,
                   setup_seconds=p.setup_seconds, metrics={})
        saved = dict(x=x, y=y, solve_state=p.stokes_state, projection_state=best)
        for label, state in [('solve', p.stokes_state), ('projection', best)]:
            grid = p.grid(state, x, y)
            for name in ('u', 'v', 'vorticity'):
                grid[name] = np.where(fluid, grid[name], np.nan)
                saved[label + '_' + ('omega' if name == 'vorticity' else name)] = grid[name]
            du, dv, dw = grid['u']-reference[0], grid['v']-reference[1], grid['vorticity']-reference[3]
            errors = [o@state+lift-target for o,lift,target in zip(
                p.operators_fluid, p.lift_fields[1:], targets)]
            boundary, _ = p.arc.sample(512, offset=.371)
            wall = p.evaluate(state, boundary)
            row['metrics'][label] = dict(
                velocity_relative_l2=float(np.sqrt(np.nansum(du*du+dv*dv)/np.nansum(reference[0]**2+reference[1]**2))),
                vorticity_relative_l2=float(np.sqrt(np.nansum(dw*dw)/np.nansum(reference[3]**2))),
                vorticity_max=float(np.nanmax(abs(dw))),
                velocity_relative_h1=float(h1norm(errors)/h1norm(targets)),
                wall_max_speed=float(np.max(np.hypot(wall[1],wall[2]))))
        a=(saved['solve_omega']-reference[3])[fluid]
        b=(saved['projection_omega']-reference[3])[fluid]
        row['vorticity_error_cosine_similarity'] = float(a@b/(la.norm(a)*la.norm(b)))
        row['solve_projection_vorticity_gap_over_solve_error'] = float(la.norm(a-b)/la.norm(a))
        residual = p.linear@p.stokes_state+p.linear_lift
        row['linear_relative_backward_error'] = float(la.norm(residual)/max(
            la.norm(p.linear)*la.norm(p.stokes_state)+la.norm(p.linear_lift), np.finfo(float).tiny))
        if p.rational is not None:
            row['rational_lift_quadrature_residual'] = p.rational.info['quadrature_diffusion_lift_norm']
            row['note'] = ('Zero-force, zero-sponge rational solve is the homogeneous Stokes lift; '
                           'this tests initialization, not general volume accuracy or NS evolution.')
        row['total_seconds'] = perf_counter()-start
        np.savez_compressed(args.out / f'{method}.npz', **saved)
        report['results'].append(row)
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print('RESULT', json.dumps(row), flush=True)
        render(args.out, x, y, reference, report['results'])
        del p, grid, r, targets, differences, rhs, gram, best
        gc.collect()


if __name__ == '__main__':
    main()
