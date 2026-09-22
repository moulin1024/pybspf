"""Exact B-spline annulus: homogeneous and domain-only forced elliptic checks."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import hankel1, k0

from bspf_models.elliptic.embedded_poisson import SplineDomain
from bspf_models.elliptic.spline_annulus import SplineAnnulus, AnnulusPanelPlan, FourierSourcePlan, adaptive_dirichlet


def geometry(gap=False):
    t = np.arange(12)*2*np.pi/12
    outer = SplineDomain(np.column_stack(((1+.12*np.cos(3*t))*np.cos(t),
                                          .85*(1+.12*np.cos(3*t))*np.sin(t))))
    s = np.arange(8)*2*np.pi/8
    center = np.array([.48, -.03]) if gap else np.array([.13, -.07])
    inner = SplineDomain(center+np.column_stack((.26*np.cos(s), .21*np.sin(s))))
    return SplineAnnulus(outer.curve, inner.curve), center


def homogeneous(sigma, center):
    if sigma == 0:
        return lambda x: np.log(np.linalg.norm(x-center, axis=1))+.2*x[:, 0]-.1*x[:, 1]
    k = np.sqrt(abs(sigma))
    if sigma > 0:
        return lambda x: k0(k*np.linalg.norm(x-center, axis=1))+.2*np.exp(k*x[:, 0])
    return lambda x: hankel1(0, k*np.linalg.norm(x-center, axis=1))+.2*np.exp(1j*k*x[:, 0])


def forced(sigma):
    def exact(x):
        return np.exp(.3*x[:, 0]+.2*x[:, 1])+.2*np.sin(3*x[:, 0])*np.cos(2*x[:, 1])
    def source(x):
        return (sigma-.13)*np.exp(.3*x[:, 0]+.2*x[:, 1])+.2*(sigma+13)*np.sin(3*x[:, 0])*np.cos(2*x[:, 1])
    return exact, source


def probes(domain):
    bulk = domain.sample(9, .371)
    near = []
    for component, boundary in enumerate(domain.boundaries):
        t = boundary.a+.173*(boundary.b-boundary.a)
        y = boundary.curve(t)
        normal = boundary.normal(t, 1 if component == 0 else -1)
        near.extend(y-eps*normal for eps in (1e-2, 1e-4, 1e-6))
    return bulk, np.asarray(near)


def check(solution, exact):
    domain = solution.plan.domain
    bulk, near = probes(domain)
    result = {}
    for label, points in [('bulk', bulk), ('near', near)]:
        actual, expected = solution.interior(points), exact(points)
        result[label+'_relative_max'] = float(np.max(abs(actual-expected))/max(1., np.max(abs(expected))))
    errors, scales = [], []
    for component, boundary in enumerate(domain.boundaries):
        t = boundary.a+(np.arange(29)+.371)/29*(boundary.b-boundary.a)
        t = np.r_[t, boundary.knots[:-1]+1e-6]
        expected = exact(boundary.curve(t))
        errors.extend(abs(solution.boundary(component, t)-expected))
        scales.extend(abs(expected))
    result['boundary_relative_max'] = float(max(errors)/max(1., max(scales)))
    result['training_residual'] = solution.training_residual
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('build/spline_annulus'))
    parser.add_argument('--orders', type=int, nargs='+', default=[6, 10, 14])
    parser.add_argument('--modes', type=int, default=12)
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--resonance-scan', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    domain, center = geometry()
    records, adaptive_records = [], []
    for sigma in (0., 4., -4.):
        for order in args.orders:
            start = perf_counter()
            print(f'building sigma={sigma} order={order}', flush=True)
            plan = AnnulusPanelPlan(domain, sigma, order=order)
            exact = homogeneous(sigma, center)
            solution = plan.solve((exact, exact))
            record = dict(problem='homogeneous', **plan.stats, **check(solution, exact))
            record['total_seconds'] = perf_counter()-start
            records.append(record)
            print(json.dumps(record), flush=True)
            (args.out/'results.json').write_text(json.dumps(records, indent=2)+'\n')
        if sigma == 0:
            print('fitting source extension', flush=True)
            source_plan = FourierSourcePlan(domain, modes=args.modes, samples=max(48, 4*args.modes+8))
        exact, source = forced(sigma)
        # The solver only receives f and boundary g; no exact particular callback.
        solution = plan.solve((exact, exact), source=source, source_plan=source_plan, source_tolerance=1e-7)
        record = dict(problem='forced', **plan.stats, **check(solution, exact),
                      source=solution.particular.stats)
        records.append(record)
        print(json.dumps(record), flush=True)
        (args.out/'results.json').write_text(json.dumps(records, indent=2)+'\n')
        if args.adaptive:
            def callback(sol, row):
                print('adaptive', sigma, row['level'], row['unknowns'], row['boundary_indicator'], flush=True)
            solution, history = adaptive_dirichlet(domain, sigma, (exact, exact), source=source,
                source_plan=source_plan, order=10, tolerance=1e-9, max_refinements=8, callback=callback)
            adaptive_records.append(dict(sigma=sigma, history=history, errors=check(solution, exact),
                                         source=solution.particular.stats))
            (args.out/'adaptive.json').write_text(json.dumps(adaptive_records, indent=2)+'\n')
    if args.resonance_scan:
        resonance_scan(domain, center, args.out)
    render(domain, records, adaptive_records, args.out)
    import bspf_models.elliptic.spline_annulus as implementation
    paths = [Path(__file__), Path(implementation.__file__)]
    (args.out/'sources.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2)+'\n')


def resonance_scan(domain, center, out):
    from scipy.optimize import minimize_scalar
    rows = []
    def sample(k, order=8):
        plan = AnnulusPanelPlan(domain, -k*k, order=order)
        rows.append(dict(k=float(k), order=order, rcond=plan.stats['reciprocal_condition_estimate']))
        return rows[-1]['rcond']
    ks = np.linspace(3., 7., 17)
    values = [sample(k) for k in ks]
    i = int(np.argmin(values))
    minimum = minimize_scalar(lambda k: np.log(sample(k)),
        bounds=(ks[max(0, i-1)], ks[min(len(ks)-1, i+1)]), method='bounded',
        options={'xatol': 1e-6, 'maxiter': 24})
    validation = []
    for order in (8, 10, 14):
        plan = AnnulusPanelPlan(domain, -minimum.x**2, order=order)
        exact = homogeneous(plan.sigma, center)
        validation.append(dict(order=order, **check(plan.solve((exact, exact)), exact),
                               rcond=plan.stats['reciprocal_condition_estimate']))
    result = dict(records=rows, candidate=float(minimum.x), validation=validation,
                  note='Discrete conditioning minimum, not a certified continuum eigenvalue.')
    (out/'resonance_scan.json').write_text(json.dumps(result, indent=2)+'\n')
    print('resonance', result['candidate'], validation, flush=True)


def render(domain, records, adaptive, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, sigma in zip(axes, (0., 4., -4.)):
        rows = [r for r in records if r['sigma'] == sigma and r['problem'] == 'homogeneous']
        for key, label in [('boundary_relative_max', 'boundary'), ('bulk_relative_max', 'interior'),
                           ('near_relative_max', 'distance >= 1e-6')]:
            ax.loglog([r['unknowns'] for r in rows], [r[key] for r in rows], 'o-', label=label)
        ax.set(title=f'-Laplacian + ({sigma:g})', xlabel='Boundary unknowns', ylabel='Max error / max(1, |exact|)')
        ticks = [r['unknowns'] for r in rows]
        ax.set_xticks(ticks, labels=[str(n) for n in ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(True, which='both', alpha=.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'convergence.png', dpi=170)
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, boundary in zip(('outer', 'inner'), domain.boundaries):
        t = np.linspace(boundary.a, boundary.b, 1000)
        x = boundary.curve(t)
        ax.plot(x[:, 0], x[:, 1], label=label)
        knots = boundary.curve(boundary.knots)
        ax.plot(knots[:, 0], knots[:, 1], '.', color='black', ms=4)
    x = domain.sample(18, .371)
    ax.scatter(x[:, 0], x[:, 1], s=5, alpha=.3, label='interior source samples (illustrative)')
    ax.set(aspect='equal', title='Exact spline annulus; no fitted volume mesh')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'geometry.png', dpi=170)


if __name__ == '__main__':
    main()
