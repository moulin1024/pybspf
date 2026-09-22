"""Record domain-only spline source accuracy/cost, including rejected resolutions.

This audits extension only, not complete PDE solutions. The exact MMS field is
never passed to the source plan. High-frequency failures are expected at coarse
resolution and are recorded, not silently accepted.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from bspf_models.elliptic.bspline_source import BSplineSourcePlan
from spline_annulus_convergence import geometry
from spline_annulus_turbulent_mms import RandomWaveMMS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--degrees', type=int, nargs='+', choices=(5, 7), default=[5, 7])
    parser.add_argument('--spans', type=int, nargs='+', default=[3, 5, 7])
    parser.add_argument('--cutoffs', type=float, nargs='+', default=[12., 24., 48.])
    parser.add_argument('--fft-grid', type=int, default=1025)
    parser.add_argument('--sample-grid', type=int, default=129)
    parser.add_argument('--radius', type=float)
    parser.add_argument('--tolerance', type=float, default=1e-7)
    parser.add_argument('--out', type=Path, default=Path('build/bspline_extension'))
    args = parser.parse_args()
    if not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error('tolerance must be finite and positive')
    domain, _ = geometry()
    fields = [RandomWaveMMS(k) for k in args.cutoffs]
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(geometry_degree=3, mms=[f.metadata() for f in fields], runs=[])
    def save():
        (args.out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    for degree in args.degrees:
        for spans in args.spans:
            start = perf_counter()
            plan = BSplineSourcePlan(domain, degree=degree, spans=spans,
                                     grid_size=args.fft_grid, sample_grid=args.sample_grid,
                                     radius=args.radius)
            run = dict(plan=plan.stats, setup_seconds=perf_counter()-start, sources=[])
            report['runs'].append(run)
            for field in fields:
                for sigma in (0., 64., -64.):
                    def source(x):
                        if not np.all(domain.contains(x)):
                            raise AssertionError('source queried outside physical domain')
                        return field.forcing(x, sigma)
                    start = perf_counter()
                    # Diagnostic fit exposes the error even for rejected cases.
                    part = plan.fit(source, sigma, tolerance=np.inf)
                    error = part.stats['validation_relative_max']
                    row = dict(cutoff=field.maximum_wave, sigma=sigma,
                               relative_max=error, tolerance=args.tolerance,
                               accepted=bool(error <= args.tolerance),
                               fit_seconds=perf_counter()-start,
                               exterior_relative_max=part.stats['exterior_relative_max'])
                    run['sources'].append(row)
                    print(degree, spans, row, flush=True)
                    save()
    save()


if __name__ == '__main__':
    main()
