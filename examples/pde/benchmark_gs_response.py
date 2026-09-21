"""Same-TSVD fixed-boundary GS: cached tensor output vs compiled response."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
import jax
import numpy as np
from bspf_models.plasma.grad_shafranov import FixedBoundaryGSPlan
from bspf_models.plasma.solovev import SolovevEquilibrium
from bspf_models.plasma.solovev import SolovevFluxDomain
jax.config.update('jax_enable_x64', True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', type=int, nargs='+', default=[25, 33, 49])
    parser.add_argument('--repeats', type=int, default=80)
    parser.add_argument('--output', type=Path, default=Path('build/gs_response/results.json'))
    args = parser.parse_args()
    rows = []
    for n in args.nodes:
        eq = SolovevEquilibrium(logarithmic=.08)
        plan = FixedBoundaryGSPlan(SolovevFluxDomain(eq), nodes=n,
                                   volume_order=16, boundary_count=512)
        fast = plan.compile_response()
        x, z = plan.source_basis[0][0], plan.source_basis[1][0]
        # Fair baseline: cached basis and flux ONLY, no derivative evaluation.
        def reference(f, g):
            c = plan.solve(f, g).coefficients.reshape(n, n)
            return np.sum((x @ c)*z, axis=1)
        points, edge = plan.points, plan.boundary
        data = [(eq.source(points)*(1+.1*np.sin(k*points[:, 1])),
                 .01*np.sin(k*edge[:, 1])) for k in (1, 3, 7, 11)]
        errors = [float(np.max(np.abs(reference(f,g)-fast.solve(f,g).flux))) for f,g in data]
        analytic_error = float(np.max(np.abs(fast.solve(eq.source).flux-eq.jets(points)[0])))
        samples = [[], []]
        for i in range(args.repeats+4):
            f, g = data[i % len(data)]
            # Alternate order to reduce systematic cache/timing bias.
            for method in ((0,1) if i % 2 else (1,0)):
                start = perf_counter()
                value = reference(f,g) if method == 0 else fast.solve(f,g).flux
                if i >= 4:
                    samples[method].append(perf_counter()-start)
        old, new = [float(np.median(s)) for s in samples]
        row = dict(nodes=n, plan=plan.diagnostics, response=fast.diagnostics,
                   cached_reference_seconds=old, compiled_seconds=new, speedup=old/new,
                   incremental_break_even_solves=(fast.diagnostics['setup_seconds']/(old-new) if old>new else None),
                   max_flux_difference=max(errors), solovev_flux_linf=analytic_error,
                   timing_samples_seconds=samples)
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('plan','response','timing_samples_seconds')}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(rows=rows, numpy=np.__version__,
            threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
            baseline='same TSVD; cached tensor flux only; both include RHS and training residual'), indent=2)+'\n')


if __name__ == '__main__':
    main()
