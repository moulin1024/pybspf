"""Multiscale random-phase scalar MMS on the exact B-spline annulus.

A deterministic stress test, not a Navier--Stokes turbulence simulation. Physical
wavevectors have random angles/radii and are NOT chosen from the solver's Fourier
dictionary. Only f and boundary values are passed to the solver.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from bspf_models.elliptic.spline_annulus import FourierSourcePlan, adaptive_dirichlet
if __package__:
    from .spline_annulus_convergence import geometry
else:
    from spline_annulus_convergence import geometry


@dataclass
class RandomWaveMMS:
    maximum_wave: float
    seed: int = 20260922
    directions: int = 12

    def __post_init__(self):
        if not np.isfinite(self.maximum_wave) or self.maximum_wave <= 0 or self.directions < 3:
            raise ValueError('positive finite wave cutoff and at least three directions required')
        rng = np.random.default_rng(self.seed)
        # Log-spaced shell bands. k^-5/3 refers to the CHOSEN shell energy weights,
        # not a claim of a continuous Kolmogorov spectrum or a turbulence closure.
        radii, angles, bands = [], [], []
        for band, radius in enumerate(self.maximum_wave*np.array([1/8, 1/4, 1/2, 1.])):
            theta = (np.arange(self.directions)+rng.uniform(-.3, .3, self.directions))*2*np.pi/self.directions
            # Positive band width; the highest radius is <= requested cutoff.
            r = radius*rng.uniform(.82, 1., self.directions)
            radii.extend(r)
            angles.extend(theta)
            bands.extend([band]*self.directions)
        radii, angles = np.asarray(radii), np.asarray(angles)
        self.waves = radii[:, None]*np.column_stack((np.cos(angles), np.sin(angles)))
        self.bands = np.asarray(bands)
        self.phases = rng.uniform(0, 2*np.pi, len(radii))
        self.amplitudes = (radii/self.maximum_wave)**(-5/6)
        self.amplitudes *= np.sqrt(2/np.sum(self.amplitudes**2))
        self.norm2 = np.sum(self.waves**2, axis=1)

    def exact(self, x):
        return np.cos(np.asarray(x)@self.waves.T+self.phases)@self.amplitudes

    def gradient(self, x):
        return (-np.sin(np.asarray(x)@self.waves.T+self.phases)*self.amplitudes)@self.waves

    def forcing(self, x, sigma):
        return np.cos(np.asarray(x)@self.waves.T+self.phases)@(self.amplitudes*(self.norm2+sigma))

    def metadata(self):
        return dict(maximum_wave=self.maximum_wave, actual_maximum_wave=float(np.sqrt(self.norm2.max())),
                    minimum_wavelength=float(2*np.pi/np.sqrt(self.norm2.max())), seed=self.seed,
                    waves=self.waves.tolist(), phases=self.phases.tolist(), amplitudes=self.amplitudes.tolist(),
                    bands=self.bands.tolist())


def diagnostics(solution, mms, validation_n=48):
    domain = solution.plan.domain
    bulk = domain.sample(validation_n, .219)
    truth = mms.exact(bulk)
    actual = solution.interior(bulk)
    error = actual-truth
    # Uniform Cartesian samples approximate volume norms; no adaptive training points reused.
    metrics = dict(interior_points=len(bulk), relative_l2=float(np.linalg.norm(error)/np.linalg.norm(truth)),
                   relative_max=float(np.max(abs(error))/np.max(abs(truth))),
                   imaginary_max=float(np.max(abs(actual.imag))),
                   training_residual=solution.training_residual)
    near, boundary_error, boundary_truth = [], [], []
    for component, b in enumerate(domain.boundaries):
        t = b.a+(np.arange(96)+.413)/96*(b.b-b.a)
        t = np.r_[t, b.knots[:-1]+1e-6]
        expected = mms.exact(b.curve(t))
        boundary_error.extend(abs(solution.boundary(component, t)-expected))
        boundary_truth.extend(abs(expected))
        tn = b.a+(np.arange(8)+.173)/8*(b.b-b.a)
        for distance in (1e-2, 1e-4, 1e-6):
            near.extend(b.curve(tn)-distance*b.normal(tn, 1 if component == 0 else -1))
    near = np.asarray(near)
    if not np.all(domain.contains(near)):
        raise AssertionError('near-wall validation points left the physical domain')
    expected = mms.exact(near)
    actual_near = solution.interior(near)
    metrics['boundary_relative_max'] = float(max(boundary_error)/max(boundary_truth))
    metrics['near_relative_max'] = float(np.max(abs(actual_near-expected))/max(abs(expected)))
    # A separate quadrature check, including the closest targets.
    chosen = near[::4]
    low = actual_near[::4]
    high = solution.interior(chosen, quadrature_order=solution.plan.qorder+20)
    metrics['quadrature_relative_change'] = float(np.max(abs(high-low))/max(1., max(abs(high))))
    return metrics, (bulk, actual, error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('build/spline_annulus_turbulent'))
    parser.add_argument('--cutoffs', type=float, nargs='+', default=[12., 24., 48.])
    parser.add_argument('--source-modes', type=int, nargs='+', default=[12, 18, 24])
    parser.add_argument('--source-padding', type=float, default=1.5)
    parser.add_argument('--helmholtz-wave', type=float, default=8.)
    parser.add_argument('--source-only', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    domain, _ = geometry()
    sigmas = (0., args.helmholtz_wave**2, -args.helmholtz_wave**2)
    fields = [RandomWaveMMS(k) for k in args.cutoffs]
    (args.out/'mms.json').write_text(json.dumps([m.metadata() for m in fields], indent=2)+'\n')
    records, done = [], set()
    def save():
        (args.out/'results.json').write_text(json.dumps(records, indent=2)+'\n')
    for modes in args.source_modes:
        print('source setup', modes, flush=True)
        start = perf_counter()
        source_plan = FourierSourcePlan(domain, modes=modes, samples=4*modes+8,
                                       padding=args.source_padding)
        print('source ready', modes, perf_counter()-start, flush=True)
        for mms in fields:
            if mms.maximum_wave in done:
                continue
            audits = []
            accepted = True
            for sigma in sigmas:
                source = lambda x, sig=sigma: mms.forcing(x, sig)
                try:
                    particular = source_plan.fit(source, sigma, tolerance=1e-7)
                    audits.append(dict(sigma=sigma, accepted=True, **particular.stats))
                except ValueError as exc:
                    accepted = False
                    audits.append(dict(sigma=sigma, accepted=False, reason=str(exc)))
            record = dict(cutoff=mms.maximum_wave, source_modes=modes, source_audits=audits,
                          status='source_accepted' if accepted else 'source_rejected')
            records.append(record)
            print(json.dumps(record), flush=True)
            save()
            if not accepted:
                continue
            done.add(mms.maximum_wave)
            if args.source_only:
                continue
            for sigma in sigmas:
                start = perf_counter()
                def callback(sol, row):
                    print('boundary', mms.maximum_wave, sigma, row['level'], row['unknowns'], row['boundary_indicator'], flush=True)
                source = lambda x, sig=sigma: mms.forcing(x, sig)
                sol, history = adaptive_dirichlet(domain, sigma, (mms.exact, mms.exact),
                    source=source, source_plan=source_plan, source_tolerance=1e-7,
                    order=10, tolerance=1e-7, max_refinements=7, callback=callback)
                metrics, data = diagnostics(sol, mms)
                row = dict(cutoff=mms.maximum_wave, sigma=sigma, source_modes=modes,
                           status='solved', source=sol.particular.stats, history=history,
                           **metrics, total_seconds=perf_counter()-start)
                row['passed'] = bool(history[-1]['converged'] and metrics['relative_l2'] < 1e-6
                                     and metrics['relative_max'] < 1e-5 and metrics['near_relative_max'] < 1e-5
                                     and metrics['boundary_relative_max'] < 1e-5
                                     and metrics['quadrature_relative_change'] < 1e-8)
                records.append(row)
                np.savez_compressed(args.out/f'field_k{mms.maximum_wave:g}_sigma{sigma:g}.npz',
                                    points=data[0], actual=data[1], error=data[2], exact=mms.exact(data[0]))
                print(json.dumps({k:v for k,v in row.items() if k != 'history'}), flush=True)
                save()
        # Release the large SVD before building the next resolution.
        particular = sol = None
        del source_plan
        if len(done) == len(fields):
            break
    metadata = dict(source_tolerance=1e-7, boundary_indicator_tolerance=1e-7, source_only=args.source_only,
                    failed_pde_runs=[dict(cutoff=r['cutoff'], sigma=r['sigma']) for r in records if r['status']=='solved' and not r['passed']],
                    unresolved_cutoffs=[m.maximum_wave for m in fields if m.maximum_wave not in done],
                    note='Random-phase scalar MMS, not a turbulent NS trajectory. Source rejection is a failed resolution level, not a PDE pass.')
    import bspf_models.elliptic.spline_annulus as model
    metadata['sources'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(model.__file__))}
    (args.out/'summary.json').write_text(json.dumps(metadata, indent=2)+'\n')
    render(domain, fields, records, args.out)


def render(domain, fields, records, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = 180
    xx, yy = np.meshgrid(np.linspace(*domain.bounds[0], n), np.linspace(*domain.bounds[1], n))
    points = np.column_stack((xx.ravel(), yy.ravel()))
    inside = domain.contains(points)
    fig, axes = plt.subplots(1, len(fields), figsize=(5*len(fields), 4), squeeze=False)
    for ax, mms in zip(axes[0], fields):
        z = np.full(len(points), np.nan)
        z[inside] = mms.exact(points[inside])
        image = ax.pcolormesh(xx, yy, z.reshape(n,n), shading='auto', cmap='RdBu_r')
        for b in domain.boundaries:
            c = b.curve(np.linspace(b.a,b.b,500))
            ax.plot(c[:,0],c[:,1],color='black',lw=.7)
        ax.set(aspect='equal', title=f'Exact MMS: max |q| <= {mms.maximum_wave:g}')
        fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out/'exact_fields.png', dpi=160)
    solved = [r for r in records if r['status'] == 'solved']
    if solved:
        fig, axes = plt.subplots(1, 2, figsize=(10,4))
        for sigma in sorted(set(r['sigma'] for r in solved)):
            rows = [r for r in solved if r['sigma']==sigma]
            for ax, key in zip(axes, ('relative_l2','near_relative_max')):
                ax.semilogy([r['cutoff'] for r in rows],[r[key] for r in rows], 'o-',label=f'sigma={sigma:g}')
                ax.set(xlabel='MMS wave cutoff',ylabel=key)
                ax.grid(alpha=.3); ax.legend()
        fig.tight_layout()
        fig.savefig(out/'errors.png',dpi=160)


if __name__ == '__main__':
    main()
