"""Compare independent trace/interior resolution for triangular Poisson FEM.

Research-only NumPy/SciPy implementation. Polynomial trace lifts plus either
polynomial or bubble-windowed Fourier interiors. All candidates share assembly,
static condensation and sparse factorization; there is no FFT fast solve claim.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

if __package__:
    from .simplex_poisson import (bernstein, indices, triangle_rule, square_mesh,
                                 topology, boundary_coefficients)
else:
    from simplex_poisson import (bernstein, indices, triangle_rule, square_mesh,
                                 topology, boundary_coefficients)


@dataclass(frozen=True)
class Space:
    trace_degree: int
    interior_degree: int
    family: str = 'polynomial'

    def __post_init__(self):
        if self.trace_degree < 3 or self.interior_degree < self.trace_degree:
            raise ValueError('require 3 <= trace_degree <= interior_degree')
        if self.family not in ('polynomial', 'fourier'):
            raise ValueError('unknown interior family')

    @property
    def interior_count(self):
        q = self.interior_degree
        return (q-1)*(q-2)//2

    @property
    def label(self):
        if self.family == 'polynomial' and self.trace_degree == self.interior_degree:
            return f'P{self.trace_degree}'
        return f'{self.family[:4]}-t{self.trace_degree}-i{self.interior_count}'


def wave_modes(count):
    """Fixed, solution-independent ordering on the reference bounding square.

    Real sine/cosine pairs from one half of the integer lattice, sorted by |k|.
    A fixed budget can end in half a pair; all retained modes are reported.
    """
    waves = [(i, j) for i in range(0, count+1) for j in range(-count, count+1)
             if i > 0 or (i == 0 and j > 0)]
    waves.sort(key=lambda k: (k[0]**2+k[1]**2, k[0], k[1]))
    return [(kind, k) for k in waves for kind in ('cos', 'sin')][:count]


def raw_basis(space, xy):
    p = space.trace_degree
    alpha = indices(p)
    trace = np.any(alpha == 0, axis=1)
    b, g = bernstein(alpha, xy)
    bt, gt = b[:, trace], g[:, trace]
    if space.family == 'polynomial':
        a = indices(space.interior_degree)
        bi, gi = bernstein(a[np.all(a > 0, axis=1)], xy)
    else:
        bi, gi = b[:, ~trace], g[:, ~trace]
        extra = space.interior_count-bi.shape[1]
        bubble, dbubble = bernstein(np.array([[1, 1, 1]]), xy)
        bubble, dbubble = bubble[:, 0], dbubble[:, 0]
        vals, grads = [bi], [gi]
        for kind, k in wave_modes(extra):
            freq = 2*np.pi*np.asarray(k)
            theta = xy@freq
            value = np.cos(theta) if kind == 'cos' else np.sin(theta)
            deriv = -np.sin(theta) if kind == 'cos' else np.cos(theta)
            vals.append((bubble*value)[:, None])
            grads.append((dbubble*value[:, None] + (bubble*deriv)[:, None]*freq)[:, None, :])
        bi, gi = np.concatenate(vals, axis=1), np.concatenate(grads, axis=1)
    return np.concatenate((bt, bi), axis=1), np.concatenate((gt, gi), axis=1)


class Plan:
    """Geometry/operator setup once; independent solves for multiple loads."""

    def __init__(self, points, cells, space, *, quadrature=24):
        started = perf_counter()
        self.points = np.asarray(points)
        # Canonical ordering makes the anisotropic Fourier mode policy invariant
        # to input cell permutations (polynomial spaces are invariant anyway).
        self.cells = np.sort(np.asarray(cells), axis=1)
        self.space = space
        self.quadrature = quadrature
        self.nt = 3*space.trace_degree
        self.ni = space.interior_count
        self.xy, self.w = triangle_rule(quadrature)
        raw, grad = raw_basis(space, self.xy)
        # QR of reference gradient samples, not a Gram-matrix eigendecomposition.
        # Whitening changes coordinates only; no modes are discarded.
        sample = (grad[:, self.nt:, :]*np.sqrt(self.w)[:, None, None]).transpose(0, 2, 1).reshape(-1, self.ni)
        _, r = np.linalg.qr(sample, mode='reduced')
        singular = np.linalg.svd(r, compute_uv=False)
        if singular[-1] <= singular[0]*1e-12:
            raise ValueError('interior basis numerically rank deficient')
        self.transform = solve_triangular(r, np.eye(self.ni))
        self.b = np.column_stack((raw[:, :self.nt], raw[:, self.nt:]@self.transform))
        self.grad = np.concatenate((grad[:, :self.nt],
                                   np.einsum('qid,ij->qjd', grad[:, self.nt:], self.transform)), axis=1)
        # Three reusable reference contractions for every affine triangle.
        gx, gy = self.grad[:, :, 0], self.grad[:, :, 1]
        self.reference = np.array([gx.T@(self.w[:, None]*gx),
                                   gx.T@(self.w[:, None]*gy)+gy.T@(self.w[:, None]*gx),
                                   gy.T@(self.w[:, None]*gy)])
        ref_done = perf_counter()
        alpha = indices(space.trace_degree)
        maps, edges, edge_ids, _ = topology(self.points, self.cells, alpha)
        self.maps = maps[:, np.any(alpha == 0, axis=1)]
        self.edges, self.edge_ids = edges, edge_ids
        self.skeleton = len(points)+len(edges)*(space.trace_degree-1)
        self.total = self.skeleton+len(cells)*self.ni
        rows, cols, values = [], [], []
        self.locals = []
        orthogonality = raw_cond = white_cond = 0.
        for cell, ids in zip(self.cells, self.maps):
            v = self.points[cell]
            jac = (v[1:]-v[0]).T
            det = abs(np.linalg.det(jac))
            inv = np.linalg.inv(jac)
            metric = inv@inv.T
            k = det*np.einsum('a,aij->ij', [metric[0, 0], metric[0, 1], metric[1, 1]], self.reference)
            kii = k[self.nt:, self.nt:]
            factor = cho_factor(kii)
            lift = -cho_solve(factor, k[self.nt:, :self.nt])
            schur = k[:self.nt, :self.nt]+k[:self.nt, self.nt:]@lift
            rows.extend(np.repeat(ids, self.nt))
            cols.extend(np.tile(ids, self.nt))
            values.extend(schur.ravel())
            self.locals.append(dict(jac=jac, inv=inv, det=det,
                                    physical=v[0]+self.xy@jac.T,
                                    factor=factor, lift=lift, k=k))
            orthogonality = max(orthogonality, float(np.linalg.norm(k[self.nt:, :self.nt]+kii@lift)/np.linalg.norm(k)))
            # These diagnostics are included in setup time, identically for all candidates.
            raw_cond = max(raw_cond, float(np.linalg.cond(r.T@kii@r)))
            white_cond = max(white_cond, float(np.linalg.cond(kii)))
        self.matrix = coo_matrix((values, (rows, cols)), shape=(self.skeleton, self.skeleton)).tocsr()
        fixed = boundary_coefficients(points, edges, edge_ids, space.trace_degree, lambda x: np.zeros(len(x)))
        self.fixed = np.array(sorted(fixed), dtype=int)
        self.free = np.setdiff1d(np.arange(self.skeleton), self.fixed)
        geometry_done = perf_counter()
        self.factor = splu(self.matrix[self.free][:, self.free].tocsc())
        done = perf_counter()
        self.stats = dict(label=space.label, family=space.family,
                          trace_degree=space.trace_degree, interior_count=self.ni,
                          triangles=len(cells), total_dofs=self.total,
                          skeleton_dofs=self.skeleton, free_solved_dofs=len(self.free),
                          quadrature_points=quadrature**2,
                          reference_seconds=ref_done-started,
                          geometry_seconds=geometry_done-ref_done,
                          factor_seconds=done-geometry_done, setup_seconds=done-started,
                          raw_interior_condition_max=raw_cond,
                          white_interior_condition_max=white_cond,
                          reference_whitening_error=float(np.linalg.norm(self.reference[0][self.nt:, self.nt:]+self.reference[2][self.nt:, self.nt:]-np.eye(self.ni))),
                          lift_orthogonality=orthogonality)
        if space.family == 'fourier':
            base_interior = (space.trace_degree-1)*(space.trace_degree-2)//2
            self.stats['waves'] = wave_modes(self.ni-base_interior)

    def evaluate_basis(self, xy):
        b, g = raw_basis(self.space, xy)
        return (np.column_stack((b[:, :self.nt], b[:, self.nt:]@self.transform)),
                np.concatenate((g[:, :self.nt], np.einsum('qid,ij->qjd', g[:, self.nt:], self.transform)), axis=1))

    def solve(self, forcing, boundary):
        start = perf_counter()
        rhs = np.zeros(self.skeleton)
        particulars = []
        for local, ids in zip(self.locals, self.maps):
            f = self.b.T@(self.w*local['det']*forcing(local['physical']))
            particular = cho_solve(local['factor'], f[self.nt:])
            load = f[:self.nt]-local['k'][:self.nt, self.nt:]@particular
            np.add.at(rhs, ids, load)
            particulars.append(particular)
        trace = np.zeros(self.skeleton)
        prescribed = boundary_coefficients(self.points, self.edges, self.edge_ids,
                                           self.space.trace_degree, boundary)
        trace[self.fixed] = [prescribed[i] for i in self.fixed]
        load = (rhs-self.matrix@trace)[self.free]
        load_done = perf_counter()
        trace[self.free] = self.factor.solve(load)
        solved = perf_counter()
        coeff = np.array([np.r_[trace[ids], particular+local['lift']@trace[ids]]
                          for ids, particular, local in zip(self.maps, particulars, self.locals)])
        end = perf_counter()
        stats = dict(load_seconds=load_done-start, global_solve_seconds=solved-load_done,
                     recovery_seconds=end-solved, rhs_seconds=end-start,
                     residual=float(np.linalg.norm((self.matrix@trace-rhs)[self.free])/max(1., np.linalg.norm(load))))
        return coeff, stats

    def errors(self, coefficients, exact, exact_gradient, *, order=36):
        xy, w = triangle_rule(order)
        b, grad = self.evaluate_basis(xy)
        l2 = h1 = norm_l2 = norm_h1 = 0.
        for cell, local, c in zip(self.cells, self.locals, coefficients):
            physical = self.points[cell[0]]+xy@local['jac'].T
            weights = w*local['det']
            u, g = exact(physical), exact_gradient(physical)
            diff = np.einsum('qid,i->qd', grad@local['inv'], c)-g
            l2 += weights@(b@c-u)**2
            h1 += weights@np.sum(diff**2, axis=1)
            norm_l2 += weights@(u*u)
            norm_h1 += weights@np.sum(g*g, axis=1)
        return dict(l2=float(np.sqrt(l2)), h1_seminorm=float(np.sqrt(h1)),
                    relative_l2=float(np.sqrt(l2/norm_l2)), relative_h1=float(np.sqrt(h1/norm_h1)))

    def best_trace_error(self, exact, order=40):
        """Best edgewise L2 P_p error on all mesh edges, normalized by length.

        Edgewise projections need not agree at vertices: this is an optimistic
        trace approximation diagnostic, not a bound on the volume L2 error.
        """
        z, weights = np.polynomial.legendre.leggauss(order)
        t, weights = (z+1)/2, weights/2
        basis = np.polynomial.legendre.legvander(z, self.space.trace_degree)
        sample = np.sqrt(weights)[:, None]*basis
        squared = length_sum = 0.
        for a, b in self.edges:
            x = (1-t[:, None])*self.points[a]+t[:, None]*self.points[b]
            values = exact(x)
            c = np.linalg.lstsq(sample, np.sqrt(weights)*values, rcond=None)[0]
            length = np.linalg.norm(self.points[b]-self.points[a])
            squared += length*(weights@(basis@c-values)**2)
            length_sum += length
        return float(np.sqrt(squared/length_sum))


def cases():
    def sine(k, l):
        a, b = np.pi*k, np.pi*l
        def u(x):
            return np.sin(a*x[:, 0])*np.sin(b*x[:, 1])
        def g(x):
            return np.column_stack((a*np.cos(a*x[:, 0])*np.sin(b*x[:, 1]),
                                    b*np.sin(a*x[:, 0])*np.cos(b*x[:, 1])))
        return u, lambda x: (a*a+b*b)*u(x), g

    def gaussian_data(x):
        # Zero boundary data and a localized, smooth interior feature.
        a = 120.
        z = x-np.array([.43, .57])
        e = np.exp(-a*np.sum(z*z, axis=1))
        v = x*(1-x)
        d = 1-2*x
        u = np.prod(v, axis=1)*e
        g = np.column_stack((v[:, 1]*(d[:, 0]-2*a*z[:, 0]*v[:, 0]),
                             v[:, 0]*(d[:, 1]-2*a*z[:, 1]*v[:, 1])))*e[:, None]
        second = (-2-4*a*z*d+(4*a*a*z*z-2*a)*v)
        f = -e*(v[:, 1]*second[:, 0]+v[:, 0]*second[:, 1])
        return u, f, g
    return dict(smooth=sine(1, 1), oscillatory=sine(5, 4),
                localized=(lambda x: gaussian_data(x)[0], lambda x: gaussian_data(x)[1],
                           lambda x: gaussian_data(x)[2]))


def audit_quadrature(points, cells, *, low=24, high=36):
    """Change assembly rule and compare fields in a third, fixed quadrature."""
    records = []
    for space in (Space(6, 6), Space(3, 7), Space(3, 7, 'fourier')):
        a = Plan(points, cells, space, quadrature=low)
        b = Plan(points, cells, space, quadrature=high)
        xy, w = triangle_rule(40)
        ba, ga = a.evaluate_basis(xy)
        bb, gb = b.evaluate_basis(xy)
        for name, (u, f, grad) in cases().items():
            ca, _ = a.solve(f, u)
            cb, _ = b.solve(f, u)
            delta = reference = 0.
            for local, ac, bc in zip(a.locals, ca, cb):
                ag = np.einsum('qid,i->qd', ga@local['inv'], ac)
                bg = np.einsum('qid,i->qd', gb@local['inv'], bc)
                delta += (w*local['det'])@np.sum((ag-bg)**2, axis=1)
                reference += (w*local['det'])@np.sum(bg**2, axis=1)
            change = float(np.sqrt(delta/reference))
            if change > 1e-8:
                raise AssertionError(f'quadrature audit failed: {space.label}, {name}: {change}')
            records.append(dict(label=space.label, case=name, low_order=low,
                                high_order=high, relative_h1_solution_change=change))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('build/simplex_bspf'))
    parser.add_argument('--meshes', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--quadrature', type=int, default=24)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spaces = [Space(p, p) for p in (3, 4, 5, 6)]
    spaces += [Space(3, q, family) for q in (5, 6, 7) for family in ('polynomial', 'fourier')]
    records = []
    for n in args.meshes:
        points, cells = square_mesh(n)
        for space in spaces:
            plan = Plan(points, cells, space, quadrature=args.quadrature)
            for name, (u, f, grad) in cases().items():
                timings = []
                for _ in range(args.repeats):
                    coeff, timing = plan.solve(f, u)
                    timings.append(timing)
                medians = {key: float(np.median([t[key] for t in timings])) for key in timing}
                record = dict(n=n, case=name, **plan.stats, **medians,
                              **plan.errors(coeff, u, grad), best_trace_rms=plan.best_trace_error(u))
                records.append(record)
                print(json.dumps(record), flush=True)
            (args.out/'results.json').write_text(json.dumps(records, indent=2)+'\n')
    metadata = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    platform=platform.platform(), seed=42, repeats=args.repeats,
                    thread_environment={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
                    sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             (Path(__file__), Path(__file__).with_name('simplex_poisson.py'))})
    try:
        from threadpoolctl import threadpool_info
        metadata['threadpools'] = threadpool_info()
    except ImportError:
        metadata['threadpools'] = 'threadpoolctl unavailable'
    (args.out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    plot(records, args.out)
    points, cells = square_mesh(min(args.meshes))
    audit = audit_quadrature(points, cells, low=args.quadrature, high=args.quadrature+12)
    (args.out/'quadrature_audit.json').write_text(json.dumps(audit, indent=2)+'\n')


def plot(records, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = max(r['n'] for r in records)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for col, name in enumerate(cases()):
        subset = [r for r in records if r['n'] == n and r['case'] == name]
        for family, marker, color in [('standard', 'o', 'C0'), ('polynomial', 's', 'C1'), ('fourier', '^', 'C2')]:
            rows = [r for r in subset if ('standard' if r['label'].startswith('P') else r['family']) == family]
            label = {'standard': 'Full Pp', 'polynomial': 'P3 trace + polynomial interior',
                     'fourier': 'P3 trace + Fourier interior'}[family]
            for ax, key in zip(axes[:, col], ('free_solved_dofs', 'total_dofs')):
                ax.loglog([r[key] for r in rows], [r['relative_h1'] for r in rows],
                          marker+'-', color=color, label=label)
                ax.set(xlabel=key.replace('_', ' '), ylabel='Relative H1 seminorm error')
                ax.grid(True, which='both', alpha=.25)
        axes[0, col].set_title(name)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'Trace/interior comparison: {2*n*n} perturbed Delaunay triangles')
    fig.tight_layout()
    fig.savefig(out/'comparison.png', dpi=160)


if __name__ == '__main__':
    main()
