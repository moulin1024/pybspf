"""Residual-selected edge modes on a fixed rich triangular Poisson interior.

An exploratory global Schur-complement greedy selector, NOT a scalable local
error estimator. Reports selection cost separately. Exact solutions are used
only after solving, never by mode selection. Homogeneous Dirichlet data only.
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import json
from math import comb
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import cho_solve
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

if __package__:
    from .simplex_bspf_comparison import Plan, Space, cases
    from .simplex_poisson import square_mesh
else:
    from simplex_bspf_comparison import Plan, Space, cases
    from simplex_poisson import square_mesh


class Skeleton:
    def __init__(self, plan):
        self.plan = plan
        p = plan.space.trace_degree
        t = np.linspace(0, 1, p+1)
        vand = np.column_stack([comb(p, k)*t**k*(1-t)**(p-k) for k in range(p+1)])
        modes = {}
        for degree in range(2, p+1):
            poly = np.polynomial.legendre.Legendre.basis(degree-2)(2*t-1)
            modes[degree] = np.linalg.solve(vand, t*(1-t)*poly)[1:-1]
        vertices = [int(v) for v in plan.free if v < len(plan.points)]
        rows, cols, values = [], [], []
        self.labels = []
        for vertex in vertices:
            col = len(self.labels)
            self.labels.append(('vertex', vertex, 1))
            rows.append(vertex); cols.append(col); values.append(1.)
            for (a, b), start in plan.edge_ids.items():
                if vertex not in (a, b):
                    continue
                for k in range(1, p):
                    rows.append(start+k-1); cols.append(col)
                    values.append(1-k/p if vertex == a else k/p)
        for edge, start in sorted(plan.edge_ids.items()):
            if plan.edges[edge] != 2:
                continue
            for degree in range(2, p+1):
                col = len(self.labels)
                self.labels.append(('edge', edge, degree))
                rows.extend(range(start, start+p-1))
                cols.extend([col]*(p-1))
                values.extend(modes[degree])
        self.prolongation = coo_matrix((values, (rows, cols)),
                                      shape=(plan.skeleton, len(self.labels))).tocsr()
        self.matrix = (self.prolongation.T@plan.matrix@self.prolongation).tocsc()
        if self.matrix.shape[0] != len(plan.free):
            raise ValueError('hierarchical basis does not span the free trace space')

    def uniform(self, degree):
        return np.array([i for i, label in enumerate(self.labels) if label[2] <= degree], dtype=int)

    def load(self, forcing):
        rhs = np.zeros(self.plan.skeleton)
        self.particulars = []
        for local, ids in zip(self.plan.locals, self.plan.maps):
            f = self.plan.b.T@(self.plan.w*local['det']*forcing(local['physical']))
            z = cho_solve(local['factor'], f[self.plan.nt:])
            np.add.at(rhs, ids, f[:self.plan.nt]-local['k'][:self.plan.nt, self.plan.nt:]@z)
            self.particulars.append(z)
        self.rhs = np.asarray(self.prolongation.T@rhs)

    def solve(self, selected):
        selected = np.sort(np.asarray(selected, dtype=int))
        factor = splu(self.matrix[selected][:, selected])
        coefficients = np.zeros(len(self.labels))
        coefficients[selected] = factor.solve(self.rhs[selected])
        return coefficients, factor

    def gain(self, selected, coefficients, factor, *, score="schur"):
        """Exact energy-error decrease for EACH individual candidate mode.

        In the finite enriched reference space, adding one mode has squared
        energy improvement r_j^2/(A_jj-A_jS A_SS^{-1} A_Sj). Batched selection
        uses these single-mode scores as a heuristic; gains are not additive.
        score="diagonal" uses r_j^2/A_jj, omitting relaxation of selected modes;
        this avoids solving for every candidate but is only a proxy.
        """
        selected = np.sort(np.asarray(selected, dtype=int))
        candidates = np.setdiff1d(np.arange(len(self.labels)), selected)
        denominator = self.matrix.diagonal()[candidates].copy()
        if score == 'schur':
            coupling = self.matrix[selected][:, candidates].toarray()
            response = factor.solve(coupling)
            denominator -= np.sum(coupling*response, axis=0)
        elif score != 'diagonal':
            raise ValueError('unknown score')
        if np.any(denominator <= 0):
            raise ValueError('nonpositive candidate Schur energy')
        residual = self.rhs-self.matrix@coefficients
        scores = residual[candidates]**2/denominator
        return candidates, scores

    def recover(self, coefficients):
        trace = np.asarray(self.prolongation@coefficients)
        return np.array([np.r_[trace[ids], z+local['lift']@trace[ids]] for ids, z, local in
                         zip(self.plan.maps, self.particulars, self.plan.locals)])

    def record(self, selected, coefficients, exact, gradient, label, elapsed):
        error = self.plan.errors(self.recover(coefficients), exact, gradient)
        residual = self.rhs-self.matrix@coefficients
        return dict(label=label, free_solved_dofs=len(selected),
                    interior_dofs=len(self.plan.cells)*self.plan.ni,
                    represented_free_dofs=len(selected)+len(self.plan.cells)*self.plan.ni,
                    selected_residual=float(np.linalg.norm(residual[selected])/max(1., np.linalg.norm(self.rhs[selected]))),
                    remaining_residual=float(np.linalg.norm(residual)),
                    selection_and_solve_seconds=elapsed, **error,
                    selected_modes=[self.labels[int(i)] for i in sorted(selected)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('build/simplex_bspf_adaptive'))
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--score', choices=['schur', 'diagonal'], default='schur')
    parser.add_argument('--interior-degree', type=int, default=6)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error('--batch must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    points, cells = square_mesh(args.n, seed=args.seed)
    plan = Plan(points, cells, Space(6, args.interior_degree))
    start = perf_counter()
    skeleton = Skeleton(plan)
    hierarchy_seconds = perf_counter()-start
    records = []
    for name, (u, f, grad) in cases().items():
        if np.max(np.abs(u(points[np.any((points == 0) | (points == 1), axis=1)]))) > 1e-12:
            raise ValueError('this selector supports only zero boundary data')
        skeleton.load(f)
        for degree in (3, 4, 5, 6):
            chosen = skeleton.uniform(degree)
            start = perf_counter()
            coefficients, factor = skeleton.solve(chosen)
            elapsed = perf_counter()-start
            records.append(dict(case=name, **skeleton.record(chosen, coefficients, u, grad,
                                                             f'uniform-t{degree}', elapsed)))
        chosen = skeleton.uniform(3)
        cumulative = 0.
        for degree in (4, 5):
            target = len(skeleton.uniform(degree))
            start = perf_counter()
            while len(chosen) < target:
                coefficients, factor = skeleton.solve(chosen)
                candidates, scores = skeleton.gain(chosen, coefficients, factor, score=args.score)
                count = min(args.batch, target-len(chosen))
                best = np.argsort(-scores, kind='stable')[:count]
                chosen = np.sort(np.r_[chosen, candidates[best]])
            coefficients, factor = skeleton.solve(chosen)
            cumulative += perf_counter()-start
            records.append(dict(case=name, **skeleton.record(chosen, coefficients, u, grad,
                                                             f'adaptive-budget-t{degree}', cumulative)))
        (args.out/'results.json').write_text(json.dumps(records, indent=2)+'\n')
        for r in records:
            if r['case'] == name:
                print(json.dumps({k: v for k, v in r.items() if k != 'selected_modes'}), flush=True)
    (args.out/'setup.json').write_text(json.dumps(dict(n=args.n, batch=args.batch, seed=args.seed, score=args.score,
                                hierarchy_seconds=hierarchy_seconds, **plan.stats), indent=2)+'\n')
    (args.out/'metadata.json').write_text(json.dumps(dict(python=platform.python_version(),
        sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 (Path(__file__), Path(__file__).with_name('simplex_bspf_comparison.py'),
                  Path(__file__).with_name('simplex_poisson.py'))}), indent=2)+'\n')
    plot(records, skeleton, args.out)


def plot(records, skeleton, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, case in zip(axes, cases()):
        for prefix, marker in [('uniform', 'o'), ('adaptive', 's')]:
            rows = [r for r in records if r['case'] == case and r['label'].startswith(prefix)]
            ax.semilogy([r['free_solved_dofs'] for r in rows], [r['relative_h1'] for r in rows], marker+'-', label=prefix)
        ax.set(title=case, xlabel='Free skeleton DOFs', ylabel='Relative H1 seminorm error')
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out/'accuracy.png', dpi=160)
    r = next(r for r in records if r['case'] == 'localized' and r['label'] == 'adaptive-budget-t4')
    counts = {edge: 0 for edge in skeleton.plan.edges}
    for kind, key, degree in r['selected_modes']:
        if kind == 'edge':
            counts[tuple(key)] += 1
    fig, ax = plt.subplots(figsize=(6, 5))
    edges = sorted(counts)
    lines = LineCollection([skeleton.plan.points[list(e)] for e in edges],
                           array=np.array([counts[e] for e in edges]), cmap='viridis', linewidths=1.7)
    lines.set_clim(0, 5)
    ax.add_collection(lines)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', title='Residual-selected edge modes: localized load')
    fig.colorbar(lines, ax=ax, label='Active edge modes (excluding vertices)')
    fig.tight_layout()
    fig.savefig(out/'selected_edges.png', dpi=160)


if __name__ == '__main__':
    main()
