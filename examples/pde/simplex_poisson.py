"""Host-side research prototype: harmonic trace lifting on triangular P_p spaces.

This is a Bernstein finite element/static-condensation baseline, not a Fourier
method or a replacement for the maintained JAX core. Solve -Delta u = f.
"""
from __future__ import annotations

import argparse
import json
from math import factorial
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def triangle_rule(order):
    """Positive Duffy quadrature on {(x,y): x,y>=0, x+y<=1}."""
    t, w = leggauss(order)
    t, w = (t + 1) / 2, w / 2
    r, s = np.meshgrid(t, t, indexing="ij")
    weights = (w[:, None] * w[None, :] * (1 - r)).ravel()
    xy = np.column_stack((r.ravel(), ((1 - r) * s).ravel()))
    return xy, weights


def indices(p):
    if not isinstance(p, int) or p < 1:
        raise ValueError("degree must be a positive integer")
    return np.array([(i, j, p-i-j) for i in range(p+1)
                     for j in range(p+1-i)], dtype=int)


def bernstein(alpha, xy):
    """Values and reference gradients; no divisions by barycentric coordinates."""
    lam = np.column_stack((1-xy.sum(axis=1), xy))
    gl = np.array([[-1., -1.], [1., 0.], [0., 1.]])
    p = int(alpha[0].sum())
    b = np.empty((len(xy), len(alpha)))
    grad = np.zeros((len(xy), len(alpha), 2))
    for k, a in enumerate(alpha):
        c = factorial(p) / np.prod([factorial(int(v)) for v in a])
        b[:, k] = c * np.prod(lam ** a, axis=1)
        for d in range(3):
            if a[d]:
                powers = a.copy()
                powers[d] -= 1
                grad[:, k] += (c*a[d]*np.prod(lam**powers, axis=1))[:, None]*gl[d]
    return b, grad


def square_mesh(n, seed=42):
    """Reproducible Delaunay mesh with perturbed interior vertices."""
    if n < 2:
        raise ValueError("n must be >= 2")
    x, y = np.meshgrid(np.linspace(0, 1, n+1), np.linspace(0, 1, n+1))
    pts = np.column_stack((x.ravel(), y.ravel()))
    interior = np.all((pts > 0) & (pts < 1), axis=1)
    pts[interior] += np.random.default_rng(seed).uniform(-.2/n, .2/n, (interior.sum(), 2))
    return pts, Delaunay(pts).simplices


def topology(points, cells, alpha):
    """Globally oriented Bernstein edge coefficients, shared across neighbors."""
    p = int(alpha[0].sum())
    edge_counts = {}
    for cell in cells:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(cell[i]), int(cell[j]))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    edge_ids = {e: len(points)+k*(p-1) for k, e in enumerate(sorted(edge_counts))}
    next_id = len(points)+len(edge_ids)*(p-1)
    maps = []
    for cell in cells:
        ids = []
        for a in alpha:
            active = np.flatnonzero(a)
            if len(active) == 1:
                ids.append(int(cell[active[0]]))
            elif len(active) == 2:
                edge = tuple(sorted(int(cell[k]) for k in active))
                hi = int(np.flatnonzero(cell == edge[1])[0])
                ids.append(edge_ids[edge] + int(a[hi])-1)
            else:
                ids.append(next_id)
                next_id += 1
        maps.append(ids)
    return np.asarray(maps), edge_counts, edge_ids, next_id


def boundary_coefficients(points, edge_counts, edge_ids, p, boundary):
    """Interpolate Dirichlet data in the shared edge Bernstein basis."""
    values = {}
    t = np.linspace(0, 1, p+1)
    vand = np.column_stack([factorial(p)/factorial(k)/factorial(p-k)*t**k*(1-t)**(p-k)
                            for k in range(p+1)])
    for (a, b), count in edge_counts.items():
        if count != 1:
            continue
        xy = (1-t[:, None])*points[a] + t[:, None]*points[b]
        c = np.linalg.solve(vand, np.asarray(boundary(xy)))
        values[a], values[b] = c[0], c[-1]
        for k in range(1, p):
            values[edge_ids[(a, b)]+k-1] = c[k]
    return values


def solve(points, cells, p, forcing, boundary, *, condensed=True):
    alpha = indices(p)
    trace = np.flatnonzero(np.any(alpha == 0, axis=1))
    bubble = np.flatnonzero(np.all(alpha > 0, axis=1))
    maps, edges, edge_ids, total = topology(points, cells, alpha)
    skeleton = len(points)+len(edges)*(p-1)
    size = skeleton if condensed else total
    xy, w = triangle_rule(p+4)
    b, gr = bernstein(alpha, xy)
    rows, cols, data = [], [], []
    rhs = np.zeros(size)
    local = []
    orthogonality = 0.
    for cell, ids in zip(cells, maps):
        vertices = points[cell]
        jac = (vertices[1:]-vertices[0]).T
        weight = w*abs(np.linalg.det(jac))
        grad = gr @ np.linalg.inv(jac)
        physical = vertices[0]+xy@jac.T
        k = np.einsum('qid,qjd,q->ij', grad, grad, weight)
        f = b.T @ (weight*np.asarray(forcing(physical)))
        if len(bubble):
            kii = k[np.ix_(bubble, bubble)]
            kit = k[np.ix_(bubble, trace)]
            lift = -np.linalg.solve(kii, kit)
            particular = np.linalg.solve(kii, f[bubble])
            orthogonality = max(orthogonality, float(np.linalg.norm(kit+kii@lift) / np.linalg.norm(k)))
        else:
            lift, particular = np.empty((0, len(trace))), np.empty(0)
        if condensed:
            matrix = k[np.ix_(trace, trace)] + k[np.ix_(trace, bubble)]@lift
            load = f[trace] - k[np.ix_(trace, bubble)]@particular
            assembly_ids = ids[trace]
        else:
            matrix, load, assembly_ids = k, f, ids
        rows.extend(np.repeat(assembly_ids, len(assembly_ids)))
        cols.extend(np.tile(assembly_ids, len(assembly_ids)))
        data.extend(matrix.ravel())
        np.add.at(rhs, assembly_ids, load)
        local.append((lift, particular))
    matrix = coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()
    prescribed = boundary_coefficients(points, edges, edge_ids, p, boundary)
    fixed = np.array(sorted(prescribed), dtype=int)
    solution = np.zeros(size)
    solution[fixed] = [prescribed[i] for i in fixed]
    free = np.setdiff1d(np.arange(size), fixed)
    load = (rhs-matrix@solution)[free]
    solution[free] = spsolve(matrix[free][:, free], load)
    residual = np.linalg.norm((matrix@solution-rhs)[free])/max(np.linalg.norm(load), 1.)
    coefficients = []
    for ids, (lift, particular) in zip(maps, local):
        if condensed:
            c = np.empty(len(alpha))
            c[trace] = solution[ids[trace]]
            c[bubble] = particular+lift@c[trace]
        else:
            c = solution[ids]
        coefficients.append(c)
    return dict(coefficients=np.array(coefficients), alpha=alpha,
                stats=dict(degree=p, triangles=len(cells), total_dofs=total,
                           skeleton_dofs=skeleton, free_solved_dofs=len(free),
                           residual=float(residual), lift_orthogonality=orthogonality))


def error(points, cells, result, exact, exact_gradient):
    p = result['stats']['degree']
    xy, w = triangle_rule(p+9)  # independent, higher-order validation quadrature
    b, gr = bernstein(result['alpha'], xy)
    l2 = h1 = 0.
    for cell, c in zip(cells, result['coefficients']):
        v = points[cell]
        jac = (v[1:]-v[0]).T
        physical = v[0]+xy@jac.T
        weights = w*abs(np.linalg.det(jac))
        l2 += weights @ (b@c-exact(physical))**2
        g = np.einsum('qid,i->qd', gr@np.linalg.inv(jac), c)
        h1 += weights @ np.sum((g-exact_gradient(physical))**2, axis=1)
    return dict(l2=float(np.sqrt(l2)), h1_seminorm=float(np.sqrt(h1)))


def manufactured(x):
    return x[:, 0]+x[:, 1]+np.sin(np.pi*x[:, 0])*np.sin(np.pi*x[:, 1])


def forcing(x):
    return 2*np.pi**2*np.sin(np.pi*x[:, 0])*np.sin(np.pi*x[:, 1])


def gradient(x):
    return 1+np.pi*np.column_stack((np.cos(np.pi*x[:, 0])*np.sin(np.pi*x[:, 1]),
                                   np.sin(np.pi*x[:, 0])*np.cos(np.pi*x[:, 1])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('build/simplex_poisson'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for p in (1, 2, 3, 4, 5):
        for n in (4, 8, 16):
            points, cells = square_mesh(n)
            result = solve(points, cells, p, forcing, manufactured)
            record = dict(n=n, **result['stats'], **error(points, cells, result, manufactured, gradient))
            if n == 4:
                full = solve(points, cells, p, forcing, manufactured, condensed=False)
                record['full_condensed_difference'] = float(np.max(np.abs(full['coefficients']-result['coefficients'])))
            records.append(record)
            print(json.dumps(record), flush=True)
    (args.out/'convergence.json').write_text(json.dumps(records, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for p in (1, 2, 3, 4, 5):
        subset = [r for r in records if r['degree'] == p]
        for ax, key in zip(axes, ('l2', 'h1_seminorm')):
            ax.loglog([1/r['n'] for r in subset], [r[key] for r in subset], 'o-', label=f'p={p}')
            ax.set(xlabel='Nominal h', ylabel=key)
            ax.grid(True, which='both', alpha=.3)
            ax.legend()
    fig.tight_layout()
    fig.savefig(args.out/'convergence.png', dpi=170)


if __name__ == '__main__':
    main()
