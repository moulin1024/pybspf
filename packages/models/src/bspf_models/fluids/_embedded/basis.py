"""Local tensor basis and deterministic cut-cell aggregation (host setup)."""

import numpy as np

from scipy.special import roots_legendre

from numpy.polynomial.legendre import legvander, legder, legval


def unique(x, tol=1e-12):
    x = np.sort(np.asarray(x))
    return x[np.r_[True, np.diff(x) > tol]] if len(x) else x


def gauss(n):
    q, w = roots_legendre(n)
    return (q + 1) / 2, w / 2


class Cuts:
    def aggregate(self):
        # Multi-source face-neighbour growth; every aggregate contains a full root.
        owner = {k: k for k in sorted(self.full)}
        pending = set(self.cut)
        neighbours = {k: set() for k in self.active}
        for a, b, *_ in self.faces:
            neighbours[a].add(b)
            neighbours[b].add(a)
        while pending:
            progress = False
            for k in sorted(pending.copy()):
                candidates = []
                for nb in sorted(neighbours[k]):
                    if nb in owner:
                        candidates.append(owner[nb])
                if candidates:
                    owner[k] = min(
                        candidates,
                        key=lambda r: ((r[0] - k[0]) ** 2 + (r[1] - k[1]) ** 2, r),
                    )
                    pending.remove(k)
                    progress = True
            if not progress:
                raise ValueError("cut component has no full-cell neighbour")
        return owner


class Basis:
    def __init__(self, root, grid, line, polynomial, aggregate_degree=7):
        self.origin = np.array([grid.edges[i][root[i]] for i in range(2)])
        self.h = grid.h
        self.line = line
        self.polynomial = polynomial
        self.scale = np.full(2, self.h)
        self.degree = aggregate_degree
        self.size = (aggregate_degree + 1) ** 2 if polynomial else line.n**2
        self.transform = None

    def evaluate(self, p, order=1, cell=None):
        xy = (np.asarray(p) - self.origin) / self.scale
        factors = []
        for x in xy.T:
            if self.polynomial:
                count = self.degree + 1
                v = legvander(2 * x - 1, self.degree) * np.sqrt(
                    2 * np.arange(count) + 1
                )
                if order == 0:
                    factors.append((v,))
                    continue
                d = np.column_stack(
                    [
                        legval(2 * x - 1, legder(np.eye(count)[k]))
                        * 2
                        * np.sqrt(2 * k + 1)
                        for k in range(count)
                    ]
                )
                if order == 2:
                    dd = np.column_stack(
                        [
                            legval(2 * x - 1, legder(np.eye(count)[k], 2))
                            * 4
                            * np.sqrt(2 * k + 1)
                            for k in range(count)
                        ]
                    )
                    factors.append((v, d, dd))
                else:
                    factors.append((v, d))
            else:
                # Physical slice quadrature repeats one coordinate many times.
                # Reuse only exactly equal coordinates; no rounding/interpolation.
                unique_x, inverse = np.unique(x, return_inverse=True)
                if len(unique_x) < 0.8 * len(x):
                    factors.append(
                        [a[inverse] for a in self.line.values(unique_x, order)]
                    )
                else:
                    factors.append(self.line.values(x, order))

        def pair(a, b):
            return (a[:, :, None] * b[:, None, :]).reshape(len(p), -1)

        norm = np.sqrt(np.prod(self.scale))
        sx, sy = self.scale
        if order == 0:
            value = pair(factors[0][0], factors[1][0]) / norm
            return (value if self.transform is None else value @ self.transform,)
        x, dx = factors[0][:2]
        y, dy = factors[1][:2]
        values = (
            pair(x, y) / norm,
            pair(dx, y) / (norm * sx),
            pair(x, dy) / (norm * sy),
        )
        if order == 2:
            values += (
                (pair(factors[0][2], y) / sx**2 + pair(x, factors[1][2]) / sy**2)
                / norm,
            )
        return (
            values
            if self.transform is None
            else tuple(a @ self.transform for a in values)
        )

    def load(self, p, weighted_values):
        """Adjoint value evaluation without a point-by-tensor-DOF table."""
        factors = []
        for axis in range(2):
            coordinates = (np.asarray(p)[:, axis] - self.origin[axis]) / self.scale[
                axis
            ]
            unique_x, inverse = np.unique(coordinates, return_inverse=True)
            if self.polynomial:
                v = legvander(2 * unique_x - 1, self.degree) * np.sqrt(
                    2 * np.arange(self.degree + 1) + 1
                )
            else:
                v = self.line.values(unique_x, 0)[0]
            factors.append(v[inverse])
        rhs = (
            factors[0].T @ (np.asarray(weighted_values)[:, None] * factors[1])
        ).ravel() / np.sqrt(np.prod(self.scale))
        return rhs if self.transform is None else self.transform.T @ rhs

    def field(self, p, c, order=1):
        """Evaluate a field by tensor contraction, without point-by-DOF arrays."""
        xy = (np.asarray(p) - self.origin) / self.scale
        factors = []
        indices = []
        for u in xy.T:
            x, idx = np.unique(u, return_inverse=True)
            indices.append(idx)
            if self.polynomial:
                count = self.degree + 1
                unit = np.eye(count)
                norm = np.sqrt(2 * np.arange(count) + 1)
                v = legvander(2 * x - 1, self.degree) * norm
                d = np.column_stack(
                    [
                        legval(2 * x - 1, legder(unit[k])) * 2 * norm[k]
                        for k in range(count)
                    ]
                )
                f = [v, d]
                if order == 2:
                    f.append(
                        np.column_stack(
                            [
                                legval(2 * x - 1, legder(unit[k], 2)) * 4 * norm[k]
                                for k in range(count)
                            ]
                        )
                    )
            else:
                f = self.line.values(x, order)
            factors.append(f)
        c = c if self.transform is None else self.transform @ c
        cc = c.reshape(factors[0][0].shape[1], factors[1][0].shape[1]) / np.sqrt(
            np.prod(self.scale)
        )

        def pair(i, j):
            return np.sum(
                (factors[0][i] @ cc)[indices[0]] * factors[1][j][indices[1]], axis=1
            )

        sx, sy = self.scale
        values = (pair(0, 0), pair(1, 0) / sx, pair(0, 1) / sy)
        if order == 2:
            values += (pair(2, 0) / sx**2 + pair(0, 2) / sy**2,)
        return values


def owners(grid, threshold):
    if not 0 <= threshold <= 1:
        raise ValueError("Small-cut fraction must be in [0,1]")
    supported = grid.full | {c for c in grid.cut if grid.fraction[c] >= threshold}
    owner = {c: c for c in supported}
    pending = grid.active - supported
    neighbors = {c: [] for c in grid.active}
    for a, b, p, w, n in grid.faces:
        neighbors[a].append((b, w.sum()))
        neighbors[b].append((a, w.sum()))
    while pending:
        additions = {}
        for c in sorted(pending):
            candidates = [(nb, length) for nb, length in neighbors[c] if nb in owner]
            if candidates:
                nb, _ = max(
                    candidates,
                    key=lambda x: (
                        -sum(abs(owner[x[0]][j] - c[j]) for j in range(2)),
                        x[1],
                        grid.fraction[owner[x[0]]],
                    ),
                )
                additions[c] = owner[nb]
        if not additions:
            raise ValueError("Small-cut component has no supported neighboring cell")
        owner.update(additions)
        pending -= additions.keys()
    return owner
