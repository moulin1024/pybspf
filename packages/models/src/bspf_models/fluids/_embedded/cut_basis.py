"""Physical-domain graded polynomial recurrence plus observable BSPF residuals.

Only cut-root residual modes are filtered. Complete total-degree P_p is kept
explicitly; full-cell BSPF spaces are unchanged. No PDE solution is consulted.
"""

import numpy as np
import scipy.linalg as la


class GradedPolynomial:
    def __init__(self, p, w, degree, origin, scale):
        self.origin = origin
        self.scale = scale
        self.area = w.sum()
        powers = [(i, d - i) for d in range(degree + 1) for i in range(d + 1)]
        ids = {a: i for i, a in enumerate(powers)}
        self.size = len(powers)
        z = 2 * (p - origin) / scale - 1
        q = np.empty((len(p), self.size))
        q[:, 0] = np.sqrt(w / self.area)
        self.recurrence = []
        for k, (i, j) in enumerate(powers[1:], 1):
            axis = 0 if i else 1
            parent = ids[i - 1, j] if i else ids[i, j - 1]
            v = z[:, axis] * q[:, parent]
            h = np.zeros(k)
            for _ in range(2):
                delta = q[:, :k].T @ v
                v -= q[:, :k] @ delta
                h += delta
            norm = la.norm(v)
            if norm < 1e-12:
                raise ValueError("Physical polynomial recurrence lost rank")
            q[:, k] = v / norm
            self.recurrence.append((axis, parent, h, norm))
        self.q = q

    def evaluate(self, p, order=1, cell=None):
        if order not in (0, 1):
            raise ValueError("Cut basis supports values and first derivatives")
        p = np.asarray(p)
        z = 2 * (p - self.origin) / self.scale - 1
        dz = 2 / self.scale
        v = np.zeros((len(p), self.size))
        x = np.zeros_like(v)
        y = np.zeros_like(v)
        v[:, 0] = 1 / np.sqrt(self.area)
        for k, (axis, parent, h, norm) in enumerate(self.recurrence, 1):
            t = z[:, axis]
            v[:, k] = (t * v[:, parent] - v[:, :k] @ h) / norm
            if order:
                x[:, k] = (
                    t * x[:, parent]
                    + (dz[0] * v[:, parent] if axis == 0 else 0)
                    - x[:, :k] @ h
                ) / norm
                y[:, k] = (
                    t * y[:, parent]
                    + (dz[1] * v[:, parent] if axis == 1 else 0)
                    - y[:, :k] @ h
                ) / norm
        return (v,) if order == 0 else (v, x, y)


class ObservableBasis:
    def __init__(self, base, p, w, degree, tolerance, allow_enrichment=False):
        if not 0 < tolerance < 1:
            raise ValueError("Cut tolerance must be between zero and one")
        self.base = base
        self.origin = base.origin
        self.scale = base.scale
        self.transform = None
        self.poly = GradedPolynomial(p, w, degree, self.origin, self.scale)
        raw = np.sqrt(w[:, None]) * base.evaluate(p, 0)[0]
        q = self.poly.q
        self.projection = q.T @ raw
        residual = raw - q @ self.projection
        rr = la.qr(residual, mode="economic")[1]
        _, s, vh = la.svd(rr, full_matrices=False)
        keep = s > tolerance * max(s[0], 1)
        self.map = vh[keep].T / s[keep]
        self.size = self.poly.size + int(keep.sum())
        self.space_growth = max(self.size - base.size, 0)
        self.discarded = max(base.size - self.size, 0)
        if self.space_growth and not allow_enrichment:
            raise RuntimeError("Inconsistent residual numerical rank")
        del self.poly.q

    def evaluate(self, p, order=1, cell=None):
        if order not in (0, 1):
            raise ValueError("Cut basis supports values and first derivatives")
        pv = self.poly.evaluate(p, order)
        bv = self.base.evaluate(p, order)
        out = tuple(
            np.column_stack((a, (b - a @ self.projection) @ self.map))
            for a, b in zip(pv, bv)
        )
        return out if self.transform is None else tuple(a @ self.transform for a in out)
