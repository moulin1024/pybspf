"""Exact ellipse cut out of a Cartesian rectangle; physical Gaussian quadrature."""

import numpy as np
from .basis import gauss, unique, Cuts


class ObstacleGrid:
    def __init__(
        self,
        cells=6,
        order=24,
        center=(0.13, -0.07),
        axes=(0.27, 0.19),
        edges=None,
        full_order=None,
        dense_full=(),
    ):
        self.cells = cells
        self.order = order
        self.center = np.array(center)
        self.axes = np.array(axes)
        if cells < 2 or order < 4 or np.any(self.axes <= 0):
            raise ValueError("Invalid grid or ellipse parameters")
        self.edges = (
            [np.linspace(-1, 1, cells + 1)] * 2
            if edges is None
            else [np.asarray(e) for e in edges]
        )
        if any(
            len(e) != cells + 1 or not np.all(np.isfinite(e)) or np.any(np.diff(e) <= 0)
            for e in self.edges
        ):
            raise ValueError("Invalid Cartesian edge arrays")
        self.lower = np.array([e[0] for e in self.edges])
        self.upper = np.array([e[-1] for e in self.edges])
        self.h = float(min(self.upper - self.lower)) / cells
        if np.any(self.center - self.axes <= self.lower) or np.any(
            self.center + self.axes >= self.upper
        ):
            raise ValueError("Ellipse must be strictly inside the rectangle")
        self.volume = {}
        self.boundary = {}
        self.faces = []
        q, w = gauss(order)
        full_q, full_w = gauss(order if full_order is None else full_order)
        dense_full = set(dense_full)
        cx, cy = self.center
        a, b = self.axes
        for i in range(cells):
            for j in range(cells):
                x0, x1 = self.edges[0][i : i + 2]
                y0, y1 = self.edges[1][j : j + 2]
                nearest = np.clip(self.center, [x0, y0], [x1, y1])
                if np.sum(((nearest - self.center) / self.axes) ** 2) >= 1:
                    fq, fw = (q, w) if (i, j) in dense_full else (full_q, full_w)
                    xx, yy = np.meshgrid(
                        x0 + (x1 - x0) * fq, y0 + (y1 - y0) * fq, indexing="ij"
                    )
                    self.volume[i, j] = (
                        np.column_stack((xx.ravel(), yy.ravel())),
                        (x1 - x0) * (y1 - y0) * np.outer(fw, fw).ravel(),
                    )
                    continue
                events = [x0, x1, cx - a, cx + a]
                for y in (y0, y1):
                    t = 1 - ((y - cy) / b) ** 2
                    if t > 0:
                        events.extend([cx - a * np.sqrt(t), cx + a * np.sqrt(t)])
                events = unique(np.clip(events, x0, x1))
                pp = []
                ww = []
                for lo, hi in zip(events[:-1], events[1:]):
                    # Quadratic endpoint map resolves square-root slice endpoints.
                    xs = lo + (hi - lo) * np.sin(np.pi * q / 2) ** 2
                    ws = (hi - lo) * np.pi / 2 * np.sin(np.pi * q) * w
                    for x, wx in zip(xs, ws):
                        for yl, yh in self.intervals(0, x, y0, y1):
                            pp.append(
                                np.column_stack(
                                    (np.full_like(q, x), yl + (yh - yl) * q)
                                )
                            )
                            ww.append(wx * (yh - yl) * w)
                if pp:
                    self.volume[i, j] = (np.vstack(pp), np.concatenate(ww))
        self.active = set(self.volume)
        # Split ellipse arcs at every Cartesian crossing and all extrema.
        ts = list(np.arange(5) * np.pi / 2)
        for x in self.edges[0]:
            r = (x - cx) / a
            if abs(r) < 1:
                t = np.arccos(r)
                ts.extend([t, 2 * np.pi - t])
        for y in self.edges[1]:
            r = (y - cy) / b
            if abs(r) < 1:
                t = np.arcsin(r)
                ts.extend([t % (2 * np.pi), (np.pi - t) % (2 * np.pi)])
        ts = unique(ts)
        for lo, hi in zip(ts[:-1], ts[1:]):
            t = lo + (hi - lo) * q
            p, n, speed = self.ellipse(t)
            key = self.locate(p[len(p) // 2 : len(p) // 2 + 1])[0]
            self.boundary.setdefault(tuple(key), []).append(
                (p, (hi - lo) * w * speed, n, "hole")
            )
        self.cut = set(self.boundary)
        self.full = self.active - self.cut
        self.fraction = {
            k: float(
                w.sum()
                / (
                    (self.edges[0][k[0] + 1] - self.edges[0][k[0]])
                    * (self.edges[1][k[1] + 1] - self.edges[1][k[1]])
                )
            )
            for k, (p, w) in self.volume.items()
        }
        # Outer boundary; does not make a cell geometrically cut.
        for axis in range(2):
            other = 1 - axis
            for side in (0, cells):
                for j in range(cells):
                    p = np.empty((order, 2))
                    p[:, axis] = self.edges[axis][side]
                    width = self.edges[other][j + 1] - self.edges[other][j]
                    p[:, other] = self.edges[other][j] + width * q
                    n = np.zeros_like(p)
                    n[:, axis] = -1 if side == 0 else 1
                    key = (
                        (0 if side == 0 else cells - 1, j)
                        if axis == 0
                        else (j, 0 if side == 0 else cells - 1)
                    )
                    self.boundary.setdefault(key, []).append((p, width * w, n, "outer"))
            for i in range(1, cells):
                value = self.edges[axis][i]
                for j in range(cells):
                    for lo, hi in self.intervals(
                        axis, value, *self.edges[other][j : j + 2]
                    ):
                        p = np.empty((order, 2))
                        p[:, axis] = value
                        p[:, other] = lo + (hi - lo) * q
                        n = np.zeros_like(p)
                        n[:, axis] = 1
                        left = (i - 1, j) if axis == 0 else (j, i - 1)
                        right = (i, j) if axis == 0 else (j, i)
                        if left not in self.active or right not in self.active:
                            raise ValueError("missing fluid cell")
                        self.faces.append((left, right, p, (hi - lo) * w, n))
        if not self.full:
            raise ValueError("Refine grid to obtain full support cells")

    def intervals(self, axis, value, lo, hi):
        other = 1 - axis
        t = 1 - ((value - self.center[axis]) / self.axes[axis]) ** 2
        if t <= 0:
            return [(lo, hi)]
        d = self.axes[other] * np.sqrt(t)
        a = self.center[other] - d
        b = self.center[other] + d
        return [(x, y) for x, y in [(lo, min(hi, a)), (max(lo, b), hi)] if y > x]

    def ellipse(self, t):
        a, b = self.axes
        p = self.center + np.column_stack((a * np.cos(t), b * np.sin(t)))
        normal = -np.column_stack((np.cos(t) / a, np.sin(t) / b))
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        return p, normal, np.hypot(a * np.sin(t), b * np.cos(t))

    def locate(self, p):
        return np.clip(
            np.column_stack(
                [
                    np.searchsorted(self.edges[a], np.asarray(p)[:, a], side="right")
                    - 1
                    for a in range(2)
                ]
            ),
            0,
            self.cells - 1,
        )

    aggregate = Cuts.aggregate
