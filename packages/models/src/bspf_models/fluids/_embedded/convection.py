"""Host convection backend for large sparse CPU runs."""

import numpy as np
from .geometry import ObstacleGrid
from .solver import Blocks


class Convection:
    def __init__(self, plan, boundary, order=48):
        self.plan, self.boundary = plan, boundary
        dense = [
            r for r in plan.grid.full if getattr(plan.vbasis[plan.owner[r]], "added", 0)
        ]
        grid = ObstacleGrid(
            plan.grid.cells,
            order,
            plan.grid.center,
            plan.grid.axes,
            edges=plan.grid.edges,
            full_order=max(20, plan.info["degree"] + 6),
            dense_full=dense,
        )
        self.volume, self.faces, self.inflow = [], [], []
        for cell, (p, w) in grid.volume.items():
            r = plan.owner[cell]
            v, dx, dy = plan.vbasis[r].evaluate(p)
            self.volume.append((plan.vi[r], w, v, dx, dy))
        for aa, bb, p, w, n in grid.faces:
            a, b = plan.owner[aa], plan.owner[bb]
            if a == b:
                continue
            va, vb = [plan.vbasis[r].evaluate(p, 0)[0] for r in (a, b)]
            self.faces.append((plan.vi[a], plan.vi[b], w, n, va, vb))
        # Use the same high-order Dirichlet boundary samples as the Stokes plan.
        for r, p, w, n, tag, v, dn, q in plan.curves:
            g = np.asarray(boundary(p, tag))
            speed = np.maximum(-np.sum(g * n, axis=1), 0)
            if np.any(speed):
                self.inflow.append((plan.vi[r], w * speed, v, g))

    def assemble(self, coefficients):
        """Oseen matrix for a frozen advecting velocity and boundary inflow load."""
        nv = self.plan.nv
        velocity = np.column_stack((coefficients[:nv], coefficients[nv : 2 * nv]))
        blocks = Blocks((nv, nv))
        load = np.zeros((nv, 2))
        for ids, w, v, dx, dy in self.volume:
            wind = v @ velocity[ids]
            directional = wind[:, 0, None] * dx + wind[:, 1, None] * dy
            blocks.add(ids, ids, v.T @ (w[:, None] * directional))
        for ia, ib, w, n, va, vb in self.faces:
            wind = 0.5 * (va @ velocity[ia] + vb @ velocity[ib])
            wn = np.sum(wind * n, axis=1)
            jump = np.column_stack((va, -vb))
            test = np.column_stack(
                (np.maximum(-wn, 0)[:, None] * va, -np.maximum(wn, 0)[:, None] * vb)
            )
            ids = np.r_[ia, ib]
            blocks.add(ids, ids, test.T @ (w[:, None] * jump))
        for ids, w, v, g in self.inflow:
            blocks.add(ids, ids, v.T @ (w[:, None] * v))
            load[ids] += v.T @ (w[:, None] * g)
        return blocks.matrix(), load.T.ravel()

    def residual(self, coefficients):
        """Nonlinear convection minus prescribed incoming convective data."""
        nv = self.plan.nv
        velocity = np.column_stack((coefficients[:nv], coefficients[nv : 2 * nv]))
        result = np.zeros((nv, 2))
        for ids, w, v, dx, dy in self.volume:
            local = velocity[ids]
            wind = v @ local
            adv = wind[:, 0, None] * (dx @ local) + wind[:, 1, None] * (dy @ local)
            result[ids] += v.T @ (w[:, None] * adv)
        for ia, ib, w, n, va, vb in self.faces:
            ua, ub = va @ velocity[ia], vb @ velocity[ib]
            wn = np.sum(0.5 * (ua + ub) * n, axis=1)
            jump = ua - ub
            result[ia] += va.T @ ((w * np.maximum(-wn, 0))[:, None] * jump)
            result[ib] -= vb.T @ ((w * np.maximum(wn, 0))[:, None] * jump)
        for ids, w, v, g in self.inflow:
            result[ids] += v.T @ (w[:, None] * (v @ velocity[ids] - g))
        return result.T.ravel()
