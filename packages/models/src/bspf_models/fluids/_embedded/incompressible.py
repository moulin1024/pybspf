"""Rank-revealed divergence and normal-trace constraints for fixed BSPF spaces.

The implicit backend eliminates divergence locally and applies global sparse QR
only to normal traces. The bounded dense backend remains a reference. Neither
puts a pressure penalty in continuity. Independent quadrature audits both.
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from .geometry import ObstacleGrid
from .runtime import sample_matrix


class ConstrainedSpace:
    def __init__(
        self,
        base,
        boundary,
        *,
        rank_tolerance=1e-8,
        tolerance=1e-9,
        max_velocity_dofs=2048,
        backend="dense_reference",
        wall_enforcement="nitsche",
    ):
        self.base = base
        self.nv = base.nv
        n = 2 * self.nv
        if backend not in ("implicit_qr", "dense_reference"):
            raise ValueError("Unknown constraint backend")
        self.backend = backend
        if wall_enforcement not in ("nitsche", "constraint"):
            raise ValueError("wall_enforcement must be 'nitsche' or 'constraint'")
        self.wall_enforcement = wall_enforcement
        if backend == "dense_reference" and n > max_velocity_dofs:
            raise ValueError(
                f"Structural setup needs a dense rank decomposition and is limited to {max_velocity_dofs} velocity DOFs (got {n}). A scalable compatible basis is required; no stabilized fallback was used."
            )
        self.tolerance = tolerance
        self.rank_tolerance = rank_tolerance
        rows, targets = [], []
        local_dimension = 0
        local_maps = {}
        self.polynomial_maps = {}

        def append(ids, table, weights, target=None):
            weighted = np.sqrt(weights)[:, None] * table
            u, sigma, vt = la.svd(weighted, full_matrices=False)
            local_tolerance = (
                min(rank_tolerance, 1e-12)
                if backend == "implicit_qr"
                else rank_tolerance
            )
            keep = sigma > local_tolerance * max(sigma[0], 1)
            scale = max(sigma[0], 1)
            u, sigma, vt = u[:, keep], sigma[keep], vt[keep]
            b = np.zeros(len(weights)) if target is None else np.sqrt(weights) * target
            projected = u.T @ b
            if la.norm(b - u @ projected) > tolerance * max(la.norm(b), 1):
                raise ValueError(
                    "Prescribed normal velocity is not representable in the structural trace space; increase the approximation space"
                )
            rows.append((ids, (sigma / scale)[:, None] * vt))
            targets.append(projected / scale)

        for root, (p, w) in base.volume.items():
            _, dx, dy = base.vbasis[root].evaluate(p)
            ids = np.r_[base.vi[root], base.nv + base.vi[root]]
            if backend == "implicit_qr":
                weighted = np.sqrt(w)[:, None] * np.column_stack((-dx, -dy))
                polynomial = self._local_polynomial_kernel(root, p, w)
                self.polynomial_maps[root] = polynomial
                orthogonal = la.qr(polynomial, mode="full")[0]
                complement = orthogonal[:, polynomial.shape[1]:]
                _, singular, vectors = la.svd(weighted @ complement, full_matrices=False)
                residual_null = vectors[singular <= 1e-12 * max(singular[0], 1)].T
                null = np.column_stack((polynomial, complement @ residual_null))
                if null.shape[1] == 0:
                    raise ValueError(
                        "Local velocity space has no resolvable divergence-free modes"
                    )
                local_maps[root] = (ids, null)
                local_dimension += null.shape[1]
            else:
                append(ids, np.column_stack((-dx, -dy)), w)
        for aa, bb, p, w, normal in base.grid.faces:
            a, b = base.owner[aa], base.owner[bb]
            if a == b:
                continue
            va = base.vbasis[a].evaluate(p, 0)[0]
            vb = base.vbasis[b].evaluate(p, 0)[0]
            ids = np.r_[
                base.vi[a], base.nv + base.vi[a], base.vi[b], base.nv + base.vi[b]
            ]
            table = np.column_stack(
                (
                    normal[:, 0, None] * va,
                    normal[:, 1, None] * va,
                    -normal[:, 0, None] * vb,
                    -normal[:, 1, None] * vb,
                )
            )
            append(ids, table, w)
        for root, p, w, normal, tag, v, *_ in base.curves:
            ids = np.r_[base.vi[root], base.nv + base.vi[root]]
            table = np.column_stack((normal[:, 0, None] * v, normal[:, 1, None] * v))
            append(ids, table, w, np.sum(boundary(p, tag) * normal, axis=1))
            if tag == "hole" and wall_enforcement == "constraint":
                tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
                table = np.column_stack((tangent[:, 0, None] * v, tangent[:, 1, None] * v))
                append(ids, table, w, np.sum(boundary(p, tag) * tangent, axis=1))
        raw = sample_matrix(rows, n)
        target = np.concatenate(targets)
        if backend == "implicit_qr":
            from .sparse_qr import LocalDivergenceFreeNullSpace

            rr, cc, vv = [], [], []
            offset = 0
            for ids, null in local_maps.values():
                width = null.shape[1]
                rr.append(np.repeat(ids, width))
                cc.append(np.tile(np.arange(offset, offset + width), len(ids)))
                vv.append(null.ravel())
                offset += width
            local_basis = sp.coo_matrix(
                (np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                shape=(n, offset),
            ).tocsr()
            self.projector = LocalDivergenceFreeNullSpace(
                local_basis, raw, rank_tolerance
            )
            self.constraint_matrix, self.constraint_target = raw, target
            self.raw_constraints, self.raw_target = raw, target
            self.np = self.projector.rank
            self.affine_velocity = self._polynomial_boundary_lift(raw, target, local_maps)
            self.velocity_reaction = True
        else:
            u, sigma, vt = la.svd(raw.toarray(), full_matrices=False)
            keep = sigma > rank_tolerance * max(sigma[0], 1)
            u, sigma, vt = u[:, keep], sigma[keep], vt[keep]
            if la.norm(target - u @ (u.T @ target)) > tolerance * max(
                la.norm(target), 1
            ):
                raise ValueError("Incompatible divergence/normal-flux constraints")
            self.constraint_matrix = sp.csr_matrix(vt)
            self.constraint_target = (u.T @ target) / sigma
            self.np = len(sigma)
        self.A = (
            None
            if backend == "implicit_qr"
            else sp.bmat(
                [
                    [sp.block_diag((base.K, base.K)), self.constraint_matrix.T],
                    [self.constraint_matrix, None],
                ],
                format="csc",
            )
        )
        self.info = dict(
            base.info,
            incompressibility="structural",
            wall_enforcement=wall_enforcement,
            pressure_stabilization_active=False,
            velocity_dofs=n,
            constraint_rank=self.np,
            unconstrained_velocity_dofs=n - self.np,
            pressure_dofs=None,
            multiplier_dofs=None if backend == "implicit_qr" else self.np,
            reaction_dofs=n if backend == "implicit_qr" else None,
            total_dofs=2 * n if backend == "implicit_qr" else self.A.shape[0],
            constraint_rank_tolerance=rank_tolerance,
            constraint_backend=backend,
            constraint_rank_history=self.projector.rank_history if backend == "implicit_qr" else None,
            effective_svd_tolerance=self.projector.effective_svd_tolerance if backend == "implicit_qr" else None,
            constraint_affine_gain=self.projector.affine_gain if backend == "implicit_qr" else None,
            polynomial_lift_residual=getattr(self, "polynomial_lift_residual", None),
            local_divergence_free_dofs=local_dimension
            if backend == "implicit_qr"
            else None,
            incompressibility_tolerance=tolerance,
        )
        if n - self.np == 0:
            raise ValueError(
                "The trial space has no free divergence-free velocity modes; increase/redesign the compatible basis"
            )
        self._prepare_audit(boundary)
        if backend == "dense_reference":
            self._prepare_pressure_recovery()
        else:
            self.pressure_recovery = None
            self.info["pressure_representation"] = (
                "separate sparse weak-gradient pressure recovery"
            )

    def _local_polynomial_kernel(self, root, points, weights):
        """Protect all divergence-free total-degree polynomials before SVD."""
        from .cut_basis import GradedPolynomial
        basis = self.base.vbasis[root]
        potential = GradedPolynomial(points, weights, self.base.info["degree"] + 1,
                                     basis.origin, basis.scale)
        _, dx, dy = potential.evaluate(points)
        values = basis.evaluate(points, 0)[0]
        weighted = np.sqrt(weights)[:, None]
        fitted = la.lstsq(weighted * values, weighted * np.column_stack((dy[:, 1:], -dx[:, 1:])),
                          cond=1e-13, lapack_driver="gelsd")[0]
        width = dx.shape[1] - 1
        vector = np.vstack((fitted[:, :width], fitted[:, width:]))
        return la.qr(vector, mode="economic")[0]

    def _polynomial_boundary_lift(self, raw, target, local_maps):
        """Use the guaranteed polynomial subspace for representable trace data.

        This is a boundary lifting only, not a replacement of the BSPF solution
        space or a streamfunction PDE solve. Curl polynomials provide an analytic
        local divergence-free dictionary; the full BSPF null space remains free.
        """
        from .sparse_qr import LocalDivergenceFreeNullSpace

        rr, cc, vv = [], [], []
        offset = 0
        for root, (points, weights) in self.base.volume.items():
            local = self.polynomial_maps[root]
            width = local.shape[1]
            ids = local_maps[root][0]
            rr.append(np.repeat(ids, width));cc.append(np.tile(np.arange(offset, offset + width), len(ids)))
            vv.append(local.ravel());offset += width
        poly = sp.coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                             shape=(2 * self.nv, offset)).tocsr()
        lifting = LocalDivergenceFreeNullSpace(poly, raw, 1e-9)
        try:
            candidate = lifting.affine(target)
            residual = target - raw @ candidate
            self.polynomial_lift_residual = float(la.norm(residual))
            if la.norm(residual) <= 1e-11 * max(la.norm(target), 1.):
                return candidate
            # Non-polynomial data still use the original full trace space.
            return candidate + self.projector.affine(residual)
        finally:
            lifting.close()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def rhs(self, boundary, forcing=None, traction=None):
        return np.r_[
            self.base.rhs(boundary, forcing, traction)[: 2 * self.nv],
            self.constraint_target,
        ]

    def _prepare_audit(self, boundary):
        s = self.base
        order = max(s.grid.order + 7, 2 * s.info["degree"] + 9)
        grid = ObstacleGrid(
            s.grid.cells,
            order,
            s.grid.center,
            s.grid.axes,
            edges=s.grid.edges,
            full_order=max(24, 2 * s.info["degree"] + 9),
            dense_full=[
                r for r in s.grid.full if getattr(s.vbasis[s.owner[r]], "added", 0)
            ],
        )

        class CompressedRecords(list):
            def append(self, item):
                ids, table = item
                r = la.qr(table, mode="r")[0][: min(table.shape)].copy()
                super().append((ids, r))

        div, grad, jumps, traces = (
            CompressedRecords(),
            CompressedRecords(),
            CompressedRecords(),
            [],
        )
        normal_data = []
        tangential_traces, tangential_data = [], []
        for cell, (p, w) in grid.volume.items():
            r = s.owner[cell]
            _, dx, dy = s.vbasis[r].evaluate(p)
            ids = np.r_[s.vi[r], s.nv + s.vi[r]]
            div.append((ids, np.sqrt(w)[:, None] * np.column_stack((dx, dy))))
            for derivative in (dx, dy):
                for offset in (0, s.nv):
                    grad.append((s.vi[r] + offset, np.sqrt(w)[:, None] * derivative))
        for aa, bb, p, w, normal in grid.faces:
            a, b = s.owner[aa], s.owner[bb]
            if a == b:
                continue
            va = s.vbasis[a].evaluate(p, 0)[0]
            vb = s.vbasis[b].evaluate(p, 0)[0]
            ids = np.r_[s.vi[a], s.nv + s.vi[a], s.vi[b], s.nv + s.vi[b]]
            jumps.append(
                (
                    ids,
                    np.sqrt(w)[:, None]
                    * np.column_stack(
                        (
                            normal[:, 0, None] * va,
                            normal[:, 1, None] * va,
                            -normal[:, 0, None] * vb,
                            -normal[:, 1, None] * vb,
                        )
                    ),
                )
            )
        for cell, segments in grid.boundary.items():
            r = s.owner[cell]
            for p, w, normal, tag in segments:
                if s.outflow and tag == "outer" and np.all(normal[:, 0] > 0.5):
                    continue
                v = s.vbasis[r].evaluate(p, 0)[0]
                ids = np.r_[s.vi[r], s.nv + s.vi[r]]
                traces.append(
                    (
                        ids,
                        np.sqrt(w)[:, None]
                        * np.column_stack(
                            (normal[:, 0, None] * v, normal[:, 1, None] * v)
                        ),
                    )
                )
                normal_data.append(
                    np.sqrt(w) * np.sum(boundary(p, tag) * normal, axis=1)
                )
                if tag == "hole":
                    tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
                    tangential_traces.append((ids, np.sqrt(w)[:, None] *
                        np.column_stack((tangent[:, 0, None] * v, tangent[:, 1, None] * v))))
                    tangential_data.append(np.sqrt(w) * np.sum(boundary(p, tag) * tangent, axis=1))

        def compressed(records):
            return sample_matrix(records, 2 * self.nv)

        self.audit_divergence = compressed(div)
        del div
        self.audit_gradient = compressed(grad)
        del grad
        self.audit_jumps = compressed(jumps)
        del jumps
        new_traces, new_data = [], []
        for (ids, a), b in zip(traces, normal_data):
            r = la.qr(np.column_stack((a, b)), mode="r")[0][: min(len(a), len(ids) + 1)]
            new_traces.append((ids, r[:, :-1]))
            new_data.append(r[:, -1])
        self.audit_boundary = sample_matrix(new_traces, 2 * self.nv)
        self.audit_boundary_data = np.concatenate(new_data)
        new_traces, new_data = [], []
        for (ids, a), b in zip(tangential_traces, tangential_data):
            r = la.qr(np.column_stack((a, b)), mode="r")[0][: min(len(a), len(ids) + 1)]
            new_traces.append((ids, r[:, :-1]))
            new_data.append(r[:, -1])
        self.audit_wall_tangent = sample_matrix(new_traces, 2 * self.nv)
        self.audit_wall_tangent_data = np.concatenate(new_data)
        self.info["constraint_audit_order"] = order

    def _prepare_pressure_recovery(self):
        # Generalized divergence/trace reactions need not themselves identify a
        # well-conditioned scalar pressure basis. Recover pressure separately
        # from the original weak pressure gradient, with jump smoothing. This
        # diagnostic recovery never enters the velocity constraint or evolution.
        b = self.base.B.toarray()
        c = (self.base.C * self.base.viscosity).toarray()
        eigen, vectors = la.eigh((c + c.T) / 2)
        root = np.sqrt(np.maximum(eigen, 0))[:, None] * vectors.T
        matrix = np.vstack((b.T, root))
        right = np.vstack(
            (self.constraint_matrix.T.toarray(), np.zeros((len(eigen), self.np)))
        )
        if not self.outflow:
            matrix = np.vstack((matrix, self.base.mean[None, :]))
            right = np.vstack((right, np.zeros((1, self.np))))
        self.pressure_recovery = la.lstsq(
            matrix, right, cond=1e-12, lapack_driver="gelsd"
        )[0]
        self.info["pressure_representation"] = (
            "separate regularized weak-gradient recovery from constraint reactions"
        )

    def recover_pressure(self, coefficients):
        if self.backend == "implicit_qr":
            from scipy.sparse.linalg import splu

            if self.pressure_recovery is None:
                g = (
                    self.base.B @ self.base.B.T + self.base.viscosity * self.base.C
                ).tocsc()
                if not self.outflow:
                    mean = sp.csc_matrix(self.base.mean[:, None])
                    g = sp.bmat([[g, mean], [mean.T, None]], format="csc")
                self._pressure_matrix = g
                self.pressure_recovery = splu(g, permc_spec="MMD_AT_PLUS_A")
            reaction = coefficients[2 * self.nv :]
            rhs = self.base.B @ reaction
            if not self.outflow:
                rhs = np.r_[rhs, 0.0]
            pressure = self.pressure_recovery.solve(rhs)
            for _ in range(2):
                defect = rhs - self._pressure_matrix @ pressure
                if la.norm(defect) <= 1e-11 * max(la.norm(rhs), 1):
                    break
                pressure += self.pressure_recovery.solve(defect)
            if la.norm(rhs - self._pressure_matrix @ pressure) > 1e-8 * max(
                la.norm(rhs), 1
            ):
                raise RuntimeError(
                    "Sparse pressure recovery residual exceeded tolerance"
                )
            pressure = pressure[: self.base.np]
        else:
            pressure = self.pressure_recovery @ coefficients[2 * self.nv :]
        if not self.outflow:
            # Physical L2 basis includes the constant, represented by mean.
            pressure -= (
                self.base.mean
                * (self.base.mean @ pressure)
                / (self.base.mean @ self.base.mean)
            )
        return pressure

    def evaluate(self, points, coefficients):
        old = np.zeros(self.base.A.shape[0])
        old[: 2 * self.nv] = coefficients[: 2 * self.nv]
        pressure = self.recover_pressure(coefficients)
        old[2 * self.nv : 2 * self.nv + self.base.np] = pressure
        return self.base.evaluate(points, old)
