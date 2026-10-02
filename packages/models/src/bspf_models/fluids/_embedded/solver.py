"""Aggregated BSPF SIPDG Stokes, pressure-jump stabilized mixed formulation.

Constant viscosity (default 1), weak Dirichlet data and optional natural right
outflow. Exact ellipse geometry; complete BSPF space on each aggregate, no
sampled exterior PDE. A natural outlet fixes the pressure level without a gauge.
Pressure defaults to discontinuous Q_(degree-1), with consistent jump penalty.
Optional pressure degrees, nonuniform grids and polynomial-preserving cut
spaces support the independent high-order experiments in high_order.py.
This is a research discretization, without a proved uniform inf-sup bound.
"""

from time import perf_counter
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from .basis import Basis
from .stable_line import StableLine
from .geometry import ObstacleGrid


class Blocks:
    def __init__(self, shape):
        self.shape = shape
        self.parts = []

    def add(self, i, j, a):
        self.parts.append(
            (np.repeat(i, len(j)), np.tile(j, len(i)), np.asarray(a).ravel())
        )

    def matrix(self):
        i, j, a = map(np.concatenate, zip(*self.parts))
        m = sp.coo_matrix((a, (i, j)), shape=self.shape).tocsc()
        m.eliminate_zeros()
        return m


class StokesPlan:
    def __init__(
        self,
        cells=6,
        degree=5,
        modes=1,
        order=24,
        pressure_penalty=0.05,
        center=(0.13, -0.07),
        axes=(0.27, 0.19),
        cut_fraction=None,
        pressure_degree=None,
        factorize=True,
        ordering="COLAMD",
        aggregate_frame=False,
        edges=None,
        cut_tolerance=None,
        divergence_form="divergence",
        cut_pressure_degree=None,
        pivot_threshold=1.0,
        linear_solver="direct",
        prior_levels=0,
        prior_tolerance=1e-5,
        prior_obstacle=True,
        prior_corners=True,
        prior_modulation=0,
        prior_surface_samples=24,
        full_order=None,
        pressure_gradient_penalty=0.0,
        cut_velocity_tolerance=None,
        prior_velocity_tolerance=None,
        cut_velocity_degree=None,
        prior_velocity_halo=0.0,
        viscosity=1.0,
        outflow=False,
        wall_penalty_factor=1.0,
    ):
        start = perf_counter()
        if not np.isfinite(viscosity) or viscosity <= 0:
            raise ValueError("Viscosity must be finite and positive")
        self.viscosity = viscosity
        self.outflow = outflow
        if not np.isfinite(wall_penalty_factor) or wall_penalty_factor < 1:
            raise ValueError("wall_penalty_factor must be finite and >= 1")
        self.wall_penalty_factor = float(wall_penalty_factor)
        if degree < 2 or modes < 1 or pressure_penalty <= 0:
            raise ValueError("Invalid approximation parameters")
        if not np.isfinite(pressure_gradient_penalty) or pressure_gradient_penalty < 0:
            raise ValueError("Pressure gradient penalty must be finite and nonnegative")
        for tolerance in (cut_velocity_tolerance, prior_velocity_tolerance):
            if tolerance is not None and not 0 < tolerance < 1:
                raise ValueError("Velocity residual tolerances must lie in (0, 1)")
        if cut_velocity_degree is not None and cut_velocity_degree < degree:
            raise ValueError(
                "Cut velocity degree must preserve the original polynomials"
            )
        if divergence_form not in ("divergence", "gradient"):
            raise ValueError("Unknown divergence form")
        if not np.isfinite(prior_velocity_halo) or prior_velocity_halo < 0:
            raise ValueError("Velocity enrichment halo must be finite and nonnegative")
        if linear_solver not in ("direct", "block"):
            raise ValueError("Unknown linear solver")
        if prior_surface_samples < 8:
            raise ValueError("At least eight geometric surface samples are required")
        if (
            prior_levels < 0
            or not 0 < prior_tolerance < 1
            or prior_modulation not in (0, 1, 2)
        ):
            raise ValueError("Invalid geometry-prior parameters")
        dense_full = (
            {(i, j) for i in (0, cells - 1) for j in (0, cells - 1)}
            if prior_levels
            else ()
        )
        if prior_levels and prior_velocity_halo and prior_obstacle:
            from .prior_basis import obstacle_near_box

            dense_full = set(dense_full)
            ee = [np.linspace(-1, 1, cells + 1)] * 2 if edges is None else edges
            for i in range(cells):
                for j in range(cells):
                    lo = np.array([ee[0][i], ee[1][j]])
                    scale = np.array([ee[0][i + 1], ee[1][j + 1]]) - lo
                    if obstacle_near_box(lo, scale, center, axes, prior_velocity_halo):
                        dense_full.add((i, j))
        self.grid = g = ObstacleGrid(
            cells,
            order,
            center,
            axes,
            edges=edges,
            full_order=full_order,
            dense_full=dense_full,
        )
        if cut_fraction is None:
            self.owner = g.aggregate()
        else:
            from .basis import owners

            self.owner = owners(g, cut_fraction)
        self.roots = sorted(set(self.owner.values()))
        self.line = StableLine(modes=modes, degree=degree, quadrature=max(24, order))
        self.vbasis = {}
        self.pbasis = {}
        self.vi = {}
        self.pi = {}
        self.volume = {}
        self.curves = []
        self.outflow_curves = []
        self.faces = []
        pressure_degree = degree - 1 if pressure_degree is None else pressure_degree
        if pressure_degree < 0 or (
            cut_pressure_degree is not None and cut_pressure_degree < 0
        ):
            raise ValueError("Pressure degree must be nonnegative")
        self.basis_conditions = []
        nv = np_ = 0
        for root in self.roots:
            vb = Basis(root, g, self.line, False)
            local_pd = (
                pressure_degree
                if cut_pressure_degree is None or root not in g.cut
                else cut_pressure_degree
            )
            pb = Basis(root, g, self.line, True, aggregate_degree=local_pd)
            for basis in (vb, pb):
                basis.origin = np.array([g.edges[a][root[a]] for a in range(2)])
                basis.scale = np.array(
                    [g.edges[a][root[a] + 1] - g.edges[a][root[a]] for a in range(2)]
                )
            if aggregate_frame:
                members = np.array([k for k, r in self.owner.items() if r == root])
                lo, hi = members.min(axis=0), members.max(axis=0) + 1
                origin = np.array([g.edges[a][lo[a]] for a in range(2)])
                scale = np.array([g.edges[a][hi[a]] for a in range(2)]) - origin
                for basis in (vb, pb):
                    basis.origin = origin
                    basis.scale = scale
            pp = []
            ww = []
            for cell in sorted(k for k, r in self.owner.items() if r == root):
                p, w = g.volume[cell]
                pp.append(p)
                ww.append(w)
            p = np.vstack(pp)
            w = np.concatenate(ww)
            if cut_tolerance is not None and root in g.cut:
                from .cut_basis import ObservableBasis

                vb = ObservableBasis(
                    vb,
                    p,
                    w,
                    degree if cut_velocity_degree is None else cut_velocity_degree,
                    cut_tolerance
                    if cut_velocity_tolerance is None
                    else cut_velocity_tolerance,
                    allow_enrichment=cut_velocity_degree is not None
                    and cut_velocity_degree > degree,
                )
                pb = ObservableBasis(pb, p, w, local_pd, cut_tolerance)
            # Physical-mass QR is an invertible scaling of the selected space.
            for basis in (vb, pb):
                v = basis.evaluate(p, 0)[0]
                _, r = la.qr(np.sqrt(w[:, None]) * v, mode="economic")
                self.basis_conditions.append(float(np.linalg.cond(r)))
                basis.transform = la.solve_triangular(r, np.eye(basis.size))
            if prior_levels:
                from .prior_basis import geometry_dictionary, PriorBasis

                dictionary = geometry_dictionary(
                    vb.origin,
                    vb.scale,
                    g,
                    prior_levels,
                    prior_corners,
                    prior_obstacle,
                    prior_modulation,
                    prior_surface_samples,
                )
                velocity_dictionary = dictionary
                if prior_velocity_halo:
                    velocity_dictionary = geometry_dictionary(
                        vb.origin,
                        vb.scale,
                        g,
                        prior_levels,
                        prior_corners,
                        prior_obstacle,
                        prior_modulation,
                        prior_surface_samples,
                        obstacle_halo=prior_velocity_halo,
                    )
                if dictionary is not None or velocity_dictionary is not None:
                    enriched = []
                    for component, basis in enumerate((vb, pb)):
                        local_dictionary = (
                            velocity_dictionary if component == 0 else dictionary
                        )
                        if local_dictionary is None:
                            enriched.append(basis)
                            continue
                        tolerance = (
                            prior_velocity_tolerance
                            if component == 0 and prior_velocity_tolerance is not None
                            else prior_tolerance
                        )
                        basis = PriorBasis(basis, local_dictionary, p, w, tolerance)
                        v = basis.evaluate(p, 0)[0]
                        _, rr = la.qr(np.sqrt(w[:, None]) * v, mode="economic")
                        self.basis_conditions.append(float(np.linalg.cond(rr)))
                        basis.transform = la.solve_triangular(rr, np.eye(basis.size))
                        enriched.append(basis)
                    vb, pb = enriched
            self.vbasis[root] = vb
            self.pbasis[root] = pb
            self.vi[root] = np.arange(nv, nv + vb.size)
            nv += vb.size
            self.pi[root] = np.arange(np_, np_ + pb.size)
            np_ += pb.size
            self.volume[root] = (p, w)
        self.nv = nv
        self.np = np_
        self.mean = np.zeros(np_)
        k = Blocks((nv, nv))
        bx = Blocks((np_, nv))
        by = Blocks((np_, nv))
        c = Blocks((np_, np_))
        local = {}
        normal = {}
        self.mass = Blocks((nv, nv))
        for r, (p, w) in self.volume.items():
            v, x, y = self.vbasis[r].evaluate(p)
            q = self.pbasis[r].evaluate(p, 0)[0]
            iv = self.vi[r]
            ip = self.pi[r]
            local[r] = x.T @ (w[:, None] * x) + y.T @ (w[:, None] * y)
            normal[r] = np.zeros_like(local[r])
            k.add(iv, iv, local[r])
            self.mass.add(iv, iv, v.T @ (w[:, None] * v))
            self.mean[ip] = w @ q
            if divergence_form == "divergence":
                bx.add(ip, iv, -q.T @ (w[:, None] * x))
                by.add(ip, iv, -q.T @ (w[:, None] * y))
            else:
                _, qx, qy = self.pbasis[r].evaluate(p)
                bx.add(ip, iv, qx.T @ (w[:, None] * v))
                by.add(ip, iv, qy.T @ (w[:, None] * v))
        for cell, segments in g.boundary.items():
            r = self.owner[cell]
            for p, w, n, tag in segments:
                v, x, y = self.vbasis[r].evaluate(p)
                q = self.pbasis[r].evaluate(p, 0)[0]
                dn = x * n[:, 0, None] + y * n[:, 1, None]
                if outflow and tag == "outer" and np.all(n[:, 0] > 0.5):
                    if divergence_form == "gradient":
                        bx.add(
                            self.pi[r], self.vi[r], -q.T @ ((w * n[:, 0])[:, None] * v)
                        )
                        by.add(
                            self.pi[r], self.vi[r], -q.T @ ((w * n[:, 1])[:, None] * v)
                        )
                    self.outflow_curves.append((r, p, w, n, tag, v, dn, q))
                    continue
                normal[r] += dn.T @ (w[:, None] * dn)
                if divergence_form == "divergence":
                    bx.add(self.pi[r], self.vi[r], q.T @ ((w * n[:, 0])[:, None] * v))
                    by.add(self.pi[r], self.vi[r], q.T @ ((w * n[:, 1])[:, None] * v))
                self.curves.append((r, p, w, n, tag, v, dn, q))
        for aa, bb, p, w, n in g.faces:
            a, b = self.owner[aa], self.owner[bb]
            if a == b:
                continue
            va, xa, ya = self.vbasis[a].evaluate(p)
            vb, xb, yb = self.vbasis[b].evaluate(p)
            da = xa * n[:, 0, None] + ya * n[:, 1, None]
            db = xb * n[:, 0, None] + yb * n[:, 1, None]
            qa = self.pbasis[a].evaluate(p, 0)[0]
            qb = self.pbasis[b].evaluate(p, 0)[0]
            normal[a] += da.T @ (w[:, None] * da)
            normal[b] += db.T @ (w[:, None] * db)
            jump = np.column_stack((va, -vb))
            avgq = 0.5 * np.column_stack((qa, qb))
            jq = np.column_stack((qa, -qb))
            iv = np.r_[self.vi[a], self.vi[b]]
            ip = np.r_[self.pi[a], self.pi[b]]
            if divergence_form == "divergence":
                bx.add(ip, iv, avgq.T @ ((w * n[:, 0])[:, None] * jump))
                by.add(ip, iv, avgq.T @ ((w * n[:, 1])[:, None] * jump))
            else:
                avgv = 0.5 * np.column_stack((va, vb))
                bx.add(ip, iv, -jq.T @ ((w * n[:, 0])[:, None] * avgv))
                by.add(ip, iv, -jq.T @ ((w * n[:, 1])[:, None] * avgv))
            face_axis = int(abs(n[0, 1]) > 0.5)
            face_h = min(
                self.vbasis[a].scale[face_axis], self.vbasis[b].scale[face_axis]
            )
            c.add(ip, ip, pressure_penalty * face_h * jq.T @ (w[:, None] * jq))
            if pressure_gradient_penalty:
                # Consistent for smooth pressure: penalize only interior jumps
                # of its gradient, never the physical pressure gradient itself.
                ga = self.pbasis[a].evaluate(p)[1:]
                gb = self.pbasis[b].evaluate(p)[1:]
                for qa_grad, qb_grad in zip(ga, gb):
                    jg = np.column_stack((qa_grad, -qb_grad))
                    c.add(
                        ip,
                        ip,
                        pressure_gradient_penalty
                        * face_h**3
                        * jg.T
                        @ (w[:, None] * jg),
                    )
            self.faces.append((a, b, w, jump, 0.5 * np.column_stack((da, db)), iv))
        self.penalty = {}
        for r in self.roots:
            kk = local[r][1:, 1:]
            nn = normal[r][1:, 1:]
            bound = la.eigh(
                (nn + nn.T) / 2,
                (kk + kk.T) / 2,
                eigvals_only=True,
                subset_by_index=[len(kk) - 1] * 2,
            )[0]
            self.penalty[r] = 4 * bound
        for a, b, w, jump, dn, iv in self.faces:
            k.add(
                iv,
                iv,
                -jump.T @ (w[:, None] * dn)
                - dn.T @ (w[:, None] * jump)
                + max(self.penalty[a], self.penalty[b]) * jump.T @ (w[:, None] * jump),
            )
        for r, p, w, n, tag, v, dn, q in self.curves:
            k.add(
                self.vi[r],
                self.vi[r],
                -v.T @ (w[:, None] * dn)
                - dn.T @ (w[:, None] * v)
                + self.boundary_penalty(r, tag) * v.T @ (w[:, None] * v),
            )
        self.K = viscosity * k.matrix()
        self.B = sp.hstack((bx.matrix(), by.matrix()), format="csc")
        self.C = c.matrix() / viscosity
        m = sp.csc_matrix(self.mean[:, None])
        zg = sp.csc_matrix((2 * nv, 1))
        self.A = sp.bmat(
            [
                [sp.block_diag((self.K, self.K)), self.B.T, zg],
                [self.B, -self.C, m],
                [zg.T, m.T, None],
            ],
            format="csc",
        )
        if outflow:
            self.A = sp.bmat(
                [[sp.block_diag((self.K, self.K)), self.B.T], [self.B, -self.C]],
                format="csc",
            )
        self.assembly_seconds = perf_counter() - start
        start = perf_counter()
        self.factor = None
        if factorize:
            if linear_solver == "block":
                from .block_solver import BlockStokesFactor

                self.factor = BlockStokesFactor(
                    self.A, self.K, self.B, self.C, self.mean, viscosity=viscosity
                )
            else:
                self.factor = splu(
                    self.A, permc_spec=ordering, diag_pivot_thresh=pivot_threshold
                )
        self.factor_seconds = perf_counter() - start
        self.info = dict(
            cells=cells,
            viscosity=viscosity,
            outflow=outflow,
            pressure_gauge="natural outlet" if outflow else "zero mean",
            degree=degree,
            modes=modes,
            quadrature=order,
            full_quadrature=order if full_order is None else full_order,
            pressure_penalty=pressure_penalty,
            wall_penalty_factor=self.wall_penalty_factor,
            pressure_gradient_penalty=pressure_gradient_penalty,
            cut_velocity_tolerance=cut_velocity_tolerance,
            prior_velocity_tolerance=prior_velocity_tolerance,
            cut_velocity_degree=cut_velocity_degree,
            prior_velocity_halo=prior_velocity_halo,
            pressure_degree=pressure_degree,
            divergence_form=divergence_form,
            cut_pressure_degree=cut_pressure_degree,
            cut_tolerance=cut_tolerance,
            cut_filtered_velocity_modes=sum(
                getattr(b, "discarded", 0) for b in self.vbasis.values()
            ),
            cut_fraction=cut_fraction,
            ordering=ordering,
            pivot_threshold=pivot_threshold,
            linear_solver=linear_solver,
            prior_levels=prior_levels,
            prior_modulation=prior_modulation,
            prior_surface_samples=prior_surface_samples,
            prior_design=("distance-v5" if prior_velocity_halo else "distance-v4")
            if prior_levels
            else None,
            prior_tolerance=prior_tolerance,
            prior_obstacle=prior_obstacle,
            prior_corners=prior_corners,
            prior_velocity_modes=sum(
                getattr(b, "added", 0) for b in self.vbasis.values()
            ),
            prior_pressure_modes=sum(
                getattr(b, "added", 0) for b in self.pbasis.values()
            ),
            block_ordering="MMD_AT_PLUS_A" if linear_solver == "block" else None,
            matrix_nnz=self.A.nnz,
            factor_nnz=(
                self.factor.factor_nnz
                if linear_solver == "block"
                else self.factor.L.nnz + self.factor.U.nnz
            )
            if factorize
            else None,
            aggregate_frame=aggregate_frame,
            edges=[e.tolist() for e in g.edges],
            max_basis_condition=max(self.basis_conditions),
            velocity_dofs=2 * nv,
            pressure_dofs=np_,
            total_dofs=self.A.shape[0],
            aggregates=len(self.roots),
            cut_cells=len(g.cut),
            min_fraction=min(g.fraction.values()),
            assembly_seconds=self.assembly_seconds,
            factor_seconds=self.factor_seconds,
        )

    def boundary_penalty(self, root, tag):
        """Strengthen obstacle Dirichlet data without changing interior SIPDG.

        The fallback permits old cached spatial plans to retain their original
        boundary operator and matching load.
        """
        factor = getattr(self, "wall_penalty_factor", 1.) if tag == "hole" else 1.
        return factor * self.penalty[root]

    def rhs(self, boundary, forcing=None, traction=None):
        f = np.zeros(self.A.shape[0])
        nv = self.nv
        flux = 0.0
        flux_scale = 0.0
        if forcing is not None:
            for r, (p, w) in self.volume.items():
                v = self.vbasis[r].evaluate(p, 0)[0]
                force = forcing(p)
                for d in range(2):
                    f[self.vi[r] + d * nv] += v.T @ (w * force[:, d])
        for r, p, w, n, tag, v, dn, q in self.curves:
            val = np.asarray(boundary(p, tag))
            lift = self.viscosity * (-dn + self.boundary_penalty(r, tag) * v)
            if val.shape != p.shape or not np.all(np.isfinite(val)):
                raise ValueError("Expected finite two-component boundary velocities")
            flux += w @ np.sum(val * n, axis=1)
            flux_scale += w @ np.linalg.norm(val, axis=1)
            for d in range(2):
                f[self.vi[r] + d * nv] += lift.T @ (w * val[:, d])
            f[2 * nv + self.pi[r]] += q.T @ (w * np.sum(val * n, axis=1))
        if traction is not None:
            for r, p, w, n, tag, v, dn, q in self.outflow_curves:
                val = traction(p, n)
                for d in range(2):
                    f[self.vi[r] + d * nv] += v.T @ (w * val[:, d])
        if not self.outflow and abs(flux) > 1e-10 * max(flux_scale, 1):
            raise ValueError("Incompatible all-Dirichlet net volume flux")
        return f

    def solve(self, boundary, forcing=None, traction=None):
        if self.factor is None:
            raise RuntimeError("Plan was built without factorization")
        start = perf_counter()
        f = self.rhs(boundary, forcing, traction)
        load = perf_counter() - start
        start = perf_counter()
        self.coefficients = self.factor.solve(f)
        solve = perf_counter() - start
        residual = np.linalg.norm(self.A @ self.coefficients - f) / max(
            np.linalg.norm(f), 1
        )
        self.last = dict(
            load_seconds=load,
            solve_seconds=solve,
            residual=float(residual),
            gauge_multiplier=None if self.outflow else float(self.coefficients[-1]),
        )
        if hasattr(self.factor, "stats"):
            self.last.update(self.factor.stats)
        if not np.all(np.isfinite(self.coefficients)) or residual > 1e-8:
            raise RuntimeError(self.last)
        return self.coefficients

    def evaluate(self, p, coefficients=None):
        """u,v,p,ux,uy,vx,vy on physical points; no pressure gauge shift here."""
        p = np.asarray(p)
        coef = self.coefficients if coefficients is None else coefficients
        out = np.empty((len(p), 7))
        if p.ndim != 2 or p.shape[1] != 2 or not np.all(np.isfinite(p)):
            raise ValueError("Expected finite physical points")
        if (
            np.any(p < self.grid.lower - 1e-12)
            or np.any(p > self.grid.upper + 1e-12)
            or np.any(
                np.sum(((p - self.grid.center) / self.grid.axes) ** 2, axis=1)
                < 1 - 1e-10
            )
        ):
            raise ValueError("Evaluation point lies outside fluid")
        cells = self.grid.locate(p)
        groups = {}
        for i, cell in enumerate(cells):
            groups.setdefault(self.owner[tuple(cell)], []).append(i)
        for r, ids in groups.items():
            for sl in np.array_split(ids, max(1, int(np.ceil(len(ids) / 1024)))):
                v, x, y = self.vbasis[r].evaluate(p[sl])
                q = self.pbasis[r].evaluate(p[sl], 0)[0]
                u = coef[self.vi[r]]
                vv = coef[self.nv + self.vi[r]]
                out[sl] = np.column_stack(
                    (
                        v @ u,
                        v @ vv,
                        q @ coef[2 * self.nv + self.pi[r]],
                        x @ u,
                        y @ u,
                        x @ vv,
                        y @ vv,
                    )
                )
        return out


def channel_boundary(p, tag):
    return (
        np.column_stack((1 - p[:, 1] ** 2, np.zeros(len(p))))
        if tag == "outer"
        else np.zeros_like(p)
    )
