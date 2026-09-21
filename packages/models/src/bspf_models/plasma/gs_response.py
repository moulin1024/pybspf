"""Precompiled fixed-point response of the existing BSPF GS factorization.

Same quadrature, space, H2 scaling and TSVD as FixedBoundaryGSPlan. Setup
contracts basis evaluation with the retained right factors once. Each RHS
then uses skinny matrix products, with no MPFR basis evaluation, 2D triangular
solve or coefficient-space field reconstruction. This accelerates repeated
solves/outputs, not the initial GS factorization; no nonlinear iteration here.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.linalg as la

from bspf_models.elliptic.convex_poisson import trace_transform
from bspf_models.plasma.grad_shafranov import FixedBoundaryGSSolution
from bspf_models.plasma.grad_shafranov import _pair
from bspf_models.plasma.grad_shafranov import _sample


class GSFixedPointResponse:
    """Compile physical (R,Z) output points; defaults to source quadrature points.

    ``derivatives=False`` stores only flux, sufficient for profile updates.
    ``derivatives=True`` also stores both gradient and three Hessian components.
    Retained singular modes are unchanged; this is an algebraic reassociation,
    not a new discretization or an additional low-rank truncation.
    """

    def __init__(self, plan, points=None, *, derivatives=False, chunk_size=256):
        start = perf_counter()
        if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.plan = plan
        self.points = np.array(plan.points if points is None else points, dtype=float, copy=True)
        self.derivatives = bool(derivatives)
        basis = plan.source_basis if points is None else plan.prepare(self.points)
        basis_seconds = perf_counter()-start
        # Keep 1/sigma out of this product. Smooth RHS projections on tiny
        # singular directions stay small until the per-RHS division; never
        # explicitly multiply out a dense data-to-output pseudoinverse.
        transform = la.solve_triangular(plan.root, plan.right.T)
        (x, dx, xx), (z, dz, zz) = basis
        components = [(x, z)]
        if derivatives:
            components += [(dx, z), (x, dz), (xx, z), (dx, dz), (x, zz)]
        self.response = np.empty((len(components), len(self.points), len(plan.singular)))
        for field, (a, b) in enumerate(components):
            for begin in range(0, len(self.points), chunk_size):
                end = begin+chunk_size
                self.response[field, begin:end] = _pair(a[begin:end], b[begin:end]) @ transform
        self.points.setflags(write=False)
        self.response.setflags(write=False)
        self.diagnostics = dict(
            setup_seconds=perf_counter()-start, basis_seconds=basis_seconds,
            contraction_seconds=perf_counter()-start-basis_seconds,
            response_bytes=self.response.nbytes, output_points=len(self.points),
            components=len(components), rank=len(plan.singular),
            backend="compiled_fixed_point_response", same_tsvd=True,
        )

    def solve(self, source, boundary_flux=0.0):
        """Solve and return requested fields; only physical RHS data are used."""
        start = perf_counter()
        plan = self.plan
        f = _sample(source, plan.points, "source")
        g = _sample(boundary_flux, plan.boundary, "boundary_flux")
        rhs = np.r_[np.sqrt(plan.weights)*f, trace_transform(g, plan.arc.length)]
        projected = plan.left.T @ rhs
        reduced = projected/plan.singular
        fields = self.response @ reduced
        residual = rhs-plan.left @ projected
        if not np.all(np.isfinite(fields)):
            raise FloatingPointError("Nonfinite GS response")
        info = dict(
            solve_and_output_seconds=perf_counter()-start,
            training_relative_residual=float(la.norm(residual)/max(la.norm(rhs), 1e-300)),
            physical_data_only=True, backend="compiled_fixed_point_response",
        )
        return GSResponseResult(self, fields, reduced, info)


@dataclass
class GSResponseResult:
    response_plan: GSFixedPointResponse
    fields: np.ndarray
    reduced_coefficients: np.ndarray
    diagnostics: dict

    @property
    def points(self):
        return self.response_plan.points

    @property
    def flux(self):
        return self.fields[0]

    def _require_derivatives(self):
        if not self.response_plan.derivatives:
            raise ValueError("Compile with derivatives=True to obtain magnetic fields and GS residuals")

    @property
    def gradient(self):
        self._require_derivatives()
        return self.fields[1:3].T

    @property
    def hessian(self):
        self._require_derivatives()
        xx, xz, zz = self.fields[3:6]
        return np.stack((xx, xz, xz, zz), axis=1).reshape(-1, 2, 2)

    @property
    def delta_star(self):
        self._require_derivatives()
        return self.fields[3]+self.fields[5]-self.fields[1]/self.points[:, 0]

    def magnetic_field(self, toroidal_function):
        self._require_derivatives()
        f = np.broadcast_to(np.asarray(toroidal_function(self.flux)), self.flux.shape)
        return np.column_stack((-self.fields[2], f, self.fields[1]))/self.points[:, :1]

    def as_solution(self):
        """Recover BSPF coefficients only when arbitrary-point output is needed."""
        plan = self.response_plan.plan
        scaled = plan.right.T @ self.reduced_coefficients
        coefficients = la.solve_triangular(plan.root, scaled)
        info = dict(self.diagnostics, coefficient_norm=float(la.norm(coefficients)),
                    box_h2_norm=float(la.norm(scaled)))
        return FixedBoundaryGSSolution(plan, coefficients, info)
