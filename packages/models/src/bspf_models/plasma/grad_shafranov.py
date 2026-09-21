"""Fixed-boundary BSPF Grad--Shafranov solve on smooth convex domains, R>0.

Solves -Delta* psi = source(R,Z), with prescribed Dirichlet flux. Solov'ev
profiles (constant p' and FF') are supported directly. This is a linear GS
backend; no general nonlinear-profile iteration or free-boundary update.
Discretization: physical strong residual + arclength H^(3/2) boundary residual,
box-H2 coefficient scaling and reusable TSVD, matching the BSPF Poisson path.
"""

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import jax
import numpy as np
import scipy.linalg as la

from bspf_models.elliptic.convex_poisson import ArcLengthBoundary
from bspf_models.elliptic.convex_poisson import box_h2_root
from bspf_models.elliptic.convex_poisson import trace_transform
from bspf_models.elliptic.convex_poisson import validate_convex
from bspf_models.elliptic.embedded_poisson import SplineDomain
from bspf_models.elliptic.smooth_extension import factors
from bspf_models.plasma.solovev import SolovevFluxDomain
from bspf_models._numerics.trial_spaces import _stream_line


@lru_cache(maxsize=8)
def _gs_line(nodes, half_width):
    return _stream_line(np.linspace(-half_width, half_width, nodes),
                        clamped=False, dirichlet=False,
                        endpoint_points=12, chebyshev_modes=12)


def _pair(x, y):
    return (x[:, :, None]*y[:, None, :]).reshape(len(x), -1)


def _sample(data, points, name):
    value = np.asarray(data(points) if callable(data) else data)
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    value = np.asarray(value, dtype=float)
    if value.ndim == 0:
        value = np.full(len(points), value)
    if value.shape != (len(points),) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must provide one finite value per physical point")
    return value


def evaluate_gs_factors(basis, coefficients):
    (x, dx, xx), (z, dz, zz) = basis
    c = coefficients.reshape(x.shape[1], z.shape[1])
    apply = lambda a, b: np.sum((a @ c)*b, axis=1)
    psi = apply(x, z)
    grad = np.column_stack((apply(dx, z), apply(x, dz)))
    h = np.empty((len(x), 2, 2))
    h[:, 0, 0], h[:, 1, 1] = apply(xx, z), apply(x, zz)
    h[:, 0, 1] = h[:, 1, 0] = apply(dx, dz)
    return psi, grad, h


class FixedBoundaryGSPlan:
    """Reusable linear GS factors. Callbacks and evaluation use physical (R,Z).

    ``domain`` is described in local (R-major_radius,Z) coordinates. Supported
    geometries are convex SplineDomain and exact analytic SolovevFluxDomain.
    The auxiliary box must stay at R>0; no physical condition is set on it.
    """

    def __init__(self, domain, *, major_radius=2.0, nodes=33, half_width=1.2,
                 volume_order=16, boundary_count=512, rcond=1e-13):
        start = perf_counter()
        if not jax.config.x64_enabled:
            raise ValueError("Enable jax_enable_x64 before BSPF setup")
        if not np.isfinite(half_width) or not np.isfinite(major_radius) or not 0 < half_width < major_radius:
            raise ValueError("Require major_radius > half_width > 0 so the box stays at R>0")
        if not isinstance(nodes, (int, np.integer)) or nodes < 17:
            raise ValueError("Require integer nodes >=17")
        if not isinstance(volume_order, (int, np.integer)) or volume_order < 2:
            raise ValueError("Require integer volume_order >=2")
        if not isinstance(boundary_count, (int, np.integer)) or boundary_count < 8 or boundary_count % 2:
            raise ValueError("Require an even boundary_count >=8")
        if not 0 < rcond < 1:
            raise ValueError("Require 0<rcond<1")
        if isinstance(domain, SplineDomain):
            geometry_checks = validate_convex(domain)
            bound = np.max(np.abs(domain.controls))
        elif isinstance(domain, SolovevFluxDomain):
            if major_radius != domain.major_radius:
                raise ValueError("major_radius differs from the analytic domain")
            geometry_checks, bound = domain.geometry_checks, np.max(np.abs(domain.bounds))
        else:
            raise TypeError("Expected a convex SplineDomain or SolovevFluxDomain")
        if bound >= half_width:
            raise ValueError("Physical boundary must lie strictly inside the auxiliary box")
        self.domain, self.major_radius = domain, float(major_radius)
        self.nodes, self.half_width, self.rcond = nodes, half_width, rcond
        self.boundary_count, self.volume_order = boundary_count, volume_order
        self.geometry_checks = dict(geometry_checks)
        self.line = _gs_line(nodes, half_width)
        line_seconds = perf_counter()-start
        self.local_points, self.weights, _ = domain.volume_rule(volume_order)
        self.points = self.local_points + [major_radius, 0]
        self.arc = ArcLengthBoundary(domain)
        self.local_boundary, self.parameters = self.arc.sample(boundary_count)
        self.boundary = self.local_boundary + [major_radius, 0]
        (x, dx, xx), (z, dz, zz) = factors(self.line, self.local_points)
        self.source_basis = ((x, dx, xx), (z, dz, zz))
        # -Delta* = -d_RR - d_ZZ + (1/R)d_R. Physical R is never local x.
        gs = -_pair(xx, z)-_pair(x, zz)+_pair(dx, z)/self.points[:, :1]
        del x, dx, xx, z, dz, zz
        (x, _, _), (z, _, _) = factors(self.line, self.local_boundary)
        trace = _pair(x, z)
        operator = np.vstack((np.sqrt(self.weights[:, None])*gs,
                              trace_transform(trace, self.arc.length)))
        del gs, trace, x, z
        self.root = box_h2_root(self.line)
        operator = la.solve_triangular(self.root.T, operator.T, lower=True, overwrite_b=True).T
        assembly_seconds = perf_counter()-start-line_seconds
        svd_start = perf_counter()
        u, singular, vh = la.svd(operator, full_matrices=False, overwrite_a=True)
        keep = singular > rcond*singular[0]
        self.left, self.singular, self.right = u[:, keep], singular[keep], vh[keep]
        self.spectrum = singular
        self.validation_cache = {}
        self.diagnostics = dict(
            nodes=nodes, ndofs=nodes**2, rank=int(keep.sum()), rcond=rcond,
            volume_points=len(self.points), boundary_count=boundary_count,
            line_seconds=line_seconds, assembly_seconds=assembly_seconds,
            svd_seconds=perf_counter()-svd_start, setup_seconds=perf_counter()-start,
            factor_bytes=sum(a.nbytes for a in (self.root, self.left, self.singular, self.right)),
            source_basis_bytes=sum(a.nbytes for axis in self.source_basis for a in axis),
            smallest_relative_singular=float(singular[-1]/singular[0]),
            geometry=self.geometry_checks,
        )

    def solve(self, source, boundary_flux=0.0):
        """Only source and boundary flux are inputs; no interior exact jets."""
        start = perf_counter()
        f = _sample(source, self.points, "source")
        g = _sample(boundary_flux, self.boundary, "boundary_flux")
        rhs = np.r_[np.sqrt(self.weights)*f, trace_transform(g, self.arc.length)]
        projection = self.left.T @ rhs
        a = self.right.T @ (projection/self.singular)
        c = la.solve_triangular(self.root, a)
        residual = rhs-self.left @ projection
        info = dict(
            solve_seconds=perf_counter()-start,
            training_relative_residual=float(la.norm(residual)/max(la.norm(rhs), 1e-300)),
            box_h2_norm=float(la.norm(a)), coefficient_norm=float(la.norm(c)),
            physical_data_only=True,
        )
        return FixedBoundaryGSSolution(self, c, info)

    def solve_solovev(self, *, p_prime, ff_prime, mu0=1.0, boundary_flux=0.0):
        """Constant profile derivatives; both pressure and toroidal source kept."""
        if not np.all(np.isfinite([p_prime, ff_prime, mu0])) or mu0 <= 0:
            raise ValueError("Require finite profile derivatives and mu0>0")
        return self.solve(lambda p: mu0*p_prime*p[:, 0]**2+ff_prime, boundary_flux)

    def prepare(self, points):
        """Reusable analytic basis factors at physical evaluation points."""
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or not np.all(np.isfinite(p)) or np.any(p[:, 0] <= 0):
            raise ValueError("Expected finite physical (R,Z) points with R>0")
        local = p-[self.major_radius, 0]
        if np.any(np.abs(local) > self.half_width+1e-14):
            raise ValueError("Evaluation points must lie in the auxiliary BSPF box")
        return factors(self.line, local)

    def compile_response(self, points=None, *, derivatives=False, chunk_size=256):
        """Precompile repeated RHS-to-field solves at fixed physical points."""
        from bspf_models.plasma.gs_response import GSFixedPointResponse

        return GSFixedPointResponse(self, points, derivatives=derivatives, chunk_size=chunk_size)


@dataclass
class FixedBoundaryGSSolution:
    plan: FixedBoundaryGSPlan
    coefficients: np.ndarray
    diagnostics: dict

    def jets(self, points):
        return evaluate_gs_factors(self.plan.prepare(points), self.coefficients)

    def evaluate(self, points):
        """Return psi, grad psi, and positive Delta* psi (not plain Laplacian)."""
        psi, gradient, hessian = self.jets(points)
        delta_star = hessian[:, 0, 0]+hessian[:, 1, 1]-gradient[:, 0]/np.asarray(points)[:, 0]
        return psi, gradient, delta_star

    def magnetic_field(self, points, toroidal_function):
        """B components in cylindrical order (R,phi,Z), with F=R B_phi."""
        psi, grad, _ = self.jets(points)
        f = np.broadcast_to(np.asarray(toroidal_function(psi)), psi.shape)
        return np.column_stack((-grad[:, 1], f, grad[:, 0]))/np.asarray(points)[:, :1]

    def validate(self, source, boundary_flux=0.0, *, volume_order=22, boundary_count=1024):
        key = (volume_order, boundary_count)
        if key not in self.plan.validation_cache:
            local, w, _ = self.plan.domain.volume_rule(volume_order)
            p = local+[self.plan.major_radius, 0]
            edge, _ = self.plan.arc.sample(boundary_count, offset=0.371)
            edge = edge+[self.plan.major_radius, 0]
            self.plan.validation_cache[key] = (p, w, self.plan.prepare(p), edge, self.plan.prepare(edge))
        p, w, basis, edge, trace = self.plan.validation_cache[key]
        _, g, h = evaluate_gs_factors(basis, self.coefficients)
        delta_star = h[:, 0, 0]+h[:, 1, 1]-g[:, 0]/p[:, 0]
        f = _sample(source, p, "source")
        residual = -delta_star-f
        boundary_error = evaluate_gs_factors(trace, self.coefficients)[0]-_sample(boundary_flux, edge, "boundary_flux")
        return dict(
            gs_residual_l2=float(np.sqrt(w @ residual**2)),
            gs_relative_residual_l2=float(np.sqrt(w @ residual**2/max(w @ f**2, 1e-300))),
            boundary_linf=float(np.max(np.abs(boundary_error))),
            boundary_h32=float(la.norm(trace_transform(boundary_error, self.plan.arc.length))),
            volume_order=volume_order, volume_points=len(p), boundary_count=boundary_count,
        )
