"""BSPF Poisson on convex spline domains with an H^(3/2) Dirichlet trace.

Dense reference solver: exact spline geometry, analytic BSPF differentiation,
uniform-arclength trace samples, box-H2 coefficient scaling, truncated SVD.
No exterior continuation equations or exact solution derivatives are imposed.
"""

from dataclasses import dataclass

import jax
import numpy as np
import scipy.linalg as la
from numpy.polynomial import polynomial as poly
from scipy.optimize import brentq
from scipy.special import roots_legendre

from bspf_models._numerics.trial_spaces import _stream_line
from pybspf.tensor import tensor_product
from bspf_models.elliptic.smooth_extension import basis_operators
from bspf_models.elliptic.smooth_extension import evaluate_field
from bspf_models.elliptic.smooth_extension import factors
from bspf_models.elliptic.smooth_extension import evaluate_factors


def validate_convex(domain):
    """Check polynomial curvature sign, regularity and total turning.

    Requires the SplineDomain convention: a closed periodic cubic curve.
    Polynomial extrema are checked on every span, not just at sample points.
    """
    q, w = roots_legendre(48)
    turning = 0.0
    minimum = np.inf
    for span in range(domain.period):
        c = np.array(
            [domain.curve(span, 1), domain.curve(span, 2), domain.curve(span, 3) / 2]
        )
        d = np.array([c[1], 2 * c[2]])
        cross = poly.polysub(
            poly.polymul(c[:, 0], d[:, 1]), poly.polymul(c[:, 1], d[:, 0])
        )
        speed2 = poly.polyadd(
            poly.polymul(c[:, 0], c[:, 0]), poly.polymul(c[:, 1], c[:, 1])
        )

        def extrema(p):
            roots = poly.polyroots(poly.polyder(p))
            candidates = [0.0, 1.0] + [
                z.real for z in roots if abs(z.imag) < 1e-10 and 0 < z.real < 1
            ]
            return np.min(poly.polyval(candidates, p))

        if extrema(speed2) <= 1e-20:
            raise ValueError("Spline boundary is not regular")
        minimum = min(minimum, float(extrema(cross)))
        turning += np.sum(
            w / 2 * poly.polyval((q + 1) / 2, cross) / poly.polyval((q + 1) / 2, speed2)
        )
    if minimum < -1e-12 or abs(turning - 2 * np.pi) > 1e-7:
        raise ValueError(
            "Expected a convex counterclockwise boundary with turning 2*pi"
        )
    return dict(minimum_curvature_numerator=minimum, total_turning=float(turning))


class ArcLengthBoundary:
    def __init__(self, domain, quadrature_order=32):
        self.domain = domain
        q, w = roots_legendre(quadrature_order)
        self.q, self.w = q, w
        lengths = [self.integral(k, 1.0) for k in range(domain.period)]
        self.offsets = np.r_[0.0, np.cumsum(lengths)]
        self.length = float(self.offsets[-1])

    def integral(self, span, t):
        p = span + t * (self.q + 1) / 2
        return float(
            t / 2 * np.sum(self.w * np.linalg.norm(self.domain.curve(p, 1), axis=1))
        )

    def sample(self, count, offset=0.0):
        if count < 8 or count % 2:
            raise ValueError("An even boundary count >=8 is required")
        s = (np.arange(count) + offset) * self.length / count % self.length
        spans = np.searchsorted(self.offsets, s, side="right") - 1
        t = np.array(
            [
                k
                + brentq(
                    lambda a: self.integral(k, a) - (v - self.offsets[k]),
                    0.0,
                    1.0,
                    xtol=5e-15,
                )
                for k, v in zip(spans, s)
            ]
        )
        return self.domain.curve(t), t


def trace_transform(values, length, sobolev_order=1.5):
    """Real matrix for periodic arclength H^s norm, including normalization."""
    count = values.shape[0]
    if count % 2:
        raise ValueError("Even trace sample count required")
    spectrum = np.fft.rfft(values, axis=0, norm="ortho")
    frequency = 2 * np.pi * np.fft.rfftfreq(count, d=length / count)
    multiplicity = np.full(len(frequency), 2.0)
    multiplicity[[0, -1]] = 1
    scale = np.sqrt(length / count * multiplicity) * (1 + frequency**2) ** (
        sobolev_order / 2
    )
    weighted = spectrum * scale.reshape((-1,) + (1,) * (values.ndim - 1))
    return np.concatenate((weighted.real, weighted.imag), axis=0)


def box_h2_root(line):
    """R with ||R c||^2 = integral_B (u^2+|grad u|^2+|Hess u|_F^2)."""
    k = np.diag(np.asarray(line.lam))
    j = np.asarray(line.bending)
    eye = np.eye(len(k))
    gram = np.eye(len(k) ** 2) + np.kron(k, eye) + np.kron(eye, k)
    gram += np.kron(j, eye) + 2 * np.kron(k, k) + np.kron(eye, j)
    return la.cholesky((gram + gram.T) / 2, lower=False)


def evaluate_hessian_factors(basis, coefficients):
    """Cartesian Hessian from the same analytic tensor factors as the solver."""
    (x, dx, xx), (y, dy, yy) = basis
    c = coefficients.reshape(x.shape[1], y.shape[1])
    h = np.empty((len(x), 2, 2))
    h[:, 0, 0] = np.sum((xx @ c) * y, axis=1)
    h[:, 1, 1] = np.sum((x @ c) * yy, axis=1)
    h[:, 0, 1] = h[:, 1, 0] = np.sum((dx @ c) * dy, axis=1)
    return h


@dataclass
class ConvexPoissonSolution:
    plan: object
    coefficients: np.ndarray
    diagnostics: dict

    def evaluate(self, points):
        """Return value, Cartesian gradient, and Delta u (positive Laplacian)."""
        return evaluate_field(self.plan.line, np.asarray(points), self.coefficients)

    def hessian(self, points):
        return evaluate_hessian_factors(
            factors(self.plan.line, np.asarray(points)), self.coefficients
        )

    def validate(self, forcing, boundary_data, *, volume_order=22, boundary_count=None):
        """Independent quadrature and shifted, refined boundary sampling."""
        count = boundary_count or 2 * self.plan.boundary_count
        key = (volume_order, count)
        if key not in self.plan.validation_cache:
            p, w, _ = self.plan.domain.volume_rule(volume_order)
            bp, _ = self.plan.arc.sample(count, offset=0.371)
            self.plan.validation_cache[key] = (
                p,
                w,
                factors(self.plan.line, p),
                bp,
                factors(self.plan.line, bp),
            )
        p, w, volume_basis, bp, boundary_basis = self.plan.validation_cache[key]
        _, _, lap = evaluate_factors(volume_basis, self.coefficients)
        f = np.asarray(forcing(p))
        b = evaluate_factors(boundary_basis, self.coefficients)[0] - boundary_data(bp)
        spectrum = np.fft.rfft(b, norm="ortho")
        freq = 2 * np.pi * np.fft.rfftfreq(count, d=self.plan.arc.length / count)
        mult = np.full(len(freq), 2.0)
        mult[[0, -1]] = 1
        energy = (
            self.plan.arc.length
            / count
            * mult
            * (1 + freq**2) ** 1.5
            * abs(spectrum) ** 2
        )
        return dict(
            pde_l2=float(np.sqrt(w @ ((-lap - f) ** 2))),
            forcing_l2=float(np.sqrt(w @ (f * f))),
            boundary_h32=float(np.sqrt(energy.sum())),
            boundary_linf=float(np.max(abs(b))),
            boundary_high_frequency_h32=float(
                np.sqrt(energy[2 * len(energy) // 3 :].sum())
            ),
            volume_order=volume_order,
            boundary_count=count,
        )


class ConvexPoissonPlan:
    def __init__(
        self,
        domain,
        *,
        nodes=49,
        half_width=1.2,
        volume_order=16,
        boundary_count=None,
        rcond=1e-13,
        sobolev_order=1.5,
        coefficient_scaling="h2",
        geometry=None,
    ):
        if not jax.config.x64_enabled:
            raise ValueError("Enable jax_enable_x64 before constructing BSPF factors")
        if not 0 < rcond < 1 or sobolev_order < 0:
            raise ValueError("Require 0<rcond<1 and nonnegative trace order")
        self.geometry_checks = validate_convex(domain)
        self.validation_cache = {}
        self.domain = domain
        self.arc = ArcLengthBoundary(domain)
        self.boundary_count = boundary_count or 2 ** int(np.ceil(np.log2(8 * nodes)))
        self.rcond = rcond
        self.sobolev_order = sobolev_order
        if geometry is None:
            if np.max(abs(domain.controls)) >= half_width:
                raise ValueError("Boundary must lie strictly inside the auxiliary box")
            self.line = _stream_line(
                np.linspace(-half_width, half_width, nodes),
                clamped=False,
                dirichlet=False,
                endpoint_points=12,
                chebyshev_modes=12,
            )
            self.points, self.weights, _ = domain.volume_rule(volume_order)
            # Do not allocate the unused volume value matrix (large at high N).
            (x, _, xx), (y, _, yy) = factors(self.line, self.points)
            laplace = -tensor_product(xx, y, paired=True)
            laplace -= tensor_product(x, yy, paired=True)
            del x, xx, y, yy
        else:
            if not np.array_equal(domain.controls, geometry.domain.controls):
                raise ValueError("Geometry cache is for a different domain")
            if (
                geometry.nodes != nodes
                or geometry.half_width != half_width
                or geometry.volume_order != volume_order
            ):
                raise ValueError("Geometry cache discretization differs")
            self.line = geometry.line
            self.points = geometry.points
            self.weights = geometry.weights
            laplace = geometry.laplace
        self.boundary, self.parameters = self.arc.sample(self.boundary_count)
        trace = basis_operators(self.line, self.boundary)[0]
        operator = np.vstack(
            (
                np.sqrt(self.weights[:, None]) * laplace,
                trace_transform(trace, self.arc.length, sobolev_order),
            )
        )
        del laplace, trace
        self.root = box_h2_root(self.line) if coefficient_scaling == "h2" else None
        if coefficient_scaling not in ("h2", "column"):
            raise ValueError("coefficient_scaling must be h2 or column")
        self.column_scale = None
        if self.root is not None:
            operator = la.solve_triangular(
                self.root.T, operator.T, lower=True, overwrite_b=True
            ).T
        else:
            self.column_scale = np.maximum(la.norm(operator, axis=0), 1e-30)
            operator = operator / self.column_scale
        left, singular, right = la.svd(operator, full_matrices=False)
        keep = singular > rcond * singular[0]
        self.left = left[:, keep]
        self.singular = singular[keep]
        self.right = right[keep]
        self.ndofs = operator.shape[1]
        self.rank = int(keep.sum())
        self.smallest_relative_singular = float(singular[-1] / singular[0])

    def solve(self, forcing, boundary_data):
        """Callbacks take points (n,2) and return physical f or g (n,)."""
        f = np.asarray(forcing(self.points), float)
        g = np.asarray(boundary_data(self.boundary), float)
        if f.shape != (len(self.points),) or g.shape != (len(self.boundary),):
            raise ValueError("f and g must return one value per supplied point")
        if not np.all(np.isfinite(f)) or not np.all(np.isfinite(g)):
            raise ValueError("Nonfinite Poisson data")
        rhs = np.r_[
            np.sqrt(self.weights) * f,
            trace_transform(g, self.arc.length, self.sobolev_order),
        ]
        projected = self.left.T @ rhs
        a = self.right.T @ (projected / self.singular)
        c = (
            la.solve_triangular(self.root, a)
            if self.root is not None
            else a / self.column_scale
        )
        info = dict(
            ndofs=self.ndofs,
            rank=self.rank,
            rcond=self.rcond,
            trace_order=self.sobolev_order,
            boundary_count=self.boundary_count,
            coefficient_norm=float(la.norm(c)),
            scaled_coefficient_norm=float(la.norm(a)),
            smallest_relative_singular=self.smallest_relative_singular,
            retained_relative_residual=float(
                la.norm(rhs - self.left @ projected) / max(la.norm(rhs), 1e-300)
            ),
            physical_data_only=True,
        )
        return ConvexPoissonSolution(self, c, info)
