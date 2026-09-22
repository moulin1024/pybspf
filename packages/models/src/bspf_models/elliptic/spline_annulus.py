"""High-order reference solver on a full spline section or spline annulus.

L_sigma = -Delta + sigma, constant real sigma, Dirichlet data on every boundary.
Poisson: augmented single layer; modified Helmholtz: single layer;
Helmholtz: interior trace of D + i*eta*S, eta>0, using outward domain normals.
Panel quadrature follows every geometry knot. Near-target integration subdivides
quadrature only, without moving geometry or changing density unknowns.

Host-side experimental model, not a JAX/FFT/FMM implementation. General sources
are approximated by an independently audited Fourier extension from domain-only
samples. No exact solution or supplied particular solution is needed by solve().
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from time import perf_counter

import numpy as np
from numpy.polynomial.legendre import legvander
from scipy.interpolate import BSpline, PPoly
from scipy.linalg import lu_factor, lu_solve, svd
from scipy.linalg.lapack import get_lapack_funcs
from scipy.optimize import brentq
from scipy.special import hankel1, j0, j1, k0, i0

from .panel_poisson import rule, log_moments


class SplineBoundary:
    """Exact planar BSpline, regular closed C1 or smoother, with arbitrary knots.

    Closure is checked; regularity and nesting checks are sampled. Simplicity
    is a caller precondition, not certified.
    Both orientations are accepted. Corners and rational splines are unsupported.
    """

    def __init__(self, curve):
        if (not isinstance(curve, BSpline) or curve.c.ndim != 2
                or curve.c.shape[1] != 2 or curve.axis != 0):
            raise ValueError('expected a planar scipy.interpolate.BSpline')
        if curve.k < 2:
            raise ValueError('require degree >= 2 and C1 regular geometry')
        self.curve = curve
        self.a, self.b = float(curve.t[curve.k]), float(curve.t[-curve.k-1])
        self.knots = np.unique(curve.t[(curve.t >= self.a) & (curve.t <= self.b)])
        scale = max(1., float(np.ptp(curve.c, axis=0).max()))
        for d in (0, 1):
            if np.linalg.norm(curve(self.a, d)-curve(self.b, d)) > 1e-10*scale:
                raise ValueError('curve must close with matching first derivative')
        internal = curve.t[(curve.t > self.a) & (curve.t < self.b)]
        if internal.size and np.max(np.unique(internal, return_counts=True)[1]) > curve.k-1:
            raise ValueError('interior knots must preserve at least C1 continuity')
        q, w = rule(max(12, curve.k+1))
        mids = (self.knots[1:]+self.knots[:-1])/2
        halves = np.diff(self.knots)/2
        t = (mids[:, None]+halves[:, None]*q).ravel()
        v, d = curve(t), curve(t, 1)
        speed = np.linalg.norm(d, axis=1)
        if np.min(speed) < 1e-10*scale/(self.b-self.a):
            raise ValueError('sampled nonregular curve')
        self.area = float(np.sum((halves[:, None]*w).ravel()*(v[:, 0]*d[:, 1]-v[:, 1]*d[:, 0]))/2)
        if abs(self.area) < 1e-12*scale*scale:
            raise ValueError('degenerate curve area')
        self.orientation = np.sign(self.area)
        self.bounds = []
        for axis in (0, 1):
            poly = PPoly.from_spline((curve.t, curve.c[:, axis], curve.k), extrapolate=False)
            roots = poly.derivative().roots(extrapolate=False)
            roots = roots[np.isfinite(roots) & (roots >= self.a) & (roots <= self.b)]
            breaks = np.unique(np.r_[self.knots, roots])
            values = curve(breaks)[:, axis]
            self.bounds.append((float(values.min()), float(values.max())))
            if axis == 0:
                self.xbreaks = breaks
        self.bounds = np.asarray(self.bounds)

    def normal(self, t, sign=1):
        d = self.curve(t, 1)
        return sign*self.orientation*np.stack((d[..., 1], -d[..., 0]), axis=-1)/np.linalg.norm(d, axis=-1)[..., None]

    def intervals(self, x):
        roots = []
        for a, b in zip(self.xbreaks[:-1], self.xbreaks[1:]):
            xa, xb = self.curve([a, b])[:, 0]
            if abs(xa-xb) > 1e-14 and min(xa, xb) <= x <= max(xa, xb):
                roots.append(brentq(lambda t: self.curve(t)[0]-x, a, b, xtol=5e-15))
        if not roots:
            return np.empty((0, 2))
        roots = np.sort((np.asarray(roots)-self.a) % (self.b-self.a)+self.a)
        roots = roots[np.r_[True, np.diff(roots) > 1e-10*(self.b-self.a)]]
        ys = np.sort(self.curve(roots)[:, 1])
        if len(ys) % 2:
            # A tangency column can have zero-measure intervals; callers use
            # shifted sampling grids to avoid it. Do not silently misclassify.
            raise ValueError('tangency/invalid geometry in vertical intersections')
        return ys.reshape(-1, 2)

    def contains(self, points):
        points = np.asarray(points)
        result = np.zeros(len(points), dtype=bool)
        for x in np.unique(points[:, 0]):
            ids = np.flatnonzero(points[:, 0] == x)
            if not self.bounds[0, 0] < x < self.bounds[0, 1]:
                continue
            for lo, hi in self.intervals(x):
                result[ids] |= (points[ids, 1] > lo) & (points[ids, 1] < hi)
        return result


class SplineAnnulus:
    def __init__(self, outer, inner):
        self.boundaries = tuple(b if isinstance(b, SplineBoundary) else SplineBoundary(b)
                                for b in (outer, inner))
        self.bounds = self.boundaries[0].bounds
        inner_points = self.boundaries[1].curve(np.linspace(self.boundaries[1].a,
                    self.boundaries[1].b, 127, endpoint=False)+1e-8)
        if not np.all(self.boundaries[0].contains(inner_points)):
            raise ValueError('inner boundary must lie strictly inside outer boundary')

    def contains(self, points):
        return self.boundaries[0].contains(points) & ~self.boundaries[1].contains(points)

    def sample(self, n, shift=.5):
        x = self.bounds[0, 0]+(np.arange(n)+shift)/n*np.diff(self.bounds[0])[0]
        y = self.bounds[1, 0]+(np.arange(n)+shift)/n*np.diff(self.bounds[1])[0]
        xx, yy = np.meshgrid(x, y, indexing='ij')
        points = np.column_stack((xx.ravel(), yy.ravel()))
        return points[self.contains(points)]


class SplineSection:
    """Simply connected full poloidal section, including its magnetic axis.

    Coordinates may be local (R-R0,Z); R=0 is not required or supported by GS.
    """
    def __init__(self, outer):
        self.boundaries = (outer if isinstance(outer, SplineBoundary) else SplineBoundary(outer),)
        self.bounds = self.boundaries[0].bounds

    def contains(self, points):
        return self.boundaries[0].contains(points)

    sample = SplineAnnulus.sample


class FourierSourcePlan:
    """Least-squares source extension from points strictly inside the annulus.

    SVD truncation concerns the source representation only, not the PDE solve.
    Sampling residuals are diagnostics, not a certified continuous error bound.
    ``padding`` scales the Fourier period relative to the bounding box: larger
    padding reduces physical bandwidth at fixed ``modes``. Interior collars on
    both boundaries constrain continuation through the grid's unsampled strips.
    """
    def __init__(self, domain, modes=6, samples=40, rcond=1e-12, padding=2.0):
        if modes < 1 or samples < 4 or not 0 < rcond < 1 or padding <= 1:
            raise ValueError('invalid Fourier source parameters')
        self.domain = domain
        self.center = domain.bounds.mean(axis=1)
        self.lengths = np.ptp(domain.bounds, axis=1)*padding
        mesh = np.arange(-modes, modes+1)
        a, b = np.meshgrid(mesh, mesh, indexing='ij')
        self.frequencies = 2*np.pi*np.column_stack((a.ravel(), b.ravel()))/self.lengths
        self.points = domain.sample(samples)
        bulk_count = len(self.points)
        # A Cartesian grid leaves an unsampled strip next to curved boundaries.
        # Fitting there only by extrapolation amplifies truncated-SVD errors.
        # Resolve that strip using interior normal offsets on both components;
        # validation uses different parameters and a different offset below.
        collar = []
        width = min(np.ptp(domain.bounds, axis=1))
        for component, boundary in enumerate(domain.boundaries):
            t = boundary.a+(np.arange(4*samples)+.137)/(4*samples)*(boundary.b-boundary.a)
            for distance in (1e-6*width, .25*width/samples):
                collar.append(boundary.curve(t)-distance*boundary.normal(t, 1 if component == 0 else -1))
        collar = np.vstack(collar)
        self.points = np.vstack((self.points, collar[domain.contains(collar)]))
        if len(self.points) < len(self.frequencies):
            raise ValueError('need at least as many source samples as Fourier modes')
        self.validation = domain.sample(samples+3, shift=.317)
        # Include near-wall validation without evaluating in the hole/exterior.
        near = []
        delta = .0001*min(np.ptp(domain.bounds, axis=1))
        validation_count = max(67, 4*samples+3)
        for component, boundary in enumerate(domain.boundaries):
            t = boundary.a+(np.arange(validation_count)+.371)/validation_count*(boundary.b-boundary.a)
            near.append(boundary.curve(t)-delta*boundary.normal(t, 1 if component == 0 else -1))
        near = np.vstack(near)
        self.validation = np.vstack((self.validation, near[domain.contains(near)]))
        start = perf_counter()
        matrix = self.basis(self.points)
        u, s, vh = svd(matrix, full_matrices=False)
        keep = s > rcond*s[0]
        # Apply the SVD in factored order; an explicitly formed pseudoinverse
        # loses accuracy through cancellation when the extension is ill-conditioned.
        self.left = u[:, keep].conj().T
        self.right = vh[keep].conj().T
        self.singular = s[keep]
        self.stats = dict(modes=modes, columns=len(s), samples=len(self.points),
                          bulk_samples=bulk_count, collar_samples=len(self.points)-bulk_count,
                          padding=padding, angular_bandwidth=np.max(abs(self.frequencies), axis=0).tolist(),
                          retained_rank=int(keep.sum()), rcond=rcond,
                          retained_condition=float(s[0]/s[keep][-1]), setup_seconds=perf_counter()-start)

    def basis(self, points):
        return np.exp(1j*(np.asarray(points)-self.center)@self.frequencies.T)

    def fit(self, source, sigma, *, tolerance=1e-7):
        values = np.asarray(source(self.points))
        if values.shape != (len(self.points),) or not np.all(np.isfinite(values)):
            raise ValueError('source must return one finite value per domain point')
        coefficients = self.right@((self.left@values)/self.singular)
        reference = np.asarray(source(self.validation))
        if reference.shape != (len(self.validation),) or not np.all(np.isfinite(reference)):
            raise ValueError('invalid validation source values')
        error = np.max(np.abs(self.basis(self.validation)@coefficients-reference))
        scale = max(float(np.max(np.abs(reference))), float(np.max(np.abs(values))), 1e-300)
        relative = float(error/scale)
        if relative > tolerance:
            raise ValueError(f'source validation failed: {relative:.3e} > {tolerance:.3e}; increase modes/sampling')
        stats = dict(self.stats, validation_relative_max=relative,
                     validation_absolute_max=float(error), tolerance=tolerance)
        return FourierParticular(self, coefficients, float(sigma), stats)


@dataclass
class FourierParticular:
    plan: FourierSourcePlan
    coefficients: np.ndarray
    sigma: float
    stats: dict

    def source_values(self, points):
        return self.plan.basis(points)@self.coefficients

    def __call__(self, points):
        x = np.asarray(points)-self.plan.center
        freq = self.plan.frequencies
        norm = np.sum(freq*freq, axis=1)
        denom = norm+self.sigma
        resonant = np.abs(denom) < 1e-11*np.maximum(1., norm+abs(self.sigma))
        phi = self.plan.basis(points)
        value = phi[:, ~resonant]@(self.coefficients[~resonant]/denom[~resonant])
        for j in np.flatnonzero(resonant):
            if norm[j] == 0:
                # -Delta(-|x|^2/4)=1. Only sigma=0 has this zero mode.
                mode = -np.sum(x*x, axis=1)/4
            else:
                # Polynomial times plane wave avoids an artificial BOX resonance.
                mode = 1j*(x@freq[j])/(2*norm[j])*phi[:, j]
            value += self.coefficients[j]*mode
        return value


class _Panel:
    def __init__(self, boundary, component, left, right, order):
        self.boundary, self.component = boundary, component
        self.left, self.right, self.order = left, right, order
        self.mid, self.half = (left+right)/2, (right-left)/2
        q, w = rule(order)
        self.t = self.mid+self.half*q
        self.points = boundary.curve(self.t)
        self.speed = np.linalg.norm(boundary.curve(self.t, 1), axis=1)
        self.ds = self.half*w*self.speed
        self.coeff = (2*np.arange(order)+1)[:, None]/2*legvander(q, order-1).T*w
        self.cache = {}

    def basis(self, t):
        return legvander((np.asarray(t)-self.mid)/self.half, self.order-1)@self.coeff

    def quadrature(self, left, right, order):
        key = (left, right, order)
        if key not in self.cache:
            q, w = rule(order)
            t = (left+right)/2+(right-left)/2*q
            curve = self.boundary.curve
            d = curve(t, 1)
            speed = np.linalg.norm(d, axis=1)
            normal = self.boundary.normal(t, 1 if self.component == 0 else -1)
            self.cache[key] = (t, (right-left)/2*w, curve(t), speed, normal, self.basis(t))
        return self.cache[key]


class AnnulusPanelPlan:
    """Dirichlet boundary correction for -Delta+sigma on an exact spline annulus."""
    def __init__(self, domain, sigma=0., order=12, subdivisions=1, quadrature_order=None, breaks=None):
        start = perf_counter()
        if not np.isfinite(sigma) or order < 3 or subdivisions < 1:
            raise ValueError('invalid operator/panel parameters')
        self.domain, self.sigma, self.order = domain, float(sigma), int(order)
        self.wave = np.sqrt(abs(self.sigma))
        self.eta = max(1., self.wave) if sigma < 0 else 0.
        self.qorder = quadrature_order or max(32, 2*self.order+8)
        if self.qorder < self.order+2:
            raise ValueError('quadrature order must exceed density order by at least two')
        self.panels = []
        for component, boundary in enumerate(domain.boundaries):
            spans = boundary.knots if breaks is None else np.asarray(breaks[component], dtype=float)
            if (spans.ndim != 1 or len(spans) < 2 or np.any(np.diff(spans) <= 0)
                    or spans[0] != boundary.a or spans[-1] != boundary.b
                    or any(not np.any(np.isclose(spans, knot, atol=1e-13, rtol=0)) for knot in boundary.knots)):
                raise ValueError('breaks must include every geometry knot and both endpoints')
            for a, b in zip(spans[:-1], spans[1:]):
                # Kernel splitting I0 grows rapidly on large alpha-panels.
                speed = np.max(np.linalg.norm(boundary.curve(np.linspace(a, b, 9), 1), axis=1))
                pieces = max(subdivisions, int(np.ceil(self.wave*(b-a)*speed/2.)))
                edges = np.linspace(a, b, pieces+1)
                self.panels.extend(_Panel(boundary, component, l, r, self.order)
                                   for l, r in zip(edges[:-1], edges[1:]))
        self.points = np.vstack([p.points for p in self.panels])
        self.parameters = np.concatenate([p.t for p in self.panels])
        self.components = np.concatenate([np.full(self.order, p.component) for p in self.panels])
        self.weights = np.concatenate([p.ds for p in self.panels])
        self.count = len(self.points)
        self.matrix = self.potential_matrix(self.points, self.components, self.parameters)
        if self.sigma == 0:
            augmented = np.zeros((self.count+1, self.count+1), dtype=complex)
            augmented[:-1, :-1] = self.matrix
            augmented[:-1, -1] = 1.
            augmented[-1, :-1] = self.weights/self.weights.sum()
            self.system = augmented
        else:
            self.system = self.matrix
        self.factor = lu_factor(self.system)
        gecon = get_lapack_funcs('gecon', (self.factor[0],))
        rcond, info = gecon(self.factor[0], np.linalg.norm(self.system, 1))
        if info:
            raise RuntimeError('condition estimate failed')
        self.stats = dict(sigma=self.sigma, order=order, panels=len(self.panels), unknowns=self.count,
                          reciprocal_condition_estimate=float(rcond), setup_seconds=perf_counter()-start,
                          formulation=('augmented_single_layer' if sigma == 0 else
                                       'single_layer' if sigma > 0 else 'double_plus_i_eta_single'))
        if not np.isfinite(rcond) or rcond < 1e-13:
            raise ValueError(f'boundary system is numerically singular (rcond={rcond:.3e}); inspect resonance/geometry/resolution')

    def _kernel(self, distance, dot):
        if self.sigma == 0:
            return -np.log(distance)/(2*np.pi)
        if self.sigma > 0:
            return k0(self.wave*distance)/(2*np.pi)
        single = .25j*hankel1(0, self.wave*distance)
        double = .25j*self.wave*hankel1(1, self.wave*distance)*dot/distance
        return double+1j*self.eta*single

    def _self_block(self, panel, parameter, order):
        # Unwrap the target to this panel; both periodic seam endpoints allowed.
        boundary = panel.boundary
        period = boundary.b-boundary.a
        tau = parameter+period*np.round((panel.mid-parameter)/period)
        tau = float(np.clip(tau, panel.left, panel.right))
        t, weights, y, speed, normal, basis = panel.quadrature(panel.left, panel.right, order)
        delta = t-tau
        # Exact within-span Taylor polynomial, evaluated from one-sided jets.
        # Use midpoint polynomial jets to avoid selecting the other knot side.
        degree = boundary.curve.k
        jets = []
        for d in range(1, degree+1):
            jet = sum(boundary.curve(panel.mid, j)*(tau-panel.mid)**(j-d)/factorial(j-d)
                      for j in range(d, degree+1))
            jets.append(jet)
        divided = sum(delta[:, None]**(d-1)*jet/factorial(d)
                      for d, jet in enumerate(jets, start=1))
        ratio = np.linalg.norm(divided, axis=1)
        r = np.abs(delta)*ratio
        dot = np.sum((-delta[:, None]*divided)*normal, axis=1)
        safe = np.maximum(r, 1e-150)
        if self.sigma == 0:
            logcoefficient = np.full_like(r, -1/(2*np.pi))
            regular = np.zeros_like(r)
        elif self.sigma > 0:
            logcoefficient = -i0(self.wave*r)/(2*np.pi)
            regular = k0(self.wave*safe)/(2*np.pi)-logcoefficient*np.log(safe)
        else:
            s_log = -j0(self.wave*r)/(2*np.pi)
            d_log = -self.wave*j1(self.wave*r)*dot/(2*np.pi*safe)
            logcoefficient = d_log+1j*self.eta*s_log
            regular = self._kernel(safe, dot)-logcoefficient*np.log(safe)
        exact = r < 1e-14
        if np.any(exact):
            s_regular = -(np.log(self.wave/2)+np.euler_gamma)/(2*np.pi) if self.wave else 0.
            if self.sigma < 0:
                s_regular += .25j
                curvature = np.sum(jets[1]*normal[exact], axis=1)/(4*np.pi*np.dot(jets[0], jets[0]))
                regular[exact] = curvature+1j*self.eta*s_regular
            else:
                regular[exact] = s_regular
        q, w = rule(order)
        to_coeff = (2*np.arange(order)+1)[:, None]/2*legvander(q, order-1).T*w
        logweights = log_moments((tau-panel.mid)/panel.half, order-1)@to_coeff + np.log(panel.half)*w
        return panel.half*((logweights*logcoefficient*speed)@basis +
                          (w*(regular+logcoefficient*np.log(ratio))*speed)@basis)

    def _off_block(self, panel, target, order):
        pending = [(panel.left, panel.right, 0)]
        result = np.zeros(self.order, dtype=complex)
        while pending:
            a, b, depth = pending.pop()
            tprobe = np.linspace(a, b, 5)
            probe = panel.boundary.curve(tprobe)
            center = probe[2]
            radius = 1.25*np.max(np.linalg.norm(probe-center, axis=1))
            if np.linalg.norm(target-center) < 2.5*radius:
                if depth >= 45:
                    raise ValueError('target is on/too close to boundary; use boundary() for boundary values')
                mid = (a+b)/2
                pending.extend(((a, mid, depth+1), (mid, b, depth+1)))
                continue
            t, weights, y, speed, normal, basis = panel.quadrature(a, b, order)
            diff = target-y
            distance = np.linalg.norm(diff, axis=1)
            dot = np.sum(diff*normal, axis=1)
            result += (weights*speed*self._kernel(distance, dot))@basis
        return result

    def potential_matrix(self, points, components=None, parameters=None, quadrature_order=None, *, batched=True):
        """Batch regular far interactions; preserve adaptive near/self quadrature.

        batched=False retains the scalar reference path for numerical audits.
        Far classification uses exactly the root test of _off_block.
        """
        points = np.asarray(points)
        matrix = np.zeros((len(points), self.count), dtype=complex)
        if components is not None:
            components = np.asarray(components)
            parameters = np.asarray(parameters)
        order = quadrature_order or self.qorder
        for j, panel in enumerate(self.panels):
            self_mask = np.zeros(len(points), dtype=bool)
            if components is not None:
                period = panel.boundary.b-panel.boundary.a
                tau = parameters+period*np.round((panel.mid-parameters)/period)
                self_mask = ((components == panel.component) &
                             (tau >= panel.left-1e-13) & (tau <= panel.right+1e-13))
            probe = panel.boundary.curve(np.linspace(panel.left, panel.right, 5))
            center = probe[2]
            radius = 1.25*np.max(np.linalg.norm(probe-center, axis=1))
            far = (~self_mask) & (np.linalg.norm(points-center, axis=1) >= 2.5*radius)
            if not batched:
                far[:] = False
            columns = slice(j*self.order, (j+1)*self.order)
            indices = np.flatnonzero(far)
            if len(indices):
                _, weights, y, speed, normal, basis = panel.quadrature(panel.left, panel.right, order)
                weighted_basis = (weights*speed)[:, None]*basis
                for offset in range(0, len(indices), 512):
                    rows = indices[offset:offset+512]
                    difference = points[rows, None, :]-y[None, :, :]
                    distance = np.linalg.norm(difference, axis=2)
                    dot = np.sum(difference*normal, axis=2)
                    matrix[rows, columns] = self._kernel(distance, dot)@weighted_basis
            for i in np.flatnonzero(~far):
                if self_mask[i]:
                    row = self._self_block(panel, parameters[i], order)
                else:
                    row = self._off_block(panel, points[i], order)
                matrix[i, columns] = row
        if self.sigma < 0 and components is not None:
            # Interior jump of the double layer. At shared panel endpoints use
            # the average of the two density traces, rather than double counting.
            for i in range(len(points)):
                matches = []
                for j, panel in enumerate(self.panels):
                    if components[i] != panel.component:
                        continue
                    period = panel.boundary.b-panel.boundary.a
                    tau = parameters[i]+period*np.round((panel.mid-parameters[i])/period)
                    if panel.left-1e-13 <= tau <= panel.right+1e-13:
                        matches.append((j, panel, tau))
                for j, panel, tau in matches:
                    matrix[i, j*self.order:(j+1)*self.order] -= .5/len(matches)*panel.basis([tau])[0]
        return matrix

    def apply_layer(self, points, density, components=None, parameters=None,
                    quadrature_order=None, *, block_size=512):
        """Apply the layer without a target-by-boundary matrix.

        Far interactions reduce quadrature charges directly in bounded target
        blocks. Near/self interactions retain the reference adaptive quadrature.
        Only the output and optional boundary jump accumulators scale with the
        target count. This is the host implementation, not a GPU kernel.
        """
        points, density = np.asarray(points), np.asarray(density)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('points must have shape (n, 2)')
        if density.shape != (self.count,):
            raise ValueError('density must have one value per boundary unknown')
        if not isinstance(block_size, (int, np.integer)) or block_size <= 0:
            raise ValueError('block_size must be a positive integer')
        if (components is None) != (parameters is None):
            raise ValueError('supply both boundary components and parameters')
        if components is not None:
            components, parameters = np.asarray(components), np.asarray(parameters)
            if components.shape != (len(points),) or parameters.shape != (len(points),):
                raise ValueError('boundary metadata must match target count')
        result = np.zeros(len(points), dtype=complex)
        has_jump = self.sigma < 0 and components is not None
        if has_jump:
            jump = np.zeros(len(points), dtype=complex)
            matches = np.zeros(len(points), dtype=int)
        order = quadrature_order or self.qorder
        for j, panel in enumerate(self.panels):
            local_density = density[j*self.order:(j+1)*self.order]
            probe = panel.boundary.curve(np.linspace(panel.left, panel.right, 5))
            center = probe[2]
            radius = 1.25*np.max(np.linalg.norm(probe-center, axis=1))
            _, weights, y, speed, normal, basis = panel.quadrature(panel.left, panel.right, order)
            charges = (weights*speed)*(basis@local_density)
            for first in range(0, len(points), block_size):
                last = min(first+block_size, len(points))
                targets = points[first:last]
                self_mask = np.zeros(len(targets), dtype=bool)
                if components is not None:
                    period = panel.boundary.b-panel.boundary.a
                    tau = parameters[first:last]+period*np.round((panel.mid-parameters[first:last])/period)
                    self_mask = ((components[first:last] == panel.component) &
                                 (tau >= panel.left-1e-13) & (tau <= panel.right+1e-13))
                far = (~self_mask) & (np.linalg.norm(targets-center, axis=1) >= 2.5*radius)
                rows = np.flatnonzero(far)
                if len(rows):
                    difference = targets[rows, None, :]-y[None, :, :]
                    distance = np.linalg.norm(difference, axis=2)
                    dot = np.sum(difference*normal, axis=2)
                    result[first+rows] += self._kernel(distance, dot)@charges
                for i in np.flatnonzero(~far):
                    if self_mask[i]:
                        row = self._self_block(panel, parameters[first+i], order)
                    else:
                        row = self._off_block(panel, targets[i], order)
                    result[first+i] += row@local_density
                if has_jump:
                    rows = first+np.flatnonzero(self_mask)
                    if len(rows):
                        jump[rows] += panel.basis(parameters[rows])@local_density
                        matches[rows] += 1
        if has_jump:
            result -= .5*np.divide(jump, matches, out=np.zeros_like(jump), where=matches>0)
        return result

    def solve(self, boundary_data, *, source=None, source_plan=None, source_tolerance=1e-7):
        """boundary_data is a pair of callables (outer, inner); source is f(x)."""
        if len(boundary_data) != len(self.domain.boundaries):
            raise ValueError('supply one Dirichlet function per boundary')
        g = np.empty(self.count, dtype=complex)
        for component in range(len(self.domain.boundaries)):
            mask = self.components == component
            g[mask] = boundary_data[component](self.points[mask])
        if not np.all(np.isfinite(g)):
            raise ValueError('nonfinite boundary data')
        particular = None
        if source is not None:
            if source_plan is None or source_plan.domain is not self.domain:
                raise ValueError('provide a source plan for this domain')
            particular = source_plan.fit(source, self.sigma, tolerance=source_tolerance)
        h = g if particular is None else g-particular(self.points)
        rhs = np.r_[h, 0.] if self.sigma == 0 else h
        answer = lu_solve(self.factor, rhs)
        density = answer[:self.count]
        constant = answer[-1] if self.sigma == 0 else 0.j
        residual = np.linalg.norm(self.system@answer-rhs)/max(1., np.linalg.norm(rhs))
        return AnnulusSolution(self, density, constant, particular, float(residual))


@dataclass
class AnnulusSolution:
    plan: AnnulusPanelPlan
    density: np.ndarray
    constant: complex
    particular: object | None
    training_residual: float

    def interior(self, points, *, quadrature_order=None, block_size=512):
        points = np.asarray(points)
        values = self.plan.apply_layer(points, self.density, quadrature_order=quadrature_order,
                                       block_size=block_size)+self.constant
        return values if self.particular is None else values+self.particular(points)

    def boundary(self, component, parameters, *, quadrature_order=None, block_size=512):
        if component not in range(len(self.plan.domain.boundaries)):
            raise ValueError('invalid boundary component')
        parameters = np.atleast_1d(parameters)
        points = self.plan.domain.boundaries[component].curve(parameters)
        values = self.plan.apply_layer(points, self.density, np.full(len(points), component),
                                       parameters, quadrature_order, block_size=block_size)+self.constant
        return values if self.particular is None else values+self.particular(points)


def adaptive_dirichlet(domain, sigma, boundary_data, *, source=None, source_plan=None,
                       order=10, tolerance=1e-8, source_tolerance=1e-7,
                       max_refinements=4, callback=None):
    """Refine density panels using independent boundary residuals on both curves.

    Source and boundary gates are independent. This sampled boundary indicator
    is NOT a certified volume error estimate, especially near Helmholtz resonance.
    Returns (solution, history); exhaustion reports converged=False.
    """
    if tolerance <= 0 or max_refinements < 0:
        raise ValueError('invalid adaptive tolerances/limit')
    breaks = [b.knots.copy() for b in domain.boundaries]
    history = []
    for level in range(max_refinements+1):
        plan = AnnulusPanelPlan(domain, sigma, order=order, breaks=breaks)
        solution = plan.solve(boundary_data, source=source, source_plan=source_plan,
                              source_tolerance=source_tolerance)
        probes = np.unique(np.r_[rule(order+3)[0], -.999, -.99, .99, .999])
        indicators, magnitudes = [], []
        for panel in plan.panels:
            t = panel.mid+panel.half*probes
            expected = np.asarray(boundary_data[panel.component](panel.boundary.curve(t)))
            actual = solution.boundary(panel.component, t, quadrature_order=plan.qorder+8)
            indicators.append(float(np.max(np.abs(actual-expected))))
            magnitudes.append(float(np.max(np.abs(expected))))
        scale = max(1., max(magnitudes))
        indicators = np.asarray(indicators)/scale
        maximum = float(indicators.max())
        record = dict(level=level, **plan.stats, boundary_indicator=maximum,
                      converged=maximum <= tolerance, tolerance=tolerance,
                      panel_indicators=indicators.tolist())
        history.append(record)
        if callback:
            callback(solution, record)
        if record['converged'] or level == max_refinements:
            break
        marked = indicators > max(tolerance, .3*maximum)
        breaks = [[] for _ in domain.boundaries]
        for panel, split in zip(plan.panels, marked):
            breaks[panel.component].extend((panel.left, panel.right))
            if split:
                breaks[panel.component].append(panel.mid)
        breaks = [np.unique(b) for b in breaks]
    return solution, history
