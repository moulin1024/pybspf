"""Domain-only local polynomial partition-of-unity extension + FFT/NUFFT.

Local TSVD fits are geometry-only and reusable. This experimental backend does
not certify continuous errors: every returned source is independently audited.
FINUFFT is an optional dependency, imported only when this backend is selected.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.polynomial.chebyshev import chebvander2d
from scipy.fft import fft2, fftshift, fftfreq
from scipy.linalg import svd
from scipy.spatial import cKDTree
from scipy.special import betainc, comb


def _step(x):
    return betainc(12, 12, np.clip(x, 0., 1.))


class LocalSourcePlan:
    def __init__(self, domain, *, grid_size=257, radius=None, degree=12,
                 patch_samples=None, padding=1.5, rcond=1e-12, eps=1e-12,
                 sample_grid=129, interpolation_order=12):
        import finufft
        if grid_size < 33 or grid_size % 2 != 1 or degree < 2 or padding <= 1:
            raise ValueError('require odd grid_size>=33, degree>=2 and padding>1')
        if not 0 < rcond < 1 or not 0 < eps < 1:
            raise ValueError('invalid local truncation/NUFFT tolerance')
        start = perf_counter()
        self.domain, self.n, self.eps = domain, int(grid_size), eps
        self.center = domain.bounds.mean(axis=1)
        self.lengths = np.ptp(domain.bounds, axis=1)*padding
        self.origin = self.center-self.lengths/2
        radius = float(radius or .17*min(np.ptp(domain.bounds, axis=1)))
        if radius <= 0 or radius > .45*min(self.lengths-np.ptp(domain.bounds, axis=1)):
            raise ValueError('patch radius must be positive and fit inside the periodic padding')
        self.radius, self.degree = radius, int(degree)
        axes = [self.origin[j]+np.arange(self.n)*self.lengths[j]/self.n for j in (0, 1)]
        xx, yy = np.meshgrid(*axes, indexing='ij')
        self.grid = np.column_stack((xx.ravel(), yy.ravel()))
        self.inside = domain.contains(self.grid)
        if sample_grid < interpolation_order+1 or sample_grid % 2 != 1 or sample_grid > grid_size:
            raise ValueError('require odd sample_grid between interpolation_order+1 and grid_size')
        if interpolation_order < 4 or interpolation_order % 2:
            raise ValueError('require even interpolation_order>=4')
        self.sample_n = int(sample_grid)
        sx, sy = np.meshgrid(*[self.origin[j]+np.arange(sample_grid)*self.lengths[j]/sample_grid
                               for j in (0, 1)], indexing='ij')
        coarse = np.column_stack((sx.ravel(), sy.ravel()))
        self.sample_inside = domain.contains(coarse)
        self.bulk_count = int(self.sample_inside.sum())
        pieces = [coarse[self.sample_inside]]
        grid_tree = cKDTree(self.grid)
        self._terms = (np.indices((degree+1, degree+1)).sum(axis=0) <= degree).ravel()
        count = patch_samples or 2*degree+6
        if count < degree+2:
            raise ValueError('too few local sampling nodes')
        q = np.cos(np.pi*(np.arange(count)+.5)/count)
        q = q[abs(q)>1e-10]  # omit the center line: patch centers lie on the boundary
        px, py = np.meshgrid(q, q, indexing='ij')
        stencil = np.column_stack((px.ravel(), py.ravel()))
        stencil = stencil[np.linalg.norm(stencil, axis=1) < .99]
        self.patches = []
        weight_sum = np.zeros(len(self.grid))
        ranks = []
        all_centers = []
        boundary_samples = []
        offset = self.bulk_count
        # Boundary centers follow arclength, independent of the source values.
        for b in domain.boundaries:
            t = np.linspace(b.a, b.b, 4097)
            curve = b.curve(t)
            boundary_samples.append(curve)
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
            centers = b.curve(np.interp(np.arange(0., arc[-1], radius*.5), arc, t))
            for center in centers:
                all_centers.append(center)
                local = center+radius*stencil
                local = local[domain.contains(local)]
                matrix = self._basis((local-center)/radius)
                if len(local) < matrix.shape[1]:
                    raise ValueError('insufficient physical samples in local patch; increase patch_samples or reduce degree')
                left, singular, right = self._factor(matrix, rcond)
                indices = np.asarray(grid_tree.query_ball_point(center, radius), dtype=int)
                normalized = (self.grid[indices]-center)/radius
                squared = np.sum(normalized**2, axis=1)
                weights = np.maximum(1-squared, 0.)**8
                evaluation = self._basis(normalized) if len(indices)*matrix.shape[1] < 250000 else None
                self.patches.append((slice(offset, offset+len(local)), left,
                                     singular, right, indices, weights, evaluation, center))
                offset += len(local)
                pieces.append(local)
                weight_sum[indices] += weights
                ranks.append(len(singular))
        self.points = np.vstack(pieces)
        # Smooth zero patches, whose entire support lies outside the physical
        # domain, replace a distance-to-boundary taper. The latter inherits knot
        # regularity and creates unnecessary Fourier tails even for constant f.
        sampled_boundary = np.vstack(boundary_samples)
        center_tree = cKDTree(sampled_boundary)
        zero_axes = [np.arange(self.origin[j], self.origin[j]+self.lengths[j], radius*.5) for j in (0, 1)]
        zx, zy = np.meshgrid(*zero_axes, indexing='ij')
        zeros = np.column_stack((zx.ravel(), zy.ravel()))
        zeros = zeros[~domain.contains(zeros)]
        distances = center_tree.query(zeros)[0]
        margin = 2*max(np.linalg.norm(np.diff(c, axis=0), axis=1).max() for c in boundary_samples)
        zero_sum = np.zeros(len(self.grid))
        for center, distance in zip(zeros, distances-margin):
            if distance < .8*radius:
                continue
            support = min(radius, .75*distance)
            indices = np.asarray(grid_tree.query_ball_point(center, support), dtype=int)
            squared = np.sum(((self.grid[indices]-center)/support)**2, axis=1)
            zero_sum[indices] += np.maximum(1-squared, 0.)**8
        if np.any(zero_sum[self.inside] != 0):
            raise ValueError('zero extension patch intersects the physical grid')
        boundary_weights = np.zeros(len(sampled_boundary))
        for center in all_centers:
            squared = np.sum(((sampled_boundary-center)/radius)**2, axis=1)
            boundary_weights += np.where(squared < 1, np.maximum(1-squared, 0.)**8, 0.)
        threshold = .0001*boundary_weights.min()
        if threshold < 1e-8:
            raise ValueError('data patches do not cover the boundary')
        self._blend = 1-_step(weight_sum/threshold)
        self._weight_sum = weight_sum+zero_sum
        collar_end = (~self.inside) & (weight_sum > 1e-14) & (weight_sum < 1e-4)
        if np.any(zero_sum[collar_end] < 1e-4):
            raise ValueError('zero patches do not cover the end of the extension collar')
        # Resample only the deep interior. Near a boundary the patch fit already
        # supplies the field, so every interpolation stencil must stay physical.
        self._bulk_targets = np.flatnonzero(self.inside & (self._blend > 0))
        coordinates = (self.grid[self._bulk_targets]-self.origin)/self.lengths*sample_grid
        base = np.floor(coordinates).astype(int)
        offsets = np.arange(-interpolation_order//2+1, interpolation_order//2+1)
        barycentric = (-1.)**np.arange(interpolation_order)*comb(interpolation_order-1, np.arange(interpolation_order))
        indices, weights = [], []
        for axis in (0, 1):
            delta = coordinates[:, axis, None]-base[:, axis, None]-offsets
            exact = abs(delta) < 1e-12
            w = np.divide(barycentric, delta, out=np.zeros_like(delta), where=~exact)
            rows = np.any(exact, axis=1)
            w[rows] = exact[rows]
            w /= w.sum(axis=1)[:, None]
            indices.append((base[:, axis, None]+offsets) % sample_grid)
            weights.append(w)
        physical = self.sample_inside.reshape(sample_grid, sample_grid)
        for i in range(interpolation_order):
            if not np.all(physical[indices[0][:, i, None], indices[1]]):
                raise ValueError('bulk interpolation stencil leaves domain; increase sample_grid or patch radius')
        self._bulk_indices, self._bulk_weights = indices, weights
        validation = [domain.sample(min(grid_size, 97)+3, shift=.317)]
        for component, b in enumerate(domain.boundaries):
            t = b.a+(np.arange(4*grid_size+3)+.371)/(4*grid_size+3)*(b.b-b.a)
            for d in (1e-6, 1e-3):
                points = b.curve(t)-d*radius*b.normal(t, 1 if component == 0 else -1)
                validation.append(points[domain.contains(points)])
        self.validation = np.vstack(validation)
        wave = fftshift(fftfreq(self.n)*self.n)
        a, b = np.meshgrid(wave, wave, indexing='ij')
        self.frequencies = 2*np.pi*np.stack((a/self.lengths[0], b/self.lengths[1]), axis=-1)
        self._plans = {}
        self.stats = dict(backend='local_pu_fft', grid_size=self.n, columns=self.n**2,
                          samples=len(self.points), bulk_samples=self.bulk_count, sample_grid=sample_grid,
                          interpolation_order=interpolation_order,
                          patches=len(self.patches), degree=degree, radius=radius,
                          min_local_rank=min(ranks), max_local_rank=max(ranks),
                          padding=padding, rcond=rcond, setup_seconds=perf_counter()-start)

    def _factor(self, matrix, rcond):
        """Physical-data application of a patch solve; subclasses may stabilize it."""
        u, singular, vh = svd(matrix, full_matrices=False)
        keep = singular > rcond*singular[0]
        return u[:, keep].conj().T, singular[keep], vh[keep].conj().T

    def _basis(self, normalized):
        return chebvander2d(normalized[:, 0], normalized[:, 1], [self.degree, self.degree])[:, self._terms]

    def _evaluate(self, coefficients, points):
        import finufft
        # Only the two fixed iteration grids are cached. Other queries are bounded
        # one-shot plans; no unbounded cache of arbitrary target arrays.
        key = 'fit' if points is self.points else 'validation' if points is self.validation else None
        plan = self._plans.get(key) if key is not None else None
        if plan is None:
            x = 2*np.pi*(np.asarray(points)-self.origin)/self.lengths
            plan = finufft.Plan(2, (self.n, self.n), eps=self.eps, isign=1, dtype='complex128')
            plan.setpts(np.ascontiguousarray(x[:, 0]), np.ascontiguousarray(x[:, 1]))
            if key is not None:
                self._plans[key] = plan
        return plan.execute(np.ascontiguousarray(coefficients, dtype=complex))

    def fit(self, source, sigma, *, tolerance=1e-7):
        if np.isnan(tolerance) or tolerance <= 0 or not np.isfinite(sigma):
            raise ValueError('require positive tolerance (or infinity for diagnostics) and finite sigma')
        values = np.asarray(source(self.points))
        if values.shape != (len(self.points),) or not np.all(np.isfinite(values)):
            raise ValueError('source must return one finite value per physical point')
        numerator = np.zeros(len(self.grid), dtype=complex)
        for sl, left, singular, right, indices, weights, basis, center in self.patches:
            c = right@((left@values[sl])/singular)
            if basis is not None:
                numerator[indices] += weights*(basis@c)
            else:
                for first in range(0, len(indices), 2048):
                    selected = indices[first:first+2048]
                    b = self._basis((self.grid[selected]-center)/self.radius)
                    numerator[selected] += weights[first:first+2048]*(b@c)
        extended = np.divide(numerator, self._weight_sum, out=np.zeros_like(numerator),
                             where=self._weight_sum>1e-14)
        coarse = np.zeros(self.sample_n**2, dtype=complex)
        coarse[self.sample_inside] = values[:self.bulk_count]
        coarse = coarse.reshape(self.sample_n, self.sample_n)
        ix, iy = self._bulk_indices
        wx, wy = self._bulk_weights
        interpolated = np.zeros(len(ix), dtype=complex)
        for j in range(wx.shape[1]):
            interpolated += wx[:, j]*np.sum(coarse[ix[:, j, None], iy]*wy, axis=1)
        target = self._bulk_targets
        beta = self._blend[target]
        extended[target] = (1-beta)*extended[target]+beta*interpolated
        coefficients = fftshift(fft2(extended.reshape(self.n, self.n), workers=-1))/self.n**2
        reference = np.asarray(source(self.validation))
        if reference.shape != (len(self.validation),) or not np.all(np.isfinite(reference)):
            raise ValueError('invalid source validation values')
        error = np.max(abs(self._evaluate(coefficients, self.validation)-reference))
        scale = max(float(np.max(abs(values))), float(np.max(abs(reference))), 1e-300)
        relative = float(error/scale)
        if not np.isfinite(relative) or relative > tolerance:
            raise ValueError(f'local source validation failed: {relative:.3e} > {tolerance:.3e}; refine grid/patches/degree')
        stats = dict(self.stats, validation_relative_max=relative, validation_absolute_max=float(error),
                     tolerance=tolerance, validation_samples=len(self.validation),
                     exterior_relative_max=float(np.max(abs(extended[~self.inside]))/scale))
        return LocalParticular(self, coefficients, float(sigma), stats)


@dataclass
class LocalParticular:
    plan: LocalSourcePlan
    coefficients: np.ndarray
    sigma: float
    stats: dict

    def source_values(self, points):
        return self.plan._evaluate(self.coefficients, points)

    def __call__(self, points):
        norm = np.sum(self.plan.frequencies**2, axis=2)
        denom = norm+self.sigma
        resonant = abs(denom) < 1e-11*np.maximum(1., norm+abs(self.sigma))
        c = np.divide(self.coefficients, denom, out=np.zeros_like(self.coefficients), where=~resonant)
        result = self.plan._evaluate(c, points)
        x = np.asarray(points)-self.plan.origin
        for i, j in np.argwhere(resonant):
            if norm[i, j] == 0:
                mode = -np.sum(x*x, axis=1)/4
            else:
                dot = x@self.plan.frequencies[i, j]
                mode = 1j*dot/(2*norm[i, j])*np.exp(1j*dot)
            result += self.coefficients[i, j]*mode
        return result
