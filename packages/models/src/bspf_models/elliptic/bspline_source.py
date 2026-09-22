"""Domain-only, stabilized local tensor B-spline extension with FFT inversion.

Compact PU patches reuse the quadrature-independent sampling and FFT machinery
of LocalSourcePlan. Interior data determine each patch; normalized jumps of the
highest spline derivative select a smooth continuation into unobserved spans.
No exterior values or manufactured-solution derivatives are requested.
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import svd

from .local_source import LocalSourcePlan


class BSplineSourcePlan(LocalSourcePlan):
    """Reusable quintic/septic extension plan on a non-body-fitted background.

    ``spans`` controls uniform knot intervals per patch coordinate. ``stability``
    weights derivative-jump rows relative to RMS data residuals. It is not an
    accuracy tolerance: ``fit`` still checks independent physical points and
    rejects unresolved sources. Geometry, patch factorizations and FFT plans
    are reused across right-hand sides. Host-side NumPy/SciPy + optional FINUFFT.
    """
    def __init__(self, domain, *, degree=7, spans=5, stability=1e-6,
                 grid_size=1025, radius=None, patch_samples=None, padding=1.5,
                 rcond=1e-12, eps=1e-12, sample_grid=129,
                 interpolation_order=12):
        if degree not in (5, 7):
            raise ValueError('B-spline degree must be 5 or 7')
        if int(spans) != spans or spans < 2:
            raise ValueError('spans must be an integer >=2')
        if not np.isfinite(stability) or not 0 < stability < 1:
            raise ValueError('stability must lie in (0,1)')
        self.spans, self.stability = int(spans), float(stability)
        breaks = np.linspace(-1., 1., self.spans+1)
        knots = np.r_[np.repeat(-1., degree), breaks, np.repeat(1., degree)]
        self._spline = BSpline(knots, np.eye(len(knots)-degree-1), degree,
                               extrapolate=False)
        # On simple interior knots the p-th derivative is piecewise constant.
        # Penalizing its jumps leaves global degree-p polynomials unpenalized.
        derivative = self._spline.derivative(degree)
        delta = (breaks[1]-breaks[0])*.01
        jump = derivative(breaks[1:-1]+delta)-derivative(breaks[1:-1]-delta)
        jump /= np.linalg.norm(jump, axis=1)[:, None]
        n = self._spline.c.shape[0]
        self._penalty = np.vstack((np.kron(jump, np.eye(n)), np.kron(np.eye(n), jump)))
        # More samples than spline coefficients even after clipping a half patch.
        count = patch_samples or 4*(degree+self.spans)
        super().__init__(domain, degree=degree, grid_size=grid_size, radius=radius,
                         patch_samples=count, padding=padding, rcond=rcond,
                         eps=eps, sample_grid=sample_grid,
                         interpolation_order=interpolation_order)
        self.stats.update(backend='bspline_pu_fft', spans=self.spans,
                          stability=self.stability, patch_columns=n*n,
                          patch_samples=count)

    def _basis(self, normalized):
        x = self._spline(np.clip(normalized[:, 0], -1., 1.))
        y = self._spline(np.clip(normalized[:, 1], -1., 1.))
        return (x[:, :, None]*y[:, None, :]).reshape(len(normalized), -1)

    def _factor(self, matrix, rcond):
        scale = np.sqrt(len(matrix))
        augmented = np.vstack((matrix/scale, self.stability*self._penalty))
        u, singular, vh = svd(augmented, full_matrices=False)
        keep = singular > rcond*singular[0]
        # Zero RHS on penalty rows; keep only the physical-data application.
        return u[:len(matrix), keep].T/scale, singular[keep], vh[keep].T
