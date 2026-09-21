"""FFT Fourier extension: Algorithm 1 of Matthysen--Huybrechs (1706.04848).

CPU NumPy/SciPy implementation. No full sampling matrix or full sampling SVD
is formed. The randomized range and compressed SVD concern (A A* - I) A.
Axes are x, y (indexing='ij'); coefficients use centered integer frequencies.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.linalg as la


class FourierExtensionPlan:
    """Oversampled, restricted unitary DFT on a square periodic box.

    ``mask`` has shape (grid_size, grid_size); only its true entries are data.
    ``modes`` is odd and counts frequencies PER AXIS. ``cutoff`` is an
    absolute singular-value threshold for PA, whose norm is at most 0.385.
    Construction is reusable across right hand sides, including complex data.
    """

    def __init__(self, mask, modes, *, half_width=1.2, cutoff=1e-12,
                 range_tolerance=None, block_size=32, seed=0, max_rank=None):
        start = perf_counter()
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
            raise ValueError("mask must be a square two-dimensional array")
        if not isinstance(modes, (int, np.integer)) or modes < 1 or modes % 2 != 1:
            raise ValueError("modes must be a positive odd integer")
        if modes > mask.shape[0] or not mask.any():
            raise ValueError("Require modes <= grid size and a nonempty mask")
        if not np.isfinite(half_width) or half_width <= 0:
            raise ValueError("half_width must be positive and finite")
        if not 1e-15 <= cutoff < 0.1:
            raise ValueError("cutoff must lie in [1e-15, 0.1)")
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.mask = mask.copy()
        self.modes, self.grid_size = modes, mask.shape[0]
        self.half_width, self.cutoff = float(half_width), float(cutoff)
        self.frequencies = np.arange(-(modes // 2), modes // 2 + 1)
        self.indices = self.frequencies % self.grid_size
        self.size, self.samples = modes**2, int(mask.sum())
        self.normalization = float(self.grid_size)  # sqrt(number of box samples)
        axis = np.linspace(-half_width, half_width, self.grid_size, endpoint=False)
        self.points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1)[mask]
        self.range_tolerance = cutoff * 0.05 if range_tolerance is None else range_tolerance
        if not 0 < self.range_tolerance < cutoff:
            raise ValueError("range_tolerance must be positive and below cutoff")
        limit = min(self.samples, self.size)
        if max_rank is not None:
            if not isinstance(max_rank, (int, np.integer)) or max_rank < 1:
                raise ValueError("max_rank must be a positive integer")
            limit = min(limit, max_rank)
        rng = np.random.default_rng(seed)
        # Independent Gaussian probes monitor the omitted range. Dividing by
        # sqrt(probe count), not sqrt(input dimension), estimates Frobenius error.
        # A unitary change from complex exponentials to real tensor cos/sin
        # coordinates leaves singular values and Algorithm 1 unchanged, while
        # permitting the expensive QR/SVD to use real arithmetic.
        probes = rng.normal(size=(self.size, 8))
        check = self.plunge(self._to_fourier(probes)).real
        q = np.empty((self.samples, 0))
        residual_estimate = float(la.norm(check) / np.sqrt(check.shape[1]))
        while residual_estimate > self.range_tolerance and q.shape[1] < limit:
            width = min(block_size, limit - q.shape[1])
            omega = rng.normal(size=(self.size, width))
            y = self.plunge(self._to_fourier(omega)).real
            # Two projections are essential when capturing near-cutoff modes.
            for _ in range(2):
                y -= q @ (q.conj().T @ y)
            u, s, _ = la.svd(y, full_matrices=False)
            keep = s > 5e-16
            if not np.any(keep):
                break
            new = u[:, keep]
            # The SVD normalizes tiny columns, amplifying their roundoff-sized
            # components along q. Reorthogonalize AFTER this normalization too.
            for _ in range(2):
                new -= q @ (q.conj().T @ new)
            new, _ = la.qr(new, mode="economic")
            q = np.column_stack((q, new))
            residual_estimate = float(la.norm(check - q @ (q.conj().T @ check)) / np.sqrt(check.shape[1]))
        # A failed rank budget is explicit; do not silently substitute a dense solve.
        if residual_estimate > max(self.range_tolerance * 5, 2e-14):
            raise RuntimeError(
                f"Plunge range unresolved: rank={q.shape[1]}, probe residual="
                f"{residual_estimate:.3g}; increase max_rank or cutoff"
            )
        if q.shape[1]:
            # PA* Q = A* (A A* - I) Q, evaluated in FFT blocks.
            compressed = np.empty((q.shape[1], self.size))
            for begin in range(0, q.shape[1], block_size):
                end = min(begin + block_size, q.shape[1])
                compressed[begin:end] = self._to_real_modes(self.adjoint(self.project(q[:, begin:end]))).real.T
            u, s, vh = la.svd(compressed, full_matrices=False, overwrite_a=True)
            keep = s > cutoff
            self.left = q @ u[:, keep]
            self.singular_values = s[keep]
            self.right = vh[keep].conj().T
        else:
            self.left, self.right = q, np.empty((self.size, 0))
            self.singular_values = np.empty(0)
        self.diagnostics = dict(
            modes=modes, grid_size=self.grid_size, samples=self.samples,
            range_rank=q.shape[1], plunge_rank=len(self.singular_values),
            range_probe_residual=residual_estimate,
            factor_bytes=self.left.nbytes + self.right.nbytes + self.singular_values.nbytes,
            setup_seconds=perf_counter() - start,
        )

    def _modal_transform(self, coefficients, inverse=False):
        c = np.asarray(coefficients)
        tail = c.shape[1:]
        a = c.reshape((self.modes, self.modes) + tail).astype(complex)
        k = np.arange(1, self.modes // 2 + 1)
        center = self.modes // 2
        for axis in (0, 1):
            x = np.moveaxis(a, axis, 0)
            out = np.empty_like(x)
            if inverse:
                out[0] = x[center]
                out[2*k-1] = (x[center+k] + x[center-k]) / np.sqrt(2)
                out[2*k] = 1j * (x[center+k] - x[center-k]) / np.sqrt(2)
            else:
                out[center] = x[0]
                out[center+k] = (x[2*k-1] - 1j*x[2*k]) / np.sqrt(2)
                out[center-k] = (x[2*k-1] + 1j*x[2*k]) / np.sqrt(2)
            a = np.moveaxis(out, 0, axis)
        return a.reshape((self.size,) + tail)

    def _to_fourier(self, coefficients):
        return self._modal_transform(coefficients)

    def _to_real_modes(self, coefficients):
        return self._modal_transform(coefficients, inverse=True)

    def forward(self, coefficients):
        """A c: normalized Fourier samples, for a vector or column batch."""
        c = np.asarray(coefficients)
        if c.ndim not in (1, 2) or c.shape[0] != self.size:
            raise ValueError("Expected coefficients of shape (modes**2[, batch])")
        tail = c.shape[1:]
        box = np.zeros((self.grid_size, self.grid_size) + tail, complex)
        box[self.indices[:, None], self.indices[None, :]] = c.reshape((self.modes, self.modes) + tail)
        return np.fft.ifft2(box, axes=(0, 1), norm="ortho")[self.mask]

    def adjoint(self, values):
        """A* b, with zero insertion in physical space and FFT restriction."""
        b = np.asarray(values)
        if b.ndim not in (1, 2) or b.shape[0] != self.samples:
            raise ValueError("Expected values of shape (mask.sum()[, batch])")
        box = np.zeros((self.grid_size, self.grid_size) + b.shape[1:], complex)
        box[self.mask] = b
        spectrum = np.fft.fft2(box, axes=(0, 1), norm="ortho")
        return spectrum[self.indices[:, None], self.indices[None, :]].reshape((self.size,) + b.shape[1:])

    def project(self, values):
        """P = A A* - I (not an orthogonal projector)."""
        return self.forward(self.adjoint(values)) - values

    def plunge(self, coefficients):
        return self.project(self.forward(coefficients))

    def solve(self, values):
        """Extend physical function samples. No exterior function values used."""
        start = perf_counter()
        values = np.asarray(values)
        if values.shape != (self.samples,) or not np.all(np.isfinite(values)):
            raise ValueError("Expected finite function values at mask points")
        b = values / self.normalization
        pb = self.project(b)
        if np.isrealobj(values):
            pb = pb.real
        y = self._to_fourier(self.right @ ((self.left.T @ pb) / self.singular_values))
        remainder = b - self.forward(y)
        if np.isrealobj(values):
            remainder = remainder.real
        c = y + self.adjoint(remainder)
        residual = self.normalization * self.forward(c) - values
        return FourierExtension(self, c, dict(
            sample_relative_residual=float(la.norm(residual) / max(la.norm(values), 1e-300)),
            sample_max_residual=float(np.max(np.abs(residual))),
            coefficient_norm=float(la.norm(c)), solve_seconds=perf_counter() - start,
        ), is_real=bool(np.isrealobj(values)))

    def evaluate(self, coefficients, points, derivative=(0, 0), chunk_size=256):
        """Off-grid analytic Fourier evaluation (chunked direct sums, not NUFFT)."""
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or not np.all(np.isfinite(p)):
            raise ValueError("Expected finite points of shape (count, 2)")
        if len(derivative) != 2 or any(int(d) != d or d < 0 for d in derivative):
            raise ValueError("derivative must be a pair of nonnegative integers")
        c = np.asarray(coefficients).reshape(self.modes, self.modes)
        k = np.pi * self.frequencies / self.half_width
        c = c * (1j * k[:, None])**derivative[0] * (1j * k[None, :])**derivative[1]
        out = np.empty(len(p), complex)
        for begin in range(0, len(p), chunk_size):
            stop = begin + chunk_size
            phase = (p[begin:stop] + self.half_width)[:, :, None] * k
            ex, ey = np.exp(1j * phase[:, 0]), np.exp(1j * phase[:, 1])
            out[begin:stop] = np.sum((ex @ c) * ey, axis=1)
        return out


@dataclass
class FourierExtension:
    plan: FourierExtensionPlan
    coefficients: np.ndarray
    diagnostics: dict
    is_real: bool = False

    def evaluate(self, points, derivative=(0, 0)):
        values = self.plan.evaluate(self.coefficients, points, derivative)
        return values.real if self.is_real else values
