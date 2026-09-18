"""QR-constrained BSPF lines extracted from the 20260917 NS benchmark.

Keep this construction separate from BSPF1D: it uses an unregularized QR spline
fit, not its KKT defaults. Taylor jets reproduce the benchmark; the optional
Chebyshev jets port the improved estimator from the differentiation work.
"""

from math import factorial
from numbers import Integral

import numpy as np
import scipy.linalg as la
from scipy.interpolate import BSpline


def real_array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real.")
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


class PressureLine:
    """FFT plus low-rank D1, its endpoint elimination, and two null modes."""

    def __init__(
        self,
        x,
        *,
        q,
        n_basis,
        degree,
        baseline_points,
        endpoint_method="taylor",
        chebyshev_modes=12,
        endpoint_regularization=1e-12,
    ):
        x = real_array(x, "grid")
        if x.ndim != 1 or x.size < 5:
            raise ValueError(
                "Each grid must be one-dimensional with at least five nodes."
            )
        h = (x[-1] - x[0]) / (x.size - 1)
        if h <= 0 or not np.allclose(np.diff(x), h, rtol=1e-10, atol=0):
            raise ValueError("Each grid must be strictly increasing and uniform.")
        for name, value in dict(
            q=q, n_basis=n_basis, degree=degree, baseline_points=baseline_points
        ).items():
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not (q <= degree + 1 <= n_basis < x.size) or n_basis <= 2 * q:
            raise ValueError(
                "Require q <= degree+1 <= n_basis < grid size and n_basis > 2*q."
            )
        if not q <= baseline_points <= x.size:
            raise ValueError("Require q <= baseline_points <= grid size.")
        if endpoint_method not in ("taylor", "chebyshev"):
            raise ValueError("endpoint_method must be 'taylor' or 'chebyshev'.")
        if endpoint_method == "chebyshev":
            if (
                isinstance(chebyshev_modes, bool)
                or not isinstance(chebyshev_modes, Integral)
                or not q <= chebyshev_modes <= baseline_points
            ):
                raise ValueError("Require q <= chebyshev_modes <= baseline_points.")
            if not np.isfinite(endpoint_regularization) or endpoint_regularization < 0:
                raise ValueError(
                    "endpoint_regularization must be finite and nonnegative."
                )
        self.x = x.copy()
        self.weights = np.full(x.size, h)
        self.weights[[0, -1]] *= 0.5
        interior = np.linspace(x[0], x[-1], n_basis - degree + 1)[1:-1]
        knots = np.r_[
            np.repeat(x[0], degree + 1), interior, np.repeat(x[-1], degree + 1)
        ]
        spline = BSpline(knots, np.eye(n_basis), degree)
        B = spline(x)
        C = np.vstack(
            [spline(x[0], nu=k) for k in range(q)]
            + [spline(x[-1], nu=k) for k in range(q)]
        )
        scales = 1 / np.max(abs(C), axis=1)
        Q, R = la.qr((C * scales[:, None]).T, mode="full")
        Q1, Q2 = Q[:, : 2 * q], Q[:, 2 * q :]
        T = Q1 @ la.solve_triangular(R[: 2 * q, : 2 * q].T, np.diag(scales), lower=True)
        BW = B.T * self.weights
        H = BW @ B
        H22 = Q2.T @ H @ Q2
        factor = la.cho_factor((H22 + H22.T) / 2)
        F0 = Q2 @ la.cho_solve(factor, Q2.T @ BW)
        J = T - Q2 @ la.cho_solve(factor, Q2.T @ H @ T)
        if endpoint_method == "chebyshev":
            jets = chebyshev_jets(
                x, q, baseline_points, chebyshev_modes, endpoint_regularization
            )
        else:
            j = np.arange(baseline_points, dtype=float)
            left = np.column_stack([j**k / factorial(k) for k in range(q)])
            right = np.column_stack([(-j) ** k / factorial(k) for k in range(q)])
            jets = np.zeros((2 * q, x.size))
            units = (h ** np.arange(q))[:, None]
            jets[:q, :baseline_points] = np.linalg.pinv(left, rcond=1e-14) / units
            jets[q:, -baseline_points:] = (
                np.linalg.pinv(right, rcond=1e-14)[:, ::-1] / units
            )
        self.P = F0 + J @ jets
        self.mult = 2j * np.pi * np.fft.fftfreq(x.size - 1, d=h)
        if (x.size - 1) % 2 == 0:
            self.mult[(x.size - 1) // 2] = 0
        self.low = spline(x, nu=1) - self._fourier(B)
        self.D = self.apply(np.eye(x.size))
        mask = np.ones(x.size)
        mask[[0, -1]] = 0
        H = self.D @ (mask[:, None] * self.D)
        ends = [0, x.size - 1]
        self.Hei = H[ends, 1:-1]
        self.Ei = la.solve(H[np.ix_(ends, ends)], np.eye(2))
        self.C = H[1:-1, ends] @ self.Ei
        A = H[1:-1, 1:-1] - self.C @ self.Hei
        lam, V = la.eig(A)
        order = np.argsort(abs(lam))
        self.lam, self.V = lam[order], V[:, order]
        self.Vi = la.solve(self.V, np.eye(x.size - 2))
        scale = max(abs(lam).max(), 1.0)
        if max(abs(self.lam[:2])) > 1e-8 * scale or abs(self.lam[2]) < 1e-8 * scale:
            raise ValueError("BSPF line must have exactly two null eigenmodes.")
        Z = la.null_space(self.D[1:-1], rcond=1e-12)
        if Z.shape[1] != 2:
            raise ValueError("Interior BSPF gradient must have nullity two.")
        z = Z @ (Z.T @ ((-1.0) ** np.arange(x.size)))
        z -= np.dot(self.weights, z) / self.weights.sum()
        norm = np.sqrt(np.dot(self.weights, z * z) / self.weights.sum())
        if norm < 1e-8:
            raise ValueError("Unable to construct the second gradient-null mode.")
        self.Z = np.column_stack([np.ones(x.size), z / norm])

    def _fourier(self, values):
        d = np.fft.ifft(np.fft.fft(values[:-1], axis=0) * self.mult[:, None], axis=0)
        return np.concatenate([d, d[:1]], axis=0).real

    def apply(self, values, axis=0):
        v = np.moveaxis(values, axis, 0)
        shape = v.shape
        v = v.reshape(shape[0], -1)
        out = self._fourier(v) + self.low @ (self.P @ v)
        return np.moveaxis(out.reshape(shape), 0, axis)


def chebyshev_jets(x, order, points, modes, alpha):
    """Local Chebyshev endpoint map; NumPy port of bspf_jax/endpoints.py.

    Use an augmented QR fit, a normalized fourth-power modal penalty, and
    exact endpoint values. This map is linear and can be folded into P.
    """
    if order == 1:
        return np.eye(x.size)[[0, -1]]
    xi = np.linspace(-1.0, 1.0, points)
    V = np.polynomial.chebyshev.chebvander(xi, modes - 1)
    index = np.arange(modes, dtype=float)
    penalty = (index / max(1, modes - 1)) ** 4
    penalty[:2] = 0
    augmented = np.vstack([V, np.sqrt(alpha) * np.diag(penalty)])
    Q, R = la.qr(augmented, mode="economic")
    projector = la.solve_triangular(R, Q[:points].T)
    endpoint = np.ones(modes)
    left = np.empty((order, points))
    width = x[points - 1] - x[0]
    for k in range(order):
        if k:
            endpoint *= (index**2 - (k - 1) ** 2) / (2 * k - 1)
        left[k] = (2 / width) ** k * (((-1.0) ** (index - k) * endpoint) @ projector)
    left[0] = 0
    left[0, 0] = 1
    jets = np.zeros((2 * order, x.size))
    jets[:order, :points] = left
    jets[order:, -points:] = left[:, ::-1] * (-1.0) ** np.arange(order)[:, None]
    return jets
