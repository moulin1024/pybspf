"""Derivative-closed BSPF trial spaces on the unit interval.

Host-side construction only. The scalar basis is endpoint-nodalized QR BSPF;
clamped streamfunctions are integrals of its zero-mean tangential subspace.
No pressure spaces, 3D solvers, or time integration are constructed here.
"""

from numbers import Integral
import numpy as np
import scipy.linalg as la
from numpy.polynomial.legendre import leggauss
from ._qr_trial import qr_spline_fit


class ClosedBSPFLine:
    """Scalar (N), zero-trace (N-2), and clamped (N-3) BSPF spaces.

    Uses the slope benchmark's q=9, degree=13, at most 32 splines and
    14-point Taylor endpoint jets. ``values`` returns clamped values and
    derivatives; their first derivatives lie exactly in the tangent space.
    Arrays are NumPy float64 and may be uploaded to any compute backend.
    """

    def __init__(self, n, *, quadrature_order=24):
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 24:
            raise ValueError("Require an integer n >= 24.")
        if not isinstance(quadrature_order, Integral) or quadrature_order < 2:
            raise ValueError("Require quadrature_order >= 2.")
        self.x, _, self.spline, projection = qr_spline_fit(
            np.linspace(0.0, 1.0, n),
            q=9,
            n_basis=min(32, n - 1),
            degree=13,
            baseline_points=14,
            derivative_splines=True,
            lower=True,
        )
        self.n = n
        b = self.spline(self.x)
        nodal = np.eye(n)
        nodal[-1] = np.eye(n)[0] + (b[-1] - b[0]) @ projection
        inverse = la.solve(nodal, np.eye(n))
        self.residual = (np.eye(n)[:-1] - b[:-1] @ projection) @ inverse
        self.spline_coefficients = projection @ inverse
        self.frequency = np.fft.fftfreq(n - 1, d=1 / (n - 1))
        self.fourier = np.fft.fft(np.eye(n - 1), axis=0) / (n - 1)
        gx, gw = leggauss(quadrature_order)
        knots = np.unique(self.spline.t)
        points = np.concatenate(
            [(a + b) / 2 + (b - a) / 2 * gx for a, b in zip(knots[:-1], knots[1:])]
        )
        weights = np.concatenate(
            [(b - a) / 2 * gw for a, b in zip(knots[:-1], knots[1:])]
        )
        r = self.scalar_values(points, 0)[0]
        mass = r.T @ (weights[:, None] * r)
        chol = la.cholesky(mass[1:-1, 1:-1], lower=True)
        self.tangent_transform = np.eye(n)[:, 1:-1] @ la.solve_triangular(
            chol.T, np.eye(n - 2), lower=False
        )
        self.primitive = self.spline.antiderivative()
        self.fourier_tangent = self.fourier @ (self.residual @ self.tangent_transform)
        self.spline_tangent = self.spline_coefficients @ self.tangent_transform
        mean = self.integral_tangent(np.array([1.0]))[0]
        zero_mean = la.null_space(mean[None, :])
        primitive = self.integral_tangent(points) @ zero_mean
        normal_mass = primitive.T @ (weights[:, None] * primitive)
        chol = la.cholesky(normal_mass, lower=True)
        self.derivative_map = zero_mean @ la.solve_triangular(
            chol.T, np.eye(n - 3), lower=False
        )

    def scalar_values(self, points, order=1):
        points = np.asarray(points)
        e = np.exp(2j * np.pi * points[:, None] * self.frequency)
        return [
            np.real((e * (2j * np.pi * self.frequency) ** k) @ self.fourier)
            @ self.residual
            + self.spline.derivative(k)(points) @ self.spline_coefficients
            for k in range(order + 1)
        ]

    def integral_tangent(self, points):
        points = np.asarray(points)
        freq = self.frequency
        e = np.empty((len(points), len(freq)), complex)
        nz = freq != 0
        e[:, ~nz] = points[:, None]
        e[:, nz] = np.expm1(2j * np.pi * points[:, None] * freq[nz]) / (
            2j * np.pi * freq[nz]
        )
        return (
            np.real(e @ self.fourier_tangent)
            + (self.primitive(points) - self.primitive(0.0)) @ self.spline_tangent
        )

    def tangent_values(self, points, order=1):
        points = np.asarray(points)
        values = [r @ self.tangent_transform for r in self.scalar_values(points, order)]
        values[0][(points == 0.0) | (points == 1.0)] = 0.0
        return values

    def values(self, points, order=2):
        points = np.asarray(points)
        normal = self.integral_tangent(points) @ self.derivative_map
        normal[(points == 0.0) | (points == 1.0)] = 0.0
        return [normal] + [
            r @ self.derivative_map
            for r in self.tangent_values(points, max(order - 1, 0))[:order]
        ]
