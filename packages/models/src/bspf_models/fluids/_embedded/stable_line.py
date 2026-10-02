"""Stable independent polynomial + Fourier space, evaluated through Chebyshev tails.

Subtracting the low-degree Chebyshev projection from each Fourier function
preserves span while avoiding spline/Fourier cancellation. Analytic Bessel
coefficients give the tails without subtracting nearly equal sampled values.
"""

import numpy as np
import scipy.linalg as la
from scipy.special import jv
from scipy.interpolate import BSpline
from numpy.polynomial.chebyshev import chebval, chebder
from .basis import gauss


class StableLine:
    def __init__(self, modes=4, period=1.5, quadrature=32, degree=7):
        self.n = degree + 1 + 2 * modes
        self.modes = modes
        self.period = period
        self.degree = degree
        polynomial_size = degree + 1
        self.x = np.linspace(0, 1, self.n)
        self.cardinal_defect = None  # Galerkin modes, not cardinal interpolation
        self.spline = BSpline(
            np.r_[np.zeros(polynomial_size), np.ones(polynomial_size)],
            np.eye(polynomial_size),
            degree,
        )
        self.omega = 2 * np.pi * np.arange(1, modes + 1) / period
        count = 96
        c = np.zeros((count, self.n))
        c[:polynomial_size, :polynomial_size] = np.eye(polynomial_size)
        for j, omega in enumerate(self.omega):
            for k in range(polynomial_size, count):
                if k % 2 == 0:
                    c[k, polynomial_size + j] = (
                        2 * (-1.0) ** (k // 2) * jv(k, omega / 2)
                    )
                else:
                    c[k, polynomial_size + modes + j] = (
                        2 * (-1.0) ** ((k - 1) // 2) * jv(k, omega / 2)
                    )
        # Normalize tails before QR; does not alter approximation space.
        c /= np.maximum(np.linalg.norm(c, axis=0), 1e-300)
        z, w = gauss(quadrature)
        edges = np.linspace(0, 1, 5)
        h = np.diff(edges)
        self.q = (edges[:-1, None] + h[:, None] * z).ravel()
        self.w = (h[:, None] * w).ravel()
        b = chebval(2 * self.q - 1, c).T
        _, r = la.qr(np.sqrt(self.w[:, None]) * b, mode="economic")
        self.condition = float(np.linalg.cond(r))
        white = la.solve_triangular(r, np.eye(self.n))
        cw = c @ white
        # Reorthogonalize the already transformed coefficient representation;
        # this removes loss from the ill-conditioned tail change of coordinates.
        for _ in range(2):
            vq = chebval(2 * self.q - 1, cw).T
            _, rr = la.qr(np.sqrt(self.w[:, None]) * vq, mode="economic")
            cw = cw @ la.solve_triangular(rr, np.eye(self.n))
        d = chebval(2 * self.q - 1, chebder(cw, axis=0) * 2).T
        self.eigen, v = la.eigh(d.T @ (self.w[:, None] * d))
        self.coefficients = cw @ v
        self.coefficients[:, 0] = 0
        self.coefficients[0, 0] = 1
        self.eigen[0] = 0
        self.transform = np.eye(self.n)
        self.derivatives = [
            self.coefficients,
            chebder(self.coefficients, axis=0) * 2,
            chebder(self.coefficients, m=2, axis=0) * 4,
        ]

    def raw(self, x, order=1):
        return self.values(x, order)

    def values(self, x, order=1):
        return [
            chebval(2 * np.asarray(x) - 1, c).T for c in self.derivatives[: order + 1]
        ]
