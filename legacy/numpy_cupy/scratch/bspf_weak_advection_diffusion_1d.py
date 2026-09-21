"""1D research prototype: same NS BSPF trial space, energy-stable weak form.

Host assembly uses the NumPy port of the NS pressure-line reconstruction.
No finite-difference replacement, modal filter, or artificial viscosity.
This is not wired into the production NS solver.
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
from scipy.interpolate import BSpline
from scipy.special import roots_legendre

from pybspf.solvers._pressure_bspf import PressureLine


@dataclass
class WeakBSPF1D:
    x: np.ndarray
    quadrature_x: np.ndarray
    quadrature_weights: np.ndarray
    values: np.ndarray
    gradients: np.ndarray
    mass: np.ndarray
    convection: np.ndarray
    stiffness: np.ndarray
    strong_d1: np.ndarray
    strong_d2: np.ndarray
    raw_sbp_defect: float
    projector_norm: float
    quadrature_order: int
    line: PressureLine
    spline: BSpline
    assembly_bits: int | None = None

    def generator(self, speed=1.0, viscosity=0.002, weak=True):
        if weak:
            return la.solve(
                self.mass,
                -speed * self.convection - viscosity * self.stiffness,
                assume_a="pos",
            )
        return (-speed * self.strong_d1 + viscosity * self.strong_d2)[1:-1, 1:-1]

    def load(self, values):
        """True quadrature load, not a forcing manufactured with discrete A."""
        return self.values.T @ (self.quadrature_weights * values)

    def project_load(self, values):
        return la.solve(self.mass, self.load(values), assume_a="pos")

    def evaluate(self, points, coefficients):
        if self.assembly_bits is None:
            values, _ = trial_values(self.line, self.spline, points)
        else:
            from bspf_mp_basis import mp_trial_values

            values, _ = mp_trial_values(
                self.line, self.spline, points, bits=self.assembly_bits
            )
        return values[:, 1:-1] @ coefficients


def trial_values(line, spline, points):
    """Continuous cardinal basis of the same spline + real Fourier extension.

    The periodic part has n-1 samples; the last physical node is the duplicate
    endpoint. For an even periodic count, real Fourier evaluation includes the
    Nyquist cosine and its off-grid derivative (not a complex-valued field).
    """
    x = line.x
    m = x.size - 1
    omega = 2 * np.pi * np.fft.fftfreq(m, d=(x[-1] - x[0]) / m)
    residual = np.eye(x.size)[:m] - spline(x[:m]) @ line.P
    spectrum = np.fft.fft(residual, axis=0) / m
    phase = np.exp(1j * (np.asarray(points)[:, None] - x[0]) * omega)
    values = spline(points) @ line.P + (phase @ spectrum).real
    gradients = (
        spline(points, nu=1) @ line.P + (phase @ (1j * omega[:, None] * spectrum)).real
    )
    return values, gradients


def assemble(
    n,
    *,
    window=16,
    modes=12,
    q=9,
    n_basis=32,
    degree=13,
    quadrature_order=None,
    domain=(-3.0, 3.0),
    assembly_bits=None,
):
    """Keep original BSPF interpolation/endpoint settings, change weak assembly.

    M = integral Phi.T Phi, K = integral Phi'.T Phi',
    C = (integral Phi.T Phi' - integral Phi'.T Phi)/2.
    Homogeneous essential conditions remove both endpoint cardinal functions.
    Gauss integration is split at every spline knot and resolves Fourier modes.
    """
    x = np.linspace(*domain, n)
    line = PressureLine(
        x,
        q=q,
        n_basis=n_basis,
        degree=degree,
        baseline_points=window,
        endpoint_method="chebyshev",
        chebyshev_modes=modes,
    )
    breaks = np.linspace(x[0], x[-1], n_basis - degree + 1)
    knots = np.r_[
        np.repeat(x[0], degree + 1), breaks[1:-1], np.repeat(x[-1], degree + 1)
    ]
    spline = BSpline(knots, np.eye(n_basis), degree)
    if quadrature_order is None:
        quadrature_order = max(
            40, int(np.ceil(np.pi * (n - 1) / (len(breaks) - 1))) + 12
        )
    nodes, weights = roots_legendre(quadrature_order)
    points = np.concatenate(
        [(a + b) / 2 + (b - a) / 2 * nodes for a, b in zip(breaks[:-1], breaks[1:])]
    )
    weights = np.concatenate(
        [(b - a) / 2 * weights for a, b in zip(breaks[:-1], breaks[1:])]
    )
    if assembly_bits is None:
        values, gradients = trial_values(line, spline, points)
    else:
        from bspf_mp_basis import mp_trial_values

        values, gradients = mp_trial_values(line, spline, points, bits=assembly_bits)
    values, gradients = values[:, 1:-1], gradients[:, 1:-1]
    mass = values.T @ (weights[:, None] * values)
    convection = values.T @ (weights[:, None] * gradients)
    defect = np.max(abs(convection + convection.T))
    convection = (convection - convection.T) / 2
    stiffness = gradients.T @ (weights[:, None] * gradients)
    omega = 2 * np.pi * np.fft.fftfreq(n - 1, d=(x[-1] - x[0]) / (n - 1))

    def fourier_second(v):
        interior = np.fft.ifft(
            np.fft.fft(v[:-1], axis=0) * (-(omega**2))[:, None], axis=0
        ).real
        return np.concatenate([interior, interior[:1]])

    d2 = (
        fourier_second(np.eye(n))
        + (spline(x, nu=2) - fourier_second(spline(x))) @ line.P
    )
    return WeakBSPF1D(
        x,
        points,
        weights,
        values,
        gradients,
        mass,
        convection,
        stiffness,
        line.D,
        d2,
        float(defect),
        float(la.norm(line.P, 2)),
        quadrature_order,
        line,
        spline,
        assembly_bits,
    )


def analytic_field(x, case):
    """phi, phi_x, phi_xx; exact u(x,t)=exp(-t)*phi(x), zero endpoint data."""
    z = np.asarray(x) / 3
    if case == "nonperiodic":
        f = (1 - z * z) * np.exp(z)
        fp = (1 - 2 * z - z * z) * np.exp(z)
        fpp = (-1 - 4 * z - z * z) * np.exp(z)
        k = 3 * np.pi
        c, s = np.cos(k * z), np.sin(k * z)
        return (
            f * c,
            (fp * c - k * f * s) / 3,
            ((fpp - k * k * f) * c - 2 * k * fp * s) / 9,
        )
    if case == "oscillatory":
        phi = np.sin(np.pi * z) + 0.2 * np.sin(7 * np.pi * z)
        px = (np.pi * np.cos(np.pi * z) + 1.4 * np.pi * np.cos(7 * np.pi * z)) / 3
        pxx = (
            -(np.pi**2 * np.sin(np.pi * z) + 9.8 * np.pi**2 * np.sin(7 * np.pi * z)) / 9
        )
        return phi, px, pxx
    if case == "gaussian":
        g = np.exp(-16 * (z - 0.1) ** 2)
        gp = -32 * (z - 0.1) * g
        gpp = (1024 * (z - 0.1) ** 2 - 32) * g
        return (
            (1 - z * z) * g,
            (-2 * z * g + (1 - z * z) * gp) / 3,
            (-2 * g - 4 * z * gp + (1 - z * z) * gpp) / 9,
        )
    raise ValueError(case)


def exact_semidiscrete_error(generator, nodal_phi, forcing, time):
    """Exponential solve isolates spatial error from RK time-step error.

    w = u-exp(-t)*phi obeys w'=A w+exp(-t)*(A phi+forcing+phi).
    The block exponential avoids cancellation of two O(1) solutions and an
    inverse of A+I. Forcing is independently evaluated from analytic derivatives.
    """
    residual = generator @ nodal_phi + forcing + nodal_phi
    block = np.zeros((len(nodal_phi) + 1, len(nodal_phi) + 1))
    block[:-1, :-1] = generator
    block[:-1, -1] = residual
    block[-1, -1] = -1
    return la.expm(time * block)[:-1, -1]
