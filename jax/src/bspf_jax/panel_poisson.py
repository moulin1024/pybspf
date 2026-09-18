"""Knot-aligned high-order single-layer solver on exact cubic spline curves.

Dense reference implementation: reusable LU, no Fourier density truncation.
The geometry must be a regular, simple, CCW periodic cubic spline with integer
knots (the SplineDomain convention). A supplied particular solution handles a
nonzero Poisson source; this module does not compute arbitrary volume potentials.
"""

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import numpy as np
from numpy.polynomial.legendre import legvander
from scipy.linalg import lu_factor, lu_solve
from scipy.special import roots_legendre, xlogy


@lru_cache(maxsize=64)
def rule(order):
    return roots_legendre(order)


def log_moments(x, degree):
    """Integral_{-1}^1 log|x-t| P_n(t) dt, for x in [-1,1]."""
    x = float(np.clip(x, -1, 1))
    moments = np.empty(degree + 1)
    moments[0] = xlogy(1 + x, 1 + x) + xlogy(1 - x, 1 - x) - 2
    if abs(x) == 1:
        n = np.arange(1, degree + 1)
        moments[1:] = -2 * x**n / (n * (n + 1))
    else:
        q = np.empty(degree + 2)
        q[0] = np.log((1 + x) / (1 - x)) / 2
        q[1] = x * q[0] - 1
        for n in range(1, degree + 1):
            q[n + 1] = ((2 * n + 1) * x * q[n] - n * q[n - 1]) / (n + 1)
        for n in range(1, degree + 1):
            moments[n] = 2 * (q[n + 1] - q[n - 1]) / (2 * n + 1)
    return moments


class PanelPoissonPlan:
    def __init__(self, domain, order=16, breaks=None, quadrature_order=None):
        start = perf_counter()
        if getattr(domain.curve, "k", None) != 3:
            raise ValueError("Exact cubic B-spline geometry is required")
        self.domain, self.order = domain, int(order)
        if self.order < 2:
            raise ValueError("Panel order must be >=2")
        self.breaks = np.asarray(
            np.arange(domain.period + 1) if breaks is None else breaks, dtype=float
        )
        if (
            self.breaks[0] != 0
            or self.breaks[-1] != domain.period
            or np.any(np.diff(self.breaks) <= 0)
            or not all(np.any(self.breaks == k) for k in range(domain.period + 1))
        ):
            raise ValueError(
                "Increasing breaks must contain every original spline knot"
            )
        self.panels = len(self.breaks) - 1
        self.count = self.panels * self.order
        self.qorder = quadrature_order or max(32, 2 * self.order + 8)
        self.nodes, weights = rule(self.order)
        self.to_coefficients = (
            (2 * np.arange(self.order) + 1)[:, None]
            / 2
            * legvander(self.nodes, self.order - 1).T
            * weights
        )
        mid = (self.breaks[:-1] + self.breaks[1:]) / 2
        half = np.diff(self.breaks) / 2
        self.parameters = (mid[:, None] + half[:, None] * self.nodes).ravel()
        self.points = domain.curve(self.parameters)
        self.weights_dt = (half[:, None] * weights).ravel()
        self.length = float(
            self.weights_dt @ np.linalg.norm(domain.curve(self.parameters, 1), axis=1)
        )
        self.radius = self.length / (2 * np.pi)
        self.matrix = self.boundary_matrix(self.parameters)
        augmented = np.zeros((self.count + 1, self.count + 1))
        augmented[:-1, :-1] = self.matrix
        augmented[:-1, -1] = 1
        augmented[-1, :-1] = self.weights_dt / self.length
        self.factor = lu_factor(augmented)
        self.setup_seconds = perf_counter() - start

    def _basis(self, t, left, right):
        return (
            legvander((2 * t - left - right) / (right - left), self.order - 1)
            @ self.to_coefficients
        )

    def _boundary_block(self, target, parameter, left, right, qorder):
        midpoint, half = (left + right) / 2, (right - left) / 2
        # Unwrap the target parameter to the nearest periodic copy of this panel.
        tau = parameter + self.domain.period * np.round(
            (midpoint - parameter) / self.domain.period
        )
        q, w = rule(qorder)
        if left - 1e-14 <= tau <= right + 1e-14:
            tau = float(np.clip(tau, left, right))
            t = midpoint + half * q
            delta = t - tau
            # Exact cubic divided difference avoids cancellation at t=tau.
            divided = (
                self.domain.curve(tau, 1)
                + delta[:, None] * self.domain.curve(tau, 2) / 2
                + delta[:, None] ** 2 * self.domain.curve(midpoint, 3) / 6
            )
            regular = -np.log(np.linalg.norm(divided, axis=1) / self.radius) / (
                2 * np.pi
            )
            moments = log_moments((tau - midpoint) / half, self.order - 1)
            moments[0] += 2 * np.log(half)
            singular = -(moments @ self.to_coefficients) / (2 * np.pi)
            return half * (singular + (w * regular) @ self._basis(t, left, right))

        # Subdivide near adjacent-panel endpoints, including the periodic seam.
        pending, intervals = [(left, right)], []
        while pending:
            a, b = pending.pop()
            distance = min(abs(tau - a), abs(tau - b))
            if b - a > 2 * distance and b - a > 1e-13:
                m = (a + b) / 2
                pending.extend(((a, m), (m, b)))
            else:
                intervals.append((a, b))
        result = np.zeros(self.order)
        for a, b in intervals:
            t = (a + b) / 2 + (b - a) / 2 * q
            distance = np.linalg.norm(self.domain.curve(t) - target, axis=1)
            kernel = -np.log(distance / self.radius) / (2 * np.pi)
            result += (b - a) / 2 * (w * kernel) @ self._basis(t, left, right)
        return result

    def boundary_matrix(self, parameters, quadrature_order=None):
        parameters = np.asarray(parameters).reshape(-1)
        points = self.domain.curve(parameters)
        matrix = np.empty((len(points), self.count))
        for j, (left, right) in enumerate(zip(self.breaks[:-1], self.breaks[1:])):
            for i, (point, parameter) in enumerate(zip(points, parameters)):
                matrix[i, j * self.order : (j + 1) * self.order] = self._boundary_block(
                    point, parameter, left, right, quadrature_order or self.qorder
                )
        return matrix

    def solve(self, boundary_data, particular=None):
        start = perf_counter()
        g = np.asarray(
            boundary_data(self.points) if callable(boundary_data) else boundary_data
        )
        if g.shape != (self.count,):
            raise ValueError("Expected one boundary value per collocation point")
        h = g.copy()
        if particular is not None:
            h -= particular(self.points)
        answer = lu_solve(self.factor, np.r_[h, 0.0])
        density, constant = answer[:-1], float(answer[-1])
        return PanelPoissonSolution(
            self,
            density,
            constant,
            particular,
            perf_counter() - start,
            float(np.max(abs(self.matrix @ density + constant - h))),
        )


@dataclass
class PanelPoissonSolution:
    plan: PanelPoissonPlan
    density: np.ndarray  # mu(gamma(t))*|gamma'(t)|, polynomial in each panel
    constant: float
    particular: object
    solve_seconds: float
    training_residual: float

    def boundary(self, parameters, quadrature_order=None):
        values = (
            self.plan.boundary_matrix(parameters, quadrature_order) @ self.density
            + self.constant
        )
        if self.particular is not None:
            values += self.particular(self.plan.domain.curve(parameters))
        return values

    def interior(self, points, quadrature_order=None):
        """Ordinary panel quadrature for separated interior targets only."""
        points = np.asarray(points)
        values = np.full(len(points), self.constant)
        q, w = rule(quadrature_order or max(64, 3 * self.plan.order))
        for j, (left, right) in enumerate(
            zip(self.plan.breaks[:-1], self.plan.breaks[1:])
        ):
            half = (right - left) / 2
            t = (left + right) / 2 + half * q
            rho = (
                self.plan._basis(t, left, right)
                @ self.density[j * self.plan.order : (j + 1) * self.plan.order]
            )
            distance = np.linalg.norm(
                points[:, None, :] - self.plan.domain.curve(t)[None, :, :], axis=2
            )
            values -= (
                half / (2 * np.pi) * (np.log(distance / self.plan.radius) @ (w * rho))
            )
        if self.particular is not None:
            values += self.particular(points)
        return values

    def panel_indicators(self, boundary_data):
        # Independent Gauss points plus explicit endpoint-near probes.
        q = np.unique(np.r_[rule(self.plan.order + 3)[0], -0.999, -0.99, 0.99, 0.999])
        mid = (self.plan.breaks[:-1] + self.plan.breaks[1:]) / 2
        half = np.diff(self.plan.breaks) / 2
        t = (mid[:, None] + half[:, None] * q).ravel()
        expected = boundary_data(self.plan.domain.curve(t))
        residual = (self.boundary(t, self.plan.qorder + 12) - expected).reshape(
            self.plan.panels, -1
        )
        scale = max(float(np.max(abs(expected))), 1e-100)
        return np.max(abs(residual), axis=1) / scale, q[
            np.argmax(abs(residual), axis=1)
        ]


def adaptive_solve(
    domain,
    boundary_data,
    *,
    order=12,
    tolerance=1e-10,
    max_refinements=6,
    particular=None,
    callback=None,
):
    """Bisect panels with large independent residuals, including endpoint probes.

    This is an a posteriori sampling indicator, not a certified error bound.
    Returns (solution, history); history[-1]['converged'] reports success/failure.
    """
    breaks = np.arange(domain.period + 1, dtype=float)
    history = []
    for level in range(max_refinements + 1):
        plan = PanelPoissonPlan(domain, order, breaks)
        solution = plan.solve(boundary_data, particular)
        indicators, locations = solution.panel_indicators(boundary_data)
        maximum = float(indicators.max())
        history.append(
            dict(
                level=level,
                panels=plan.panels,
                unknowns=plan.count,
                max_indicator=maximum,
                converged=maximum <= tolerance,
                breaks=breaks.tolist(),
                indicators=indicators.tolist(),
                peak_local_coordinates=locations.tolist(),
                setup_seconds=plan.setup_seconds,
            )
        )
        if callback is not None:
            callback(solution, history[-1])
        if maximum <= tolerance or level == max_refinements:
            break
        marked = indicators > max(tolerance, 0.3 * maximum)
        midpoints = (breaks[:-1] + breaks[1:]) / 2
        breaks = np.sort(np.r_[breaks, midpoints[marked]])
    return solution, history
