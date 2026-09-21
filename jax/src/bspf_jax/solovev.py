"""Analytic static Solov'ev equilibria with exact smooth flux-surface domains.

Convention: -Delta* psi = mu0 R^2 p'(psi) + F F'(psi), psi=0 at the edge.
Both source terms are nonzero. A logarithmic homogeneous addition provides a
nonpolynomial test without changing p' or FF'. Coordinates in this model are
physical (R,Z); the domain uses local (R-R0,Z) for the background BSPF box.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import roots_legendre


@dataclass(frozen=True)
class SolovevEquilibrium:
    major_radius: float = 2.0
    axis_flux: float = 0.2
    a: float = 0.1
    b: float = 0.15
    c: float = 0.2
    logarithmic: float = 0.0
    mu0: float = 1.0
    edge_f: float = 3.0

    def __post_init__(self):
        values = (self.major_radius, self.axis_flux, self.a, self.b, self.c, self.mu0, self.edge_f)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("Solovev parameters must be finite and positive")
        if not np.isfinite(self.logarithmic) or not 0 <= self.logarithmic < self.c:
            raise ValueError("Require 0 <= logarithmic < c")
        if self.axis_flux >= self.a * self.major_radius**4:
            raise ValueError("Flux surface must stay away from R=0")

    @property
    def p_prime(self):
        return (8*self.a + 2*self.b) / self.mu0

    @property
    def ff_prime(self):
        return 2*self.c

    def radial(self, r):
        r = np.asarray(r)
        if np.any(r <= 0):
            raise ValueError("Require physical R>0")
        r0, d = self.major_radius, self.logarithmic
        log = np.log1p((r-r0)/r0)
        u = self.a*(r*r-r0*r0)**2 + d*(r*r*log - (r*r-r0*r0)/2)
        du = 4*self.a*r*(r*r-r0*r0) + 2*d*r*log
        ddu = 4*self.a*(3*r*r-r0*r0) + 2*d*(log+1)
        v = self.b*r*r + self.c-d
        return u, du, ddu, v

    def jets(self, points):
        """Return psi, gradient (R,Z), and Cartesian meridional Hessian."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("Expected finite physical points (count,2)")
        r, z = points.T
        u, du, ddu, v = self.radial(r)
        psi = self.axis_flux - u - v*z*z
        grad = np.column_stack((-du-2*self.b*r*z*z, -2*v*z))
        h = np.empty((len(points), 2, 2))
        h[:, 0, 0], h[:, 1, 1] = -ddu-2*self.b*z*z, -2*v
        h[:, 0, 1] = h[:, 1, 0] = -4*self.b*r*z
        return psi, grad, h

    def source(self, points):
        """Independent closed-form source; does not differentiate exact jets."""
        r = np.asarray(points)[:, 0]
        return (8*self.a+2*self.b)*r*r + 2*self.c

    def toroidal_function(self, psi):
        radicand = self.edge_f**2 + 2*self.ff_prime*np.asarray(psi)
        if np.any(radicand <= 0):
            raise ValueError("F^2 is not positive")
        return np.sqrt(radicand)

    def magnetic_field(self, points):
        psi, grad, _ = self.jets(points)
        r = np.asarray(points)[:, 0]
        return np.column_stack((-grad[:, 1], self.toroidal_function(psi), grad[:, 0])) / r[:, None]


class SolovevFluxDomain:
    """Exact analytic psi=0 boundary; no polygon or fitted spline geometry.

    Implements the geometry interface used by ArcLengthBoundary. Only the
    unknown field uses BSPF. Convexity is certified here with conservative
    Hessian bounds over an enclosing rectangle, not merely sampled curvature.
    """

    period = 4

    def __init__(self, equilibrium):
        self.equilibrium = equilibrium
        self.major_radius = r0 = equilibrium.major_radius
        delta = np.sqrt(equilibrium.axis_flux / equilibrium.a)
        lower, upper = np.sqrt(r0*r0-delta), np.sqrt(r0*r0+delta)
        target = lambda r: float(equilibrium.radial(r)[0] - equilibrium.axis_flux)
        self.rmin = brentq(target, lower*(1-1e-12), r0, xtol=5e-15)
        self.rmax = brentq(target, r0, upper*(1+1e-12), xtol=5e-15)
        vmin = float(equilibrium.radial(self.rmin)[3])
        self.zbound = np.sqrt(equilibrium.axis_flux / vmin)
        radial_hessian_min = float(equilibrium.radial(self.rmin)[2])
        cross_bound = 4*equilibrium.b*self.rmax*self.zbound
        determinant_lower = 2*vmin*radial_hessian_min - cross_bound**2
        if radial_hessian_min <= 0 or determinant_lower <= 0:
            raise ValueError("Parameters do not satisfy the sufficient convexity bound")
        self.geometry_checks = dict(
            radial_min=self.rmin, radial_max=self.rmax,
            level_hessian_rr_lower=radial_hessian_min,
            level_hessian_determinant_lower=determinant_lower,
            exact_flux_surface=True,
        )
        self.bounds = np.array([[self.rmin-r0, self.rmax-r0], [-self.zbound, self.zbound]])
        self.name = "solovev_logarithmic" if equilibrium.logarithmic else "solovev_polynomial"

    def curve(self, parameter, nu=0):
        """Periodic polar parameterization and analytic first/second derivatives."""
        if nu not in (0, 1, 2):
            raise ValueError("Curve derivatives supported through order two")
        t = np.asarray(parameter, dtype=float)
        shape = t.shape
        theta = t.ravel() * (2*np.pi/self.period)
        e = np.column_stack((np.cos(theta), np.sin(theta)))
        et = np.column_stack((-np.sin(theta), np.cos(theta)))
        # The surrounding rectangle gives an exterior radial endpoint. A
        # safeguarded vector Newton solve keeps all R strictly positive.
        radial_distance = np.where(e[:, 0] >= 0, self.rmax-self.major_radius,
                                   self.major_radius-self.rmin)
        xmax = radial_distance / np.maximum(np.abs(e[:, 0]), 1e-300)
        zmax = self.zbound / np.maximum(np.abs(e[:, 1]), 1e-300)
        lo, hi = np.zeros(len(e)), np.minimum(xmax, zmax)
        rho = 0.7*hi
        for _ in range(60):
            local = rho[:, None]*e
            physical = local + np.array([self.major_radius, 0])
            psi, gradient, hessian = self.equilibrium.jets(physical)
            residual, outward = -psi, -gradient
            slope = np.sum(outward*e, axis=1)
            if np.max(np.abs(residual), initial=0) < 3e-15:
                break
            lo = np.where(residual < 0, rho, lo)
            hi = np.where(residual >= 0, rho, hi)
            step = rho-residual/slope
            candidate = np.where((step > lo) & (step < hi), step, (lo+hi)/2)
            rho = np.where(np.abs(residual) < 3e-15, rho, candidate)
        else:
            raise RuntimeError("Analytic flux-boundary root solve did not converge")
        if nu == 0:
            return local.reshape(shape+(2,))
        drho = -rho*np.sum(outward*et, axis=1)/slope
        tangent = drho[:, None]*e + rho[:, None]*et
        rate = 2*np.pi/self.period
        if nu == 1:
            return (rate*tangent).reshape(shape+(2,))
        known = 2*drho[:, None]*et - rho[:, None]*e
        dd = -(np.einsum("qi,qij,qj->q", tangent, -hessian, tangent)
               + np.sum(outward*known, axis=1))/slope
        return (rate**2*(dd[:, None]*e+known)).reshape(shape+(2,))

    def normal(self, parameter):
        tangent = self.curve(parameter, 1)
        return np.stack((tangent[..., 1], -tangent[..., 0]), axis=-1) / np.linalg.norm(tangent, axis=-1)[..., None]

    def intersections(self, x):
        r = float(x)+self.major_radius
        if not self.rmin < r < self.rmax:
            return np.empty((0, 2))
        u, _, _, v = self.equilibrium.radial(r)
        h = np.sqrt(max(0.0, (self.equilibrium.axis_flux-u)/v))
        return np.array([[-h, h]])

    def volume_rule(self, order=16):
        """Composite Gauss integration in exact slices with endpoint smoothing."""
        if not isinstance(order, (int, np.integer)) or order < 2:
            raise ValueError("Quadrature order must be an integer >=2")
        q, w = roots_legendre(order)
        edges = np.linspace(-1, 1, 5)
        s = np.concatenate([(a+b)/2+(b-a)*q/2 for a, b in zip(edges[:-1], edges[1:])])
        ws = np.tile(w/4, 4)
        angle = np.pi*s/2
        middle, half = (self.rmin+self.rmax)/2, (self.rmax-self.rmin)/2
        r = middle+half*np.sin(angle)
        wr = ws * half*np.pi/2*np.cos(angle)
        u, _, _, v = self.equilibrium.radial(r)
        h = np.sqrt(np.maximum(0, (self.equilibrium.axis_flux-u)/v))
        x = np.broadcast_to((r-self.major_radius)[:, None], (len(r), len(s)))
        z = h[:, None]*s[None, :]
        weights = wr[:, None]*h[:, None]*ws[None, :]
        return np.column_stack((x.ravel(), z.ravel())), weights.ravel(), 1
