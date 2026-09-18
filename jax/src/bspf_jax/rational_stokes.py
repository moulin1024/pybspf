"""Homogeneous Goursat Stokes boundary extension for BSPF volume spaces.

Rational/AAA/lightning construction follows Xue, Waters and Trefethen,
SISC 46 (2024), doi:10.1137/23M1576876. The independent benchmark remains in
examples/pde/lightning_stokes_reference.py. This module adds reusable boundary
response matrices, a single-valued streamfunction and arbitrary rectangles.
"""

import numpy as np
import scipy.linalg as la
from scipy.interpolate import AAA


class RationalBasis:
    """Block Arnoldi with differentiated recurrences through second order."""

    def __init__(self, z, degree, poles, laurent):
        self.blocks = [(None, degree), (np.zeros(laurent, complex), laurent)]
        self.blocks += [(np.asarray(p), len(p)) for p in poles]
        self.hessenberg = []
        for p, n in self.blocks:
            q = np.ones((len(z), n + 1), complex)
            h = np.zeros((n + 1, n), complex)
            for k in range(n):
                value = z * q[:, k] if p is None else q[:, k] / (z - p[k])
                # Twice modified Gram-Schmidt; the stored recurrence contains
                # both passes, so evaluation describes exactly the same basis.
                for _ in range(2):
                    for j in range(k + 1):
                        v = np.vdot(q[:, j], value) / len(z)
                        h[j, k] += v
                        value -= v * q[:, j]
                h[k + 1, k] = la.norm(value) / np.sqrt(len(z))
                q[:, k + 1] = value / h[k + 1, k]
            self.hessenberg.append(h)

    def evaluate(self, z):
        output = [[], [], []]
        for ib, ((p, n), h) in enumerate(zip(self.blocks, self.hessenberg)):
            q = np.ones((len(z), n + 1), complex)
            d = np.zeros_like(q)
            dd = np.zeros_like(q)
            for k in range(n):
                if p is None:
                    r, dr, ddr = z, 1, 0
                else:
                    r = 1 / (z - p[k])
                    dr, ddr = -(r**2), 2 * r**3
                hk, scale = h[: k + 1, k], h[k + 1, k]
                q[:, k + 1] = (r * q[:, k] - q[:, : k + 1] @ hk) / scale
                d[:, k + 1] = (r * d[:, k] + dr * q[:, k] - d[:, : k + 1] @ hk) / scale
                dd[:, k + 1] = (
                    r * dd[:, k]
                    + 2 * dr * d[:, k]
                    + ddr * q[:, k]
                    - dd[:, : k + 1] @ hk
                ) / scale
            for out, val in zip(output, (q, d, dd)):
                out.append(val[:, int(ib != 0) :])
        return tuple(np.column_stack(x) for x in output)


class RationalStokesExtension:
    """Map hole velocity data to zero-outer-data homogeneous Stokes fields.

    Inlet/top/bottom velocity is zero; right Laplacian traction is zero.
    The hole flux must be zero. No body forcing or particular solution is used.
    Pressure rows represent p/nu; the velocity extension is independent of nu.
    """

    def __init__(
        self,
        bounds,
        hole,
        *,
        degree=96,
        corner_poles=32,
        laurent=64,
        samples=800,
        rcond=1e-13,
    ):
        left, right, h = bounds
        self.bounds, self.hole = bounds, hole
        self.center = complex(*hole.center)
        if (
            degree < 1
            or corner_poles < 1
            or laurent < 1
            or samples < max(32, 2 * degree)
        ):
            raise ValueError("Invalid rational extension sizes")
        t = np.tanh(
            np.linspace(
                -2 * np.sqrt(corner_poles) - 1, 2 * np.sqrt(corner_poles) + 1, samples
            )
        )
        theta = 2 * np.pi * np.arange(samples) / samples
        a, b = hole.axes
        self.hole_complex = self.center + a * np.cos(theta) + 1j * b * np.sin(theta)
        self.hole_points = np.column_stack(
            (self.hole_complex.real, self.hole_complex.imag)
        )
        points = np.r_[
            left + 1j * h * t,
            (left + right) / 2 + (right - left) / 2 * t + 1j * h,
            (left + right) / 2 + (right - left) / 2 * t - 1j * h,
            right + 1j * h * t,
            self.hole_complex,
        ]
        self.boundary_points = points
        eta = 2 * np.pi * np.arange(max(1000, samples)) / max(1000, samples)
        ellipse = a * np.cos(eta) + 1j * b * np.sin(eta)
        poles = AAA(ellipse, ellipse.conj(), rtol=1e-13, max_terms=80).poles()
        poles = poles[
            np.isfinite(poles)
            & ((poles.real / a) ** 2 + (poles.imag / b) ** 2 < 1 - 1e-10)
        ]
        corners = (
            np.array([left - 1j * h, left + 1j * h, right - 1j * h, right + 1j * h])
            - self.center
        )
        directions = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j]) / np.sqrt(2)
        d = (
            2
            * h
            * np.exp(
                4 * (np.sqrt(np.arange(corner_poles, 0, -1)) - np.sqrt(corner_poles))
            )
        )
        self.poles = [c + v * d for c, v in zip(corners, directions)] + [poles]
        self.basis = RationalBasis(points - self.center, degree, self.poles, laurent)
        k = degree + 1 + laurent + sum(len(g) for g in self.poles)
        # The real coefficient of g's log is the flux/source mode. Remove it
        # exactly so streamfunctions cannot acquire an angular branch jump.
        self.columns = np.delete(np.arange(4 * k + 4), 2 * k + 1)
        u, v, p, _, ux, vx = self.rows(points)
        a1, a2 = u.copy(), v.copy()
        out = slice(3 * samples, 4 * samples)
        a1[out] = ux[out] - p[out]
        a2[out] = vx[out]
        matrix = np.vstack((a1, a2))[:, self.columns]
        scale = la.norm(matrix, axis=0)
        scale[scale == 0] = 1
        u, s, vh = la.svd(matrix / scale, full_matrices=False)
        keep = s > rcond * s[0]
        hole_rows = np.r_[
            np.arange(4 * samples, 5 * samples), np.arange(9 * samples, 10 * samples)
        ]
        # Preserve the SVD solve order: an explicit pseudoinverse loses digits
        # through cancellation between nearly null columns.
        self.left_hole = u[hole_rows][:, keep].T
        self.singular_values = s[keep]
        self.right_scaled = vh[keep].T / scale[:, None]
        self.info = dict(
            degree=degree,
            corner_poles=corner_poles,
            laurent=laurent,
            aaa_poles=len(poles),
            samples_per_side=samples,
            rank=int(keep.sum()),
            retained_condition=float(s[0] / s[keep][-1]),
            rcond=rcond,
        )
        # Fix an arbitrary streamfunction gauge on the bottom wall.
        self.psi_gauge = self.stream_rows(np.array([(left + right) / 2 - 1j * h]))[0][0]

    def response(self, hole_velocity):
        projected = self.left_hole @ np.asarray(hole_velocity)
        divisor = (
            self.singular_values
            if projected.ndim == 1
            else self.singular_values[:, None]
        )
        return self.right_scaled @ (projected / divisor)

    def stream_rows(self, points):
        z = np.asarray(points).ravel() - self.center
        r, _, _ = self.basis.evaluate(z)
        logz = np.log(z)
        cz = z.conj()
        base = np.column_stack((cz[:, None] * r, r))
        psi = np.column_stack(
            (
                base.imag,
                (cz * logz - z * logz + z).imag,
                logz.imag,
                base.real,
                (cz * logz + z * logz - z).real,
                logz.real,
            )
        )[:, self.columns]
        if hasattr(self, "psi_gauge"):
            psi = psi - self.psi_gauge
        u, v, _, omega, ux, vx = self.rows(points)
        return (psi,) + tuple(a[:, self.columns] for a in (u, v, ux, vx - omega, vx))

    def evaluate(self, points, coefficients, batch_size=512):
        """psi,u,v,u_x,u_y,v_x; accepts one vector or a matrix of coefficients."""
        points = np.asarray(points)
        z = points[:, 0] + 1j * points[:, 1] if points.ndim == 2 else points.ravel()
        values = [[] for _ in range(6)]
        for start in range(0, len(z), batch_size):
            for output, row in zip(
                values, self.stream_rows(z[start : start + batch_size])
            ):
                output.append(row @ coefficients)
        return tuple(np.concatenate(v, axis=0) for v in values)

    def rows(self, points):
        z = np.asarray(points).ravel() - self.center
        r, dr, ddr = self.basis.evaluate(z)
        cz = z.conj()[:, None]
        o = 1 / z
        logz = np.log(z)
        zero = np.zeros_like(r)
        # Complex coefficient order: [f rational, g rational, f log, g log].
        # g contains -conj(a)*(z log z-z), making velocities single-valued.
        ub = np.column_stack((cz * dr - r, dr))
        vb = np.column_stack((-cz * dr - r, -dr))
        pb = np.column_stack((4 * dr, zero))
        wb = np.column_stack((-4 * dr, zero))
        u = np.column_stack(
            (
                ub.real,
                (z.conj() * o - 2 * logz).real,
                o.real,
                -ub.imag,
                -(z.conj() * o).imag,
                -o.imag,
            )
        )
        v = np.column_stack(
            (
                vb.imag,
                (-z.conj() * o).imag,
                -o.imag,
                vb.real,
                (-z.conj() * o - 2 * logz).real,
                -o.real,
            )
        )
        pressure = np.column_stack(
            (
                pb.real,
                (4 * o).real,
                np.zeros(len(z)),
                -pb.imag,
                -(4 * o).imag,
                np.zeros(len(z)),
            )
        )
        omega = np.column_stack(
            (
                wb.imag,
                (-4 * o).imag,
                np.zeros(len(z)),
                wb.real,
                (-4 * o).real,
                np.zeros(len(z)),
            )
        )
        xb = np.column_stack((cz * ddr, ddr))
        yb = np.column_stack((-2 * dr - cz * ddr, -ddr))
        ux = np.column_stack(
            (
                xb.real,
                (-z.conj() * o**2 - o).real,
                (-(o**2)).real,
                -xb.imag,
                -(-z.conj() * o**2 + o).imag,
                -(-(o**2)).imag,
            )
        )
        vx = np.column_stack(
            (
                yb.imag,
                (-o + z.conj() * o**2).imag,
                (o**2).imag,
                yb.real,
                (-3 * o + z.conj() * o**2).real,
                (o**2).real,
            )
        )
        return u, v, pressure, omega, ux, vx
