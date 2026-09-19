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

    def __init__(self, z, degree, poles, laurent, *, device=None):
        self.blocks = [(None, degree), (np.zeros(laurent, complex), laurent)]
        self.blocks += [(np.asarray(p), len(p)) for p in poles]
        self._gpu_data = {}
        self.hessenberg = []
        if device is not None:
            import jax
            from ._gpu_rational import construct_block
            if device.platform != "gpu" or not jax.config.x64_enabled:
                raise ValueError("Rational construction requires a GPU and jax_enable_x64")
            zd = jax.device_put(np.asarray(z, complex), device)
            factors = []
            for p, n in self.blocks:
                pd = jax.device_put(np.zeros(n, complex) if p is None else p, device)
                factors.append(construct_block(zd, pd, degree=n, polynomial=p is None))
            # Small recurrence tables also support the independent host evaluator.
            self.hessenberg = list(jax.device_get(factors))
            if not all(np.all(np.isfinite(h)) for h in self.hessenberg):
                raise np.linalg.LinAlgError("GPU rational construction returned non-finite factors")
            return
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

    def evaluate_gpu(self, z, device):
        """Evaluate all three recurrence orders without host intermediates."""
        import jax
        from ._gpu_rational import evaluate_blocks, unpad_blocks
        if device.platform != "gpu" or not jax.config.x64_enabled:
            raise ValueError("Rational evaluation requires a GPU and jax_enable_x64")
        sizes = tuple(n for _, n in self.blocks)
        if device not in self._gpu_data:
            width = max(sizes)
            hs = np.zeros((len(sizes), width+1, width), complex)
            ps = np.full((len(sizes), width), 1.e100, complex)
            polynomial = np.array([p is None for p, _ in self.blocks])
            for i, ((p, n), h) in enumerate(zip(self.blocks, self.hessenberg)):
                hs[i, :n+1, :n] = h
                hs[i, np.arange(n, width)+1, np.arange(n, width)] = 1
                if p is not None:
                    ps[i, :n] = p
            self._gpu_data[device] = jax.device_put((hs, ps, polynomial), device)
        z = jax.device_put(z, device)
        return unpad_blocks(evaluate_blocks(z, *self._gpu_data[device]), sizes=sizes)


class RationalStokesExtension:
    """Map hole velocity data to zero-outer-data homogeneous Stokes fields.

    Inlet/top/bottom velocity is zero; right Laplacian traction is zero.
    The hole flux must be zero. No body forcing or particular solution is used.
    Pressure rows represent p/nu; the velocity extension is independent of nu.
    assembly_device optionally runs the boundary-matrix SVD on the selected GPU;
    response factors are returned to the host geometry representation.
    basis_construction="gpu" also builds the Arnoldi recurrence on that GPU;
    this is opt-in because ordered reorthogonalization and JIT cost more than
    the small CPU construction. AAA pole selection remains on the host.
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
        assembly_device=None,
        basis_construction="cpu",
        min_pole_distance=0.0,
    ):
        if assembly_device is not None and assembly_device.platform != "gpu":
            raise ValueError("assembly_device must be a GPU device")
        if basis_construction not in ("cpu", "gpu"):
            raise ValueError("basis_construction must be 'cpu' or 'gpu'")
        if basis_construction == "gpu" and assembly_device is None:
            raise ValueError("GPU basis construction requires assembly_device")
        if not np.isfinite(min_pole_distance) or not 0 <= min_pole_distance <= 2:
            raise ValueError("min_pole_distance must lie in [0,2] (units of half-height)")
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
        if min_pole_distance and corner_poles > 1:
            fraction = (np.sqrt(corner_poles)-np.sqrt(np.arange(corner_poles, 0, -1))) / (np.sqrt(corner_poles)-1)
            d = 2*h*np.exp(np.log(min_pole_distance/2)*fraction)
        self.poles = [c + v * d for c, v in zip(corners, directions)] + [poles]
        self.basis = RationalBasis(
            points - self.center, degree, self.poles, laurent,
            device=assembly_device if basis_construction == "gpu" else None,
        )
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
        if assembly_device is None:
            u, s, vh = la.svd(matrix / scale, full_matrices=False)
        else:
            from ._gpu_linalg import gpu_svd
            u, s, vh = gpu_svd(matrix / scale, device=assembly_device)
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
            svd_backend="cpu" if assembly_device is None else "gpu",
            basis_construction=basis_construction,
            degree=degree,
            corner_poles=corner_poles,
            min_pole_distance=float(min_pole_distance),
            actual_min_pole_distance=float(d[-1]/h),
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
        basis_values = self.basis.evaluate(z)
        r = basis_values[0]
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
        u, v, _, omega, ux, vx = self._rows_from_basis(points, basis_values)
        return (psi,) + tuple(a[:, self.columns] for a in (u, v, ux, vx - omega, vx))

    def evaluate(self, points, coefficients, batch_size=512, *, device=None,
                 return_device=False):
        """psi,u,v,u_x,u_y,v_x; optional resident GPU recurrence and products.

        GPU results can be retained for subsequent volume assembly.
        """
        if return_device and device is None:
            raise ValueError("return_device requires a GPU device")
        if device is not None:
            return self._evaluate_gpu(points, coefficients, batch_size, device, return_device)
        points = np.asarray(points)
        z = points[:, 0] + 1j * points[:, 1] if points.ndim == 2 else points.ravel()
        values = [[] for _ in range(6)]
        for start in range(0, len(z), batch_size):
            for output, row in zip(
                values, self.stream_rows(z[start : start + batch_size])
            ):
                output.append(row @ coefficients)
        return tuple(np.concatenate(v, axis=0) for v in values)

    def _evaluate_gpu(self, points, coefficients, batch_size, device, return_device):
        import jax
        from ._immersed_assembly import _apply_rational_rows, _join_rational_chunks
        from ._gpu_rational import stream_rows
        if device.platform != "gpu":
            raise ValueError("Rational GPU evaluation requires a GPU device")
        if not jax.config.x64_enabled:
            raise ValueError("Enable jax_enable_x64 for rational GPU evaluation")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        points = np.asarray(points)
        z = points[:, 0]+1j*points[:, 1] if points.ndim == 2 else points.ravel()
        coefficients = jax.device_put(np.asarray(coefficients), device)
        if not len(z):
            empty = np.empty((0,)+coefficients.shape[1:])
            result = jax.device_put((empty,)*6, device)
        else:
            chunks = []
            gauge = jax.device_put(self.psi_gauge, device)
            for start in range(0, len(z), batch_size):
                batch = z[start:start+batch_size]-self.center
                count = len(batch)
                if count < batch_size:
                    batch = np.pad(batch, (0, batch_size-count), mode="edge")
                shifted = jax.device_put(batch, device)
                values = self.basis.evaluate_gpu(shifted, device)
                rows = stream_rows(shifted, values, gauge)
                chunk = _apply_rational_rows(rows, coefficients)
                chunks.append(tuple(a[:count] for a in chunk))
            result = _join_rational_chunks(tuple(chunks))
        return result if return_device else jax.device_get(result)

    def rows(self, points):
        z = np.asarray(points).ravel() - self.center
        return self._rows_from_basis(points, self.basis.evaluate(z))

    def _rows_from_basis(self, points, basis_values):
        z = np.asarray(points).ravel() - self.center
        r, dr, ddr = basis_values
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
