"""Divergence-free BSPF channel flow past a fixed smooth immersed obstacle.

Dense geometry setup with optional GPU volume products: physical-domain Galerkin integration,
SVD wall constraints, an analytic wall factor, or a rationally corrected BSPF
space, and the original cavity IMEX integrator. No solid penalization; relaxation acts only
in the explicitly added buffer.
"""

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import jax
import numpy as np
import scipy.linalg as la
from scipy.special import roots_legendre

from ._weak_basis import with_basis_workers
from ._flow_kernels import imex_midpoint, tensor_product
from .convex_poisson import ArcLengthBoundary
from .immersed_poisson import EllipticHole
from .stream_navier_stokes import _stream_line, stream_evaluate_line, _smooth_step


def channel_lift(points, half_height=1.0, peak=1.0):
    """psi, u, v, u_x, u_y, v_x; v_y=-u_x by construction."""
    y = np.asarray(points)[:, 1]
    h = half_height
    z = np.zeros_like(y)
    return (
        peak * (y - y**3 / (3 * h * h) + 2 * h / 3),
        peak * (1 - y * y / (h * h)),
        z,
        z,
        -2 * peak * y / (h * h),
        z,
    )


def _gauss(lo, hi, count):
    q, w = roots_legendre(count)
    return (lo + hi) / 2 + (hi - lo) / 2 * q, (hi - lo) / 2 * w


def _stream_product(fields, jet):
    """Analytic product rule; fields are psi,u,v,u_x,u_y,v_x."""
    p, u, v, xy, yy, minus_xx = fields
    q, qx, qy, qxx, qxy, qyy = jet
    return (
        q * p,
        q * u + qy * p,
        q * v - qx * p,
        q * xy + qx * u - qy * v + qxy * p,
        q * yy + 2 * qy * u + qyy * p,
        q * minus_xx + 2 * qx * v - qxx * p,
    )


def elliptic_wall_factor(points, bounds, hole, width=2.0):
    """Analytic q=F²/(F²+width² G²), with q=grad(q)=0 on the hole.

    F is the ellipse level minus one; G vanishes on the rectangle and equals
    one at the ellipse center. Thus q=1, grad(q)=0 on the rectangle. No mask,
    sampled differentiation, or postprocessing filter is used.
    """
    x, y = np.asarray(points).T
    left, right, h = bounds
    (cx, cy), (a, b) = hole.center, hole.axes
    f = ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 - 1
    fx, fy = 2 * (x - cx) / a**2, 2 * (y - cy) / b**2
    px, py = (x - left) * (right - x), 1 - (y / h) ** 2
    gx, gy = right + left - 2 * x, -2 * y / h**2
    norm = (cx - left) * (right - cx) * (1 - (cy / h) ** 2)
    g = px * py / norm
    g1 = np.stack((gx * py, px * gy), axis=-1) / norm
    g2 = np.stack((-2 * py, gx * gy, -2 * px / h**2), axis=-1) / norm
    f1 = np.stack((fx, fy), axis=-1)
    f2 = np.broadcast_to([2 / a**2, 0, 2 / b**2], g2.shape)

    def square_jet(v, d, dd):
        return (
            v * v,
            2 * v[:, None] * d,
            2 * v[:, None] * dd
            + 2 * np.stack((d[:, 0] ** 2, d[:, 0] * d[:, 1], d[:, 1] ** 2), axis=-1),
        )

    n, n1, n2 = square_jet(f, f1, f2)
    s, s1, s2 = square_jet(g, g1, g2)
    d, d1, d2 = n + width**2 * s, n1 + width**2 * s1, n2 + width**2 * s2
    q = n / d
    q1 = (n1 - q[:, None] * d1) / d[:, None]
    cross = np.stack(
        (
            2 * q1[:, 0] * d1[:, 0],
            q1[:, 0] * d1[:, 1] + q1[:, 1] * d1[:, 0],
            2 * q1[:, 1] * d1[:, 1],
        ),
        axis=-1,
    )
    q2 = (n2 - q[:, None] * d2 - cross) / d[:, None]
    return q, q1[:, 0], q1[:, 1], q2[:, 0], q2[:, 1], q2[:, 2]


def channel_quadrature(bounds, hole, nx, ny, factor=1.5, x_breaks=()):
    """Positive exact-geometry patch quadrature, no staircase cell mask.

    Two rectangular side patches and two curved middle patches. On the
    latter x=cx+a*cos(theta), y interpolates between the exact ellipse and
    a horizontal wall. The Jacobian vanishes smoothly at tangencies.
    """
    left, right, h = bounds
    if hole is None:
        x, wx = _gauss(left, right, int(np.ceil(factor * nx)))
        y, wy = _gauss(-h, h, int(np.ceil(factor * ny)))
        xx, yy = np.meshgrid(x, y, indexing="ij")
        return np.column_stack((xx.ravel(), yy.ravel())), np.outer(wx, wy).ravel()
    cx, cy = hole.center
    a, b = hole.axes
    points, weights = [], []
    y, wy = _gauss(-h, h, int(np.ceil(factor * ny)))
    intervals = []
    for lo, hi in ((left, cx - a), (cx + a, right)):
        cuts = [lo] + sorted(v for v in x_breaks if lo < v < hi) + [hi]
        intervals.extend(zip(cuts[:-1], cuts[1:]))
    for lo, hi in intervals:
        x, wx = _gauss(
            lo, hi, max(12, int(np.ceil(factor * nx * (hi - lo) / (right - left))))
        )
        xx, yy = np.meshgrid(x, y, indexing="ij")
        points.append(np.column_stack((xx.ravel(), yy.ravel())))
        weights.append(np.outer(wx, wy).ravel())
    nt = max(20, int(np.ceil(factor * (nx * 2 * a / (right - left) + ny * b / h))) + 8)
    theta, wt = _gauss(0, np.pi, nt)
    x = cx + a * np.cos(theta)
    for sign in (-1, 1):
        wall = sign * h
        edge = cy + sign * b * np.sin(theta)
        ns = max(12, int(np.ceil(factor * ny * (h - sign * cy) / (2 * h))) + 4)
        s, ws = _gauss(0, 1, ns)
        yy = edge[:, None] * (1 - s) + wall * s
        xx = np.broadcast_to(x[:, None], yy.shape)
        jac = a * np.sin(theta) * abs(wall - edge)
        points.append(np.column_stack((xx.ravel(), yy.ravel())))
        weights.append((wt[:, None] * ws * jac[:, None]).ravel())
    return np.vstack(points), np.concatenate(weights)


class ImmersedFlowPlan:
    """Parabolic Dirichlet inlet, no-slip horizontal walls/hole, traction outlet.

    Re uses mean inlet velocity and vertical obstacle diameter (channel height
    when hole=None). Outlet traction is nu*du/dn - p*n = min(u_n,0)*u.
    Pressure is eliminated, not reconstructed. Hole streamfunction constant
    is deliberately NOT fixed: its circulation degree of freedom is retained.
    xlim specifies the physical channel; buffer_length extends its right end.
    wall_method='factor' replaces sampled hole constraints with a rational
    geometry factor and one unknown hole streamfunction constant. It changes
    the trial space while keeping the original BSPF grid and derivatives.
    wall_method='rational' uses a homogeneous Stokes boundary extension to
    construct a fixed corrected space B+R. All time, convection, viscosity and
    sponge operators act on B+R; it is not a post-step Stokes correction.
    assembly_device moves the boundary SVDs, volume energy eigensystem, Gram,
    basis-transform, and reduced mass/stiffness/sponge products onto a GPU.
    Results return to the host plan API. basis_workers controls a bounded
    spawn-based CPU pool for independent MPFR rows, closed when setup finishes.
    basis_precision="float64" opts into GPU double-precision basis evaluation,
    bypassing MPFR and its worker pool. The default "mpfr" preserves the
    cancellation-sensitive 113-bit evaluation. FP64 accuracy is case-dependent.
    """

    @with_basis_workers
    def __init__(
        self,
        *,
        nx=73,
        ny=33,
        xlim=(-1.0, 3.0),
        half_height=1.0,
        hole=EllipticHole(),
        reynolds=20.0,
        peak=1.0,
        quadrature_factor=2.5,
        boundary_count=None,
        wall_rcond=1e-10,
        volume_rcond=1e-11,
        buffer_length=2.0,
        buffer_strength=3.0,
        wall_method="svd",
        wall_width=2.0,
        rational_options=None,
        rational_preprocessing=False,
        assembly_device=None,
        basis_workers=1,
        basis_precision="mpfr",
    ):
        start = perf_counter()
        if assembly_device is not None and assembly_device.platform != "gpu":
            raise ValueError("assembly_device must be a GPU device")
        if basis_precision not in ("mpfr", "float64"):
            raise ValueError("basis_precision must be mpfr or float64")
        if basis_precision == "float64" and assembly_device is None:
            raise ValueError("float64 basis evaluation requires assembly_device on GPU")
        self.basis_precision = basis_precision
        self.assembly_device = assembly_device
        if wall_method not in ("svd", "factor", "rational"):
            raise ValueError("wall_method must be svd, factor or rational")
        if not np.isfinite(wall_width) or wall_width <= 0:
            raise ValueError("wall_width must be finite and positive")
        self.wall_method, self.wall_width = wall_method, wall_width
        self.factored_wall = wall_method == "factor" and hole is not None
        self.rational_wall = wall_method == "rational" and hole is not None
        self.rational = None
        if not jax.config.x64_enabled:
            raise ValueError("Enable jax_enable_x64 before BSPF setup")
        if not (xlim[0] < xlim[1] and half_height > 0 and reynolds > 0 and peak > 0):
            raise ValueError("Invalid domain or flow parameters")
        if not (0 < wall_rcond < 1 and 0 < volume_rcond < 1 and quadrature_factor >= 1):
            raise ValueError("Invalid discretization parameters")
        if not (
            np.isfinite(buffer_length)
            and buffer_length >= 0
            and np.isfinite(buffer_strength)
            and buffer_strength >= 0
        ):
            raise ValueError("Buffer parameters must be finite and nonnegative")
        if hole is not None:
            cx, cy = hole.center
            a, b = hole.axes
            if not (xlim[0] < cx - a < cx + a < xlim[1] and abs(cy) + b < half_height):
                raise ValueError("Obstacle must lie strictly inside the channel")
        self.nx, self.ny = nx, ny
        self.physical_xlim = tuple(xlim)
        self.buffer_start = xlim[1]
        self.buffer_length, self.buffer_strength = buffer_length, buffer_strength
        xlim = (xlim[0], xlim[1] + buffer_length)
        self.bounds = (*map(float, xlim), float(half_height))
        self.hole, self.peak, self.reynolds = hole, float(peak), float(reynolds)
        self.mean_speed = 2 * peak / 3
        self.diameter = 2 * (hole.axes[1] if hole is not None else half_height)
        self.nu = self.mean_speed * self.diameter / reynolds
        self.volume_rcond, self.wall_rcond = volume_rcond, wall_rcond
        self.quadrature_factor = quadrature_factor

        def line(x, clamped):
            return _stream_line(
                x,
                clamped=clamped,
                endpoint_points=min(len(x), 24),
                chebyshev_modes=min(len(x), 20),
                basis_executor=self._basis_executor,
                basis_precision=self.basis_precision, basis_device=self.assembly_device,
            )

        self.x = line(np.linspace(*xlim, nx), False)
        self.y = line(np.linspace(-half_height, half_height, ny), True)
        # Keep outlet free, constrain only the inlet using the original BSPF
        # line space; re-diagonalize its existing stiffness in the null space.
        trace = np.stack((np.asarray(self.x.bn)[0], np.asarray(self.x.gn)[0]))
        trace /= la.norm(trace, axis=1)[:, None]
        q = la.null_space(trace)
        lam, rot = la.eigh(q.T @ (np.asarray(self.x.lam)[:, None] * q))
        self.x_rotation = q @ rot
        self.shape = (len(lam), len(self.y.lam))
        self.rectangular_inertia = lam[:, None] + np.asarray(self.y.lam)[None, :]
        self.scale = 1 / np.sqrt(self.rectangular_inertia.ravel())
        if self.factored_wall:
            # Extra scalar is the unknown constant streamfunction on the hole.
            self.scale = np.r_[self.scale, 1.0]
        self.ndofs = self.scale.size
        self.line_seconds = perf_counter() - start

        @lru_cache(maxsize=10)
        def factors(axis, data):
            values = stream_evaluate_line(
                self.x if axis == 0 else self.y, np.frombuffer(data, dtype=np.float64),
                basis_executor=self._basis_executor,
                basis_precision=self.basis_precision, basis_device=self.assembly_device,
            )
            return tuple(a @ self.x_rotation if axis == 0 else a for a in values)

        self._factors = factors
        if self.rational_wall:
            from .rational_stokes import RationalStokesExtension

            extension = RationalStokesExtension(
                self.bounds, hole, assembly_device=assembly_device,
                **(rational_options or {})
            )
            boundary_ops = self.operators(extension.hole_points)
            trace = np.vstack(boundary_ops[1:3]) * self.scale
            if assembly_device is None:
                utrace, strace, vhtrace = la.svd(trace, full_matrices=False)
            else:
                from ._gpu_linalg import gpu_svd
                utrace, strace, vhtrace = gpu_svd(trace, device=assembly_device)
            trace_keep = strace > 1e-13 * strace[0]
            if rational_preprocessing is not False:
                if rational_preprocessing is not True and not isinstance(rational_preprocessing, dict):
                    raise ValueError("rational_preprocessing must be True, False, or an options dict")
                from ._rational_preprocess import prepare_extension
                # All retained trace directions, in the existing scaled BSPF
                # coefficient norm. Do not divide by tiny trace singular values.
                directions = self.scale[:, None]*vhtrace[trace_keep].T
                if assembly_device is not None:
                    directions = jax.device_put(directions, assembly_device)
                @lru_cache(maxsize=4)
                def audit_data(data):
                    points = np.frombuffer(data, dtype=np.float64).reshape(-1, 2)
                    ops = self.operators(points, device_output=assembly_device is not None)[1:]
                    if assembly_device is None:
                        values = tuple(o @ directions for o in ops)
                    else:
                        from ._immersed_assembly import _apply_rational_rows
                        values = jax.device_get(_apply_rational_rows(ops, directions))
                    lift = channel_lift(points, half_height, peak)[1:]
                    return tuple(-np.column_stack((b, v)) for b, v in zip(lift, values))
                selected = prepare_extension(
                    extension, lambda points: audit_data(np.asarray(points, dtype=np.float64).tobytes()),
                    device=assembly_device,
                    options={} if rational_preprocessing is True else rational_preprocessing,
                )
                if selected.info['samples_per_side'] != extension.info['samples_per_side']:
                    boundary_ops = self.operators(selected.hole_points)
                    trace = np.vstack(boundary_ops[1:3])*self.scale
                    if assembly_device is None:
                        utrace, strace, vhtrace = la.svd(trace, full_matrices=False)
                    else:
                        utrace, strace, vhtrace = gpu_svd(trace, device=assembly_device)
                    trace_keep = strace > 1e-13*strace[0]
                extension = selected
                audit_data.cache_clear()
            self.rational_modes = extension.response(utrace[:, trace_keep])
            self.rational_map = (
                -(strace[trace_keep, None] * vhtrace[trace_keep]) / self.scale
            )
            target = -np.concatenate(
                channel_lift(extension.hole_points, half_height, peak)[1:3]
            )
            self.rational_lift = extension.response(target)
            self.rational = extension
            extension.info["trace_rank"] = int(trace_keep.sum())
            del boundary_ops, trace, utrace, strace, vhtrace
        self.arc = None if hole is None else ArcLengthBoundary(hole)
        self.boundary_count = boundary_count or 4 * max(nx, ny)
        self.boundary_count += self.boundary_count % 2
        if hole is None or self.factored_wall or self.rational_wall:
            z = np.eye(self.ndofs)
            lift = np.zeros(self.ndofs)
            self.constraint_rank = 0
        else:
            boundary, _ = self.arc.sample(self.boundary_count)
            op = self.operators(boundary)
            constraint = np.vstack((op[1], op[2])) * self.scale
            target = -np.concatenate(channel_lift(boundary, half_height, peak)[1:3])
            if assembly_device is None:
                u, s, vh = la.svd(constraint, full_matrices=True)
            else:
                from ._gpu_linalg import gpu_svd
                u, s, vh = gpu_svd(constraint, device=assembly_device, full_matrices=True)
            rank = int(np.sum(s > wall_rcond * s[0]))
            lift = self.scale * (vh[:rank].T @ ((u[:, :rank].T @ target) / s[:rank]))
            z = self.scale[:, None] * vh[rank:].T
            self.constraint_rank = rank
            self.wall_lift_error = float(
                la.norm(constraint @ (lift / self.scale) - target, np.inf)
            )
            del op, constraint, u, vh
        if hole is None or self.factored_wall or self.rational_wall:
            z = self.scale[:, None] * z
            self.wall_lift_error = 0.0
        self.lift_coefficients = lift
        self.points, self.weights = channel_quadrature(
            self.bounds, hole, nx, ny, quadrature_factor, (self.buffer_start,)
        )
        device_volume = assembly_device is not None and not self.factored_wall
        op, base = self.operators(self.points, with_base=True, device_output=device_volume)
        diagonal_constraints = hole is None or self.factored_wall or self.rational_wall
        if device_volume:
            from ._immersed_assembly import prepare_gpu_volume
            self.lift_fields, raw = prepare_gpu_volume(
                op, base, lift, self.scale, None if diagonal_constraints else z,
                assembly_device,
            )
        else:
            self.lift_fields = tuple(o @ lift + b for o, b in zip(op, base))
            raw = (tuple(o * self.scale for o in op[1:]) if diagonal_constraints
                   else tuple(o @ z for o in op[1:]))
        del op
        assembly = None
        if assembly_device is not None:
            from ._immersed_assembly import GPUVolumeAssembly
            assembly = GPUVolumeAssembly(raw, self.weights, assembly_device)
            mass, stiffness = assembly.gram()
        else:
            mass, stiffness = self._gram(raw)
        energy = mass + stiffness
        if assembly_device is None:
            ev, vec = la.eigh((energy + energy.T) / 2)
        else:
            from ._gpu_linalg import gpu_eigh
            ev, vec = gpu_eigh(energy, device=assembly_device)
        keep = ev > volume_rcond * ev[-1]
        self.energy_condition = float(ev[-1] / ev[keep][0])
        self.discarded_volume_modes = int((~keep).sum())
        normalize = vec[:, keep] / np.sqrt(ev[keep])
        self.transform = (self.scale[:, None] * normalize if diagonal_constraints
                          else z @ normalize)
        self.dofs = int(keep.sum())
        self.sigma = self.sponge_profile(self.points[:, 0])
        if assembly is not None:
            self.operators_fluid, self.mass, self.stiffness, self.sponge = (
                assembly.transform_and_reduce(normalize, mass, stiffness, self.sigma)
            )
        else:
            self.operators_fluid = tuple(o @ normalize for o in raw)
            self.mass = normalize.T @ mass @ normalize
            self.stiffness = normalize.T @ stiffness @ normalize
        del raw, assembly
        self.mass = (self.mass + self.mass.T) / 2
        self.stiffness = (self.stiffness + self.stiffness.T) / 2
        self.mass_factor = la.cho_factor(self.mass)
        uu, vv, xy, yy, minus_xx = self.operators_fluid
        _, ul, vl, xyl, yyl, minus_xxl = self.lift_fields
        self.diffusion_lift = (
            2 * xy.T @ (self.weights * xyl)
            + yy.T @ (self.weights * yyl)
            + minus_xx.T @ (self.weights * minus_xxl)
        )
        if self.rational_wall:
            # The entire fixed lift is homogeneous Stokes with zero outlet
            # traction. Green's identity makes its weak viscous load zero.
            # Do not pollute this identity with a volume quadrature residual.
            self.rational.info["quadrature_diffusion_lift_norm"] = float(
                la.norm(self.diffusion_lift)
            )
            self.diffusion_lift = np.zeros_like(self.diffusion_lift)
        sw = self.weights * self.sigma
        if assembly_device is None:
            self.sponge = uu.T @ (sw[:, None] * uu) + vv.T @ (sw[:, None] * vv)
        reference_u = channel_lift(self.points, half_height, peak)[1]
        self.sponge_lift = uu.T @ (sw * (ul - reference_u)) + vv.T @ (sw * vl)
        self.linear = self.nu * self.stiffness + self.sponge
        self.linear_lift = self.nu * self.diffusion_lift + self.sponge_lift
        out_y, self.out_weights = _gauss(
            -half_height, half_height, max(24, int(2 * ny))
        )
        self.out_points = np.column_stack((np.full_like(out_y, xlim[1]), out_y))
        oo, out_base = self.operators(self.out_points, with_base=True)
        self.out_ops = (oo[1] @ self.transform, oo[2] @ self.transform)
        self.out_lift = tuple(
            o @ lift + b
            for o, b in zip(oo[1:3], out_base[1:3])
        )
        self.stokes_state = la.solve(self.linear, -self.linear_lift, assume_a="pos")
        self.setup_seconds = perf_counter() - start

    def sponge_profile(self, x):
        if self.buffer_length == 0:
            return np.zeros_like(x, dtype=float)
        return self.buffer_strength * np.asarray(
            _smooth_step((np.asarray(x) - self.buffer_start) / self.buffer_length)[0]
        )

    def _line_values(self, axis, coordinates):
        coordinates = np.asarray(coordinates, dtype=np.float64)
        lo, hi = self.bounds[:2] if axis == 0 else (-self.bounds[2], self.bounds[2])
        if coordinates.ndim != 1 or not np.all(np.isfinite(coordinates)):
            raise ValueError("Expected finite one-dimensional coordinates")
        if np.any(coordinates < lo - 1e-13) or np.any(coordinates > hi + 1e-13):
            raise ValueError("Evaluation must stay inside the computational rectangle")
        return self._factors(axis, coordinates.tobytes())

    def operators(self, points, *, with_base=False, device_output=False):
        """psi,u,v,u_x,u_y,v_x; exact v_y=-u_x, using BSPF derivatives.

        with_base also returns lift fields, sharing rational basis evaluation.
        device_output retains tensor/correction assembly on the selected GPU;
        this is used for volume setup without a factored wall.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (count,2)")
        factors = []
        for axis in range(2):
            p, idx = np.unique(np.asarray(points)[:, axis], return_inverse=True)
            factors.append([o[idx] for o in self._line_values(axis, p)])
        if device_output:
            if self.assembly_device is None or self.factored_wall:
                raise ValueError("Device operators require GPU assembly without a factored wall")
            from ._immersed_assembly import gpu_tensor_operators
            correction = mapping = None
            if self.rational is not None:
                coefficients = (np.column_stack((self.rational_modes, self.rational_lift))
                                if with_base else self.rational_modes)
                correction = self.rational.evaluate(
                    points, coefficients, device=self.assembly_device, return_device=True
                )
                mapping = self.rational_map
            base = channel_lift(points, self.bounds[2], self.peak) if with_base else None
            return gpu_tensor_operators(factors, correction, mapping, base, self.assembly_device)
        (x, dx, xx), (y, dy, yy) = factors

        def pair(a, b):
            return tensor_product(a, b, paired=True)

        result = (
            pair(x, y),
            pair(x, dy),
            -pair(dx, y),
            pair(dx, dy),
            pair(x, yy),
            -pair(xx, y),
        )
        if self.rational is not None:
            coefficients = (np.column_stack((self.rational_modes, self.rational_lift))
                            if with_base else self.rational_modes)
            correction = self.rational.evaluate(points, coefficients)
            if with_base:
                base = tuple(a + b[:, -1] for a, b in zip(
                    channel_lift(points, self.bounds[2], self.peak), correction))
                correction = tuple(a[:, :-1] for a in correction)
            result = tuple(o + r @ self.rational_map for o, r in zip(result, correction))
        elif self.factored_wall:
            jet = self.wall_factor(points)
            result = _stream_product(result, tuple(a[:, None] for a in jet))
            q, qx, qy, qxx, qxy, qyy = jet
            circulation = (1 - q, -qy, qx, -qxy, -qyy, qxx)
            result = tuple(np.column_stack((o, c)) for o, c in zip(result, circulation))
        if with_base:
            if self.rational is None:
                base = self.base_fields(points)
            return result, base
        return result

    def wall_factor(self, points):
        return elliptic_wall_factor(points, self.bounds, self.hole, self.wall_width)

    def base_fields(self, points):
        result = channel_lift(points, self.bounds[2], self.peak)
        if self.rational is not None:
            correction = self.rational.evaluate(points, self.rational_lift)
            return tuple(a + b for a, b in zip(result, correction))
        return (
            _stream_product(result, self.wall_factor(points))
            if self.factored_wall
            else result
        )

    def _gram(self, ops):
        u, v, xy, yy, minus_xx = ops
        w = self.weights[:, None]
        return (
            u.T @ (w * u) + v.T @ (w * v),
            2 * xy.T @ (w * xy) + yy.T @ (w * yy) + minus_xx.T @ (w * minus_xx),
        )

    def explicit(self, state, time=0.0):
        u, v, xy, yy, minus_xx = [
            o @ state + lift
            for o, lift in zip(self.operators_fluid, self.lift_fields[1:])
        ]
        convection_x, convection_y = u * xy + v * yy, u * minus_xx - v * xy
        bu, bv = self.operators_fluid[:2]
        rhs = -bu.T @ (self.weights * convection_x) - bv.T @ (
            self.weights * convection_y
        )
        outu, outv = [o @ state + lift for o, lift in zip(self.out_ops, self.out_lift)]
        incoming = np.minimum(outu, 0) * self.out_weights
        rhs += self.out_ops[0].T @ (incoming * outu) + self.out_ops[1].T @ (
            incoming * outv
        )
        return rhs - self.linear_lift

    def rhs(self, state):
        return la.cho_solve(
            self.mass_factor, self.explicit(state) - self.linear @ state
        )

    def force_load(self, force):
        """Integrate a physical vector force against the same curl test space."""
        force = np.asarray(force)
        if force.shape != (len(self.points), 2) or not np.all(np.isfinite(force)):
            raise ValueError(
                "Expected a finite vector force at every fluid quadrature point"
            )
        u, v = self.operators_fluid[:2]
        return u.T @ (self.weights * force[:, 0]) + v.T @ (self.weights * force[:, 1])

    def stepper(self, dt, *, device=None):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive")
        if device is not None:
            from .immersed_flow_gpu import GPUImmersedFlowStepper

            return GPUImmersedFlowStepper(self, dt, device)
        factor = la.cho_factor(self.mass + dt / 2 * self.linear)
        return ImmersedFlowStepper(self, dt, factor)

    def coefficients(self, state):
        return self.lift_coefficients + self.transform @ state

    def grid(self, state, x, y):
        return self._grid(state, x, y)

    def grid_many(self, states, x, y, *, batch_size=64):
        """Yield snapshots, sharing rational basis evaluation across each batch.

        The spatial grid is fixed; only coefficients vary with time. This uses
        the existing multiple-RHS rational evaluator and bounds temporary
        storage by batch_size, without caching a large dense evaluation matrix.
        States must be explicitly downloaded before calling this host renderer.
        """
        states = np.asarray(states)
        if states.ndim != 2 or states.shape[1] != self.dofs:
            raise ValueError("Expected (snapshots, dofs) states")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not self.rational_wall:
            for state in states:
                yield self.grid(state, x, y)
            return
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack((xx.ravel(), yy.ravel()))
        physical = self.hole.level(points) >= 1 - 1e-13
        for start in range(0, len(states), batch_size):
            batch = states[start:start + batch_size]
            coefficients = self.lift_coefficients[:, None] + self.transform @ batch.T
            rational_coeff = self.rational_lift[:, None] + self.rational_modes @ (
                self.rational_map @ coefficients
            )
            correction = self.rational.evaluate(points[physical], rational_coeff)
            for k, state in enumerate(batch):
                yield self._grid(state, x, y, tuple(a[:, k] for a in correction))

    def _grid(self, state, x, y, rational_correction=None):
        bx, by = self._line_values(0, x), self._line_values(1, y)
        coefficients = self.coefficients(state)
        c = (coefficients[:-1] if self.factored_wall else coefficients).reshape(
            self.shape
        )

        def apply(i, j):
            return tensor_product(bx[i], by[j], c).T

        xx, yy = np.meshgrid(x, y)
        lift = [
            a.reshape(xx.shape)
            for a in channel_lift(
                np.column_stack((xx.ravel(), yy.ravel())), self.bounds[2], self.peak
            )
        ]
        psi, u, v = apply(0, 0) + lift[0], apply(0, 1) + lift[1], -apply(1, 0)
        xy = apply(1, 1)
        uy, vx = apply(0, 2) + lift[4], -apply(2, 0)
        if self.factored_wall:
            jet = tuple(
                a.reshape(xx.shape)
                for a in self.wall_factor(np.column_stack((xx.ravel(), yy.ravel())))
            )
            psi, u, v, xy, uy, vx = _stream_product((psi, u, v, xy, uy, vx), jet)
            q, qx, qy, qxx, qxy, qyy = jet
            circulation = coefficients[-1]
            psi += circulation * (1 - q)
            u -= circulation * qy
            v += circulation * qx
            xy -= circulation * qxy
            uy -= circulation * qyy
            vx += circulation * qxx
        if self.rational_wall:
            points = np.column_stack((xx.ravel(), yy.ravel()))
            physical = self.hole.level(points) >= 1 - 1e-13
            rational_coeff = self.rational_lift + self.rational_modes @ (
                self.rational_map @ coefficients
            )
            correction = (self.rational.evaluate(points[physical], rational_coeff)
                          if rational_correction is None else rational_correction)
            for field, addition in zip((psi, u, v, xy, uy, vx), correction):
                field.flat[np.flatnonzero(physical)] += addition
                field.flat[np.flatnonzero(~physical)] = np.nan
        return dict(psi=psi, u=u, v=v, vorticity=vx - uy, divergence=xy - xy)

    def evaluate(self, state, points):
        if self.rational_wall and np.any(
            self.hole.level(np.asarray(points)) < 1 - 1e-12
        ):
            raise ValueError(
                "Rational correction is defined only in the fluid and on its boundary"
            )
        ops = self.operators(points)
        return tuple(
            o @ self.coefficients(state) + lift
            for o, lift in zip(ops, self.base_fields(points))
        )

    def diagnostics(self, state):
        fields = [
            o @ state + lift
            for o, lift in zip(self.operators_fluid, self.lift_fields[1:])
        ]
        u, v, xy, uy, vx = fields
        outu, outv = [o @ state + lift for o, lift in zip(self.out_ops, self.out_lift)]
        derivative = self.rhs(state)
        du = u - channel_lift(self.points, self.bounds[2], self.peak)[1]
        return dict(
            kinetic_energy=float(np.sum(self.weights * (u * u + v * v)) / 2),
            max_speed=float(np.max(np.hypot(u, v))),
            min_outlet_u=float(np.min(outu)),
            flux_in=4 * self.peak * self.bounds[2] / 3,
            flux_out=float(self.out_weights @ outu),
            acceleration_l2=float(np.sqrt(max(derivative @ self.mass @ derivative, 0))),
            dissipation=float(
                self.nu * np.sum(self.weights * (2 * xy * xy + uy * uy + vx * vx))
            ),
            sponge_perturbation_dissipation=float(
                np.sum(self.weights * self.sigma * (du * du + v * v))
            ),
        )


@dataclass
class ImmersedFlowStepper:
    plan: ImmersedFlowPlan
    dt: float
    factor: tuple

    def step(self, state, time=0.0, load=None):
        p = self.plan

        def explicit(a, t):
            value = p.explicit(a, t)
            return value if load is None else value + load(t)

        return imex_midpoint(
            state,
            time,
            self.dt,
            lambda a: p.mass @ a,
            lambda a: p.linear @ a,
            explicit,
            lambda b: la.cho_solve(self.factor, b),
        )
