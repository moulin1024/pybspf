"""Same-quadrature matrix-free convex Poisson with coarse deflation.

The physical objective is identical to ConvexPoissonPlan. A separate explicit
box-H2 Tikhonov term stabilizes the continuation outside the physical domain.
Only the small coarse space is factored; no fine 2D matrix is assembled.
"""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from time import perf_counter
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, lsmr
from scipy.special import roots_legendre

from .convex_poisson import (
    ArcLengthBoundary,
    ConvexPoissonSolution,
    trace_transform,
    validate_convex,
)
from .stream_navier_stokes import StreamLine, _stream_line, stream_evaluate_line


@lru_cache(maxsize=3)
def _implementation_key(kind):
    # Bump geometry-v1 if grouped factor storage/assembly changes. Iteration
    # changes must not invalidate expensive, unchanged MPFR line factors.
    h = hashlib.sha256()
    sources = [
        "stream_navier_stokes.py",
        "_weak_basis.py",
        "pressure.py",
        "basis.py",
        "plans.py",
        "operators.py",
        "endpoints.py",
        "_compressed_transform.py",
    ]
    if kind != "line":
        sources += ["embedded_poisson.py", "convex_poisson.py"]
        h.update(b"tensor-geometry-v1")
    if kind == "coarse":
        sources += ["convex_poisson_tensor.py"]
    for name in sources:
        path = Path(__file__).parent / name
        h.update(path.name.encode())
        h.update(path.read_bytes())
    import gmpy2

    h.update(
        repr(
            (np.__version__, scipy.__version__, jax.__version__, gmpy2.version())
        ).encode()
    )
    return h.hexdigest()


def _cache_file(cache_dir, kind, parameters):
    if cache_dir is None:
        return None

    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Unsupported cache-key value: {type(value).__name__}")

    payload = json.dumps(
        [_implementation_key(kind), kind, parameters], sort_keys=True, default=scalar
    )
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root / (kind + "_" + hashlib.sha256(payload.encode()).hexdigest() + ".npz")


def _save_arrays(path, arrays):
    if path is None:
        return
    # Atomic numeric-only cache; never deserialize executable pickle data.
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, **arrays)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@lru_cache(maxsize=8)
def _line(nodes, half_width, cache_dir):
    path = _cache_file(cache_dir, "line", [nodes, half_width, 12, 12])
    if path is not None and path.exists():
        with np.load(path, allow_pickle=False) as data:
            return StreamLine(*(jnp.asarray(data[k]) for k in StreamLine._fields))
    line = _stream_line(
        np.linspace(-half_width, half_width, nodes),
        clamped=False,
        dirichlet=False,
        endpoint_points=12,
        chebyshev_modes=12,
    )
    _save_arrays(path, {k: np.asarray(v) for k, v in line._asdict().items()})
    return line


def trace_adjoint(values, length, count, order=1.5):
    """Euclidean adjoint of the real-packed trace_transform (even count)."""
    modes = count // 2 + 1
    frequency = 2 * np.pi * np.fft.rfftfreq(count, d=length / count)
    multiplicity = np.full(modes, 2.0)
    multiplicity[[0, -1]] = 1
    scale = np.sqrt(length / count / multiplicity) * (1 + frequency**2) ** (order / 2)
    return np.fft.irfft(
        (values[:modes] + 1j * values[modes:]) * scale, n=count, norm="ortho"
    )


class BoxH2Tensor:
    """Exact discrete box H2 factor and adjoint, using 1D Grams only."""

    def __init__(self, line):
        self.k = np.diag(np.asarray(line.lam))
        self.j = np.asarray(line.bending)

        # Compute square roots via eigensystems; clip only roundoff negatives.
        def root(a):
            s, v = la.eigh((a + a.T) / 2)
            if s.min() < -1e-11 * max(1, s.max()):
                raise ValueError("Indefinite derivative Gram")
            return (v * np.sqrt(np.maximum(s, 0))) @ v.T

        self.rk, self.rj = root(self.k), root(self.j)

    def gram(self, c):
        k, j = self.k, self.j
        return c + k @ c + c @ k + j @ c + c @ j + 2 * k @ c @ k

    def factor(self, c):
        k, j = self.rk, self.rj
        return np.concatenate(
            [v.ravel() for v in (c, k @ c, c @ k, j @ c, c @ j, np.sqrt(2) * k @ c @ k)]
        )

    def adjoint(self, z):
        n = len(self.k)
        c, x, y, xx, yy, xy = np.asarray(z).reshape(6, n, n)
        return (
            c
            + self.rk @ x
            + y @ self.rk
            + self.rj @ xx
            + yy @ self.rj
            + np.sqrt(2) * self.rk @ xy @ self.rk
        )


class PoissonIterationError(RuntimeError):
    """An iterative solve exhausted its limit; diagnostics remain inspectable."""

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(
            f"Poisson iteration did not converge: {diagnostics['stop_code']=}, "
            f"iterations={diagnostics['iterations']}, "
            f"relative data residual={diagnostics['relative_data_residual']:.3e}"
        )


def _bounding_box_scaling(line, domain):
    """Full-rank interval mass whitening; changes preconditioning, not the space."""
    from scipy.interpolate import PPoly

    transforms, roots = [], []
    n = len(line.x)
    for axis in range(2):
        poly = PPoly.from_spline((domain.curve.t, domain.curve.c[:, axis], 3))
        extrema = poly.derivative().roots(extrapolate=False)
        extrema = extrema[
            np.isfinite(extrema) & (extrema >= 0) & (extrema <= domain.period)
        ]
        values = domain.curve(np.r_[np.arange(domain.period + 1), extrema])[:, axis]
        lo, hi = values.min(), values.max()
        breaks = np.linspace(float(line.x[0]), float(line.x[-1]), 20)
        breaks = np.r_[lo, breaks[(breaks > lo) & (breaks < hi)], hi]
        q, w = roots_legendre(max(24, int(np.ceil(1.5 * np.pi * (n - 1) / 19)) + 8))
        points = np.concatenate(
            [(a + b) / 2 + (b - a) / 2 * q for a, b in zip(breaks[:-1], breaks[1:])]
        )
        weights = np.concatenate(
            [(b - a) / 2 * w for a, b in zip(breaks[:-1], breaks[1:])]
        )
        b, g, h = stream_evaluate_line(line, points)
        m = b.T @ (weights[:, None] * b)
        eigen, rotation = la.eigh((m + m.T) / 2)
        whiten = rotation / np.sqrt(np.maximum(eigen, 1e-12 * eigen.max()))
        g, h = g @ whiten, h @ whiten
        metric = (
            np.eye(n) + 2 * g.T @ (weights[:, None] * g) + h.T @ (weights[:, None] * h)
        )
        eigen, rotation = la.eigh((metric + metric.T) / 2)
        if eigen.min() <= 0:
            raise ValueError("Invalid bounding-box H2 metric")
        transforms.append(whiten @ rotation)
        roots.append(np.sqrt(eigen))
    return transforms, roots


class TensorConvexPoissonPlan:
    """Matrix-free graph least squares with projected coarse correction.

    Solves ||A c-b||² + regularization² ||c||²_box_H2. The physical A uses the
    *same* domain.volume_rule and arclength trace as the dense reference.
    ``coarse_nodes=None`` disables deflation for controlled comparisons.
    Cached line and geometry factors are independent of f/g and output grids.
    """

    def __init__(
        self,
        domain,
        *,
        nodes=33,
        half_width=1.2,
        volume_order=16,
        boundary_count=None,
        sobolev_order=1.5,
        coarse_nodes=17,
        coarse_rcond=1e-10,
        regularization=1e-12,
        tolerance=1e-12,
        maxiter=2000,
        cache_dir=None,
        preconditioner="box",
        data_tolerance=None,
        coarse_space="projected",
    ):
        start = perf_counter()
        if not jax.config.x64_enabled:
            raise ValueError("Enable jax_enable_x64 before constructing BSPF factors")
        if not isinstance(nodes, (int, np.integer)) or nodes < 17:
            raise ValueError("nodes must be an integer >=17")
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        if not 0 < tolerance < 1 or not 0 < coarse_rcond < 1 or maxiter < 1:
            raise ValueError(
                "Invalid iteration tolerance, coarse cutoff or iteration limit"
            )
        if coarse_nodes is not None and (
            not isinstance(coarse_nodes, (int, np.integer))
            or not 17 <= coarse_nodes <= nodes
        ):
            raise ValueError("Require 17 <= coarse_nodes <= nodes, or None")
        if not isinstance(volume_order, (int, np.integer)) or volume_order < 2:
            raise ValueError("volume_order must be an integer >=2")
        if not isinstance(maxiter, (int, np.integer)):
            raise ValueError("maxiter must be an integer")
        if preconditioner not in ("box", "bounding_box"):
            raise ValueError("Unknown preconditioner")
        if data_tolerance is not None and not 0 < data_tolerance < 1:
            raise ValueError("data_tolerance must lie in (0,1), or be None")
        self.data_tolerance = data_tolerance
        if coarse_space not in ("projected", "spectral"):
            raise ValueError("coarse_space must be 'projected' or 'spectral'")
        self.coarse_space = coarse_space
        if not np.isfinite(half_width) or np.max(abs(domain.controls)) >= half_width:
            raise ValueError("Boundary must lie strictly inside the auxiliary box")
        if not np.isfinite(sobolev_order) or sobolev_order < 0:
            raise ValueError("Invalid Sobolev trace order")
        self.geometry_checks = validate_convex(domain)
        self.domain, self.nodes = domain, nodes
        self.sobolev_order = sobolev_order
        self.regularization, self.tolerance, self.maxiter = (
            regularization,
            tolerance,
            maxiter,
        )
        self.coarse_nodes, self.coarse_rcond = coarse_nodes, coarse_rcond
        self.boundary_count = (
            boundary_count
            if boundary_count is not None
            else 2 ** int(np.ceil(np.log2(8 * nodes)))
        )
        if (
            not isinstance(self.boundary_count, (int, np.integer))
            or self.boundary_count < 8
            or self.boundary_count % 2
        ):
            raise ValueError("An even boundary count >=8 is required")
        self.validation_cache = {}
        cache_dir = str(Path(cache_dir).resolve()) if cache_dir is not None else None
        self.line = _line(nodes, half_width, cache_dir)
        lined = perf_counter()
        path = _cache_file(
            cache_dir,
            "geometry",
            [
                nodes,
                half_width,
                volume_order,
                self.boundary_count,
                domain.controls.tolist(),
            ],
        )
        self.arc = ArcLengthBoundary(domain)
        self.geometry_cache_hit = path is not None and path.exists()
        if self.geometry_cache_hit:
            with np.load(path, allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
        else:
            points, weights, _ = domain.volume_rule(volume_order)
            boundary, parameters = self.arc.sample(self.boundary_count)
            xs, index = np.unique(points[:, 0], return_inverse=True)
            order = np.argsort(index, kind="stable")
            starts = np.r_[0, np.cumsum(np.bincount(index))[:-1]]
            bx, _, hx = stream_evaluate_line(self.line, xs)
            ys, inverse = np.unique(points[order, 1], return_inverse=True)
            by, _, hy = stream_evaluate_line(self.line, ys)
            arrays = dict(
                points=points,
                weights=weights,
                boundary=boundary,
                parameters=parameters,
                order=order,
                starts=starts,
                index=index[order],
                bx=bx,
                hx=hx,
                by=by[inverse],
                hy=hy[inverse],
                edge_x=stream_evaluate_line(self.line, boundary[:, 0])[0],
                edge_y=stream_evaluate_line(self.line, boundary[:, 1])[0],
            )
            _save_arrays(path, arrays)
        for key, value in arrays.items():
            value.setflags(write=False)
            setattr(self, key, value)
        self.sqrt_weights = np.sqrt(self.weights)
        self.ndofs = nodes**2
        self.data_rows = len(self.points) + self.boundary_count + 2
        self.h2 = BoxH2Tensor(self.line)
        self.preconditioner = preconditioner
        if preconditioner == "box":
            metric = np.eye(nodes) + 2 * self.h2.k + self.h2.j
            eigen, rotation = la.eigh((metric + metric.T) / 2)
            if eigen.min() <= 0:
                raise ValueError("Invalid separable H2 preconditioner")
            self.tx = self.ty = rotation
            root_x = root_y = np.sqrt(eigen)
        elif preconditioner == "bounding_box":
            (self.tx, self.ty), (root_x, root_y) = _bounding_box_scaling(
                self.line, domain
            )
        else:
            raise ValueError("Unknown preconditioner")
        self.denominator = root_x[:, None] + root_y[None, :]
        self.operator = LinearOperator(
            (self.data_rows, self.ndofs),
            matvec=self.data_apply,
            rmatvec=self.data_adjoint,
            dtype=float,
        )
        self.augmented = LinearOperator(
            (self.data_rows + 6 * self.ndofs, self.ndofs),
            matvec=self._apply,
            rmatvec=self._adjoint,
            dtype=float,
        )
        factored = perf_counter()
        self.coarse_left = np.empty((self.augmented.shape[0], 0))
        self.coarse_right = np.empty((self.ndofs, 0))
        self.coarse_singular = np.empty(0)
        coarse_path = (
            _cache_file(
                cache_dir,
                "coarse",
                [
                    nodes,
                    half_width,
                    volume_order,
                    self.boundary_count,
                    domain.controls.tolist(),
                    sobolev_order,
                    coarse_nodes,
                    coarse_rcond,
                    regularization,
                    preconditioner,
                    coarse_space,
                ],
            )
            if coarse_nodes is not None
            else None
        )
        self.coarse_cache_hit = coarse_path is not None and coarse_path.exists()
        if self.coarse_cache_hit:
            with np.load(coarse_path, allow_pickle=False) as data:
                self.coarse_left = data["left"]
                self.coarse_right = data["right"]
                self.coarse_singular = data["singular"]
                self.tx, self.ty = data["tx"], data["ty"]
                self.denominator = data["denominator"]
        elif coarse_nodes is not None:
            if coarse_space == "spectral":
                # The StreamLine basis is mass-orthonormal and ordered by
                # stiffness eigenvalue. This algebraic coarse space is an
                # EXACT subspace of the fine space, without geometric coarsening.
                transfer = np.eye(nodes, coarse_nodes)
            else:
                coarse = _line(coarse_nodes, half_width, cache_dir)
                # L2 projection resolving BOTH spaces; no assumed nesting.
                qline = (
                    self.line if len(self.line.points) >= len(coarse.points) else coarse
                )
                fine_b = stream_evaluate_line(self.line, np.asarray(qline.points))[0]
                coarse_b = stream_evaluate_line(coarse, np.asarray(qline.points))[0]
                mass = fine_b.T @ (np.asarray(qline.weights)[:, None] * fine_b)
                transfer = la.solve(
                    mass,
                    fine_b.T @ (np.asarray(qline.weights)[:, None] * coarse_b),
                    assume_a="pos",
                )
            # Map coarse tensor functions into preconditioned fine coordinates.
            tx = la.solve(self.tx, transfer)
            ty = la.solve(self.ty, transfer)
            z = (tx[:, None, :, None] * ty[None, :, None, :]).reshape(
                self.ndofs, coarse_nodes**2
            )
            z *= self.denominator.ravel()[:, None]
            z, _ = la.qr(z, mode="economic")
            coarse_matrix = np.column_stack([self.augmented @ col for col in z.T])
            left, singular, right = la.svd(coarse_matrix, full_matrices=False)
            keep = singular > coarse_rcond * singular[0]
            self.coarse_left = left[:, keep]
            self.coarse_singular = singular[keep]
            self.coarse_right = z @ right[keep].T
            _save_arrays(
                coarse_path,
                dict(
                    left=self.coarse_left,
                    right=self.coarse_right,
                    singular=self.coarse_singular,
                    tx=self.tx,
                    ty=self.ty,
                    denominator=self.denominator,
                ),
            )
        self.setup_diagnostics = dict(
            line_seconds=lined - start,
            factors_seconds=factored - lined,
            coarse_seconds=perf_counter() - factored,
            setup_seconds=perf_counter() - start,
            geometry_cache_hit=self.geometry_cache_hit,
            coarse_cache_hit=self.coarse_cache_hit,
            coarse_rank=len(self.coarse_singular),
            coarse_space=self.coarse_space,
            factor_storage_bytes=sum(v.nbytes for v in arrays.values()),
            coarse_storage_bytes=self.coarse_left.nbytes
            + self.coarse_right.nbytes
            + self.coarse_singular.nbytes,
        )

    def decode(self, a):
        return (
            self.tx
            @ (np.asarray(a).reshape(self.nodes, self.nodes) / self.denominator)
            @ self.ty.T
        )

    def _decode_adjoint(self, c):
        return ((self.tx.T @ c @ self.ty) / self.denominator).ravel()

    def data_apply(self, c):
        c = np.asarray(c).reshape(self.nodes, self.nodes)
        # Group points by x to share x contractions along each curved-domain slice.
        volume = -np.sum(
            (self.hx @ c)[self.index] * self.by + (self.bx @ c)[self.index] * self.hy,
            axis=1,
        )
        result = np.empty(len(volume))
        result[self.order] = volume
        edge = np.sum((self.edge_x @ c) * self.edge_y, axis=1)
        return np.r_[
            self.sqrt_weights * result,
            trace_transform(edge, self.arc.length, self.sobolev_order),
        ]

    def data_adjoint(self, z):
        z = np.asarray(z).reshape(-1)
        count = len(self.points)
        v = (self.sqrt_weights * z[:count])[self.order]
        y = np.add.reduceat(v[:, None] * self.by, self.starts, axis=0)
        yy = np.add.reduceat(v[:, None] * self.hy, self.starts, axis=0)
        c = -(self.hx.T @ y + self.bx.T @ yy)
        edge = trace_adjoint(
            z[count:], self.arc.length, self.boundary_count, self.sobolev_order
        )
        c += self.edge_x.T @ (edge[:, None] * self.edge_y)
        return c.ravel()

    def _apply(self, a):
        c = self.decode(a)
        return np.r_[self.data_apply(c), self.regularization * self.h2.factor(c)]

    def _adjoint(self, z):
        z = np.asarray(z).reshape(-1)
        c = self.data_adjoint(z[: self.data_rows]).reshape(self.nodes, self.nodes)
        c += self.regularization * self.h2.adjoint(z[self.data_rows :])
        return self._decode_adjoint(c)

    def _project(self, z):
        return z - self.coarse_left @ (self.coarse_left.T @ z)

    def _project_domain(self, a):
        return a - self.coarse_right @ (self.coarse_right.T @ a)

    def solve(self, forcing, boundary_data, *, allow_unconverged=False, progress=False):
        start = perf_counter()
        f, g = (
            np.asarray(forcing(self.points), float),
            np.asarray(boundary_data(self.boundary), float),
        )
        if f.shape != (len(self.points),) or g.shape != (len(self.boundary),):
            raise ValueError("f and g must return one value per supplied point")
        if not np.all(np.isfinite(f)) or not np.all(np.isfinite(g)):
            raise ValueError("Nonfinite Poisson data")
        b = np.r_[
            self.sqrt_weights * f,
            trace_transform(g, self.arc.length, self.sobolev_order),
        ]
        rhs = np.r_[b, np.zeros(6 * self.ndofs)]
        coarse_a = self.coarse_right @ (
            (self.coarse_left.T @ rhs) / self.coarse_singular
        )
        coarse_residual = self.data_apply(self.decode(coarse_a)) - b
        coarse_target_met = self.data_tolerance is not None and la.norm(
            coarse_residual
        ) <= self.data_tolerance * la.norm(b)
        projected = LinearOperator(
            self.augmented.shape,
            matvec=lambda a: self._project(self._apply(self._project_domain(a))),
            rmatvec=lambda z: self._project_domain(self._adjoint(self._project(z))),
            dtype=float,
        )
        if (
            len(self.coarse_singular) == self.ndofs
            or not np.any(rhs)
            or coarse_target_met
        ):
            # A full coarse span already solves this regularized problem.
            # Iterating on its roundoff-sized complement would amplify noise.
            answer = (np.zeros(self.ndofs), 0, 0)
        else:
            projected_rhs = self._project(rhs)
            btol = self.tolerance
            if self.data_tolerance is not None:
                # Normalize a physical residual target by the FULL rhs, not
                # the much smaller rhs remaining after coarse correction.
                btol = min(
                    0.9,
                    0.5
                    * self.data_tolerance
                    * la.norm(rhs)
                    / max(la.norm(projected_rhs), 1e-300),
                )
            answer = lsmr(
                projected,
                projected_rhs,
                atol=self.tolerance,
                btol=btol,
                maxiter=self.maxiter,
                conlim=1e14,
                show=progress,
            )
        a = self._project_domain(answer[0])
        a += self.coarse_right @ (
            (self.coarse_left.T @ (rhs - self._apply(a))) / self.coarse_singular
        )
        c = self.decode(a).ravel()
        residual = self.operator @ c - b
        full_residual = self._apply(a) - rhs
        normal_residual = self._adjoint(full_residual)
        optimization_converged = answer[1] in (0, 1, 2, 4, 5) and not coarse_target_met
        relative_data_residual = float(la.norm(residual) / max(la.norm(b), 1e-300))
        data_target_met = (
            self.data_tolerance is not None
            and relative_data_residual <= self.data_tolerance
        )
        converged = (
            optimization_converged if self.data_tolerance is None else data_target_met
        )
        info = dict(
            backend="tensor_two_level_lsmr",
            preconditioner=self.preconditioner,
            converged=converged,
            convergence_criterion="regularized_least_squares"
            if self.data_tolerance is None
            else "physical_data_residual",
            optimization_converged=optimization_converged,
            data_tolerance=self.data_tolerance,
            data_target_met=data_target_met,
            stop_code=int(answer[1]),
            iterations=int(answer[2]),
            regularization=self.regularization,
            tolerance=self.tolerance,
            ndofs=self.ndofs,
            boundary_count=self.boundary_count,
            volume_samples=len(self.points),
            trace_order=self.sobolev_order,
            relative_data_residual=relative_data_residual,
            augmented_residual=float(la.norm(full_residual)),
            preconditioned_normal_residual=float(la.norm(normal_residual)),
            box_h2_norm=float(
                la.norm(self.h2.factor(c.reshape(self.nodes, self.nodes)))
            ),
            solve_seconds=perf_counter() - start,
            physical_data_only=True,
            **self.setup_diagnostics,
        )
        if not converged and not allow_unconverged:
            raise PoissonIterationError(info)
        return ConvexPoissonSolution(self, c, info)
