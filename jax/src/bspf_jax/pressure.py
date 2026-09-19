"""Masked BSPF pressure projection, with optional local Chebyshev jets.

The default backend uses JAX numerical setup and application. Optional DCT/HODLR
compression uses SciPy/NumPy host preprocessing and JAX application.
This intentionally uses the benchmark's QR fit and endpoint-exclusive FFT,
not the general calculus plan's KKT fit and FFT convention.
"""

from functools import partial
from math import factorial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np

from .basis import basis_matrix, open_knots
from .endpoints import chebyshev_boundary_blocks
from .plans import _integer
from ._compressed_transform import (
    CompressedTransforms,
    apply_transform,
    build_transforms,
)


class PressureLine(NamedTuple):
    x: jax.Array
    weights: jax.Array
    projector: jax.Array
    low: jax.Array
    multiplier: jax.Array
    hei: jax.Array
    endpoint_inverse: jax.Array
    coupling: jax.Array
    eigenvalues: jax.Array
    vectors: jax.Array | None
    inverse_vectors: jax.Array | None
    null_basis: jax.Array
    compressed: CompressedTransforms | None = None


class PressurePoisson2DPlan(NamedTuple):
    """Immutable PyTree, using (nx, ny) scalars and (nx, ny, 2) vectors."""

    x: PressureLine
    y: PressureLine
    mask: jax.Array
    weights: jax.Array
    corners: jax.Array
    wall_indices: jax.Array
    denominators: jax.Array
    delta: jax.Array
    wall_scales: jax.Array
    wall_q: jax.Array
    wall_r: jax.Array


class PressurePoisson2DResult(NamedTuple):
    """Final unlifted diagnostics; check converged even inside JIT workflows.

    wall_gradient_fit_linf is NaN when wall completion was not requested.
    converged tests the Schur equation, not exact satisfaction of the wall fit.
    """

    pressure: jax.Array
    schur_residual_linf: jax.Array
    schur_residual_l2: jax.Array
    wall_gradient_fit_linf: jax.Array
    converged: jax.Array


def _fourier(values, multiplier):
    d = jnp.fft.ifft(jnp.fft.fft(values[:-1], axis=0) * multiplier[:, None], axis=0)
    return jnp.concatenate((d, d[:1]), axis=0).real


def _differentiate(line, values, axis):
    values = jnp.moveaxis(values, axis, 0)
    shape = values.shape
    flat = values.reshape(shape[0], -1)
    out = _fourier(flat, line.multiplier) + line.low @ (line.projector @ flat)
    return jnp.moveaxis(out.reshape(shape), 0, axis)


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6))
def _line_projector(x, q, n_basis, degree, points, method, modes, alpha):
    """Compile spline/jet setup together instead of hundreds of tiny kernels.

    Returns the same constrained least-squares projector used by pressure and
    streamfunction plans. Streamfunction setup needs no pressure eigensystem.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    n = x.size
    h = x[1] - x[0]
    weights = jnp.full_like(x, h).at[0].set(h / 2).at[-1].set(h / 2)
    knots = open_knots(x[0], x[-1], degree=degree, n_basis=n_basis)
    B = basis_matrix(knots, x, degree=degree)
    ends = jnp.stack(
        [
            basis_matrix(knots, x[jnp.array([0, n - 1])], degree=degree, derivative=k)
            for k in range(q)
        ],
        axis=1,
    )
    C = ends.reshape(2 * q, n_basis)
    scales = 1 / jnp.max(abs(C), axis=1)
    Q, R = jnp.linalg.qr((C * scales[:, None]).T, mode="complete")
    Q1, Q2 = Q[:, : 2 * q], Q[:, 2 * q :]
    T = Q1 @ jl.solve_triangular(R[: 2 * q, : 2 * q].T, jnp.diag(scales), lower=True)
    BW = B.T * weights
    H = BW @ B
    H22 = Q2.T @ H @ Q2
    factor = jl.cho_factor((H22 + H22.T) / 2, lower=True)
    F0 = Q2 @ jl.cho_solve(factor, Q2.T @ BW)
    J = T - Q2 @ jl.cho_solve(factor, Q2.T @ H @ T)
    if method == "chebyshev":
        blocks = chebyshev_boundary_blocks(
            x, order=q, points=points, modes=modes, alpha=alpha, penalty_power=4
        )
        jets = jnp.zeros((2 * q, n)).at[:q, :points].set(blocks[0])
        jets = jets.at[q:, -points:].set(blocks[1])
    else:
        offsets = jnp.arange(points, dtype=x.dtype)
        left = jnp.stack([offsets**k / factorial(k) for k in range(q)], axis=1)
        right = jnp.stack([(-offsets) ** k / factorial(k) for k in range(q)], axis=1)
        units = (h ** jnp.arange(q))[:, None]
        jets = (
            jnp.zeros((2 * q, n))
            .at[:q, :points]
            .set(jnp.linalg.pinv(left, rtol=1e-14) / units)
        )
        jets = jets.at[q:, -points:].set(
            jnp.linalg.pinv(right, rtol=1e-14)[:, ::-1] / units
        )
    P = F0 + J @ jets
    return P, weights, knots, B


def _make_line(x, q, n_basis, degree, points, method, modes, alpha):
    x = jnp.asarray(x, dtype=jnp.float64)
    n, h = x.size, x[1] - x[0]
    P, weights, knots, B = _line_projector(
        x, q, n_basis, degree, points, method, modes, alpha
    )
    multiplier = 2j * jnp.pi * jnp.fft.fftfreq(n - 1, d=h)
    if (n - 1) % 2 == 0:
        multiplier = multiplier.at[(n - 1) // 2].set(0)
    low = basis_matrix(knots, x, degree=degree, derivative=1) - _fourier(B, multiplier)
    D = _fourier(jnp.eye(n), multiplier) + low @ P
    mask = jnp.ones(n).at[0].set(0).at[-1].set(0)
    H = D @ (mask[:, None] * D)
    e = jnp.array([0, n - 1])
    hei = H[e, 1:-1]
    ei = jnp.linalg.solve(H[jnp.ix_(e, e)], jnp.eye(2))
    coupling = H[1:-1, e] @ ei
    A = H[1:-1, 1:-1] - coupling @ hei
    eigenvalues, vectors = jnp.linalg.eig(A)
    order = jnp.argsort(abs(eigenvalues))
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    inverse = jnp.linalg.solve(vectors, jnp.eye(n - 2))
    scale = jnp.maximum(jnp.max(abs(eigenvalues)), 1.0)
    if not bool(
        (jnp.max(abs(eigenvalues[:2])) <= 1e-8 * scale)
        & (abs(eigenvalues[2]) >= 1e-8 * scale)
    ):
        raise ValueError("Expected exactly two line null eigenmodes.")
    _, singular, vh = jnp.linalg.svd(D[1:-1], full_matrices=True)
    if not bool(jnp.all(singular > 1e-12 * singular[0])):
        raise ValueError("Interior gradient must have nullity two.")
    Z = vh[-2:].T
    z = Z @ (Z.T @ ((-1.0) ** jnp.arange(n)))
    z -= weights @ z / jnp.sum(weights)
    norm = jnp.sqrt(weights @ (z * z) / jnp.sum(weights))
    if not bool(jnp.isfinite(norm) & (norm >= 1e-8)):
        raise ValueError("Unable to construct second gradient-null mode.")
    return PressureLine(
        x,
        weights,
        P,
        low,
        multiplier,
        hei,
        ei,
        coupling,
        eigenvalues,
        vectors,
        inverse,
        jnp.column_stack([jnp.ones(n), z / norm]),
    )


def plan_pressure_poisson2d(
    x,
    y,
    *,
    q=9,
    n_basis=32,
    degree=13,
    baseline_points=14,
    endpoint_method="taylor",
    chebyshev_modes=12,
    endpoint_regularization=1e-12,
    transform_backend="dense",
    compression_tolerance=1e-12,
    compression_leaf_size=16,
    protected_modes=8,
    compression_layout="layered",
):
    """Build the masked pressure plan outside JIT, requiring explicit JAX x64.

    Uniform, endpoint-inclusive x/y grids; no Chebyshev nodes needed. Options
    match the NumPy PressurePoisson2D solver. 'taylor' preserves its degree-q-1
    least-squares jets, not the general JAX plan's interpolating FD stencil.
    'chebyshev' reuses the local augmented-QR estimator from differentiation.
    Both gradient and divergence use the same newly assembled D1.

    Setup includes nonsymmetric eigendecompositions (backend support required)
    and host validation. Kernels support JIT, vmap, and autodiff with respect
    to fields with a fixed plan; differentiation through setup is unsupported.
    """
    if transform_backend not in ("dense", "dct_hodlr"):
        raise ValueError("transform_backend must be 'dense' or 'dct_hodlr'")
    if not jax.config.x64_enabled:
        raise ValueError("Pressure projection requires jax_enable_x64=True.")
    for name, value in dict(
        q=q, n_basis=n_basis, degree=degree, baseline_points=baseline_points
    ).items():
        _integer(name, value, 1)
    if not q <= degree + 1 <= n_basis or n_basis <= 2 * q:
        raise ValueError("Require q <= degree+1 <= n_basis and n_basis > 2*q.")
    for grid in [x, y]:
        host = np.asarray(grid)
        if (
            host.ndim != 1
            or host.size < 5
            or np.iscomplexobj(host)
            or not np.all(np.isfinite(host))
        ):
            raise ValueError(
                "Grids must be finite real 1D arrays with at least five nodes."
            )
        dx = np.diff(host)
        if np.any(dx <= 0) or not np.allclose(dx, dx[0], rtol=1e-10, atol=0):
            raise ValueError("Grids must be uniform and strictly increasing.")
        if n_basis >= host.size or not q <= baseline_points <= host.size:
            raise ValueError(
                "Require n_basis < grid size and q <= baseline_points <= grid size."
            )
    if endpoint_method not in ("taylor", "chebyshev"):
        raise ValueError("endpoint_method must be 'taylor' or 'chebyshev'.")
    if endpoint_method == "chebyshev":
        _integer("chebyshev_modes", chebyshev_modes, q)
        if chebyshev_modes > baseline_points or baseline_points < 2:
            raise ValueError(
                "Require chebyshev_modes <= baseline_points and at least two samples."
            )
        if not np.isfinite(endpoint_regularization) or endpoint_regularization < 0:
            raise ValueError("endpoint_regularization must be finite and nonnegative.")
    args = (
        q,
        n_basis,
        degree,
        baseline_points,
        endpoint_method,
        chebyshev_modes,
        endpoint_regularization,
    )
    X, Y = _make_line(x, *args), _make_line(y, *args)
    nx, ny = X.x.size, Y.x.size
    mask = jnp.zeros((nx, ny)).at[1:-1, 1:-1].set(1)
    corners = jnp.array([0, ny - 1, (nx - 1) * ny, nx * ny - 1])
    walls = jnp.flatnonzero(mask.ravel() == 0)
    den = X.eigenvalues[:, None] + Y.eigenvalues[None, :]
    delta = -10.0 - den[:2, :2]
    den = den.at[:2, :2].set(-10.0)
    if not bool(jnp.all(abs(den) >= 1e-9)):
        raise ValueError("Unexpected tensor pressure resonance.")
    plan = PressurePoisson2DPlan(
        X,
        Y,
        mask,
        jnp.outer(X.weights, Y.weights),
        corners,
        walls,
        den,
        delta,
        jnp.empty(0),
        jnp.empty(0),
        jnp.empty(0),
    )
    B = jnp.column_stack(
        [
            _wall(plan, pressure_gradient(plan, _null_field(plan, j))).ravel()
            for j in range(7)
        ]
    )
    scales = jnp.linalg.norm(B, axis=0)
    if not bool(jnp.all(jnp.isfinite(scales) & (scales > 0))):
        raise ValueError("Degenerate pressure completion basis.")
    Q, R = jnp.linalg.qr(B / scales, mode="reduced")
    singular = jnp.linalg.svd(R, compute_uv=False)
    if not bool(jnp.all(singular > 7 * jnp.finfo(R.dtype).eps * singular[0])):
        raise ValueError("Wall completion must have rank seven.")
    plan = plan._replace(wall_scales=scales, wall_q=Q, wall_r=R)
    if transform_backend == "dct_hodlr":
        plan = compress_pressure_plan(
            plan,
            tolerance=compression_tolerance,
            leaf_size=compression_leaf_size,
            protected_modes=protected_modes,
            layout=compression_layout,
        )
    return plan


def compress_pressure_plan(
    plan, *, tolerance=1e-12, leaf_size=16, protected_modes=8, layout="layered"
):
    """Return a DCT/HODLR plan, discarding its dense eigenvector arrays.

    Optional SciPy host preprocessing; install bspf-jax[compression]. Application
    stays JAX/JIT/vmap/field-autodiff compatible. Setup is not differentiable.
    Compression tolerance bounds block error, not final pressure error.
    """

    def convert(line):
        if line.compressed is not None:
            raise ValueError("Plan is already compressed; use its original dense plan")
        factors = build_transforms(
            line.vectors,
            line.inverse_vectors,
            tolerance=tolerance,
            leaf_size=leaf_size,
            protected_modes=protected_modes,
            layout=layout,
        )
        return line._replace(vectors=None, inverse_vectors=None, compressed=factors)

    result = plan._replace(x=convert(plan.x), y=convert(plan.y))
    # Preserve complex eigensystems, but do not pay complex FFT costs for real ones.
    if np.all(np.asarray(plan.denominators).imag == 0):
        result = result._replace(denominators=plan.denominators.real)
    return result


def _array(plan, values, vector=False):
    values = jnp.asarray(values)
    shape = plan.mask.shape + ((2,) if vector else ())
    if values.shape != shape or jnp.iscomplexobj(values):
        raise ValueError(f"Expected real array with shape {shape}.")
    return values.astype(plan.mask.dtype)


def pressure_remove_mean(plan, pressure):
    """Fix the trapezoidal volume-mean gauge."""
    pressure = _array(plan, pressure)
    return pressure - jnp.sum(plan.weights * pressure) / jnp.sum(plan.weights)


def pressure_gradient(plan, pressure):
    """Strong gradient; scalar (nx, ny) -> vector (nx, ny, 2), components x/y."""
    p = _array(plan, pressure)
    return jnp.stack(
        [_differentiate(plan.x, p, 0), _differentiate(plan.y, p, 1)], axis=-1
    )


def pressure_divergence(plan, vector):
    """Strong divergence at all nodes, including the boundary."""
    v = _array(plan, vector, vector=True)
    return _differentiate(plan.x, v[..., 0], 0) + _differentiate(plan.y, v[..., 1], 1)


def pressure_schur(plan, pressure):
    """Apply div(Q grad); not the general calculus plan's D2 Laplacian."""
    return pressure_divergence(
        plan, plan.mask[..., None] * pressure_gradient(plan, pressure)
    )


def _wall(plan, vector):
    return vector.reshape(-1, 2)[plan.wall_indices]


def _null_field(plan, j):
    if j < 3:
        ix, iy = [(0, 1), (1, 0), (1, 1)][j]
        return jnp.outer(plan.x.null_basis[:, ix], plan.y.null_basis[:, iy])
    return (
        jnp.zeros(plan.mask.size)
        .at[plan.corners[j - 3]]
        .set(1)
        .reshape(plan.mask.shape)
    )


def _lift(plan, p):
    X, Y = plan.x, plan.y
    vx, wx = _lift_vectors(X)
    vy, wy = _lift_vectors(Y)
    c = wx @ p[1:-1, 1:-1] @ wy.T
    interior = vx @ (plan.delta * c) @ vy.T
    out = jnp.zeros_like(p).at[1:-1, 1:-1].set(interior.real)
    return (
        out.ravel()
        .at[plan.corners]
        .set(-10.0 * p.ravel()[plan.corners])
        .reshape(p.shape)
    )


def _lift_vectors(line):
    if line.compressed is None:
        return line.vectors[:, :2], line.inverse_vectors[:2]
    return line.compressed.protected_vectors[:, :2], line.compressed.protected_inverse[
        :2
    ]


def _transform(line, values, inverse=False):
    if line.compressed is not None:
        return apply_transform(line.compressed, values, inverse=inverse)
    return (line.inverse_vectors if inverse else line.vectors) @ values


def _tensor_solve(plan, rhs):
    X, Y = plan.x, plan.y
    e = jnp.array([0, -1])
    r = rhs[1:-1, 1:-1] - X.coupling @ rhs[e, 1:-1] - rhs[1:-1, e] @ Y.coupling.T
    spectral = _transform(Y, _transform(X, r, True).T, True).T
    core = _transform(Y, _transform(X, spectral / plan.denominators).T).T
    p = jnp.zeros(rhs.shape, dtype=core.dtype).at[1:-1, 1:-1].set(core)
    p = p.at[e, 1:-1].set(X.endpoint_inverse @ (rhs[e, 1:-1] - X.hei @ core))
    p = p.at[1:-1, e].set((rhs[1:-1, e] - core @ Y.hei.T) @ Y.endpoint_inverse.T)
    p = (
        p.ravel()
        .at[plan.corners]
        .set(rhs.ravel()[plan.corners] / -10.0)
        .reshape(rhs.shape)
    )
    return p.real


def solve_pressure_poisson2d(
    plan, rhs, *, wall_gradient=None, rtol=1e-10, atol=1e-9, refinement_steps=None
):
    """Solve compatible S p = rhs, optionally fitting both wall gradient components.

    The optional target has shape (nx, ny, 2); only boundary entries are used.
    Without it, return the lift-selected zero-mean representative (not zero
    Neumann data). Dense plans default to two refinement steps; compressed plans
    default to zero (one direct application). refinement_steps is a static integer
    override under JIT. Compression never automatically retries or relaxes tolerance.

    Unlike NumPy, invalid numerical data or incompatible RHS cannot raise from
    JIT: result.converged is False if data/tolerances are nonfinite, tolerances
    are negative, or the final unlifted residual exceeds atol+rtol*norm(rhs).
    The caller must check this flag before using the result. Shape/complex-data
    errors raise at trace time. No RHS projection or tolerance relaxation occurs.
    """
    b = _array(plan, rhs)
    target = None if wall_gradient is None else _array(plan, wall_gradient, vector=True)
    rtol, atol = jnp.asarray(rtol), jnp.asarray(atol)
    if rtol.ndim or atol.ndim or jnp.iscomplexobj(rtol) or jnp.iscomplexobj(atol):
        raise ValueError("rtol and atol must be real scalars.")
    if refinement_steps is None:
        refinement_steps = (
            2 if plan.x.compressed is None and plan.y.compressed is None else 0
        )
    if (
        isinstance(refinement_steps, bool)
        or not isinstance(refinement_steps, int)
        or refinement_steps < 0
    ):
        raise ValueError("refinement_steps must be a static nonnegative integer")
    p = _tensor_solve(plan, b)
    for _ in range(refinement_steps):
        p += _tensor_solve(plan, b - pressure_schur(plan, p) - _lift(plan, p))
    if target is not None:
        residual = _wall(plan, target - pressure_gradient(plan, p)).ravel()
        coeff = (
            jl.solve_triangular(plan.wall_r, plan.wall_q.T @ residual)
            / plan.wall_scales
        )
        for j in range(7):
            p += coeff[j] * _null_field(plan, j)
    p = pressure_remove_mean(plan, p)
    residual = pressure_schur(plan, p) - b
    norm = jnp.linalg.norm(residual)
    fit = (
        jnp.asarray(jnp.nan)
        if target is None
        else jnp.max(abs(_wall(plan, pressure_gradient(plan, p) - target)))
    )
    valid = jnp.all(jnp.isfinite(b)) & jnp.all(jnp.isfinite(p))
    if target is not None:
        valid &= jnp.all(jnp.isfinite(target))
    valid &= jnp.isfinite(rtol) & jnp.isfinite(atol) & (rtol >= 0) & (atol >= 0)
    valid &= jnp.isfinite(norm) & (norm <= atol + rtol * jnp.linalg.norm(b))
    return PressurePoisson2DResult(p, jnp.max(abs(residual)), norm, fit, valid)


def project_pressure2d(
    plan, raw, *, completion=True, rtol=1e-10, atol=1e-9, refinement_steps=None
):
    """Return (Q*(raw-grad(p)), result); raw may be an NS stage acceleration.

    completion is a static Python option under JIT. Use vmap for batches.
    Always check result.converged. No time-step scaling is applied.
    """
    raw = _array(plan, raw, vector=True)
    result = solve_pressure_poisson2d(
        plan,
        pressure_divergence(plan, plan.mask[..., None] * raw),
        wall_gradient=raw if completion else None,
        rtol=rtol,
        atol=atol,
        refinement_steps=refinement_steps,
    )
    return plan.mask[..., None] * (
        raw - pressure_gradient(plan, result.pressure)
    ), result
