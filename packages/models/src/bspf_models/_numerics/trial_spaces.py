"""Shared model trial spaces; no dependency on any physical equation."""
from functools import partial
from math import factorial
from types import SimpleNamespace
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
from pybspf.basis import basis_matrix, open_knots
from pybspf.endpoints import chebyshev_boundary_blocks
from bspf_models._numerics._compressed_transform import CompressedTransforms
from bspf_models._numerics._weak_basis import evaluate_basis
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


def _fourier(values, multiplier):
    d = jnp.fft.ifft(jnp.fft.fft(values[:-1], axis=0) * multiplier[:, None], axis=0)
    return jnp.concatenate((d, d[:1]), axis=0).real


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


class StreamLine(NamedTuple):
    x: jax.Array
    points: jax.Array
    weights: jax.Array
    b: jax.Array
    g: jax.Array
    h: jax.Array
    bn: jax.Array
    gn: jax.Array
    hn: jax.Array
    lam: jax.Array
    bending: jax.Array
    transform: jax.Array
    projector: jax.Array
    layers: jax.Array
    rotation: jax.Array | None = None


def _stream_line(
    x,
    *,
    quadrature_order=None,
    layers=(),
    clamped=True,
    dirichlet=False,
    endpoint_method="chebyshev",
    endpoint_points=16,
    chebyshev_modes=12,
    endpoint_regularization=1e-12,
    basis_executor=None,
    basis_precision="mpfr",
    basis_device=None,
):
    import scipy.linalg as la
    from scipy.interpolate import BSpline
    from scipy.special import roots_legendre

    layers = tuple(float(d) for d in layers)
    x = np.asarray(x, dtype=float)
    if (
        x.ndim != 1
        or len(x) < 17
        or not np.all(np.isfinite(x))
        or x[-1] <= x[0]
        or not np.allclose(
            np.diff(x), (x[-1] - x[0]) / (len(x) - 1), rtol=1e-10, atol=1e-14
        )
    ):
        raise ValueError("Require >=17 finite increasing uniform BSPF nodes per axis")
    if not all(np.isfinite(d) and d > 0 for d in layers) or len(set(layers)) != len(
        layers
    ):
        raise ValueError("Layer lengths must be finite, positive and distinct")
    if quadrature_order is not None and (
        not isinstance(quadrature_order, int) or quadrature_order < 2
    ):
        raise ValueError("quadrature_order must be an integer >= 2")
    x = np.asarray(x)
    if endpoint_method not in ("chebyshev", "taylor"):
        raise ValueError("endpoint_method must be chebyshev or taylor")
    if not isinstance(
        endpoint_points, (int, np.integer)
    ) or not 9 <= endpoint_points <= len(x):
        raise ValueError("Require 9 <= endpoint_points <= number of grid nodes")
    if (
        not isinstance(chebyshev_modes, (int, np.integer))
        or not 9 <= chebyshev_modes <= endpoint_points
    ):
        raise ValueError("Require 9 <= chebyshev_modes <= endpoint_points")
    if not np.isfinite(endpoint_regularization) or endpoint_regularization < 0:
        raise ValueError("endpoint_regularization must be finite and nonnegative")
    projector, *_ = _line_projector(
        x,
        9,
        32,
        13,
        endpoint_points,
        endpoint_method,
        chebyshev_modes,
        endpoint_regularization,
    )
    host = SimpleNamespace(x=x, P=np.asarray(projector))
    breaks = np.linspace(x[0], x[-1], 20)
    knots = np.r_[np.repeat(x[0], 14), breaks[1:-1], np.repeat(x[-1], 14)]
    spline = BSpline(knots, np.eye(32), 13)
    order = quadrature_order or max(
        24, int(np.ceil(1.5 * np.pi * (len(x) - 1) / 19)) + 8
    )
    if layers:
        extra = [x[0] + min(layers) * a for a in (0.5, 1, 2, 4, 8, 16, 32, 64)]
        extra += [x[-1] - min(layers) * a for a in (0.5, 1, 2, 4, 8, 16, 32, 64)]
        breaks = np.unique(np.r_[breaks, [v for v in extra if x[0] < v < x[-1]]])
    q, w = roots_legendre(order)
    points = np.concatenate(
        [(a + b) / 2 + (b - a) / 2 * q for a, b in zip(breaks[:-1], breaks[1:])]
    )
    weights = np.concatenate([(b - a) / 2 * w for a, b in zip(breaks[:-1], breaks[1:])])
    # The first QR needs only values. Derivatives and nodal traces are
    # evaluated below after the sensitive normalization is available.
    (b,) = evaluate_basis(
        host, spline, points, values_only=True, executor=basis_executor,
        precision=basis_precision, device=basis_device,
    )
    if layers:
        enrichment = [
            np.exp(sign * (points - endpoint) / delta)
            for delta in layers
            for endpoint, sign in ((x[0], -1), (x[-1], 1))
        ]
        b = np.column_stack((b, np.array(enrichment).T))
    # Orthonormalize the entire enriched space BEFORE imposing the walls.
    # This keeps the boundary null-space calculation well conditioned, even
    # when an exponential is nearly represented by the original BSPF basis.
    root_w = np.sqrt(weights)
    _, rv = la.qr(root_w[:, None] * b, mode="economic")
    transform = la.solve_triangular(rv, np.eye(rv.shape[0]))
    bc, gc, hc = evaluate_basis(
        host, spline, points, second=True, transform=transform, layers=layers,
        executor=basis_executor, precision=basis_precision, device=basis_device,
    )
    bnc, gnc, hnc = evaluate_basis(
        host, spline, x, second=True, transform=transform, layers=layers,
        executor=basis_executor, precision=basis_precision, device=basis_device,
    )
    qv, rv = la.qr(root_w[:, None] * bc, mode="economic")
    correction = la.solve_triangular(rv, np.eye(rv.shape[0]))
    # A magnetic flux potential needs only its value fixed at the wall;
    # streamfunction velocity walls additionally constrain the normal derivative.
    traces = np.vstack((bnc[[0, -1]], gnc[[0, -1]])) if clamped else bnc[[0, -1]]
    constraints = traces @ correction
    constraints /= la.norm(constraints, axis=1)[:, None]
    z = (
        la.null_space(constraints)
        if clamped or dirichlet
        else np.eye(correction.shape[1])
    )
    constrained = correction @ z
    normalized_g = gc @ constrained
    stiff = normalized_g.T @ (weights[:, None] * normalized_g)
    lam, v = la.eigh(stiff)
    rotation = constrained @ v
    bc = (qv @ z @ v) / root_w[:, None]
    gc, hc = gc @ rotation, hc @ rotation
    bending = hc.T @ (weights[:, None] * hc)
    return StreamLine(
        *map(
            jnp.asarray,
            (
                x,
                points,
                weights,
                bc,
                gc,
                hc,
                bnc @ rotation,
                gnc @ rotation,
                hnc @ rotation,
                lam,
                bending,
                transform,
                host.P,
                np.asarray(layers),
                rotation,
            ),
        )
    )


def stream_evaluate_line(line, points, *, basis_executor=None,
                         basis_precision="mpfr", basis_device=None):
    """Evaluate the enriched basis and two derivatives at arbitrary host points."""
    from scipy.interpolate import BSpline

    x = np.asarray(line.x)
    host = SimpleNamespace(x=x, P=np.asarray(line.projector))
    knots = np.r_[
        np.repeat(x[0], 14), np.linspace(x[0], x[-1], 20)[1:-1], np.repeat(x[-1], 14)
    ]
    arrays = evaluate_basis(
        host,
        BSpline(knots, np.eye(32), 13),
        points,
        executor=basis_executor, precision=basis_precision, device=basis_device,
        second=True,
        transform=np.asarray(line.transform),
        layers=np.asarray(line.layers),
    )
    if line.rotation is None:
        return arrays
    return tuple(a @ np.asarray(line.rotation) for a in arrays)
