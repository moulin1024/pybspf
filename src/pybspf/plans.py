"""Explicit, immutable precomputation for constrained spline/Fourier operators.

Only checked factory functions inspect host values. All numerical assembly and
application use JAX; no NumPy/SciPy numerical implementation is called.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from numbers import Integral
import math

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np

from pybspf.validation import integer as _integer
from pybspf.basis import basis_matrix
from pybspf.basis import open_knots
from pybspf.endpoints import chebyshev_boundary_blocks
from pybspf.noise import NoisePlan
from pybspf.noise import assemble_noise


@partial(jax.tree_util.register_dataclass,
         data_fields=["x", "knots", "basis", "weights", "constraint", "boundary_blocks",
                      "weighted_basis", "gram", "lu", "omega", "regularization", "noise"],
         meta_fields=["degree", "constraint_order", "max_derivative"])
@dataclass(frozen=True)
class Plan1D:
    """A PyTree of precomputed arrays, never a mutable solver or hidden cache."""
    x: jax.Array
    knots: jax.Array
    basis: tuple[jax.Array, ...]
    weights: jax.Array
    constraint: jax.Array
    boundary_blocks: jax.Array  # (left/right, derivative order, local sample)
    weighted_basis: jax.Array
    gram: jax.Array
    lu: tuple[jax.Array, jax.Array]
    omega: jax.Array
    regularization: jax.Array
    degree: int
    constraint_order: int
    max_derivative: int
    noise: NoisePlan | None = None


@partial(jax.tree_util.register_dataclass, data_fields=["axes"], meta_fields=[])
@dataclass(frozen=True)
class TensorPlan:
    """Separable physical axes in array order: x, y, z (``indexing='ij'``)."""
    axes: tuple[Plan1D, ...]


def _factorize(gram, constraint, lam):
    m = constraint.shape[0]
    matrix = jnp.block([
        [2 * (gram + lam * jnp.eye(gram.shape[0], dtype=gram.dtype)), -constraint.T],
        [constraint, jnp.zeros((m, m), dtype=gram.dtype)],
    ])
    return jl.lu_factor(matrix)


def with_regularization(plan: Plan1D, lam) -> Plan1D:
    """Pure, differentiable refactorization at a new nonnegative scalar ``lam``.

    Unlike the checked factory, this kernel accepts tracers. Its caller must
    ensure finite, nonnegative lam and a nonsingular constrained system.
    """
    if plan.noise is not None:
        raise ValueError("lam controls only clean-data plans; noisy plans use noise_std and noise_alphas")
    lam = jnp.asarray(lam, dtype=plan.x.dtype)
    return replace(plan, lu=_factorize(plan.gram, plan.constraint, lam), regularization=lam)


def _assemble(x, knots, *, degree, constraint_order, boundary_points, max_derivative, lam,
              endpoint_method, chebyshev_modes, chebyshev_alpha, chebyshev_penalty_power,
              noisy=False):
    if noisy:
        constraint_order, boundary_points, endpoint_method = 0, 1, "finite_difference"
    basis = tuple(basis_matrix(knots, x, degree=degree, derivative=k)
                  for k in range(max(max_derivative, constraint_order - 1) + 1))
    dx = x[1] - x[0]
    weights = jnp.full_like(x, dx).at[0].set(dx / 2).at[-1].set(dx / 2)
    q = constraint_order
    constraint = jnp.stack([b[0] for b in basis[:q]] + [b[-1] for b in basis[:q]]) if q else jnp.zeros((0, basis[0].shape[1]), dtype=x.dtype)
    p = boundary_points
    if endpoint_method == "chebyshev":
        boundary = chebyshev_boundary_blocks(
            x, order=q, points=p, modes=chebyshev_modes,
            alpha=chebyshev_alpha, penalty_power=chebyshev_penalty_power)
    else:
        offsets = jnp.arange(p, dtype=x.dtype)
        powers = jnp.arange(p)[:, None]
        factorials = jnp.asarray([math.factorial(k) for k in range(p)], dtype=x.dtype)[:, None]
        vandermonde = offsets[None, :] ** powers / factorials
        # Rows map endpoint samples to physical derivatives of orders 0,...,q-1.
        left = jnp.linalg.solve(vandermonde, jnp.eye(p, dtype=x.dtype)[:, :q]).T
        left = left / dx ** jnp.arange(q)[:, None]
        right = left[:, ::-1] * (-1.0) ** jnp.arange(q)[:, None]
        boundary = jnp.stack((left, right))
    weighted = basis[0].T * weights
    gram = weighted @ basis[0]
    lu = ((jnp.empty((0, 0), dtype=x.dtype), jnp.empty((0,), dtype=jnp.int32))
          if noisy else _factorize(gram, constraint, lam))
    return Plan1D(x, knots, basis, weights, constraint, boundary, weighted, gram,
                  lu,
                  2*jnp.pi*jnp.fft.fftfreq(x.size, d=dx), jnp.asarray(lam, dtype=x.dtype),
                  degree, q, max_derivative)


def plan_1d(x, *, degree=5, n_basis=None, knots=None, constraint_order=None,
            boundary_points=None, max_derivative=4, lam=0.0, clustering=0.0,
            endpoint_method="finite_difference", chebyshev_modes=None,
            chebyshev_alpha=1e-12, chebyshev_penalty_power=4,
            noise_std=0.0, noise_penalty_order=2, noise_alphas=None) -> Plan1D:
    """Validate static geometry and precompute a BSPF operator outside ``jit``.

    ``constraint_order=q`` constrains derivatives 0,...,q-1 at both ends.
    Defaults match pybspf: q=degree-1, stencil width=degree. Float64 is required
    explicitly, since high-order endpoint stencils and KKT solves are sensitive
    to roundoff. Enable JAX x64 in your application before calling this factory.

    ``endpoint_method='chebyshev'`` fits a local Chebyshev expansion using
    ``boundary_points`` samples per end. Modes default to degree+1; the window
    defaults to min(grid size, 2*modes). Require order <= modes <= window size.
    ``chebyshev_alpha`` regularizes normalized modal coefficients independently
    of spline ``lam``; the first two modes are unpenalized. Endpoint values are
    copied exactly. Explicit boundary jets supplied to kernels override either
    estimator. The default finite-difference behavior is unchanged.

    ``noise_std`` is an a priori estimate of absolute sample-noise standard
    deviation, supplied by the caller. The operator receives observed data only;
    it does not add noise or need a clean reference. Zero preserves the
    clean-data operator exactly. A positive
    absolute RMS noise level enables joint spline/Fourier smoothing, with no
    endpoint constraints. ``lam`` must then be zero; endpoint estimator options
    do not participate in the noisy fit. Penalty order defaults to 2. Optional
    positive increasing ``noise_alphas`` override the discrepancy search grid.
    Noisy differentiation smooths transverse axes too in tensor plans.
    """
    if not jax.config.x64_enabled:
        raise ValueError("BSPF requires x64; set jax.config.update('jax_enable_x64', True)")
    _integer("degree", degree, 1)
    _integer("max_derivative", max_derivative, 0)
    if (isinstance(noise_std, str) or np.ndim(noise_std) != 0
        or np.iscomplexobj(noise_std) or not np.isfinite(noise_std) or noise_std < 0):
        raise ValueError("noise_std must be a finite nonnegative a priori standard-deviation estimate")
    sigma = noise_std
    noisy = sigma > 0
    if noisy:
        _integer("noise_penalty_order", noise_penalty_order, 1)
        if noise_penalty_order > degree or lam != 0:
            raise ValueError("noisy plans require noise_penalty_order <= degree and lam=0")
        if noise_alphas is not None:
            alpha_host = np.asarray(noise_alphas)
            if (alpha_host.ndim != 1 or alpha_host.size < 2 or np.iscomplexobj(alpha_host)
                or not np.all(np.isfinite(alpha_host)) or np.any(alpha_host <= 0)
                or np.any(np.diff(alpha_host) <= 0)):
                raise ValueError("noise_alphas must be a positive, finite, strictly increasing 1D grid")
    elif noise_alphas is not None:
        raise ValueError("noise_alphas requires positive noise_std")
    q = degree - 1 if constraint_order is None else constraint_order
    _integer("constraint_order", q, 0)
    host = np.asarray(x)
    if host.ndim != 1 or host.size < 2 or np.iscomplexobj(host) or not np.all(np.isfinite(host)):
        raise ValueError("x must be a finite real 1D grid with at least two points")
    spacing = np.diff(host)
    if np.any(spacing <= 0) or not np.allclose(spacing, spacing[0], rtol=1e-10, atol=1e-13):
        raise ValueError("x must be strictly increasing and uniformly spaced")
    if endpoint_method not in ("finite_difference", "chebyshev"):
        raise ValueError("endpoint_method must be 'finite_difference' or 'chebyshev'")
    if endpoint_method == "chebyshev":
        chebyshev_modes = degree+1 if chebyshev_modes is None else chebyshev_modes
        _integer("chebyshev_modes", chebyshev_modes, max(1, q))
        _integer("chebyshev_penalty_power", chebyshev_penalty_power, 0)
        if not np.isfinite(chebyshev_alpha) or chebyshev_alpha < 0:
            raise ValueError("chebyshev_alpha must be finite and nonnegative")
        p = min(host.size, 2*chebyshev_modes) if boundary_points is None else boundary_points
        _integer("boundary_points", p, max(2, chebyshev_modes))
    else:
        if chebyshev_modes is not None:
            raise ValueError("chebyshev_modes requires endpoint_method='chebyshev'")
        p = degree if boundary_points is None else boundary_points
        _integer("boundary_points", p, 1)
    if q > degree or p < max(1, q) or p > host.size:
        raise ValueError("require q <= degree and max(1, q) <= boundary_points <= grid size")
    if not np.isfinite(lam) or lam < 0 or not np.isfinite(clustering) or clustering < 0:
        raise ValueError("lam and clustering must be finite and nonnegative")
    x = jnp.asarray(x, dtype=jnp.float64)
    if knots is None:
        n_basis = min(4*(degree + 1), host.size) if n_basis is None else n_basis
        _integer("n_basis", n_basis, degree + 1)
        knots = open_knots(x[0], x[-1], degree=degree, n_basis=n_basis, clustering=clustering)
    else:
        if n_basis is not None:
            raise ValueError("provide knots or n_basis, not both")
        t = np.asarray(knots)
        if t.ndim != 1 or np.iscomplexobj(t) or t.size < 2*(degree+1) or not np.all(np.isfinite(t)) or np.any(np.diff(t) < 0):
            raise ValueError("knots must be a finite nondecreasing clamped knot vector")
        if not (np.all(t[:degree+1] == host[0]) and np.all(t[-degree-1:] == host[-1])):
            raise ValueError("knots must be clamped to the grid endpoints")
        interior = t[degree+1:-degree-1]
        if interior.size and (np.any(interior <= host[0]) or np.any(interior >= host[-1]) or np.any(np.diff(interior) <= 0)):
            raise ValueError("interior knots must be strictly increasing (simple knots)")
        knots = jnp.asarray(knots, dtype=x.dtype)
        n_basis = knots.size - degree - 1
    if not noisy and n_basis < 2*q:
        raise ValueError("n_basis must be >= 2*constraint_order")
    if not noisy and lam == 0 and n_basis > host.size:
        raise ValueError("unregularized fit requires n_basis <= number of samples")
    plan = _assemble(x, knots, degree=degree, constraint_order=q, boundary_points=p,
                     max_derivative=max_derivative, lam=lam, endpoint_method=endpoint_method,
                     chebyshev_modes=chebyshev_modes, chebyshev_alpha=chebyshev_alpha,
                     chebyshev_penalty_power=chebyshev_penalty_power, noisy=noisy)
    if not noisy and (not bool(jnp.all(jnp.isfinite(plan.lu[0]))) or bool(jnp.any(jnp.diag(plan.lu[0]) == 0))):
        raise ValueError("singular KKT system; use fewer basis functions or positive lam")
    if noisy:
        alphas = (jnp.geomspace(1e-14, 1e2, 81)*((x[-1]-x[0])/(2*jnp.pi))**(2*noise_penalty_order)
                  if noise_alphas is None else jnp.asarray(noise_alphas, dtype=x.dtype))
        noise = assemble_noise(x, knots, plan.basis[0], plan.omega, degree=degree,
                               order=noise_penalty_order, sigma=sigma, alphas=alphas)
        if not bool(jnp.all(jnp.isfinite(noise.projector))):
            raise ValueError("singular noisy fit; adjust basis size or noise_alphas")
        plan = replace(plan, noise=noise)
    return plan


def tensor_plan(*axes: Plan1D) -> TensorPlan:
    """Compose one to three independently configured 1D plans."""
    if not 1 <= len(axes) <= 3 or not all(isinstance(a, Plan1D) for a in axes):
        raise ValueError("provide one to three Plan1D objects")
    noisy = [a.noise is not None for a in axes]
    if any(noisy) and not all(noisy):
        raise ValueError("tensor axes must all use clean plans or all use noisy plans")
    if all(noisy) and any(float(a.noise.sigma) != float(axes[0].noise.sigma) for a in axes):
        raise ValueError("tensor axes must share the same absolute noise_std")
    return TensorPlan(tuple(axes))


def plan_2d(x, y, **kwargs):
    """Convenience constructor; use tensor_plan for per-axis options."""
    return tensor_plan(plan_1d(x, **kwargs), plan_1d(y, **kwargs))


def plan_3d(x, y, z, **kwargs):
    """Convenience constructor; fields have shape (nx, ny, nz, ...)."""
    return tensor_plan(plan_1d(x, **kwargs), plan_1d(y, **kwargs), plan_1d(z, **kwargs))
