"""Reusable fast-diagonalization solver for rectangular weak elliptic systems.

The caller supplies boundary-reduced one-dimensional mass/stiffness matrices
and integrated loads. No global spatial matrix is assembled. Setup
and solves execute on the selected JAX device; CPU is supported for reference
checks. Pure Neumann/periodic nullspaces require a separate gauge-aware solver.
"""

from typing import NamedTuple

import jax.numpy as jnp
import jax.scipy.linalg as jl

import jax


def tensor_elliptic_solve(load, denominator, left=None, right=None):
    """Apply a separable inverse to (..., nx, ny) integrated loads.

    Columns of left/right are mass-orthonormal generalized eigenvectors;
    None denotes identity. This small kernel also preserves NumPy callers.
    JAX callers can fuse it into their enclosing compiled PDE step.
    """
    transformed = load if left is None else left.T @ load
    if right is not None:
        transformed = transformed @ right
    result = transformed / denominator
    if left is not None:
        result = left @ result
    if right is not None:
        result = result @ right.T
    return result


class RectanglePoissonPlan(NamedTuple):
    """Device-resident factors; a JAX pytree reusable across RHS batches."""

    left: jax.Array
    right: jax.Array
    denominator: jax.Array


@jax.jit
def _generalized_modes(mass, stiffness):
    # K v = lambda M v, reduced using M = L L.T. Triangular solves avoid
    # explicit inverses; all O(n^3) setup work stays on the selected device.
    lower = jnp.linalg.cholesky(mass)
    reduced = jl.solve_triangular(lower, stiffness, lower=True)
    reduced = jl.solve_triangular(lower, reduced.T, lower=True).T
    values, vectors = jnp.linalg.eigh((reduced + reduced.T) / 2)
    vectors = jl.solve_triangular(lower.T, vectors, lower=False)
    return values, vectors


def plan_rectangle_poisson(
    mass_x,
    stiffness_x,
    mass_y,
    stiffness_y,
    *,
    device,
    shift=0.0,
    weights=(1.0, 1.0),
):
    """Factor shift*M + weights[0]*Kx⊗My + weights[1]*Mx⊗Ky once.

    M = Mx⊗My. Matrices must be real symmetric, masses positive definite,
    and the resulting tensor operator positive definite. For -Delta use
    shift=0, weights=(1, 1), with homogeneous Dirichlet boundary DOFs already
    eliminated. Nonzero boundary data must be lifted into the load by the
    caller. Coordinate lengths/scaling belong in the supplied axis matrices.

    Pass device=jax.devices('gpu')[0] for GPU setup and runtime. FP64 must be
    enabled explicitly. Only small validation flags transfer back to the host;
    factors remain on device. This constructor is eager, not JIT-transformable.
    """
    rotations, spectra, shift = _plan_axes(
        (mass_x, mass_y), (stiffness_x, stiffness_y), device, shift, weights
    )
    denominator = shift + spectra[0][:, None] + spectra[1][None, :]
    return RectanglePoissonPlan(*rotations, denominator)


class BoxPoissonPlan(NamedTuple):
    """Compact 2D/3D factors; no full-volume denominator stored in the plan.

    spectra contains weighted generalized eigenvalues. All arrays reside on
    the selected device and the plan is a JAX pytree.
    """

    rotations: tuple[jax.Array, ...]
    spectra: tuple[jax.Array, ...]
    shift: jax.Array

    @property
    def shape(self):
        return tuple(values.size for values in self.spectra)


def _plan_axes(masses, stiffnesses, device, shift, weights):
    """Shared eager validation/factorization for both public constructors."""
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 for rectangular Poisson setup")
    if device is None:
        raise ValueError("Select an explicit JAX device")
    ndim = len(masses)
    if ndim not in (2, 3) or len(stiffnesses) != ndim:
        raise ValueError("Provide matching mass/stiffness sequences for 2 or 3 axes")
    if weights is None:
        weights = (1.0,) * ndim
    with jax.default_device(device):

        def matrix(value, name):
            a = jax.device_put(jnp.asarray(value), device)
            if a.ndim != 2 or a.shape[0] != a.shape[1] or not a.shape[0]:
                raise ValueError(f"{name} must be a nonempty square matrix")
            if jnp.iscomplexobj(a):
                raise ValueError(f"{name} must be real")
            a = a.astype(jnp.float64)
            if not bool(
                jnp.all(jnp.isfinite(a)) & jnp.allclose(a, a.T, rtol=1e-12, atol=1e-14)
            ):
                raise ValueError(f"{name} must be finite and symmetric")
            return (a + a.T) / 2

        s = jax.device_put(
            jnp.asarray(shift, dtype=jnp.float64)
            if not jnp.iscomplexobj(shift)
            else jnp.asarray(shift),
            device,
        )
        w = jax.device_put(jnp.asarray(weights), device)
        if s.ndim or w.shape != (ndim,) or jnp.iscomplexobj(s) or jnp.iscomplexobj(w):
            raise ValueError(
                "shift must be a real scalar and weights a real value per axis"
            )
        if not bool(
            jnp.isfinite(s) & jnp.all(jnp.isfinite(w)) & (s >= 0) & jnp.all(w > 0)
        ):
            raise ValueError(
                "shift must be finite/nonnegative and weights finite/positive"
            )
        spectra, rotations, cache = [], [], []
        valid = jnp.asarray(True)
        for axis, (mass, stiffness) in enumerate(zip(masses, stiffnesses)):
            # Reusing the same input objects (e.g. cubic geometry) also reuses
            # the decomposition and rotation storage. No global mutable cache.
            found = next(
                (
                    entry
                    for entry in cache
                    if entry[0] is mass and entry[1] is stiffness
                ),
                None,
            )
            if found is None:
                m = matrix(mass, f"mass[{axis}]")
                k = matrix(stiffness, f"stiffness[{axis}]")
                if m.shape != k.shape:
                    raise ValueError(
                        "Mass and stiffness shapes must match on each axis"
                    )
                values, vectors = _generalized_modes(m, k)
                cache.append((mass, stiffness, values, vectors))
            else:
                values, vectors = found[2:]
            spectra.append(w[axis] * values)
            rotations.append(vectors)
            valid &= jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(vectors))
        # Spectral extrema validate all denominators without allocating a
        # potentially very large 3D volume during setup.
        lo = s + sum(jnp.min(values) for values in spectra)
        hi = s + sum(jnp.max(values) for values in spectra)
        cutoff = 32 * jnp.finfo(jnp.float64).eps * jnp.maximum(jnp.abs(lo), jnp.abs(hi))
        valid &= jnp.isfinite(lo) & jnp.isfinite(hi) & (lo > cutoff)
        if not bool(valid):
            raise ValueError(
                "Masses and tensor operator must be numerically positive definite"
            )
        return tuple(rotations), tuple(spectra), s


def plan_box_poisson(masses, stiffnesses, *, device, shift=0.0, weights=None):
    """Factor a positive-definite separable Poisson/Helmholtz box operator.

    Provide two or three mass matrices and matching stiffness matrices, one
    per spatial axis. In 3D the operator in C-order coefficient layout is
    shift*Mx⊗My⊗Mz + wx*Kx⊗My⊗Mz + wy*Mx⊗Ky⊗Mz + wz*Mx⊗My⊗Kz.
    Default weights are one. Boundary reduction, integrated loads, FP64 and
    positive-definiteness have the same contract as plan_rectangle_poisson.

    Setup uses only 1D matrices on the selected device. Identical input object
    pairs share their decomposition. Construction is eager, outside JIT.
    """
    rotations, spectra, s = _plan_axes(
        tuple(masses), tuple(stiffnesses), device, shift, weights
    )
    return BoxPoissonPlan(rotations, spectra, s)


@jax.jit
def solve_box_poisson(plan, load):
    """Solve for (..., nx, ny[, nz]) coefficient arrays on the plan's device.

    Leading dimensions are independent RHS batches, never spatial axes.
    Six axis matrix products in 3D and one modal division replace a full
    spatial solve. The denominator is broadcast inside JIT, not retained as
    an extra volume in the plan. Warm solves have no host callbacks/transfers.
    """
    ndim = len(plan.rotations)
    if load.ndim < ndim or load.shape[-ndim:] != plan.shape:
        raise ValueError("load must have trailing spatial shape matching the box plan")
    values = load
    for axis, rotation in enumerate(plan.rotations):
        dim = axis - ndim
        values = jnp.moveaxis(jnp.moveaxis(values, dim, -1) @ rotation, -1, dim)
    denominator = plan.shift
    for axis, spectrum in enumerate(plan.spectra):
        shape = [1] * ndim
        shape[axis] = spectrum.size
        denominator = denominator + spectrum.reshape(shape)
    values = values / denominator
    for axis in reversed(range(ndim)):
        dim = axis - ndim
        values = jnp.moveaxis(
            jnp.moveaxis(values, dim, -1) @ plan.rotations[axis].T, -1, dim
        )
    return values


@jax.jit
def solve_rectangle_poisson(plan, load):
    """Solve for coefficients from a weak load of shape (..., nx, ny).

    Leading dimensions are independent RHS batches. Repeated calls reuse the
    plan and compiled executable; device-resident inputs require no host work.
    Supports JIT, vmap, and differentiation with respect to the load.
    """
    if load.ndim < 2 or load.shape[-2:] != plan.denominator.shape:
        raise ValueError("load must have trailing shape (nx, ny) matching the plan")
    return tensor_elliptic_solve(load, plan.denominator, plan.left, plan.right)
