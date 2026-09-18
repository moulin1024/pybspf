"""Three-dimensional masked BSPF pressure direct core, with optional compression.

Includes face elimination/recovery and the eight tensor plus edge/corner lifts.
Does not implement physical wall-gradient completion. No iterative refinement.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from .pressure import (
    PressureLine,
    _make_line,
    _differentiate,
    _transform,
    _lift_vectors,
)
from ._compressed_transform import build_transforms
from .plans import _integer


class PressurePoisson3DPlan(NamedTuple):
    lines: tuple[PressureLine, PressureLine, PressureLine]


class PressurePoisson3DResult(NamedTuple):
    pressure: jax.Array
    residual_linf: jax.Array
    residual_l2: jax.Array
    converged: jax.Array


def plan_pressure_poisson3d(
    x,
    y,
    z,
    *,
    q=9,
    n_basis=32,
    degree=13,
    baseline_points=16,
    endpoint_method="chebyshev",
    chebyshev_modes=12,
    endpoint_regularization=1e-12,
    transform_backend="dense",
    compression_tolerance=1e-12,
    compression_leaf_size=16,
    protected_modes=8,
    compression_layout="layered",
):
    """Uniform endpoint grids; arrays are (nx,ny,nz). Setup runs outside JIT.

    Defaults follow the tested Chebyshev configuration. Identical axes share
    their line plan. Full-volume denominators/masks are generated on demand,
    not stored in the plan. For compressed plans SciPy is needed only at setup.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Pressure projection requires jax_enable_x64=True")
    for name, value in [
        ("q", q),
        ("n_basis", n_basis),
        ("degree", degree),
        ("baseline_points", baseline_points),
    ]:
        _integer(name, value, 1)
    if not q <= degree + 1 <= n_basis or n_basis <= 2 * q:
        raise ValueError("Require q <= degree+1 <= n_basis and n_basis > 2*q")
    if endpoint_method not in ("taylor", "chebyshev"):
        raise ValueError("Invalid endpoint_method")
    if endpoint_method == "chebyshev":
        _integer("chebyshev_modes", chebyshev_modes, q)
        if (
            chebyshev_modes > baseline_points
            or not np.isfinite(endpoint_regularization)
            or endpoint_regularization < 0
        ):
            raise ValueError("Invalid Chebyshev endpoint settings")
    if transform_backend not in ("dense", "dct_hodlr"):
        raise ValueError("Invalid transform_backend")
    lines = []
    grids = []
    for values in (x, y, z):
        grid = np.asarray(values)
        if (
            grid.ndim != 1
            or np.iscomplexobj(grid)
            or not np.all(np.isfinite(grid))
            or grid.size < 5
        ):
            raise ValueError("Grid must be finite real one-dimensional values")
        dx = np.diff(grid)
        if np.any(dx <= 0) or not np.allclose(dx, dx[0], rtol=1e-10, atol=0):
            raise ValueError("Grid must be uniform and increasing")
        if not n_basis < grid.size or not q <= baseline_points <= grid.size:
            raise ValueError("Invalid basis/window size for grid")
        same = next((i for i, g in enumerate(grids) if np.array_equal(g, grid)), None)
        if same is not None:
            line = lines[same]
        else:
            line = _make_line(
                grid,
                q,
                n_basis,
                degree,
                baseline_points,
                endpoint_method,
                chebyshev_modes,
                endpoint_regularization,
            )
        grids.append(grid)
        lines.append(line)
    plan = PressurePoisson3DPlan(tuple(lines))
    den = _denominator(plan)
    if not bool(jnp.all(abs(den) >= 1e-9)):
        raise ValueError("Unexpected pressure tensor resonance")
    if transform_backend == "dct_hodlr":
        plan = compress_pressure_plan3d(
            plan,
            tolerance=compression_tolerance,
            leaf_size=compression_leaf_size,
            protected_modes=protected_modes,
            layout=compression_layout,
        )
    return plan


def compress_pressure_plan3d(
    plan, *, tolerance=1e-12, leaf_size=16, protected_modes=8, layout="layered"
):
    """Return a compressed plan, retaining only factors and protected modes."""
    cache = {}
    lines = []
    for line in plan.lines:
        if line.compressed is not None:
            raise ValueError("Plan is already compressed")
        key = id(line)
        if key not in cache:
            factors = build_transforms(
                line.vectors,
                line.inverse_vectors,
                tolerance=tolerance,
                leaf_size=leaf_size,
                protected_modes=protected_modes,
                layout=layout,
            )
            cache[key] = line._replace(
                vectors=None, inverse_vectors=None, compressed=factors
            )
        lines.append(cache[key])
    return PressurePoisson3DPlan(tuple(lines))


def _shape(plan):
    return tuple(line.x.size for line in plan.lines)


def _array(plan, values):
    a = jnp.asarray(values)
    if a.shape != _shape(plan) or jnp.iscomplexobj(a):
        raise ValueError(f"Expected real array of shape {_shape(plan)}")
    return a.astype(jnp.float64)


def _axmul(matrix, values, axis):
    moved = jnp.moveaxis(values, axis, 0)
    out = matrix @ moved.reshape(moved.shape[0], -1)
    return jnp.moveaxis(out.reshape((matrix.shape[0],) + moved.shape[1:]), 0, axis)


def _axis_transform(line, values, axis, inverse, batch_size):
    moved = jnp.moveaxis(values, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    if batch_size is None or flat.shape[1] <= batch_size:
        out = _transform(line, flat, inverse)
    else:
        count = (flat.shape[1] + batch_size - 1) // batch_size
        padded = jnp.pad(flat, ((0, 0), (0, count * batch_size - flat.shape[1])))
        # Fixed-size workspaces per line batch. No N^3-by-r response matrix.
        dtype = jnp.result_type(flat, line.eigenvalues)

        def body(i, out):
            tile = jax.lax.dynamic_slice(
                padded, (0, i * batch_size), (flat.shape[0], batch_size)
            )
            transformed = _transform(line, tile, inverse).astype(dtype)
            return jax.lax.dynamic_update_slice(out, transformed, (0, i * batch_size))

        out = jax.lax.fori_loop(0, count, body, jnp.zeros(padded.shape, dtype=dtype))[
            :, : flat.shape[1]
        ]
    return jnp.moveaxis(out.reshape(moved.shape), 0, axis)


def _faces(a, axis):
    slices = [slice(1, -1)] * 3
    slices[axis] = jnp.array([0, a.shape[axis] - 1])
    return a[tuple(slices)]


def _setfaces(a, axis, values):
    slices = [slice(1, -1)] * 3
    slices[axis] = jnp.array([0, a.shape[axis] - 1])
    return a.at[tuple(slices)].set(values)


def _edge_mask(shape):
    ends = []
    for axis, n in enumerate(shape):
        sh = [1] * 3
        sh[axis] = n
        ends.append(
            ((jnp.arange(n) == 0) | (jnp.arange(n) == n - 1))
            .astype(jnp.int32)
            .reshape(sh)
        )
    return ends[0] + ends[1] + ends[2] >= 2


def _denominator(plan):
    a, b, c = (line.eigenvalues for line in plan.lines)
    return (
        (a[:, None, None] + b[None, :, None] + c[None, None, :])
        .at[:2, :2, :2]
        .set(-10.0)
    )


def pressure_schur3d(plan, pressure):
    """Apply the original strong div(Q grad) operator, including boundary rows."""
    p = _array(plan, pressure)
    out = jnp.zeros_like(p)
    for axis, line in enumerate(plan.lines):
        derivative = _differentiate(line, p, axis)
        masked = (
            jnp.zeros_like(p).at[1:-1, 1:-1, 1:-1].set(derivative[1:-1, 1:-1, 1:-1])
        )
        out += _differentiate(line, masked, axis)
    return out


def pressure_lift3d(plan, pressure):
    """Eight tensor-mode shifts plus edge/corner coordinate shifts, as source."""
    p = _array(plan, pressure)
    coeff = p[1:-1, 1:-1, 1:-1]
    for axis, line in enumerate(plan.lines):
        coeff = _axmul(_lift_vectors(line)[1], coeff, axis)
    a, b, c = (line.eigenvalues[:2] for line in plan.lines)
    coeff *= -10.0 - a[:, None, None] - b[None, :, None] - c[None, None, :]
    for axis, line in enumerate(plan.lines):
        coeff = _axmul(_lift_vectors(line)[0], coeff, axis)
    out = jnp.zeros_like(p).at[1:-1, 1:-1, 1:-1].set(coeff.real)
    return jnp.where(_edge_mask(p.shape), -10.0 * p, out)


def pressure_action3d(plan, pressure):
    """Invertible lifted operator S+L; useful for manufactured direct-core tests."""
    return pressure_schur3d(plan, pressure) + pressure_lift3d(plan, pressure)


def _tensor_solve3d(plan, rhs, *, batch_size=2048):
    if batch_size is not None and (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be None or a static positive integer")
    r = rhs[1:-1, 1:-1, 1:-1]
    for axis, line in enumerate(plan.lines):
        r = r - _axmul(line.coupling, _faces(rhs, axis), axis)
    for axis, line in enumerate(plan.lines):
        r = _axis_transform(line, r, axis, True, batch_size)
    r = r / _denominator(plan)
    for axis, line in enumerate(plan.lines):
        r = _axis_transform(line, r, axis, False, batch_size)
    p = jnp.zeros(rhs.shape, dtype=r.dtype).at[1:-1, 1:-1, 1:-1].set(r)
    for axis, line in enumerate(plan.lines):
        p = _setfaces(
            p,
            axis,
            _axmul(
                line.endpoint_inverse,
                _faces(rhs, axis) - _axmul(line.hei, r, axis),
                axis,
            ),
        )
    return jnp.where(_edge_mask(rhs.shape), rhs / -10.0, p).real


def solve_pressure_poisson3d(
    plan, rhs, *, lifted=False, batch_size=2048, rtol=1e-10, atol=1e-9
):
    """One direct solve, no refinement. lifted/batch_size are static under JIT.

    lifted=False solves compatible S p=b in the representative selected by lifts.
    No physical wall-gradient completion or mean removal is performed. Pressure
    is not unique for S; check its residual or the lifted equation's unique p.
    Always inspect converged. lifted=True solves (S+L)p=b without projection.
    """
    b = _array(plan, rhs)
    rtol, atol = jnp.asarray(rtol), jnp.asarray(atol)
    if rtol.ndim or atol.ndim or jnp.iscomplexobj(rtol) or jnp.iscomplexobj(atol):
        raise ValueError("Tolerances must be real scalars")
    p = _tensor_solve3d(plan, b, batch_size=batch_size)
    residual = pressure_schur3d(plan, p) - b
    if lifted:
        residual += pressure_lift3d(plan, p)
    l2 = jnp.linalg.norm(residual)
    valid = jnp.all(jnp.isfinite(b)) & jnp.all(jnp.isfinite(p)) & jnp.isfinite(l2)
    valid &= jnp.isfinite(rtol) & jnp.isfinite(atol) & (rtol >= 0) & (atol >= 0)
    valid &= l2 <= atol + rtol * jnp.linalg.norm(b)
    return PressurePoisson3DResult(p, jnp.max(abs(residual)), l2, valid)
