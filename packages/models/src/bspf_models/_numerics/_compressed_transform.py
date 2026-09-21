"""Optional host-built DCT/HODLR factors; application is pure JAX.

SciPy is imported only during compression setup. No dense reconstruction is
performed during application. The default layout batches whole tree levels with
contiguous sibling views; the original indexed grouping remains available.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.fft as jf
import numpy as np


class BlockGroup(NamedTuple):
    rows: jax.Array
    columns: jax.Array
    left: jax.Array
    right: jax.Array | None


class LayerFactors(NamedTuple):
    left: jax.Array
    right: jax.Array | None


class LayeredBlocks(NamedTuple):
    diagonal: jax.Array
    levels: tuple[LayerFactors, ...]


class CompressedTransforms(NamedTuple):
    forward: tuple[BlockGroup, ...] | LayeredBlocks
    inverse: tuple[BlockGroup, ...] | LayeredBlocks
    permutation: jax.Array
    inverse_permutation: jax.Array
    protected_vectors: jax.Array
    protected_inverse: jax.Array


def _groups(a, tol, leaf):
    from scipy.linalg import svd

    buckets = {}

    def save(lo, hi, left, right, factors):
        key = (hi - lo, right - left, -1 if factors[1] is None else factors[1].shape[0])
        buckets.setdefault(key, []).append(
            (np.arange(lo, hi), np.arange(left, right), *factors)
        )

    def visit(lo, hi):
        if hi - lo <= leaf:
            save(lo, hi, lo, hi, (a[lo:hi, lo:hi], None))
            return
        mid = (lo + hi) // 2
        visit(lo, mid)
        visit(mid, hi)
        for i, j, k, end in [(lo, mid, mid, hi), (mid, hi, lo, mid)]:
            block = a[i:j, k:end]
            u, s, vh = svd(block, full_matrices=False)
            tails = np.sqrt(np.r_[np.cumsum(s[::-1] ** 2)[::-1], 0.0])
            rank = int(np.flatnonzero(tails <= tol * np.linalg.norm(s))[0])
            # Small rank buckets reduce the number of separate XLA products.
            rank = min(len(s), 4 * ((rank + 3) // 4))
            if rank == 0:
                continue
            factors = (
                (u[:, :rank] * s[:rank], vh[:rank])
                if rank * sum(block.shape) < block.size
                else (block, None)
            )
            save(i, j, k, end, factors)

    visit(0, len(a))
    return tuple(
        BlockGroup(
            jnp.asarray(np.stack([b[0] for b in blocks])),
            jnp.asarray(np.stack([b[1] for b in blocks])),
            jnp.asarray(np.stack([b[2] for b in blocks])),
            None
            if blocks[0][3] is None
            else jnp.asarray(np.stack([b[3] for b in blocks])),
        )
        for blocks in buckets.values()
    )


def _layers(a, tol, leaf):
    """Equal-width sibling batches; padding is outside the physical DCT grid."""
    from scipy.linalg import svd

    count = 1
    while (len(a) + count - 1) // count > leaf:
        count *= 2
    width = (len(a) + count - 1) // count
    size = count * width
    padded = np.pad(a, ((0, size - len(a)), (0, size - len(a))))
    diagonal = np.stack(
        [padded[i : i + width, i : i + width] for i in range(0, size, width)]
    )
    levels = []
    parents = 1
    while parents < count:
        half = size // (2 * parents)
        decompositions = []
        blocks = []
        for parent in range(parents):
            lo = 2 * parent * half
            for destination, source in [(lo, lo + half), (lo + half, lo)]:
                block = padded[destination : destination + half, source : source + half]
                u, s, vh = svd(block, full_matrices=False)
                tails = np.sqrt(np.r_[np.cumsum(s[::-1] ** 2)[::-1], 0.0])
                rank = int(np.flatnonzero(tails <= tol * np.linalg.norm(s))[0])
                blocks.append(block)
                decompositions.append((u, s, vh, rank))
        rank = min(half, 4 * ((max(d[3] for d in decompositions) + 3) // 4))
        if rank * 2 >= half:
            levels.append(LayerFactors(jnp.asarray(np.stack(blocks)), None))
        else:
            left = np.stack([u[:, :rank] * s[:rank] for u, s, vh, _ in decompositions])
            right = np.stack([vh[:rank] for u, s, vh, _ in decompositions])
            levels.append(LayerFactors(jnp.asarray(left), jnp.asarray(right)))
        parents *= 2
    return LayeredBlocks(jnp.asarray(diagonal), tuple(levels))


def build_transforms(
    v, vi, *, tolerance=1e-12, leaf_size=16, protected_modes=8, layout="layered"
):
    """Host preprocessing; retain only factors and exact protected modes."""
    from scipy.fft import dct
    from scipy.optimize import linear_sum_assignment

    if layout not in ("layered", "grouped"):
        raise ValueError("layout must be 'layered' or 'grouped'")
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("compression tolerance must be finite and in (0, 1)")
    for name, value, minimum in [
        ("leaf_size", leaf_size, 1),
        ("protected_modes", protected_modes, 2),
    ]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    v, vi = np.asarray(v), np.asarray(vi)
    # Actual real eigensystems can use real factors; complex ones remain complex.
    if np.all(v.imag == 0) and np.all(vi.imag == 0):
        v, vi = v.real, vi.real
    c0 = dct(v, type=2, norm="ortho", axis=0)
    _, perm = linear_sum_assignment(-(abs(c0) ** 2))
    c = c0[:, perm]
    b = dct(vi[perm, :], type=2, norm="ortho", axis=1)
    s = min(protected_modes, len(v))
    build = _layers if layout == "layered" else _groups
    return CompressedTransforms(
        build(c, tolerance, leaf_size),
        build(b, tolerance, leaf_size),
        jnp.asarray(perm),
        jnp.asarray(np.argsort(perm)),
        jnp.asarray(v[:, :s]),
        jnp.asarray(vi[:s]),
    )


def _blocks(groups, x):
    if isinstance(groups, LayeredBlocks):
        count, width, _ = groups.diagonal.shape
        size, batch = count * width, x.shape[1]
        padded = jnp.pad(x, ((0, size - x.shape[0]), (0, 0)))
        out = (groups.diagonal @ padded.reshape(count, width, batch)).reshape(
            size, batch
        )
        for level in groups.levels:
            blocks, half, _ = level.left.shape
            # Siblings exchange inputs using a contiguous view/reverse. No
            # index arrays, gathers, or scatter-adds in the HODLR application.
            values = padded.reshape(blocks // 2, 2, half, batch)[:, ::-1].reshape(
                blocks, half, batch
            )
            if level.right is not None:
                values = level.right @ values
            out = out + (level.left @ values).reshape(size, batch)
        return out[: x.shape[0]]
    dtype = jnp.result_type(x, groups[0].left)
    out = jnp.zeros(x.shape, dtype=dtype)
    for group in groups:
        values = x[group.columns]
        if group.right is not None:
            values = group.right @ values
        values = group.left @ values
        out = out.at[group.rows].add(values)
    return out


def _dct(x, inverse=False):
    fn = jf.idct if inverse else jf.dct
    if jnp.iscomplexobj(x):
        return fn(x.real, type=2, norm="ortho", axis=0) + 1j * fn(
            x.imag, type=2, norm="ortho", axis=0
        )
    return fn(x, type=2, norm="ortho", axis=0)


def apply_transform(plan, x, *, inverse=False):
    """Apply along axis zero; remaining axes are independent right-hand sides.

    Fixed algebraic projections preserve selected left/right eigenmodes. No
    iterative solve/refinement or dense transform is used.
    """
    shape = x.shape
    x = x.reshape(shape[0], -1)
    v, w = plan.protected_vectors, plan.protected_inverse
    s = w.shape[0]
    if inverse:
        exact = w @ x
        residual = x - v @ exact
        result = _blocks(plan.inverse, _dct(residual))[plan.inverse_permutation]
        result = jnp.concatenate([exact, result[s:]], axis=0)
    else:
        exact = x[:s]
        residual = jnp.concatenate([jnp.zeros_like(x[:s]), x[s:]], axis=0)
        result = _dct(_blocks(plan.forward, residual[plan.permutation]), inverse=True)
        result += v @ (exact - w @ result)
    return result.reshape(shape)


def transform_storage(plan):
    """Host diagnostic, including protection and index arrays; excludes setup."""
    leaves = jax.tree.leaves(plan)
    factor_arrays = []
    batches = 0
    for gs in (plan.forward, plan.inverse):
        if isinstance(gs, LayeredBlocks):
            factor_arrays.append(gs.diagonal)
            batches += 1
            gs = gs.levels
        for group in gs:
            factor_arrays.append(group.left)
            if group.right is not None:
                factor_arrays.append(group.right)
        batches += len(gs)
    n = plan.permutation.size
    return dict(
        factor_scalars=sum(a.size for a in factor_arrays),
        factor_ratio=sum(a.size for a in factor_arrays) / (2 * n * n),
        stored_bytes=sum(a.size * a.dtype.itemsize for a in leaves),
        protected_modes=plan.protected_inverse.shape[0],
        groups=batches,
        layout="layered" if isinstance(plan.forward, LayeredBlocks) else "grouped",
    )
