"""Isolate sibling exchange before versus after the level matrix products."""

import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from bspf_jax._compressed_transform import (
    build_transforms,
    _blocks,
    LayeredBlocks,
    LayerFactors,
)

jax.config.update("jax_enable_x64", True)


def source_order(groups, x):
    count, width, _ = groups.diagonal.shape
    size, batch = count * width, x.shape[1]
    padded = jnp.pad(x, ((0, size - x.shape[0]), (0, 0)))
    out = (groups.diagonal @ padded.reshape(count, width, batch)).reshape(size, batch)
    for level in groups.levels:
        blocks, half, _ = level.left.shape
        values = padded.reshape(blocks, half, batch)
        if level.right is not None:
            values = level.right @ values
        values = (level.left @ values).reshape(blocks // 2, 2, half, batch)[:, ::-1]
        out = out + values.reshape(size, batch)
    return out[: x.shape[0]]


def swap(a):
    return a.reshape(a.shape[0] // 2, 2, *a.shape[1:])[:, ::-1].reshape(a.shape)


def time_call(fn, g, x):
    jax.block_until_ready(fn(g, x))
    result = []
    for _ in range(15):
        start = time.perf_counter()
        jax.block_until_ready(fn(g, x))
        result.append(time.perf_counter() - start)
    return float(np.median(result))


results = []
for n in [128, 256, 512]:
    data = np.load(f"build/pressure_transform_analysis/matrices_chebyshev_{n}.npz")
    plan = build_transforms(data["V"], data["Vi"])
    g = plan.forward
    sg = LayeredBlocks(
        g.diagonal,
        tuple(
            LayerFactors(
                swap(level.left), None if level.right is None else swap(level.right)
            )
            for level in g.levels
        ),
    )
    x = jnp.asarray(
        np.random.default_rng(1).normal(size=(n - 2, n))
        + 1j * np.random.default_rng(2).normal(size=(n - 2, n))
    )
    f, j = jax.jit(_blocks), jax.jit(source_order)
    np.testing.assert_allclose(f(g, x), j(sg, x), atol=1e-12)
    row = dict(
        N=n, before_seconds=time_call(f, g, x), after_seconds=time_call(j, sg, x)
    )
    results.append(row)
    print(row, flush=True)
Path("build/layered_pressure_benchmark/layer_order.json").write_text(
    json.dumps(results, indent=2)
)
