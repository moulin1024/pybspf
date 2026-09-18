"""Continuous Dirichlet PDE test using BSPF D1@D1 and direct tensor inversion.

This is a separate boundary closure, NOT the masked pressure solver. Analytic
Laplacian and boundary values are the only solve inputs; no manufactured A@p.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from bspf_jax.pressure import _make_line, _differentiate
from bspf_jax.pressure3d import _axmul, _axis_transform, _faces
from bspf_jax._compressed_transform import build_transforms


class Axis(NamedTuple):
    second: jax.Array
    eigenvalues: jax.Array
    vectors: jax.Array | None
    inverse_vectors: jax.Array | None
    compressed: object = None


def make_axis(n):
    line = _make_line(np.linspace(0, 1, n), 9, 32, 13, 16, "chebyshev", 12, 1e-12)
    d = _differentiate(line, jnp.eye(n), 0)
    d2 = d @ d
    lam, v = jnp.linalg.eig(d2[1:-1, 1:-1])
    idx = jnp.argsort(abs(lam))
    lam, v = lam[idx], v[:, idx]
    vi = jnp.linalg.solve(v, jnp.eye(n - 2))
    if not bool(jnp.all(lam.real < -1)):
        raise ValueError("Unexpected nonnegative Dirichlet spectrum")
    return Axis(d2, lam, v, vi)


def solve(axis, laplacian, boundary):
    # boundary interior must be zero; only the six faces enter the reduced RHS.
    r = laplacian[1:-1, 1:-1, 1:-1]
    coupling = axis.second[1:-1][:, jnp.array([0, axis.second.shape[0] - 1])]
    for k in range(3):
        r = r - _axmul(coupling, _faces(boundary, k), k)
    for k in range(3):
        r = _axis_transform(axis, r, k, True, 2048)
    lam = axis.eigenvalues
    r = r / (lam[:, None, None] + lam[None, :, None] + lam[None, None, :])
    for k in range(3):
        r = _axis_transform(axis, r, k, False, 2048)
    return boundary.at[1:-1, 1:-1, 1:-1].set(r.real)


def fields(n):
    g = np.linspace(0, 1, n)
    x, y, z = g[:, None, None], g[None, :, None], g[None, None, :]
    ex = np.exp(x + 0.5 * y - 0.3 * z)
    trig = np.sin(3 * np.pi * x) * np.cos(2 * np.pi * y) * np.cos(np.pi * z)
    p = ex + trig
    lap = 1.34 * ex - 14 * np.pi**2 * trig
    boundary = p.copy()
    boundary[1:-1, 1:-1, 1:-1] = 0
    return tuple(jnp.asarray(a) for a in (p, lap, boundary))


@jax.jit
def metrics(axis, p, exact, lap, boundary):
    residual = (
        sum(_axmul(axis.second, p, k) for k in range(3))[1:-1, 1:-1, 1:-1]
        - lap[1:-1, 1:-1, 1:-1]
    )
    rnorm = jnp.linalg.norm(residual)
    bnorm = jnp.linalg.norm(lap[1:-1, 1:-1, 1:-1])
    diff = p - exact
    boundary_diff = diff.at[1:-1, 1:-1, 1:-1].set(0)
    return jnp.array(
        [
            jnp.max(abs(diff)),
            jnp.linalg.norm(diff) / jnp.linalg.norm(exact),
            jnp.max(abs(residual)),
            rnorm / bnorm,
            jnp.max(abs(boundary_diff)),
            jnp.max(abs(axis.second)),
        ]
    )


def worker(args):
    jax.config.update("jax_enable_x64", True)
    t = time.perf_counter()
    axis = make_axis(args.size)
    jax.block_until_ready(axis)
    p, lap, g = fields(args.size)
    row = dict(
        n=args.size,
        problem="Delta p = analytic f; exact Dirichlet on all six faces",
        discretization="BSPF D1@D1; NOT masked pressure",
        refinement_steps=0,
        setup_seconds=time.perf_counter() - t,
        condition_vectors=float(jnp.linalg.cond(axis.vectors)),
        results={},
    )
    direct = jax.jit(solve)
    for backend in ["dense", "compressed"]:
        a = axis
        if backend == "compressed":
            factors = build_transforms(
                axis.vectors,
                axis.inverse_vectors,
                tolerance=1e-12,
                leaf_size=16,
                protected_modes=8,
                layout="layered",
            )
            a = axis._replace(vectors=None, inverse_vectors=None, compressed=factors)
        got = jax.block_until_ready(direct(a, lap, g))
        vals = np.asarray(metrics(a, got, p, lap, g))
        row["results"][backend] = dict(
            zip(
                [
                    "error_linf",
                    "error_relative_l2",
                    "residual_linf",
                    "residual_relative_l2",
                    "boundary_error_linf",
                    "max_d2",
                ],
                map(float, vals),
            )
        )
        print(args.size, backend, row["results"][backend], flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.size}.json").write_text(json.dumps(row, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[40, 48, 64, 96, 128, 192, 256]
    )
    parser.add_argument("--out", type=Path, default=Path("build/bspf3d_continuous"))
    args = parser.parse_args()
    if args.size:
        worker(args)
        return
    rows = []
    for n in args.sizes:
        subprocess.run(
            [sys.executable, __file__, "--size", str(n), "--out", str(args.out)],
            check=True,
        )
        rows.append(json.loads((args.out / f"{n}.json").read_text()))
        (args.out / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
