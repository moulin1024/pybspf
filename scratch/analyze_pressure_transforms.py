"""Analyze actual JAX pressure eigenvectors; no production solver changes.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src
MPLCONFIGDIR=/tmp/pybspf-mpl python3 scratch/analyze_pressure_transforms.py
"""

import argparse
import json
from pathlib import Path
import time

import jax
import numpy as np
import scipy.linalg as la
from scipy.fft import dct, dst
from scipy.optimize import linear_sum_assignment

from bspf_jax.pressure import _make_line

jax.config.update("jax_enable_x64", True)
TOLS = (1e-6, 1e-10, 1e-12)


def ranks(s, scale=None):
    """Minimum rank at relative Frobenius error tolerance (not spectral norm)."""
    norm = la.norm(s) if scale is None else scale
    tail = np.sqrt(np.r_[np.cumsum(s[::-1] ** 2)[::-1], 0.0])
    return {str(t): int(np.flatnonzero(tail <= t * norm)[0]) for t in TOLS}


def bases(n):
    eye = np.eye(n)
    yield "DCT-I", dct(eye, type=1, norm="ortho", axis=0).T
    yield "DCT-II", dct(eye, type=2, norm="ortho", axis=0).T
    yield "DST-I", dst(eye, type=1, norm="ortho", axis=0).T
    x = np.arange(n)
    cols = [np.ones(n) / np.sqrt(n)]
    for k in range(1, (n + 1) // 2):
        cols.extend(
            [
                np.sqrt(2 / n) * np.cos(2 * np.pi * k * x / n),
                np.sqrt(2 / n) * np.sin(2 * np.pi * k * x / n),
            ]
        )
    if n % 2 == 0:
        cols.append((-1.0) ** x / np.sqrt(n))
    yield "real-DFT", np.column_stack(cols)


def hodlr(a, tol, leaf=16):
    """Diagnostic reconstruction and factor-storage count, NOT a fast apply.

    Dense diagonal leaves; SVD-compressed sibling off-diagonal blocks. Each
    block meets relative Frobenius tolerance; use dense if factors cost more.
    """
    out = np.zeros_like(a)
    stats = []

    def visit(lo, hi, depth):
        if hi - lo <= leaf:
            out[lo:hi, lo:hi] = a[lo:hi, lo:hi]
            return (hi - lo) ** 2
        mid = (lo + hi) // 2
        cost = visit(lo, mid, depth + 1) + visit(mid, hi, depth + 1)
        for r, c in [
            (slice(lo, mid), slice(mid, hi)),
            (slice(mid, hi), slice(lo, mid)),
        ]:
            block = a[r, c]
            u, s, vh = la.svd(block, full_matrices=False)
            tail = np.sqrt(np.r_[np.cumsum(s[::-1] ** 2)[::-1], 0.0])
            rank = int(np.flatnonzero(tail <= tol * la.norm(s))[0])
            compressed = rank * sum(block.shape) < block.size
            out[r, c] = (u[:, :rank] * s[:rank]) @ vh[:rank] if compressed else block
            cost += rank * sum(block.shape) if compressed else block.size
            stats.append(
                {
                    "depth": depth,
                    "shape": list(block.shape),
                    "rank": rank,
                    "compressed": compressed,
                }
            )
        return cost

    storage = visit(0, len(a), 0)
    return out, {
        "storage_ratio": storage / a.size,
        "relative_error": float(la.norm(out - a) / la.norm(a)),
        "max_rank": max(s["rank"] for s in stats),
        "blocks": stats,
    }


def solve_check(v, vi, lam, va, via, a):
    """2D lifted interior tensor inverse; no refinement, no wall completion."""
    n = len(lam)
    den = lam[:, None] + lam[None, :]
    delta = -10 - den[:2, :2].copy()
    den[:2, :2] = -10

    def action(p):
        lift = v[:, :2] @ (delta * (vi[:2] @ p @ vi[:2].T)) @ v[:, :2].T
        return a @ p + p @ a.T + lift

    rng = np.random.default_rng(17)
    z = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(z, z, indexing="ij")
    fields = {
        "random": rng.standard_normal((n, n)),
        "smooth": np.exp(xx + 0.5 * yy)
        + np.sin(3 * np.pi * xx) * np.cos(2 * np.pi * yy),
    }
    results = {}
    for name, p in fields.items():
        b = action(p)
        exact = v @ ((vi @ b @ vi.T) / den) @ v.T
        approx = va @ ((via @ b @ via.T) / den) @ va.T
        residual = action(approx) - b
        results[name] = {
            "dense_relative_residual": float(la.norm(action(exact) - b) / la.norm(b)),
            "dense_relative_solution_error": float(la.norm(exact - p) / la.norm(p)),
            "compressed_relative_solution_error": float(
                la.norm(approx - p) / la.norm(p)
            ),
            "compressed_relative_residual": float(la.norm(residual) / la.norm(b)),
        }
    return results


def analyze(n, method, out):
    started = time.perf_counter()
    line = _make_line(
        np.linspace(0, 1, n),
        9,
        32,
        13,
        14 if method == "taylor" else 16,
        method,
        12,
        1e-12,
    )
    v, vi, lam = map(np.asarray, (line.vectors, line.inverse_vectors, line.eigenvalues))
    # Rebuild the reduced operator independently of its eigendecomposition.
    eye = np.eye(n)
    fourier = np.fft.ifft(
        np.fft.fft(eye[:-1], axis=0) * np.asarray(line.multiplier)[:, None], axis=0
    ).real
    d = np.concatenate([fourier, fourier[:1]]) + np.asarray(line.low) @ np.asarray(
        line.projector
    )
    h = d[:, 1:-1] @ d[1:-1, :]
    reduced_a = h[1:-1, 1:-1] - np.asarray(line.coupling) @ np.asarray(line.hei)
    np.savez_compressed(
        out / f"matrices_{method}_{n}.npz", V=v, Vi=vi, eigenvalues=lam, A=reduced_a
    )
    m = n - 2
    record = {
        "N": n,
        "method": method,
        "dimension": m,
        "cond_V": float(np.linalg.cond(v)),
        "max_eigenvalue_imag": float(np.max(abs(lam.imag))),
        "inverse_error": float(la.norm(vi @ v - np.eye(m)) / np.sqrt(m)),
        "eigendecomposition_relative_residual": float(
            la.norm(reduced_a @ v - v * lam) / la.norm(reduced_a)
        ),
        "global": {},
        "bases": {},
    }
    for name, a in [("V", v), ("Vi", vi)]:
        s = la.svdvals(a)
        record["global"][name] = {"ranks": ranks(s), "singular_values": s.tolist()}
    candidates = {}
    for name, t in bases(m):
        c0 = t.T @ v
        _, perm = linear_sum_assignment(-(abs(c0) ** 2))
        c = c0[:, perm]
        b = vi[perm, :] @ t
        item = {}
        for label, a in [("V", c), ("Vi", b)]:
            correction = a - np.diag(np.diag(a))
            s = la.svdvals(correction)
            item[label] = {
                "correction_ranks_relative_to_full": ranks(s, la.norm(a)),
                "correction_relative_norm": float(la.norm(s) / la.norm(a)),
                "correction_singular_values_relative_to_full": (
                    s / la.norm(a)
                ).tolist(),
                "hodlr": {},
            }
            for tol in TOLS:
                _, stats = hodlr(a, tol)
                item[label]["hodlr"][str(tol)] = stats
        record["bases"][name] = item
        candidates[name] = (t, perm, c, b)
    best = min(
        record["bases"],
        key=lambda name: sum(
            record["bases"][name][label]["hodlr"]["1e-10"]["storage_ratio"]
            for label in ["V", "Vi"]
        ),
    )
    t, perm, c, b = candidates[best]
    record["best_basis"] = best
    record["solve_checks"] = {}
    record["null_protected_solve_checks"] = {}
    for tol in TOLS:
        ca, _ = hodlr(c, tol)
        ba, _ = hodlr(b, tol)
        va = (t @ ca)[:, np.argsort(perm)]
        via = (ba @ t.T)[np.argsort(perm), :]
        record["solve_checks"][str(tol)] = solve_check(v, vi, lam, va, via, reduced_a)
        assert (
            max(
                check["dense_relative_residual"]
                for check in record["solve_checks"][str(tol)].values()
            )
            < 1e-7
        ), "Dense baseline failed; compression results are not interpretable."
        # Low-rank corrections preserve both null subspaces; no iterations.
        e = np.eye(m)[:, :2]
        va += v[:, :2] @ (e.T - vi[:2] @ va)
        va[:, :2] = v[:, :2]
        via += (e - via @ v[:, :2]) @ vi[:2]
        via[:2] = vi[:2]
        record["null_protected_solve_checks"][str(tol)] = solve_check(
            v, vi, lam, va, via, reduced_a
        )
    # Test HODLR directly on physical-to-modal matrices in eigenvalue order too.
    record["physical_hodlr"] = {}
    for label, a in [("V", v), ("Vi", vi)]:
        record["physical_hodlr"][label] = hodlr(a, 1e-10)[1]
    record["seconds"] = time.perf_counter() - started
    return record


def plot(records, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for method in ["taylor", "chebyshev"]:
        rows = [r for r in records if r["method"] == method]
        if not rows:
            continue
        ns = [r["N"] for r in rows]
        axes[0, 0].semilogy(ns, [r["cond_V"] for r in rows], "o-", label=method)
        axes[1, 0].plot(
            ns,
            [
                sum(
                    r["bases"][r["best_basis"]][k]["hodlr"]["1e-10"]["storage_ratio"]
                    for k in ["V", "Vi"]
                )
                / 2
                for r in rows
            ],
            "o-",
            label=method,
        )
        axes[1, 1].semilogy(
            ns,
            [
                max(
                    s["compressed_relative_residual"]
                    for s in r["solve_checks"]["1e-10"].values()
                )
                for r in rows
            ],
            "o-",
            label=method,
        )
    row = next(r for r in reversed(records) if r["method"] == "chebyshev")
    for name, bs in row["bases"].items():
        axes[0, 1].semilogy(
            np.arange(1, row["dimension"] + 1),
            bs["V"]["correction_singular_values_relative_to_full"],
            label=name,
        )
    axes[0, 0].set(
        title="Condition number of current V", xlabel="Grid nodes N", ylabel="cond2(V)"
    )
    axes[0, 1].set(
        title=f"V: fast-basis + diagonal correction, N={row['N']}",
        xlabel="Singular-value index",
        ylabel="sigma / ||V||F",
        ylim=(1e-16, 1),
    )
    axes[1, 0].set(
        title="HODLR coefficient storage / dense storage",
        xlabel="Grid nodes N",
        ylabel="Mean of V and inverse, tolerance 1e-10",
        ylim=(0, 1.05),
    )
    axes[1, 1].set(
        title="Compressed direct tensor solve: residual",
        xlabel="Grid nodes N",
        ylabel="Relative residual; no refinement",
    )
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(out / "compressibility.png", dpi=170)
    fig.savefig(out / "compressibility.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument(
        "--out", type=Path, default=Path("build/pressure_transform_analysis")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for n in args.sizes:
        for method in ["taylor", "chebyshev"]:
            r = analyze(n, method, args.out)
            records.append(r)
            (args.out / "results.json").write_text(json.dumps(records, indent=2))
            best = r["bases"][r["best_basis"]]
            print(
                json.dumps(
                    {
                        "N": n,
                        "method": method,
                        "cond": r["cond_V"],
                        "basis": r["best_basis"],
                        "rankV": best["V"]["correction_ranks_relative_to_full"],
                        "hodlr": [
                            best[k]["hodlr"]["1e-10"]["storage_ratio"]
                            for k in ["V", "Vi"]
                        ],
                        "solve": r["solve_checks"]["1e-10"],
                        "seconds": r["seconds"],
                    }
                ),
                flush=True,
            )
    plot(records, args.out)


if __name__ == "__main__":
    main()
