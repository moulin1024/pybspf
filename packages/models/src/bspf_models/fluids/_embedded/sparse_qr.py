"""Implicit SuiteSparseQR null-space actions in an isolated native process.

Isolation prevents conflicting OpenMP runtimes in Python/JAX distributions. Q
is never materialized. The constraint matrix stays sparse; numerical rank is
revealed by a bounded dense SVD of the QR range core during setup.
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

_program = None
_directory = None


def _executable():
    global _program, _directory
    if _program is not None:
        return _program
    prefixes = [
        Path(p)
        for p in (
            os.environ.get("BSPF_SUITESPARSE_PREFIX", ""),
            "/opt/homebrew/opt/suite-sparse",
            sys.prefix,
            "/usr",
            "/usr/local",
        )
        if p
    ]
    for prefix in prefixes:
        include = prefix / "include" / "suitesparse"
        if (include / "SuiteSparseQR_C.h").exists():
            break
    else:
        raise RuntimeError(
            "Sparse structural setup requires SuiteSparse development headers/libraries and a C++ compiler. Set BSPF_SUITESPARSE_PREFIX to their install prefix."
        )
    _directory = tempfile.TemporaryDirectory(prefix="bspf-spqr-")
    output = Path(_directory.name) / "projector"
    library = prefix / "lib"
    command = [
        os.environ.get("CXX", "c++"),
        "-std=c++17",
        "-O2",
        f"-I{include}",
        str(Path(__file__).with_name("spqr_projector.cpp")),
        f"-L{library}",
        f"-Wl,-rpath,{library}",
        "-lspqr",
        "-lcholmod",
        "-lsuitesparseconfig",
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    _program = output
    return _program


class SparseNullSpace:
    """Persistent isolated SPQR process; Q and the null basis stay implicit."""

    def __init__(self, constraints, tolerance):
        import threading

        self.lock = threading.Lock()
        self.directory = tempfile.TemporaryDirectory(prefix="bspf-null-")
        directory = Path(self.directory.name)
        executable = _executable()
        self.row_order = np.argsort(-np.asarray(constraints.power(2).sum(axis=1)).ravel(), kind="stable")
        a = constraints[self.row_order].T.tocsc().astype(float)
        a.sum_duplicates()
        a.sort_indices()
        matrix = directory / "matrix.bin"
        with matrix.open("wb") as f:
            np.array([*a.shape, a.nnz], dtype=np.int64).tofile(f)
            np.asarray(a.indptr, dtype=np.int64).tofile(f)
            np.asarray(a.indices, dtype=np.int64).tofile(f)
            a.data.tofile(f)
        self.error_file = (directory / "stderr.log").open("w+b")
        self.process = subprocess.Popen(
            [str(executable), str(matrix), str(float(tolerance))],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.error_file,
        )
        self.rank = int(np.frombuffer(self._read(8), dtype=np.int64)[0])
        dims = np.frombuffer(self._read(24), dtype=np.int64)
        nr, nc, nnz = map(int, dims)
        rp = np.frombuffer(self._read((nc + 1) * 8), dtype=np.int64).copy()
        ri = np.frombuffer(self._read(nnz * 8), dtype=np.int64).copy()
        rv = np.frombuffer(self._read(nnz * 8), dtype=float).copy()
        import scipy.sparse as sp
        self.R = sp.csc_matrix((rv, ri, rp), shape=(nr, nc))
        self.column_order = np.frombuffer(self._read(nc * 8), dtype=np.int64).copy()
        self.dimension = a.shape[0]
        self.rows = a.shape[1]
        self.free = self.dimension - self.rank
        matrix.unlink()

    def _read(self, nbytes):
        data = bytearray()
        while len(data) < nbytes:
            part = self.process.stdout.read(nbytes - len(data))
            if not part:
                self.error_file.seek(0)
                raise RuntimeError(
                    f"Sparse QR worker exited: {self.error_file.read().decode(errors='replace')}"
                )
            data.extend(part)
        return data

    def _action(self, command, x, size, expected):
        x = np.ascontiguousarray(x, dtype=np.float64)
        if x.shape != (expected,):
            raise ValueError("Implicit QR vector dimension mismatch")
        with self.lock:
            self.process.stdin.write(np.array([command], dtype=np.int64).tobytes())
            self.process.stdin.write(x.tobytes())
            self.process.stdin.flush()
            return np.frombuffer(self._read(size * 8), dtype=np.float64).copy()

    def restrict(self, x):
        return self._action(1, x, self.free, self.dimension)

    def lift(self, x):
        return self._action(2, x, self.dimension, self.free)

    def affine(self, x):
        return self._action(3, np.asarray(x)[self.row_order], self.dimension, self.rows)

    def full_qt(self, x):
        return self._action(5, x, self.dimension, self.dimension)

    def full_q(self, x):
        return self._action(6, x, self.dimension, self.dimension)

    def full_q_block(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] != self.dimension or not 1 <= x.shape[1] <= 256:
            raise ValueError("Expected a full-Q block with 1..256 columns")
        with self.lock:
            self.process.stdin.write(np.array([7, x.shape[1]], dtype=np.int64).tobytes())
            self.process.stdin.write(x.tobytes(order="F"))
            self.process.stdin.flush()
            return np.frombuffer(self._read(x.size * 8), dtype=float).reshape(x.shape, order="F").copy()

    def affine_transpose(self, x):
        ordered = self._action(4, x, self.rows, self.dimension)
        result = np.empty_like(ordered);result[self.row_order] = ordered
        return result

    def inverse_norm_estimate(self):
        # Power iteration on L L^T, L = the basic QR affine lift. This measures
        # the retained triangular system, not merely its diagonal pivots.
        if self.rank == 0:
            return 0.0
        x = np.random.default_rng(718).normal(size=self.dimension)
        x /= np.linalg.norm(x)
        estimate = 0.0
        for _ in range(24):
            y = self.affine_transpose(x)
            estimate = float(np.linalg.norm(y))
            if not np.isfinite(estimate) or estimate == 0:
                return estimate
            z = self.affine(y / estimate)
            norm = np.linalg.norm(z)
            if not np.isfinite(norm) or norm == 0:
                return float("inf")
            x = z / norm
        return float(np.linalg.norm(self.affine_transpose(x)))

    def close(self):
        process = getattr(self, "process", None)
        if process is not None:
            if process.poll() is None:
                try:
                    process.stdin.write(bytes(8))
                    process.stdin.flush()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    process.wait(timeout=5)
            process.stdin.close()
            process.stdout.close()
            self.error_file.close()
            self.process = None
        if getattr(self, "directory", None):
            self.directory.cleanup()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class SpectralTraceNullSpace:
    """Rank reveal a bounded QR core, keeping the large Householder Q implicit.

    SPQR diagonal dropping alone is unsafe for nearly dependent trace functions.
    The compact R factor is rank revealed with SVD; neither the original global
    constraint matrix nor the global null basis is made dense. Core storage has
    an explicit cap and is reported, rather than hidden as a scalable operation.
    """
    def __init__(self, constraints, tolerance):
        import scipy.linalg as la
        self.qr = SparseNullSpace(constraints, 1e-14)
        try:
            R = self.qr.R
            if R.shape[0] * R.shape[1] > 64_000_000:
                raise ValueError("Compact constraint rank core exceeds the 512 MB storage cap")
            # Only the reduced R core is dense; sparse global input stays sparse.
            core = np.zeros(R.shape)
            for j in range(R.shape[1]):
                sl = slice(R.indptr[j], R.indptr[j + 1])
                core[R.indices[sl], j] = R.data[sl]
            probe = np.random.default_rng(713).normal(size=constraints.shape[0])
            transformed = self.qr.full_qt(constraints.T @ probe)
            ordered = probe[self.qr.row_order][self.qr.column_order]
            error = np.linalg.norm(transformed[:self.qr.rank] - core @ ordered)
            if error > 1e-11 * max(np.linalg.norm(transformed), 1.):
                raise RuntimeError("Exported rank core is inconsistent with implicit Q")
            U, singular, Vt = la.svd(core, full_matrices=False, check_finite=False)
            # Limit amplification of quadrature/representation roundoff. This
            # floor is explicit in metadata; it is not the physical audit tol.
            self.effective_tolerance = max(float(tolerance), 1e-9)
            keep = singular > self.effective_tolerance * max(singular[0], 1.)
            self.rank = int(np.sum(keep));self.dimension = self.qr.dimension
            self.free = self.dimension - self.rank;self.rows = constraints.shape[0]
            self.qrank = self.qr.rank
            self.discard = U[:, ~keep]
            self.kept = U[:, keep]
            self.inverse = Vt[keep] / singular[keep, None]
            self.affine_gain = float(1. / singular[keep][-1]) if self.rank else 0.
            self.rank_history = [dict(qr_tolerance=1e-14, qr_rank=self.qrank, rank=self.rank,
                svd_tolerance=self.effective_tolerance, affine_gain=self.affine_gain,
                core_shape=list(core.shape), core_bytes=core.nbytes)]
        except BaseException:
            self.qr.close();raise
    def restrict(self, x):
        y = self.qr.full_qt(x)
        return np.r_[self.discard.T @ y[:self.qrank], y[self.qrank:]]
    def lift(self, x):
        k = self.discard.shape[1];y = np.zeros(self.dimension)
        y[:self.qrank] = self.discard @ x[:k];y[self.qrank:] = x[k:]
        return self.qr.full_q(y)
    def affine(self, target):
        ordered = np.asarray(target)[self.qr.row_order][self.qr.column_order]
        y = np.zeros(self.dimension)
        y[:self.qrank] = self.kept @ (self.inverse @ ordered)
        return self.qr.full_q(y)
    def close(self):
        self.qr.close()


class LocalDivergenceFreeNullSpace:
    """Compose sparse block-local divergence elimination and implicit trace QR."""

    def __init__(self, local_basis, constraints, tolerance):
        self.local_basis = local_basis.tocsr()
        trace = (constraints @ self.local_basis).tocsr()
        self.inner = SpectralTraceNullSpace(trace, tolerance)
        self.rank_history = self.inner.rank_history
        self.effective_svd_tolerance = self.inner.effective_tolerance
        self.affine_gain = self.inner.affine_gain
        self.dimension = self.local_basis.shape[0]
        self.free = self.inner.free
        self.rank = self.dimension - self.free
        self.rows = constraints.shape[0]

    def restrict(self, x):
        return self.inner.restrict(self.local_basis.T @ x)

    def lift(self, x):
        return self.local_basis @ self.inner.lift(x)

    def affine(self, x):
        return self.local_basis @ self.inner.affine(x)

    def close(self):
        self.inner.close()


class ArrayNullSpace:
    """Bounded explicit null basis in local divergence-free coordinates.

    Trades O(local_dimension * free) storage for device-friendly GEMV actions.
    No QR worker is needed for its runtime actions. This is an optional backend,
    not an asymptotically scalable replacement for the implicit representation.
    """
    def __init__(self, implicit, max_bytes=256 * 1024**2):
        inner = implicit.inner
        required = inner.dimension * inner.free * 8
        if required > max_bytes:
            raise ValueError(f"Array projector needs {required} bytes, exceeding cap {max_bytes}; use implicit_qr projection")
        self.local_basis = implicit.local_basis
        self.dimension, self.free, self.rank = implicit.dimension, implicit.free, implicit.rank
        self.basis = np.empty((inner.dimension, inner.free))
        k = inner.discard.shape[1]
        for first in range(0, inner.free, 64):
            last = min(first + 64, inner.free)
            coefficients = np.zeros((inner.dimension, last-first))
            for j in range(first,last):
                if j < k:
                    coefficients[:inner.qrank,j-first] = inner.discard[:,j]
                else:
                    coefficients[inner.qrank+j-k,j-first] = 1.
            self.basis[:,first:last] = inner.qr.full_q_block(coefficients)
        self.bytes = required

    def restrict(self, velocity):
        return self.basis.T @ (self.local_basis.T @ velocity)

    def lift(self, coordinates):
        return self.local_basis @ (self.basis @ coordinates)

    def jax_actions(self):
        """Pure JAX device-resident actions; no host callbacks or QR worker."""
        import jax
        import jax.numpy as jnp
        from .runtime import sparse
        if not jax.config.x64_enabled:
            raise ValueError("Array constraint projection requires JAX float64")
        local = sparse(self.local_basis)
        basis = jnp.asarray(self.basis)
        return (jax.jit(lambda x: basis.T @ (local.T @ x)),
                jax.jit(lambda z: local @ (basis @ z)))

    def close(self):
        pass
