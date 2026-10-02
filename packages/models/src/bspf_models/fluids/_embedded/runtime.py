"""Sparse JAX operators and reusable, host-factorized linear solves."""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from jax.experimental.sparse import BCOO


def sparse(matrix):
    matrix = matrix.tocoo()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    indices = np.column_stack((matrix.row, matrix.col)).astype(np.int32)
    return BCOO((jnp.asarray(matrix.data), jnp.asarray(indices)), shape=matrix.shape)


def sample_matrix(records, n):
    """Stack local quadrature evaluation tables without global dense storage."""
    rows, cols, data = [], [], []
    start = 0
    for ids, values in records:
        rows.append(np.repeat(np.arange(start, start + len(values)), len(ids)))
        cols.append(np.tile(ids, len(values)))
        data.append(values.ravel())
        start += len(values)
    if not rows:
        return sp.csr_matrix((0, n))
    return sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(start, n),
    ).tocsr()


def factor_solve(matrix, backend):
    """Factor once on the host; dense/sparse solves execute in JAX.

    The explicit host_sparse option instead calls a host LU through a callback.

    The sparse backend trades triangular parallelism for O(nnz) storage. It is
    a correctness-oriented fallback, not a GPU sparse-direct performance claim.
    """
    if backend == "dense":
        if matrix.shape[0] > 4096:
            raise ValueError("Dense backend limited to 4096 unknowns; use sparse")
        lu, piv = sl.lu_factor(matrix.toarray())
        if not np.all(np.isfinite(lu)) or np.any(np.diag(lu) == 0):
            raise ValueError("Singular time-step matrix")
        lu, piv = jnp.asarray(lu), jnp.asarray(piv)
        return lambda b: jl.lu_solve((lu, piv), b)
    if backend == "host_sparse":
        factor = splu(
            matrix.tocsc(),
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.001,
            options={"SymmetricMode": True},
        )
        shape = jax.ShapeDtypeStruct((matrix.shape[0],), jnp.float64)
        return lambda b: jax.pure_callback(
            lambda rhs: factor.solve(np.asarray(rhs)), shape, b
        )
    if backend != "sparse":
        raise ValueError("linear_backend must be 'dense' or 'sparse'")
    factor = splu(
        matrix.tocsc(),
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.001,
        options={"SymmetricMode": True},
    )

    def triangular(matrix, lower):
        diagonal = jnp.asarray(matrix.diagonal())
        strict = (sp.tril(matrix, -1) if lower else sp.triu(matrix, 1)).tocsr()
        data, ids, ptr = map(jnp.asarray, (strict.data, strict.indices, strict.indptr))
        n = matrix.shape[0]
        # Padding also makes the empty strict triangular matrix traceable.
        data = jnp.concatenate((data, jnp.zeros(1, data.dtype)))
        ids = jnp.concatenate((ids, jnp.zeros(1, ids.dtype)))

        def solve(b):
            def row(k, x):
                i = k if lower else n - 1 - k
                total = jax.lax.fori_loop(
                    ptr[i],
                    ptr[i + 1],
                    lambda j, a: a + data[j] * x[ids[j]],
                    jnp.zeros((), b.dtype),
                )
                return x.at[i].set((b[i] - total) / diagonal[i])

            return jax.lax.fori_loop(0, n, row, jnp.zeros_like(b))

        return solve

    lower, upper = triangular(factor.L, True), triangular(factor.U, False)
    rows, cols = jnp.asarray(np.argsort(factor.perm_r)), jnp.asarray(factor.perm_c)
    return lambda b: upper(lower(b[rows]))[cols]
