"""Shared sample validation and constrained spline fitting."""

from __future__ import annotations

from ..backend import get_array_module, validate_backend_array


def prepare_samples(op, f, *, ndim=1):
    """Normalize numeric samples without silently moving between CPU and GPU."""
    validate_backend_array(f, use_gpu=op.use_gpu, name="f")
    xp = get_array_module(use_gpu=op.use_gpu)
    f = xp.asarray(f)
    if f.ndim != ndim or f.shape[0] != op.grid.n:
        shape = "(n,)" if ndim == 1 else "(n, batch)"
        raise ValueError(f"Expected f with shape {shape}, where n={op.grid.n} matches grid size.")
    dtype = xp.complex128 if xp.iscomplexobj(f) else xp.float64
    return f.astype(dtype, copy=False)


def solve_spline(op, f, lam=0.0, neumann_bc=None, *, ndim=1):
    """Fit one signal or a matrix of signals using the same cached KKT system."""
    f = prepare_samples(op, f, ndim=ndim)
    xp = get_array_module(use_gpu=op.use_gpu)
    n_basis = op._BW_f.shape[0]
    shape = (n_basis + 2 * op.order,) + f.shape[1:]
    rhs = xp.empty(shape, dtype=f.dtype)
    rhs[:n_basis] = 2.0 * (op._BW_f @ f)
    rhs[n_basis:] = op._BND_f @ f
    if neumann_bc is not None:
        left, right = neumann_bc
        if (left is not None or right is not None) and op.order < 2:
            raise ValueError("Neumann BC requires order >= 2 (value and first derivative constraints).")
        if left is not None:
            rhs[n_basis + 1] = left
        if right is not None:
            rhs[n_basis + op.order + 1] = right
    coefficients = op._kkt_solver.solve(rhs, lam)[:n_basis]
    spline = op._BT0_f @ coefficients
    return coefficients, spline, f - spline
