"""Variable-coefficient elliptic solve using the stream NS tensor inverse."""

from dataclasses import dataclass
import numpy as np
import scipy.linalg as la
from .rectangle_poisson import tensor_elliptic_solve


@dataclass
class TensorPoissonPreconditioner:
    left: np.ndarray
    right: np.ndarray
    denominator: np.ndarray

    def __call__(self, rhs):
        vector = rhs.ndim == 1
        values = rhs[:, None] if vector else rhs
        batch = values.T.reshape((-1,) + self.denominator.shape)
        solved = tensor_elliptic_solve(batch, self.denominator, self.left, self.right)
        result = solved.reshape(values.shape[1], -1).T
        return result[:, 0] if vector else result


def plan_tensor_preconditioner(mr, kr, mt, kt, radial_weight, angular_weight):
    lr, vr = la.eigh((kr + kr.T) / 2, (mr + mr.T) / 2)
    lt, vt = la.eigh((kt + kt.T) / 2, (mt + mt.T) / 2)
    denominator = radial_weight * lr[:, None] + angular_weight * lt[None, :]
    if denominator.min() <= 0:
        raise ValueError("Tensor elliptic preconditioner must be positive definite")
    return TensorPoissonPreconditioner(vr, vt, denominator)


def tensor_pcg(matrix, rhs, preconditioner, *, rtol=2e-12, maxiter=500):
    """Independent PCG columns batched through the common tensor Poisson solve.

    Recompute the actual residual before accepting convergence. This is a
    variable-coefficient solve; the separable inverse is a preconditioner only.
    """
    if not np.isfinite(rtol) or rtol <= 0 or maxiter < 1:
        raise ValueError("Invalid PCG controls")
    vector = rhs.ndim == 1
    b = np.asarray(rhs[:, None] if vector else rhs)
    x = np.zeros_like(b)
    r = b.copy()
    z = preconditioner(r)
    p = z.copy()
    rho = np.sum(r * z, axis=0)
    norms = la.norm(b, axis=0)
    target = rtol * np.maximum(norms, 1e-30)
    active = la.norm(r, axis=0) > target
    counts = np.zeros(b.shape[1], dtype=int)
    for iteration in range(maxiter):
        if not active.any():
            break
        p[:, ~active] = 0
        ap = matrix @ p
        pap = np.sum(p * ap, axis=0)
        if np.any(pap[active] <= 0):
            raise RuntimeError("PCG lost positive definiteness")
        alpha = np.divide(rho, pap, out=np.zeros_like(rho), where=active)
        x += p * alpha
        r -= ap * alpha
        counts[active] += 1
        candidates = la.norm(r, axis=0) <= target
        refresh = (iteration + 1) % 40 == 0 or np.all(candidates)
        if refresh:
            r = b - matrix @ x
        new_active = la.norm(r, axis=0) > target
        z = preconditioner(r)
        new_rho = np.sum(r * z, axis=0)
        beta = np.divide(
            new_rho, rho, out=np.zeros_like(rho), where=active & new_active
        )
        p = z if refresh else z + p * beta
        rho = new_rho
        active = new_active
    residual = b - matrix @ x
    relative = la.norm(residual, axis=0) / np.maximum(norms, 1e-30)
    if np.any(relative > rtol * 1.05):
        raise RuntimeError(
            f"Tensor-preconditioned elliptic solve did not converge: {relative.max():g}"
        )
    diagnostics = dict(
        elliptic_solver="tensor_pcg",
        elliptic_max_iterations=int(counts.max()),
        elliptic_mean_iterations=float(counts.mean()),
        elliptic_max_relative_residual=float(relative.max()),
    )
    return (x[:, 0] if vector else x), diagnostics
