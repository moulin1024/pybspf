"""Tensor BSPF pressure projection extracted from BSPF_3D64_T2_20260917.

Solves S p = b with S = div(Q grad), where Q vanishes on all four walls.
This is the NS pressure Schur equation, not a general Neumann Laplacian.
Arrays use (y, x); vector components use (x, y). Only real CPU data is supported.
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la

from ._pressure_bspf import PressureLine, real_array


@dataclass(frozen=True)
class PressurePoisson2DResult:
    """Pressure and diagnostics, evaluated after optional wall completion."""

    pressure: np.ndarray
    schur_residual_linf: float
    schur_residual_l2: float
    wall_gradient_fit_linf: float | None


class PressurePoisson2D:
    """Reusable masked BSPF pressure solve and no-slip vector projection.

    Parameters
    ----------
    x, y : array_like
        Uniform, increasing, endpoint-inclusive coordinate grids. Scalars use
        shape ``(len(y), len(x))``; vectors use that shape followed by ``2``.
    q, n_basis, degree, baseline_points : int
        BSPF endpoint-jet count, spline count, spline degree, and endpoint
        least-squares sample count. Defaults reproduce the source benchmark.
        Small grids require explicitly smaller parameters; no silent clamping.
    endpoint_method : {"taylor", "chebyshev"}
        The default retains the original Taylor fit. Chebyshev uses a local
        augmented-QR fit and copies endpoint values exactly.
    chebyshev_modes : int
        Local fit degree plus one, between q and baseline_points. Used only
        for Chebyshev endpoints; increasing it can amplify roundoff or noise.
    endpoint_regularization : float
        Nonnegative normalized modal penalty strength for Chebyshev fitting.
        This regularizes the endpoint estimator, not the pressure equation.

    Notes
    -----
    The discrete gradient has eight masked null modes: four tensor modes and
    four corner coordinates. Optional wall-gradient least squares determines
    seven nonconstant coefficients. A trapezoidal mean fixes the constant.
    No global grid-by-grid matrix is assembled. Setup factors 1D matrices and
    a wall-by-seven completion matrix; each solve uses tensor transforms.
    """

    def __init__(
        self,
        x,
        y,
        *,
        q=9,
        n_basis=32,
        degree=13,
        baseline_points=14,
        endpoint_method="taylor",
        chebyshev_modes=12,
        endpoint_regularization=1e-12,
    ):
        options = dict(
            q=q,
            n_basis=n_basis,
            degree=degree,
            baseline_points=baseline_points,
            endpoint_method=endpoint_method,
            chebyshev_modes=chebyshev_modes,
            endpoint_regularization=endpoint_regularization,
        )
        self._x = PressureLine(x, **options)
        self._y = PressureLine(y, **options)
        self.x, self.y = self._x.x.copy(), self._y.x.copy()
        self.shape = (self.y.size, self.x.size)
        self.mask = np.zeros(self.shape)
        self.mask[1:-1, 1:-1] = 1
        self.walls = self.mask == 0
        self._weights = np.outer(self._y.weights, self._x.weights)
        self._corners = np.zeros(self.shape, dtype=bool)
        self._corners[np.ix_([0, self.y.size - 1], [0, self.x.size - 1])] = True
        self._shift = -10.0
        self._den = self._y.lam[:, None] + self._x.lam[None, :]
        self._delta = self._shift - self._den[:2, :2].copy()
        self._den[:2, :2] = self._shift
        if np.min(abs(self._den)) < 1e-9:
            raise ValueError("Unexpected resonance in the tensor pressure operator.")
        # Store only seven wall-gradient columns; generate null fields on demand.
        B = np.column_stack(
            [self.gradient(self._null_field(j))[self.walls].ravel() for j in range(7)]
        )
        self._scales = la.norm(B, axis=0)
        if np.any(self._scales == 0):
            raise ValueError("Degenerate wall completion basis.")
        self._wall_basis = B / self._scales
        # QR avoids the squared condition number of the source's normal equations.
        self._wall_q, self._wall_r = la.qr(self._wall_basis, mode="economic")
        if np.linalg.matrix_rank(self._wall_r) != 7:
            raise ValueError("Wall completion must have rank seven.")

    def _array(self, value, name, vector=False):
        a = real_array(value, name)
        shape = self.shape + ((2,) if vector else ())
        if a.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {a.shape}.")
        return a

    def remove_mean(self, p):
        """Return pressure with zero trapezoidal volume mean."""
        p = self._array(p, "pressure")
        return p - np.sum(self._weights * p) / self._weights.sum()

    def gradient(self, p):
        """Return the strong BSPF gradient, with components (x, y)."""
        p = self._array(p, "pressure")
        return np.stack([self._x.apply(p, 1), self._y.apply(p, 0)], axis=-1)

    def divergence(self, vector):
        """Return strong BSPF divergence at every grid node, including walls."""
        v = self._array(vector, "vector", vector=True)
        return self._x.apply(v[..., 0], 1) + self._y.apply(v[..., 1], 0)

    def schur(self, p):
        """Apply S = div(Q grad); this is not the BSPF D2 Laplacian."""
        return self.divergence(self.mask[..., None] * self.gradient(p))

    def _lift(self, p):
        X, Y = self._x, self._y
        coefficients = Y.Vi[:2] @ p[1:-1, 1:-1] @ X.Vi[:2].T
        out = np.zeros(self.shape, dtype=complex)
        out[1:-1, 1:-1] = Y.V[:, :2] @ (self._delta * coefficients) @ X.V[:, :2].T
        out[self._corners] = self._shift * p[self._corners]
        return out.real

    def _tensor_solve(self, rhs):
        X, Y = self._x, self._y
        r = rhs[1:-1, 1:-1].astype(complex)
        r -= Y.C @ rhs[[0, -1], 1:-1]
        r -= rhs[1:-1, :][:, [0, -1]] @ X.C.T
        core = Y.V @ ((Y.Vi @ r @ X.Vi.T) / self._den) @ X.V.T
        p = np.zeros(self.shape, dtype=complex)
        p[1:-1, 1:-1] = core
        p[[0, -1], 1:-1] = Y.Ei @ (rhs[[0, -1], 1:-1] - Y.Hei @ core)
        p[1:-1, [0, -1]] = (rhs[1:-1, [0, -1]] - core @ X.Hei.T) @ X.Ei.T
        p[self._corners] = rhs[self._corners] / self._shift
        if abs(p.imag).max() > 1e-6 * max(abs(p.real).max(), np.finfo(float).tiny):
            raise RuntimeError("Unexpected imaginary contamination in tensor solve.")
        return p.real

    def _null_field(self, j):
        if j < 3:
            iy, ix = ((0, 1), (1, 0), (1, 1))[j]
            return np.outer(self._y.Z[:, iy], self._x.Z[:, ix])
        p = np.zeros(self.shape)
        p.ravel()[np.flatnonzero(self._corners)[j - 3]] = 1
        return p

    def solve(self, rhs, *, wall_gradient=None, rtol=1e-10, atol=1e-9):
        """Solve a compatible masked pressure equation ``S p = rhs``.

        ``wall_gradient`` is an optional ``(ny, nx, 2)`` target; only its wall
        entries enter an unweighted least-squares fit. Without it, return the
        zero-mean representative selected by the tensor lifts. This generally
        differs from the completed physical pressure in nonconstant null modes.

        Incompatible RHS data raises ValueError rather than being silently
        projected. Compatibility is discrete membership in range(S), not simply
        a zero volume integral. The final Euclidean residual must be at most
        ``atol + rtol * norm(rhs)``. A failure can also indicate numerical loss
        of accuracy for the requested grid, parameters, or tolerances.
        """
        b = self._array(rhs, "rhs")
        if not np.isfinite(rtol) or not np.isfinite(atol) or min(rtol, atol) < 0:
            raise ValueError("rtol and atol must be finite and nonnegative.")
        target = (
            None
            if wall_gradient is None
            else self._array(wall_gradient, "wall_gradient", vector=True)
        )
        p = self._tensor_solve(b)
        for _ in range(2):
            p += self._tensor_solve(b - self.schur(p) - self._lift(p))
        if target is not None:
            residual = (target - self.gradient(p))[self.walls].ravel()
            coeff = la.solve_triangular(self._wall_r, self._wall_q.T @ residual)
            for j, value in enumerate(coeff / self._scales):
                p += value * self._null_field(j)
        p = self.remove_mean(p)
        residual = self.schur(p) - b
        residual_l2 = float(la.norm(residual))
        if not np.isfinite(residual_l2) or residual_l2 > atol + rtol * la.norm(b):
            raise ValueError(
                f"Masked Poisson RHS is incompatible or solve accuracy is insufficient: "
                f"residual {residual_l2:.3e} exceeds {atol + rtol * la.norm(b):.3e}."
            )
        fit = (
            None
            if target is None
            else float(abs((self.gradient(p) - target)[self.walls]).max())
        )
        return PressurePoisson2DResult(p, float(abs(residual).max()), residual_l2, fit)

    def project(self, raw, *, completion=True, rtol=1e-10, atol=1e-9):
        """Return ``(Q*(raw-grad(p)), result)`` with zero wall values.

        For NS, pass the non-pressure acceleration, not a time-step-scaled RHS.
        The same operation also projects a vector field. Completion fits the
        wall gradient to ``raw`` and leaves the interior projection unchanged
        up to roundoff. Divergence errors are the negative Schur residual.
        """
        raw = self._array(raw, "raw", vector=True)
        result = self.solve(
            self.divergence(self.mask[..., None] * raw),
            wall_gradient=raw if completion else None,
            rtol=rtol,
            atol=atol,
        )
        projected = self.mask[..., None] * (raw - self.gradient(result.pressure))
        return projected, result


__all__ = ["PressurePoisson2D", "PressurePoisson2DResult"]
