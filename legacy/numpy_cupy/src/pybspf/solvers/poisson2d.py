"""! @file solvers/poisson2d.py
@brief Direct tensor-product Poisson solver for homogeneous/constant Dirichlet data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.fft import dst, dstn, idst
from scipy.fft import idstn
from scipy import linalg as sla

from ..operators.bspf1d import BSPF1D

RHSInput = np.ndarray | Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PODLayerBasis2D:
    """POD-compressed zero-trace layer basis built from teacher corrector snapshots."""

    solution_basis: np.ndarray
    laplacian_basis: np.ndarray
    singular_values: np.ndarray
    n_snapshots: int

    @property
    def rank(self) -> int:
        return int(self.singular_values.size)


def _boundary_value_constraint(op: BSPF1D) -> np.ndarray:
    """Return the left/right endpoint value constraints for one 1D basis."""
    basis_values = np.asarray(op.basis.B0, dtype=np.float64)
    return np.vstack((basis_values[:, 0], basis_values[:, -1]))


def _dirichlet_nullspace(op: BSPF1D) -> np.ndarray:
    """Return a coefficient-space basis that enforces homogeneous Dirichlet data."""
    constraint = _boundary_value_constraint(op)
    null_basis = sla.null_space(constraint)
    if null_basis.size == 0:
        raise ValueError("The spline basis has no interior nullspace under Dirichlet constraints.")
    return np.asarray(null_basis, dtype=np.float64)


def _constraint_right_inverse(constraint: np.ndarray) -> np.ndarray:
    """Return a right inverse for a full-row-rank endpoint constraint matrix."""
    gram = constraint @ constraint.T
    return constraint.T @ np.linalg.inv(gram)


def _mass_matrix(op: BSPF1D) -> np.ndarray:
    """Assemble the 1D quadrature mass matrix."""
    return _quadrature_gram(op, deriv_order=0)


def _stiffness_matrix(op: BSPF1D) -> np.ndarray:
    """Assemble the 1D quadrature stiffness matrix."""
    return _quadrature_gram(op, deriv_order=1)


def _quadrature_gram(op: BSPF1D, *, deriv_order: int) -> np.ndarray:
    """Assemble a 1D Gram matrix by Gauss-Legendre quadrature on knot spans."""
    knots = np.asarray(op.basis.knots, dtype=np.float64)
    q_order = max(op.degree + 1, 2)
    ref_nodes, ref_weights = np.polynomial.legendre.leggauss(q_order)

    n_basis = len(op.basis._splines)
    gram = np.zeros((n_basis, n_basis), dtype=np.float64)
    evaluators = [
        spline.derivative(deriv_order) if deriv_order else spline
        for spline in op.basis._splines
    ]

    for left, right in zip(knots[:-1], knots[1:]):
        if right <= left:
            continue

        half = 0.5 * (right - left)
        mid = 0.5 * (right + left)
        x_nodes = mid + half * ref_nodes
        weights = half * ref_weights

        values = np.empty((n_basis, q_order), dtype=np.float64)
        for idx, evaluator in enumerate(evaluators):
            values[idx, :] = evaluator(x_nodes)

        gram += (values * weights) @ values.T

    return gram


def _quadrature_rule(op: BSPF1D) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return knot-span quadrature nodes, weights, and basis values for one 1D operator."""
    knots = np.asarray(op.basis.knots, dtype=np.float64)
    q_order = max(op.degree + 1, 2)
    ref_nodes, ref_weights = np.polynomial.legendre.leggauss(q_order)

    rule: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for left, right in zip(knots[:-1], knots[1:]):
        if right <= left:
            continue

        half = 0.5 * (right - left)
        mid = 0.5 * (right + left)
        nodes = mid + half * ref_nodes
        weights = half * ref_weights
        values = np.asarray(op.basis._evaluate_splines_vectorized(nodes, deriv_order=0), dtype=np.float64)
        rule.append((nodes, weights, values))
    return rule


def _integration_weights(op: BSPF1D) -> np.ndarray:
    """Return the 1D BSPF integration weights for sampled data on the operator grid."""
    weights = np.empty(op.grid.n, dtype=np.float64)
    for idx in range(op.grid.n):
        basis_vector = np.zeros(op.grid.n, dtype=np.float64)
        basis_vector[idx] = 1.0
        weights[idx] = op.definite_integral(basis_vector, lam=0.0)
    return weights


def _weighted_gram(mat_a: np.ndarray, mat_b: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return ``A^T W B`` for sampled basis matrices and 1D quadrature weights."""
    return mat_a.T @ (weights[:, None] * mat_b)


def _sample_rhs_on_grid(
    rhs: RHSInput,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Return ``rhs`` sampled on the tensor grid with shape ``(len(y), len(x))``."""
    if callable(rhs):
        xx, yy = np.meshgrid(x, y)
        values = np.asarray(rhs(xx, yy), dtype=np.float64)
        try:
            return np.broadcast_to(values, xx.shape).copy()
        except ValueError as exc:
            raise ValueError("callable rhs must return values broadcastable to the grid shape.") from exc

    values = np.asarray(rhs, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("rhs must be a 2D array with shape (len(y), len(x)).")
    if values.shape != (y.size, x.size):
        raise ValueError(f"rhs shape {values.shape} must match (len(y), len(x))=({y.size}, {x.size}).")
    return values


def _uniform_spacing(grid: np.ndarray, *, name: str) -> float:
    """Return the uniform spacing of ``grid`` or raise if the grid is not uniform."""
    if grid.size < 2:
        raise ValueError(f"{name} must contain at least two points.")
    spacing = np.diff(grid)
    h = float(spacing[0])
    if h <= 0.0 or not np.allclose(spacing, h, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} must be a strictly increasing uniform grid for the DST solver.")
    return h


def _stable_sinh_ratio(alpha: np.ndarray, frac: np.ndarray) -> np.ndarray:
    """Return ``sinh(alpha * frac) / sinh(alpha)`` stably for ``0 <= frac <= 1``."""
    alpha_arr = np.asarray(alpha, dtype=np.float64)[None, :]
    frac_arr = np.asarray(frac, dtype=np.float64)[:, None]
    exp_tail = np.exp(-alpha_arr * (1.0 - frac_arr))
    numerator = 1.0 - np.exp(-2.0 * alpha_arr * frac_arr)
    denominator = 1.0 - np.exp(-2.0 * alpha_arr)
    return exp_tail * (numerator / denominator)


def _left_layer_profile(s: np.ndarray, power: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a left boundary-layer profile and its second derivative in normalized coordinates."""
    s_arr = np.asarray(s, dtype=np.float64)
    values = s_arr * (1.0 - s_arr) ** power
    second = power * (1.0 - s_arr) ** (power - 2) * ((power + 1.0) * s_arr - 2.0)
    return values, second


def _right_layer_profile(s: np.ndarray, power: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a right boundary-layer profile and its second derivative in normalized coordinates."""
    s_arr = np.asarray(s, dtype=np.float64)
    values = s_arr**power * (1.0 - s_arr)
    second = power * s_arr ** (power - 2) * ((power - 1.0) - (power + 1.0) * s_arr)
    return values, second



def _negative_discrete_laplacian(field: np.ndarray, *, hx: float, hy: float) -> np.ndarray:
    """Return the 5-point ``-Delta_h`` of ``field`` on the interior grid."""
    center = field[1:-1, 1:-1]
    return (
        (2.0 * center - field[1:-1, :-2] - field[1:-1, 2:]) / (hx * hx)
        + (2.0 * center - field[:-2, 1:-1] - field[2:, 1:-1]) / (hy * hy)
    )


def _solve_zero_dirichlet_poisson_dst(rhs: np.ndarray, *, hx: float, hy: float) -> np.ndarray:
    """Solve ``Delta u = rhs`` with zero Dirichlet data by a 2D DST-I spectral method."""
    ny_int, nx_int = rhs.shape
    if nx_int == 0 or ny_int == 0:
        return np.zeros_like(rhs)

    lx = hx * (nx_int + 1)
    ly = hy * (ny_int + 1)

    kx = np.arange(1, nx_int + 1, dtype=np.float64)
    ky = np.arange(1, ny_int + 1, dtype=np.float64)
    eig_x = (np.pi * kx / lx) ** 2
    eig_y = (np.pi * ky / ly) ** 2
    denom = eig_y[:, None] + eig_x[None, :]

    # Eigenvalues of Delta are -(eig_x + eig_y), so u = -rhs_hat / denom
    rhs_hat = dstn(rhs, type=1, norm="ortho")
    sol_hat = -rhs_hat / denom
    return idstn(sol_hat, type=1, norm="ortho")


def _solve_periodic_poisson_fft(
    rhs: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    periodic_endpoint: bool = True,
) -> np.ndarray:
    """Solve ``Delta u = rhs`` by periodic FFT inversion with zero-mean gauge."""
    rhs_grid = np.asarray(rhs, dtype=np.float64)
    if rhs_grid.shape != (y.size, x.size):
        raise ValueError(f"rhs shape {rhs_grid.shape} must match (len(y), len(x))=({y.size}, {x.size}).")

    if periodic_endpoint:
        if x.size < 2 or y.size < 2:
            raise ValueError("Periodic FFT solve requires at least two points per axis.")
        rhs_core = rhs_grid[:-1, :-1]
        nx = x.size - 1
        ny = y.size - 1
        lx = float(x[-1] - x[0])
        ly = float(y[-1] - y[0])
    else:
        rhs_core = rhs_grid
        nx = x.size
        ny = y.size
        lx = float(x[-1] - x[0] + _uniform_spacing(x, name="x"))
        ly = float(y[-1] - y[0] + _uniform_spacing(y, name="y"))

    rhs_core = rhs_core - np.mean(rhs_core)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    denom = ky[:, None] ** 2 + kx[None, :] ** 2

    rhs_hat = np.fft.fft2(rhs_core)
    sol_hat = np.zeros_like(rhs_hat, dtype=np.complex128)
    mask = denom > 0.0
    sol_hat[mask] = -rhs_hat[mask] / denom[mask]
    sol_core = np.fft.ifft2(sol_hat).real

    if not periodic_endpoint:
        return sol_core

    sol_full = np.empty_like(rhs_grid)
    sol_full[:-1, :-1] = sol_core
    sol_full[:-1, -1] = sol_core[:, 0]
    sol_full[-1, :-1] = sol_core[0, :]
    sol_full[-1, -1] = sol_core[0, 0]
    return sol_full


def _normalize_trace(
    data: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
    grid: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Normalize one edge trace to samples on the requested grid."""
    if callable(data):
        values = np.asarray(data(grid), dtype=np.float64)
    elif np.isscalar(data):
        values = np.full(grid.shape, float(data), dtype=np.float64)
    else:
        values = np.asarray(data, dtype=np.float64)

    if values.shape != grid.shape:
        raise ValueError(f"{name} boundary data must have shape {grid.shape}.")
    return values


def _validate_dirichlet_corners(
    left_values: np.ndarray,
    right_values: np.ndarray,
    bottom_values: np.ndarray,
    top_values: np.ndarray,
) -> None:
    """Validate that the four Dirichlet traces agree at the rectangle corners."""
    corners_from_vertical = np.array(
        [[left_values[0], left_values[-1]], [right_values[0], right_values[-1]]],
        dtype=np.float64,
    )
    corners_from_horizontal = np.array(
        [[bottom_values[0], top_values[0]], [bottom_values[-1], top_values[-1]]],
        dtype=np.float64,
    )
    if not np.allclose(corners_from_vertical, corners_from_horizontal, rtol=0.0, atol=1.0e-10):
        raise ValueError("Dirichlet boundary traces must agree at the four corners.")


def _fit_trace_coefficients(op: BSPF1D, values: np.ndarray) -> np.ndarray:
    """Project one sampled boundary trace into the 1D spline basis with exact endpoints."""
    constraint = _boundary_value_constraint(op)
    target = np.array([values[0], values[-1]], dtype=np.float64)
    particular = _constraint_right_inverse(constraint) @ target
    null_basis = sla.null_space(constraint)
    if null_basis.size == 0:
        return particular

    sample_matrix = np.asarray(op.basis.BT0, dtype=np.float64)
    residual = values - sample_matrix @ particular
    reduced_design = sample_matrix @ null_basis
    reduced_coeffs, *_ = np.linalg.lstsq(reduced_design, residual, rcond=None)
    return particular + null_basis @ reduced_coeffs


def _fit_trace_least_squares(op: BSPF1D, values: np.ndarray) -> np.ndarray:
    """Project one sampled trace into the 1D spline basis without endpoint constraints."""
    sample_matrix = np.asarray(op.basis.BT0, dtype=np.float64)
    coeffs, *_ = np.linalg.lstsq(sample_matrix, np.asarray(values, dtype=np.float64), rcond=None)
    return coeffs


@dataclass
class Poisson2DDirichletSolver:
    """Direct solver for ``Delta u = f`` on a tensor grid.

    The solver currently targets a rectangular domain with constant-coefficient
    Poisson and homogeneous Dirichlet boundary conditions. A scalar constant
    Dirichlet value can be supplied at solve time through a trivial harmonic
    lift because ``Delta(constant) = 0``.
    """

    x: np.ndarray
    y: np.ndarray
    x_model: BSPF1D
    y_model: BSPF1D
    tx: np.ndarray
    ty: np.ndarray
    eigvals_x: np.ndarray
    eigvals_y: np.ndarray
    eigvecs_x: np.ndarray
    eigvecs_y: np.ndarray

    def __post_init__(self):
        if self.x_model.use_gpu or self.y_model.use_gpu:
            raise ValueError("Poisson2DDirichletSolver currently supports CPU-backed BSPF models only.")

        self._mass_x = _mass_matrix(self.x_model)
        self._mass_y = _mass_matrix(self.y_model)
        self._stiff_x = _stiffness_matrix(self.x_model)
        self._stiff_y = _stiffness_matrix(self.y_model)
        self._basis_x = np.asarray(self.x_model.basis.B0, dtype=np.float64)
        self._basis_y = np.asarray(self.y_model.basis.B0, dtype=np.float64)
        self._basis_xt = np.asarray(self.x_model.basis.BT0, dtype=np.float64)
        self._basis_yt = np.asarray(self.y_model.basis.BT0, dtype=np.float64)
        self._basis_x1t = np.asarray(self.x_model.basis.BkT(1), dtype=np.float64)
        self._basis_x2t = np.asarray(self.x_model.basis.BkT(2), dtype=np.float64)
        self._basis_x3t = np.asarray(self.x_model.basis.BkT(3), dtype=np.float64)
        self._basis_y1t = np.asarray(self.y_model.basis.BkT(1), dtype=np.float64)
        self._basis_y2t = np.asarray(self.y_model.basis.BkT(2), dtype=np.float64)
        self._basis_y3t = np.asarray(self.y_model.basis.BkT(3), dtype=np.float64)
        self._weights_x = np.asarray(self.x_model.grid.trap, dtype=np.float64)
        self._weights_y = np.asarray(self.y_model.grid.trap, dtype=np.float64)
        self._constraint_x = _boundary_value_constraint(self.x_model)
        self._constraint_y = _boundary_value_constraint(self.y_model)
        self._constraint_x_right_inverse = _constraint_right_inverse(self._constraint_x)
        self._constraint_y_right_inverse = _constraint_right_inverse(self._constraint_y)
        self._quadrature_x = _quadrature_rule(self.x_model)
        self._quadrature_y = _quadrature_rule(self.y_model)
        self._integration_weights_x = _integration_weights(self.x_model)
        self._integration_weights_y = _integration_weights(self.y_model)
        self._moment_projector_x = self._basis_x * self._integration_weights_x
        self._moment_projector_y = self._basis_y * self._integration_weights_y
        self._layer_basis_cache: dict[tuple[int, int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def from_grids(
        cls,
        *,
        x: np.ndarray,
        y: np.ndarray,
        degree_x: int = 10,
        degree_y: Optional[int] = None,
        knots_x: Optional[np.ndarray] = None,
        knots_y: Optional[np.ndarray] = None,
        n_basis_x: Optional[int] = None,
        n_basis_y: Optional[int] = None,
        domain_x: Optional[tuple[float, float]] = None,
        domain_y: Optional[tuple[float, float]] = None,
        use_clustering_x: bool = False,
        use_clustering_y: bool = False,
        clustering_factor_x: float = 2.0,
        clustering_factor_y: float = 2.0,
    ) -> "Poisson2DDirichletSolver":
        """Construct the direct solver from orthogonal 1D grids."""
        if degree_y is None:
            degree_y = degree_x

        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        x_model = BSPF1D.from_grid(
            degree=degree_x,
            x=x_arr,
            knots=knots_x,
            n_basis=n_basis_x,
            domain=domain_x,
            use_clustering=use_clustering_x,
            clustering_factor=clustering_factor_x,
            use_gpu=False,
        )
        y_model = BSPF1D.from_grid(
            degree=degree_y,
            x=y_arr,
            knots=knots_y,
            n_basis=n_basis_y,
            domain=domain_y,
            use_clustering=use_clustering_y,
            clustering_factor=clustering_factor_y,
            use_gpu=False,
        )

        tx = _dirichlet_nullspace(x_model)
        ty = _dirichlet_nullspace(y_model)

        mass_x_reduced = tx.T @ _mass_matrix(x_model) @ tx
        mass_y_reduced = ty.T @ _mass_matrix(y_model) @ ty
        stiff_x_reduced = tx.T @ _stiffness_matrix(x_model) @ tx
        stiff_y_reduced = ty.T @ _stiffness_matrix(y_model) @ ty

        eigvals_x, eigvecs_x = sla.eigh(stiff_x_reduced, mass_x_reduced)
        eigvals_y, eigvecs_y = sla.eigh(stiff_y_reduced, mass_y_reduced)

        return cls(
            x=x_arr,
            y=y_arr,
            x_model=x_model,
            y_model=y_model,
            tx=tx,
            ty=ty,
            eigvals_x=np.asarray(eigvals_x, dtype=np.float64),
            eigvals_y=np.asarray(eigvals_y, dtype=np.float64),
            eigvecs_x=np.asarray(eigvecs_x, dtype=np.float64),
            eigvecs_y=np.asarray(eigvecs_y, dtype=np.float64),
        )

    def _load_matrix(self, rhs: RHSInput) -> np.ndarray:
        """Assemble the weak-form load matrix from sampled values or a callable RHS."""
        if callable(rhs):
            load = np.zeros((self._basis_x.shape[0], self._basis_y.shape[0]), dtype=np.float64)
            for x_nodes, x_weights, basis_x in self._quadrature_x:
                for y_nodes, y_weights, basis_y in self._quadrature_y:
                    xx, yy = np.meshgrid(x_nodes, y_nodes, indexing="ij")
                    values = np.asarray(rhs(xx, yy), dtype=np.float64)
                    try:
                        values = np.broadcast_to(values, xx.shape)
                    except ValueError as exc:
                        raise ValueError(
                            "callable rhs must return values broadcastable to the quadrature grid shape."
                        ) from exc
                    load += (basis_x * x_weights) @ values @ (basis_y * y_weights).T
            return load

        rhs_xy = np.asarray(rhs, dtype=np.float64).T
        if rhs_xy.shape != (self.x.size, self.y.size):
            raise ValueError(
                f"rhs shape {np.asarray(rhs).shape} must match (len(y), len(x))="
                f"({self.y.size}, {self.x.size})."
            )
        return self._moment_projector_x @ rhs_xy @ self._moment_projector_y.T

    def _build_boundary_coefficients(
        self,
        *,
        left: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        right: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        bottom: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        top: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Construct a coefficient-space lift that matches the prescribed Dirichlet traces."""
        left_values = _normalize_trace(left, self.y, name="left")
        right_values = _normalize_trace(right, self.y, name="right")
        bottom_values = _normalize_trace(bottom, self.x, name="bottom")
        top_values = _normalize_trace(top, self.x, name="top")
        _validate_dirichlet_corners(left_values, right_values, bottom_values, top_values)

        left_coeffs = _fit_trace_coefficients(self.y_model, left_values)
        right_coeffs = _fit_trace_coefficients(self.y_model, right_values)
        bottom_coeffs = _fit_trace_coefficients(self.x_model, bottom_values)
        top_coeffs = _fit_trace_coefficients(self.x_model, top_values)

        vertical = self._constraint_x_right_inverse @ np.vstack((left_coeffs, right_coeffs))
        horizontal = np.column_stack((bottom_coeffs, top_coeffs)) @ self._constraint_y_right_inverse.T
        corner_values = self._constraint_x @ np.column_stack((bottom_coeffs, top_coeffs))
        return vertical + horizontal - self._constraint_x_right_inverse @ corner_values @ self._constraint_y_right_inverse.T

    def build_dirichlet_lift(
        self,
        *,
        left: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        right: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        bottom: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        top: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Return the sampled spline lift that matches the requested boundary traces."""
        coefficients = self._build_boundary_coefficients(left=left, right=right, bottom=bottom, top=top)
        return (self._basis_xt @ coefficients @ self._basis_yt.T).T

    def _evaluate_solution(self, coefficients: np.ndarray) -> np.ndarray:
        """Evaluate spline coefficients on the tensor grid."""
        return (self._basis_xt @ coefficients @ self._basis_yt.T).T

    def _evaluate_laplacian(self, coefficients: np.ndarray) -> np.ndarray:
        """Evaluate the analytic spline Laplacian on the tensor grid."""
        lap_xy = self._basis_x2t @ coefficients @ self._basis_yt.T
        lap_xy += self._basis_xt @ coefficients @ self._basis_y2t.T
        return lap_xy.T

    def _build_corrector_targets_02(
        self,
        rhs_grid: np.ndarray,
        *,
        left: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        right: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        bottom: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        top: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return sampled value and second-normal-derivative targets on the four edges."""
        left_values = _normalize_trace(left, self.y, name="left")
        right_values = _normalize_trace(right, self.y, name="right")
        bottom_values = _normalize_trace(bottom, self.x, name="bottom")
        top_values = _normalize_trace(top, self.x, name="top")
        _validate_dirichlet_corners(left_values, right_values, bottom_values, top_values)

        left_tangent_yy  = self.y_model.derivatives(left_values,   orders=2)[2]
        right_tangent_yy = self.y_model.derivatives(right_values,  orders=2)[2]
        bottom_tangent_xx = self.x_model.derivatives(bottom_values, orders=2)[2]
        top_tangent_xx   = self.x_model.derivatives(top_values,    orders=2)[2]

        # u_xx + u_yy = f  →  u_xx = f - u_yy  (second normal derivative target)
        left_second   = rhs_grid[:, 0]  - left_tangent_yy
        right_second  = rhs_grid[:, -1] - right_tangent_yy
        bottom_second = rhs_grid[0, :]  - bottom_tangent_xx
        top_second    = rhs_grid[-1, :] - top_tangent_xx

        return (
            left_values,
            right_values,
            bottom_values,
            top_values,
            left_second,
            right_second,
            bottom_second,
            top_second,
        )

    def build_harmonic_extension(
        self,
        *,
        left: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        right: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        bottom: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        top: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build a rectangle-spectral harmonic lift that matches the boundary exactly."""
        left_values = _normalize_trace(left, self.y, name="left")
        right_values = _normalize_trace(right, self.y, name="right")
        bottom_values = _normalize_trace(bottom, self.x, name="bottom")
        top_values = _normalize_trace(top, self.x, name="top")
        _validate_dirichlet_corners(left_values, right_values, bottom_values, top_values)

        xx, yy = np.meshgrid(self.x, self.y)
        x0 = float(self.x[0])
        x1 = float(self.x[-1])
        y0 = float(self.y[0])
        y1 = float(self.y[-1])
        lx = x1 - x0
        ly = y1 - y0
        sx = (xx - x0) / lx
        ty = (yy - y0) / ly
        corner_bl = float(left_values[0])
        corner_br = float(right_values[0])
        corner_tl = float(left_values[-1])
        corner_tr = float(right_values[-1])
        corner_patch = (
            (1.0 - sx) * (1.0 - ty) * corner_bl
            + sx * (1.0 - ty) * corner_br
            + (1.0 - sx) * ty * corner_tl
            + sx * ty * corner_tr
        )

        left_residual = left_values - corner_patch[:, 0]
        right_residual = right_values - corner_patch[:, -1]
        bottom_residual = bottom_values - corner_patch[0, :]
        top_residual = top_values - corner_patch[-1, :]

        harmonic_sides = np.zeros_like(corner_patch)
        nx_int = self.x.size - 2
        ny_int = self.y.size - 2

        if nx_int > 0:
            x_modes = np.arange(1, nx_int + 1, dtype=np.float64)
            x_decay = np.pi * x_modes / lx
            x_alpha = x_decay * ly
            y_int = self.y[1:-1] - y0

            bottom_coeffs = dst(bottom_residual[1:-1], type=1, norm="ortho")
            bottom_decay = _stable_sinh_ratio(x_alpha, (ly - y_int) / ly)
            harmonic_sides[1:-1, 1:-1] += idst(bottom_decay * bottom_coeffs[None, :], type=1, norm="ortho", axis=1)

            top_coeffs = dst(top_residual[1:-1], type=1, norm="ortho")
            top_decay = _stable_sinh_ratio(x_alpha, y_int / ly)
            harmonic_sides[1:-1, 1:-1] += idst(top_decay * top_coeffs[None, :], type=1, norm="ortho", axis=1)

        if ny_int > 0:
            y_modes = np.arange(1, ny_int + 1, dtype=np.float64)
            y_decay = np.pi * y_modes / ly
            y_alpha = y_decay * lx
            x_int = self.x[1:-1] - x0

            left_coeffs = dst(left_residual[1:-1], type=1, norm="ortho")
            left_decay = _stable_sinh_ratio(y_alpha, (lx - x_int) / lx)
            harmonic_sides[1:-1, 1:-1] += idst(left_decay * left_coeffs[None, :], type=1, norm="ortho", axis=1).T

            right_coeffs = dst(right_residual[1:-1], type=1, norm="ortho")
            right_decay = _stable_sinh_ratio(y_alpha, x_int / lx)
            harmonic_sides[1:-1, 1:-1] += idst(right_decay * right_coeffs[None, :], type=1, norm="ortho", axis=1).T

        harmonic_lift = corner_patch + harmonic_sides
        harmonic_lift[:, 0] = left_values
        harmonic_lift[:, -1] = right_values
        harmonic_lift[0, :] = bottom_values
        harmonic_lift[-1, :] = top_values

        return harmonic_lift, corner_patch, harmonic_sides, np.zeros_like(harmonic_lift)

    def build_zero_boundary_strip_correction(
        self,
        rhs_grid: np.ndarray,
        *,
        n_strip: int,
        strip_weight: float = 1.0,
        smoothness_weight: float = 1.0,
        ridge: float = 1.0e-10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a zero-trace boundary-layer correction from POD-compressed strip responses."""
        rhs_arr = np.asarray(rhs_grid, dtype=np.float64)
        if rhs_arr.shape != (self.y.size, self.x.size):
            raise ValueError(
                f"rhs_grid shape {rhs_arr.shape} must match (len(y), len(x))=({self.y.size}, {self.x.size})."
            )
        if n_strip <= 0:
            return np.zeros_like(rhs_arr), np.zeros_like(rhs_arr)

        nx = self.x.size
        ny = self.y.size
        strip_defs = [
            (slice(None), slice(1, n_strip + 1)),
            (slice(None), slice(nx - 1 - n_strip, nx - 1)),
            (slice(1, n_strip + 1), slice(None)),
            (slice(ny - 1 - n_strip, ny - 1), slice(None)),
        ]
        basis_solution, basis_forcing = self._get_operator_adapted_layer_basis(
            n_strip=n_strip,
            max_tangent_modes=8,
            max_normal_samples=4,
            max_rank=16,
            energy_tol=0.999,
        )

        strip_index_blocks = [np.ravel_multi_index(np.indices(rhs_arr[sy, sx_sel].shape), rhs_arr[sy, sx_sel].shape) for sy, sx_sel in []]
        strip_indices = []
        for sy, sx_sel in strip_defs:
            yy_idx, xx_idx = np.meshgrid(np.arange(ny)[sy], np.arange(nx)[sx_sel], indexing="ij")
            strip_indices.append((yy_idx * nx + xx_idx).reshape(-1))
        strip_indices_flat = np.concatenate(strip_indices)

        A = basis_forcing[strip_indices_flat, :]
        b = rhs_arr.reshape(-1)[strip_indices_flat]
        gram = strip_weight * (A.T @ A) + (smoothness_weight + ridge) * np.eye(A.shape[1], dtype=np.float64)
        rhs_vec = strip_weight * (A.T @ b)
        coeffs = np.linalg.solve(gram, rhs_vec)

        correction = (basis_solution @ coeffs).reshape(rhs_arr.shape)
        laplacian = (basis_forcing @ coeffs).reshape(rhs_arr.shape)
        return correction, laplacian

    def build_pod_layer_basis_from_02(
        self,
        training_samples: list[dict[str, object]],
        *,
        rank: Optional[int] = None,
        energy_tol: float = 0.999,
        second_normal_weight: float = 100.0,
        laplacian_weight: float = 1.0e-6,
        ridge: float = 1.0e-10,
        n_strip: int = 0,
        strip_weight: float = 1.0,
    ) -> PODLayerBasis2D:
        """Build a POD layer basis from full `02` teacher corrector snapshots.

        Each training sample is a dict with keys:
        `rhs`, `left`, `right`, `bottom`, `top`.
        The returned basis is zero-trace by construction.
        """
        if not training_samples:
            raise ValueError("training_samples must contain at least one sample.")
        if rank is not None and rank < 1:
            raise ValueError("rank must be positive when provided.")
        if not (0.0 < energy_tol <= 1.0):
            raise ValueError("energy_tol must lie in (0, 1].")

        layer_snapshots = []
        laplacian_snapshots = []
        for idx, sample in enumerate(training_samples):
            missing = {"rhs", "left", "right", "bottom", "top"} - set(sample.keys())
            if missing:
                raise ValueError(f"training_samples[{idx}] is missing keys: {sorted(missing)}")

            left_values = _normalize_trace(sample["left"], self.y, name=f"left[{idx}]")
            right_values = _normalize_trace(sample["right"], self.y, name=f"right[{idx}]")
            bottom_values = _normalize_trace(sample["bottom"], self.x, name=f"bottom[{idx}]")
            top_values = _normalize_trace(sample["top"], self.x, name=f"top[{idx}]")

            teacher_corrector, teacher_laplacian, _ = self.build_boundary_corrector_02(
                sample["rhs"],
                left=left_values,
                right=right_values,
                bottom=bottom_values,
                top=top_values,
                second_normal_weight=second_normal_weight,
                laplacian_weight=laplacian_weight,
                ridge=ridge,
                n_strip=n_strip,
                strip_weight=strip_weight,
            )
            harmonic_lift, _, _, _ = self.build_harmonic_extension(
                left=left_values,
                right=right_values,
                bottom=bottom_values,
                top=top_values,
            )
            raw_layer = teacher_corrector - harmonic_lift
            residual_trace_harmonic, _, _, _ = self.build_harmonic_extension(
                left=raw_layer[:, 0],
                right=raw_layer[:, -1],
                bottom=raw_layer[0, :],
                top=raw_layer[-1, :],
            )
            zero_trace_layer = raw_layer - residual_trace_harmonic
            zero_trace_layer[:, 0] = 0.0
            zero_trace_layer[:, -1] = 0.0
            zero_trace_layer[0, :] = 0.0
            zero_trace_layer[-1, :] = 0.0

            layer_snapshots.append(zero_trace_layer.reshape(-1))
            laplacian_snapshots.append(teacher_laplacian.reshape(-1))

        snapshot_matrix = np.column_stack(layer_snapshots)
        laplacian_matrix = np.column_stack(laplacian_snapshots)
        u, singular_values, vt = np.linalg.svd(snapshot_matrix, full_matrices=False)
        if singular_values.size == 0 or singular_values[0] <= 0.0:
            raise ValueError("Teacher snapshots are degenerate; cannot build POD layer basis.")

        if rank is None:
            cumulative_energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
            rank_eff = int(np.searchsorted(cumulative_energy, energy_tol) + 1)
        else:
            rank_eff = int(rank)
        rank_eff = min(rank_eff, singular_values.size)

        basis_solution = u[:, :rank_eff]
        combo = vt[:rank_eff, :].T / singular_values[:rank_eff][None, :]
        basis_laplacian = laplacian_matrix @ combo
        return PODLayerBasis2D(
            solution_basis=basis_solution,
            laplacian_basis=basis_laplacian,
            singular_values=singular_values[:rank_eff].copy(),
            n_snapshots=len(training_samples),
        )

    def solve_harmonic_pod_02(
        self,
        rhs: RHSInput,
        *,
        pod_basis: PODLayerBasis2D,
        left: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        right: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        bottom: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        top: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        n_strip: int = 0,
        strip_weight: float = 1.0,
        ridge: float = 1.0e-10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Solve with harmonic trace lift + POD teacher-layer + zero-boundary DST bulk."""
        rhs_grid = _sample_rhs_on_grid(rhs, self.x, self.y)
        harmonic_lift, _, _, _ = self.build_harmonic_extension(
            left=left,
            right=right,
            bottom=bottom,
            top=top,
        )

        layer = np.zeros_like(rhs_grid)
        layer_laplacian = np.zeros_like(rhs_grid)
        if pod_basis.rank > 0 and n_strip > 0:
            nx = self.x.size
            ny = self.y.size
            strip_defs = [
                (slice(None), slice(1, n_strip + 1)),
                (slice(None), slice(nx - 1 - n_strip, nx - 1)),
                (slice(1, n_strip + 1), slice(None)),
                (slice(ny - 1 - n_strip, ny - 1), slice(None)),
            ]
            strip_indices = []
            for sy, sx_sel in strip_defs:
                y_idx = np.arange(ny)[sy]
                x_idx = np.arange(nx)[sx_sel]
                yy_idx, xx_idx = np.meshgrid(y_idx, x_idx, indexing="ij")
                strip_indices.append((yy_idx * nx + xx_idx).reshape(-1))
            strip_indices_flat = np.concatenate(strip_indices)

            a = pod_basis.laplacian_basis[strip_indices_flat, :]
            b = rhs_grid.reshape(-1)[strip_indices_flat]
            gram = strip_weight * (a.T @ a) + ridge * np.eye(pod_basis.rank, dtype=np.float64)
            rhs_vec = strip_weight * (a.T @ b)
            coeffs = np.linalg.solve(gram, rhs_vec)
            layer = (pod_basis.solution_basis @ coeffs).reshape(rhs_grid.shape)
            layer_laplacian = (pod_basis.laplacian_basis @ coeffs).reshape(rhs_grid.shape)
            layer[:, 0] = 0.0
            layer[:, -1] = 0.0
            layer[0, :] = 0.0
            layer[-1, :] = 0.0

        hx = _uniform_spacing(self.x, name="x")
        hy = _uniform_spacing(self.y, name="y")
        remainder = np.zeros_like(rhs_grid)
        corrected_rhs = rhs_grid - layer_laplacian
        if self.x.size > 2 and self.y.size > 2:
            remainder[1:-1, 1:-1] = _solve_zero_dirichlet_poisson_dst(
                corrected_rhs[1:-1, 1:-1],
                hx=hx,
                hy=hy,
            )

        lift = harmonic_lift + layer
        solution = lift + remainder
        return solution, lift, remainder, harmonic_lift, layer

    def _get_operator_adapted_layer_basis(
        self,
        *,
        n_strip: int,
        max_tangent_modes: int,
        max_normal_samples: int,
        max_rank: int,
        energy_tol: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return POD-compressed zero-boundary strip-response bases for solution and forcing."""
        cache_key = (n_strip, max_tangent_modes, max_normal_samples, max_rank, int(round(energy_tol * 1000.0)))
        if cache_key in self._layer_basis_cache:
            return self._layer_basis_cache[cache_key]

        nx = self.x.size
        ny = self.y.size
        hx = _uniform_spacing(self.x, name="x")
        hy = _uniform_spacing(self.y, name="y")
        x0 = float(self.x[0])
        x1 = float(self.x[-1])
        y0 = float(self.y[0])
        y1 = float(self.y[-1])
        lx = x1 - x0
        ly = y1 - y0
        sx = (self.x - x0) / lx
        ty = (self.y - y0) / ly

        tangent_modes_x = np.arange(1, max(1, min(max_tangent_modes, nx - 2)) + 1, dtype=np.float64)
        tangent_modes_y = np.arange(1, max(1, min(max_tangent_modes, ny - 2)) + 1, dtype=np.float64)
        strip_samples = np.unique(np.round(np.linspace(1, n_strip, min(n_strip, max_normal_samples))).astype(int))

        forcing_snapshots: list[np.ndarray] = []
        solution_snapshots: list[np.ndarray] = []

        tangential_y = np.sin(np.pi * np.outer(ty, tangent_modes_y))
        tangential_x = np.sin(np.pi * np.outer(sx, tangent_modes_x))

        for offset in strip_samples:
            left_col = offset
            right_col = nx - 1 - offset
            for mode_idx in range(tangent_modes_y.size):
                q_left = np.zeros((ny, nx), dtype=np.float64)
                q_left[:, left_col] = tangential_y[:, mode_idx]
                q_right = np.zeros((ny, nx), dtype=np.float64)
                q_right[:, right_col] = tangential_y[:, mode_idx]
                for q in (q_left, q_right):
                    scale = np.linalg.norm(q)
                    if scale == 0.0:
                        continue
                    q /= scale
                    sol = np.zeros_like(q)
                    sol[1:-1, 1:-1] = _solve_zero_dirichlet_poisson_dst(q[1:-1, 1:-1], hx=hx, hy=hy)
                    forcing_snapshots.append(q.reshape(-1))
                    solution_snapshots.append(sol.reshape(-1))

        for offset in strip_samples:
            bottom_row = offset
            top_row = ny - 1 - offset
            for mode_idx in range(tangent_modes_x.size):
                q_bottom = np.zeros((ny, nx), dtype=np.float64)
                q_bottom[bottom_row, :] = tangential_x[:, mode_idx]
                q_top = np.zeros((ny, nx), dtype=np.float64)
                q_top[top_row, :] = tangential_x[:, mode_idx]
                for q in (q_bottom, q_top):
                    scale = np.linalg.norm(q)
                    if scale == 0.0:
                        continue
                    q /= scale
                    sol = np.zeros_like(q)
                    sol[1:-1, 1:-1] = _solve_zero_dirichlet_poisson_dst(q[1:-1, 1:-1], hx=hx, hy=hy)
                    forcing_snapshots.append(q.reshape(-1))
                    solution_snapshots.append(sol.reshape(-1))

        W = np.column_stack(solution_snapshots)
        Q = np.column_stack(forcing_snapshots)
        gram = 0.5 * (W.T @ W + (W.T @ W).T)
        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        positive = eigvals > max(1.0e-14 * eigvals[0], 0.0)
        eigvals = eigvals[positive]
        eigvecs = eigvecs[:, positive]
        if eigvals.size == 0:
            raise ValueError("Failed to build an operator-adapted strip basis: no positive snapshot energy.")

        cumulative_energy = np.cumsum(eigvals) / np.sum(eigvals)
        rank = min(max_rank, int(np.searchsorted(cumulative_energy, energy_tol) + 1))
        singular_values = np.sqrt(eigvals[:rank])
        combination = eigvecs[:, :rank] / singular_values[None, :]
        basis_solution = W @ combination
        basis_forcing = Q @ combination
        self._layer_basis_cache[cache_key] = (basis_solution, basis_forcing)
        return basis_solution, basis_forcing

    def build_boundary_corrector_02(
        self,
        rhs: RHSInput,
        *,
        left: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        right: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        bottom: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        top: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        second_normal_weight: float = 100.0,
        laplacian_weight: float = 1.0e-6,
        ridge: float = 1.0e-10,
        n_strip: int = 0,
        strip_weight: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a spline boundary corrector using value and second-normal-derivative edge jets.

        Boundary values are enforced as hard constraints. Second-normal edge
        data is matched by a soft least-squares penalty so the lifting surface
        can relax high-frequency edge jets instead of oscillating to satisfy
        them exactly. If ``n_strip > 0``, an additional soft penalty encourages
        the analytic Laplacian of the corrector to match ``rhs`` over the first
        ``n_strip`` interior rows/columns adjacent to each boundary edge.
        """
        rhs_grid = _sample_rhs_on_grid(rhs, self.x, self.y)
        (
            left_values,
            right_values,
            bottom_values,
            top_values,
            left_second,
            right_second,
            bottom_second,
            top_second,
        ) = self._build_corrector_targets_02(
            rhs_grid,
            left=left,
            right=right,
            bottom=bottom,
            top=top,
        )

        left_coeffs = _fit_trace_coefficients(self.y_model, left_values)
        right_coeffs = _fit_trace_coefficients(self.y_model, right_values)
        bottom_coeffs = _fit_trace_coefficients(self.x_model, bottom_values)
        top_coeffs = _fit_trace_coefficients(self.x_model, top_values)

        left_second_coeffs = _fit_trace_least_squares(self.y_model, left_second)
        right_second_coeffs = _fit_trace_least_squares(self.y_model, right_second)
        bottom_second_coeffs = _fit_trace_least_squares(self.x_model, bottom_second)
        top_second_coeffs = _fit_trace_least_squares(self.x_model, top_second)

        nbx = self._basis_x.shape[0]
        nby = self._basis_y.shape[0]
        eye_x = np.eye(nbx, dtype=np.float64)
        eye_y = np.eye(nby, dtype=np.float64)

        value_x = self._constraint_x
        value_y = self._constraint_y
        second_x = np.vstack((self._basis_x2t[0, :], self._basis_x2t[-1, :]))
        second_y = np.vstack((self._basis_y2t[0, :], self._basis_y2t[-1, :]))

        constraints = [
            np.kron(eye_y, value_x[[0], :]),
            np.kron(eye_y, value_x[[1], :]),
            np.kron(value_y[[0], :], eye_x),
            np.kron(value_y[[1], :], eye_x),
        ]
        targets = [
            left_coeffs,
            right_coeffs,
            bottom_coeffs,
            top_coeffs,
        ]

        C = np.vstack(constraints)
        d = np.concatenate(targets)

        gram_x = {
            (0, 0): _weighted_gram(self._basis_xt, self._basis_xt, self._weights_x),
            (0, 1): _weighted_gram(self._basis_xt, self._basis_x1t, self._weights_x),
            (0, 2): _weighted_gram(self._basis_xt, self._basis_x2t, self._weights_x),
            (0, 3): _weighted_gram(self._basis_xt, self._basis_x3t, self._weights_x),
            (1, 1): _weighted_gram(self._basis_x1t, self._basis_x1t, self._weights_x),
            (1, 3): _weighted_gram(self._basis_x1t, self._basis_x3t, self._weights_x),
            (2, 2): _weighted_gram(self._basis_x2t, self._basis_x2t, self._weights_x),
            (2, 3): _weighted_gram(self._basis_x2t, self._basis_x3t, self._weights_x),
            (3, 3): _weighted_gram(self._basis_x3t, self._basis_x3t, self._weights_x),
        }
        gram_y = {
            (0, 0): _weighted_gram(self._basis_yt, self._basis_yt, self._weights_y),
            (0, 1): _weighted_gram(self._basis_yt, self._basis_y1t, self._weights_y),
            (0, 2): _weighted_gram(self._basis_yt, self._basis_y2t, self._weights_y),
            (0, 3): _weighted_gram(self._basis_yt, self._basis_y3t, self._weights_y),
            (1, 1): _weighted_gram(self._basis_y1t, self._basis_y1t, self._weights_y),
            (1, 3): _weighted_gram(self._basis_y1t, self._basis_y3t, self._weights_y),
            (2, 2): _weighted_gram(self._basis_y2t, self._basis_y2t, self._weights_y),
            (2, 3): _weighted_gram(self._basis_y2t, self._basis_y3t, self._weights_y),
            (3, 3): _weighted_gram(self._basis_y3t, self._basis_y3t, self._weights_y),
        }
        gram_x[(1, 0)] = gram_x[(0, 1)].T
        gram_x[(2, 0)] = gram_x[(0, 2)].T
        gram_x[(3, 0)] = gram_x[(0, 3)].T
        gram_x[(3, 1)] = gram_x[(1, 3)].T
        gram_x[(3, 2)] = gram_x[(2, 3)].T
        gram_y[(1, 0)] = gram_y[(0, 1)].T
        gram_y[(2, 0)] = gram_y[(0, 2)].T
        gram_y[(3, 0)] = gram_y[(0, 3)].T
        gram_y[(3, 1)] = gram_y[(1, 3)].T
        gram_y[(3, 2)] = gram_y[(2, 3)].T

        H_lap = (
            np.kron(gram_y[(0, 0)], gram_x[(2, 2)])
            + np.kron(gram_y[(0, 2)], gram_x[(2, 0)])
            + np.kron(gram_y[(2, 0)], gram_x[(0, 2)])
            + np.kron(gram_y[(2, 2)], gram_x[(0, 0)])
        )
        H_grad = (
            np.kron(gram_y[(0, 0)], gram_x[(3, 3)])
            + np.kron(gram_y[(0, 2)], gram_x[(3, 1)])
            + np.kron(gram_y[(2, 0)], gram_x[(1, 3)])
            + np.kron(gram_y[(2, 2)], gram_x[(1, 1)])
            + np.kron(gram_y[(3, 3)], gram_x[(0, 0)])
            + np.kron(gram_y[(3, 1)], gram_x[(0, 2)])
            + np.kron(gram_y[(1, 3)], gram_x[(2, 0)])
            + np.kron(gram_y[(1, 1)], gram_x[(2, 2)])
        )
        H = H_grad + laplacian_weight * H_lap + ridge * np.eye(nbx * nby, dtype=np.float64)
        H = 0.5 * (H + H.T)
        g = np.zeros(nbx * nby, dtype=np.float64)

        if second_normal_weight > 0.0:
            second_penalties = [
                (np.kron(eye_y, second_x[[0], :]), left_second_coeffs),
                (np.kron(eye_y, second_x[[1], :]), right_second_coeffs),
                (np.kron(second_y[[0], :], eye_x), bottom_second_coeffs),
                (np.kron(second_y[[1], :], eye_x), top_second_coeffs),
            ]
            for A_second, target_second in second_penalties:
                H = H + second_normal_weight * (A_second.T @ A_second)
                g -= second_normal_weight * (A_second.T @ target_second)
            H = 0.5 * (H + H.T)

        # Strip Laplacian penalty: encourage Δw = rhs over interior rows/cols near each edge.
        # A_strip maps the coefficient vector (Fortran order) to Laplacian values in the strip:
        #   A_strip = kron(B_y[strip_y, :], B_x2[strip_x, :]) + kron(B_y2[strip_y, :], B_x[strip_x, :])
        # Penalty: strip_weight * ||A_strip @ c - rhs_strip||^2
        # This adds strip_weight * A_strip.T @ A_strip to H and a linear term g = -strip_weight * A_strip.T @ rhs_strip.
        if n_strip > 0:
            nx = self.x.size
            ny = self.y.size
            strip_defs = [
                (slice(None),                     slice(1, n_strip + 1)),           # left
                (slice(None),                     slice(nx - 1 - n_strip, nx - 1)), # right
                (slice(1, n_strip + 1),           slice(None)),                     # bottom
                (slice(ny - 1 - n_strip, ny - 1), slice(None)),                     # top
            ]
            for sy, sx in strip_defs:
                bx2_s = self._basis_x2t[sx, :]
                bx_s  = self._basis_xt[sx, :]
                by_s  = self._basis_yt[sy, :]
                by2_s = self._basis_y2t[sy, :]
                A_s = np.kron(by_s, bx2_s) + np.kron(by2_s, bx_s)
                rhs_s = rhs_grid[sy, sx].flatten()  # target is Δw = f in the strip
                H = H + strip_weight * (A_s.T @ A_s)
                g -= strip_weight * (A_s.T @ rhs_s)

        particular, *_ = np.linalg.lstsq(C, d, rcond=None)
        null_basis = sla.null_space(C)
        if null_basis.size == 0:
            coeff_vector = particular
        else:
            reduced_h = null_basis.T @ H @ null_basis
            reduced_rhs = -(null_basis.T @ (H @ particular + g))
            reduced_sol, *_ = np.linalg.lstsq(reduced_h, reduced_rhs, rcond=None)
            coeff_vector = particular + null_basis @ reduced_sol

        coefficients = coeff_vector.reshape((nbx, nby), order="F")

        corrector = self._evaluate_solution(coefficients)
        laplacian = self._evaluate_laplacian(coefficients)
        corrected_rhs = rhs_grid - laplacian
        return corrector, laplacian, corrected_rhs

    def solve_dst_corrected_02(
        self,
        rhs: RHSInput,
        *,
        left: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        right: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        bottom: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        top: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        second_normal_weight: float = 100.0,
        laplacian_weight: float = 1.0e-6,
        ridge: float = 1.0e-10,
        n_strip: int = 0,
        strip_weight: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build the 0/2-jet corrector and solve for the remainder with DST."""
        left_values = _normalize_trace(left, self.y, name="left")
        right_values = _normalize_trace(right, self.y, name="right")
        bottom_values = _normalize_trace(bottom, self.x, name="bottom")
        top_values = _normalize_trace(top, self.x, name="top")
        corrector, laplacian, corrected_rhs = self.build_boundary_corrector_02(
            rhs,
            left=left_values,
            right=right_values,
            bottom=bottom_values,
            top=top_values,
            second_normal_weight=second_normal_weight,
            laplacian_weight=laplacian_weight,
            ridge=ridge,
            n_strip=n_strip,
            strip_weight=strip_weight,
        )

        hx = _uniform_spacing(self.x, name="x")
        hy = _uniform_spacing(self.y, name="y")
        dst_remainder = np.zeros_like(corrected_rhs)
        if self.x.size > 2 and self.y.size > 2:
            rhs_interior = corrected_rhs[1:-1, 1:-1]
            solution_interior = _solve_zero_dirichlet_poisson_dst(rhs_interior, hx=hx, hy=hy)
            dst_remainder[1:-1, 1:-1] = solution_interior

        solution = corrector + dst_remainder
        return solution, corrector, dst_remainder, laplacian, corrected_rhs

    def solve_hybrid_dst(
        self,
        rhs: RHSInput,
        *,
        boundary_value: float = 0.0,
        left: float | np.ndarray | Callable[[np.ndarray], np.ndarray] | None = None,
        right: float | np.ndarray | Callable[[np.ndarray], np.ndarray] | None = None,
        bottom: float | np.ndarray | Callable[[np.ndarray], np.ndarray] | None = None,
        top: float | np.ndarray | Callable[[np.ndarray], np.ndarray] | None = None,
        return_parts: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve ``Delta u = rhs`` by a spline lift plus a zero-Dirichlet DST spectral solve."""
        hx = _uniform_spacing(self.x, name="x")
        hy = _uniform_spacing(self.y, name="y")
        rhs_grid = _sample_rhs_on_grid(rhs, self.x, self.y)

        if any(value is not None for value in (left, right, bottom, top)):
            if not all(value is not None for value in (left, right, bottom, top)):
                raise ValueError("Provide all four Dirichlet traces or none of them.")
            lift_coeffs = self._build_boundary_coefficients(
                left=left,
                right=right,
                bottom=bottom,
                top=top,
            )
        elif boundary_value != 0.0:
            lift_coeffs = self._build_boundary_coefficients(
                left=boundary_value,
                right=boundary_value,
                bottom=boundary_value,
                top=boundary_value,
            )
        else:
            lift_coeffs = np.zeros((self._basis_x.shape[0], self._basis_y.shape[0]), dtype=np.float64)

        lift = self._evaluate_solution(lift_coeffs)

        if self.x.size <= 2 or self.y.size <= 2:
            if return_parts:
                return lift.copy(), np.zeros_like(lift), lift.copy()
            return lift.copy()

        lap_lift = self._evaluate_laplacian(lift_coeffs)
        rhs_interior = rhs_grid[1:-1, 1:-1] - lap_lift[1:-1, 1:-1]
        remainder = np.zeros_like(lift)
        remainder[1:-1, 1:-1] = _solve_zero_dirichlet_poisson_dst(rhs_interior, hx=hx, hy=hy)
        solution = lift + remainder
        if return_parts:
            return solution, remainder, lift
        return solution

    def solve_harmonic_extension_dst(
        self,
        rhs: RHSInput,
        *,
        left: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        right: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        bottom: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        top: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        n_strip: int = 0,
        strip_weight: float = 1.0,
        smoothness_weight: float = 1.0,
        ridge: float = 1.0e-10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Solve ``Delta u = rhs`` using a harmonic lift, optional strip correction, and DST remainder."""
        rhs_grid = _sample_rhs_on_grid(rhs, self.x, self.y)
        hx = _uniform_spacing(self.x, name="x")
        hy = _uniform_spacing(self.y, name="y")

        harmonic_lift, patch, harmonic_correction, lap_patch = self.build_harmonic_extension(
            left=left,
            right=right,
            bottom=bottom,
            top=top,
        )

        strip_correction, strip_laplacian = self.build_zero_boundary_strip_correction(
            rhs_grid,
            n_strip=n_strip,
            strip_weight=strip_weight,
            smoothness_weight=smoothness_weight,
            ridge=ridge,
        )
        total_lift = harmonic_lift + strip_correction

        remainder = np.zeros_like(rhs_grid)
        if self.x.size > 2 and self.y.size > 2:
            remainder[1:-1, 1:-1] = _solve_zero_dirichlet_poisson_dst(
                (rhs_grid - strip_laplacian)[1:-1, 1:-1],
                hx=hx,
                hy=hy,
            )

        solution = total_lift + remainder
        return solution, total_lift, remainder, patch, harmonic_correction, strip_correction, strip_laplacian

    def solve(
        self,
        rhs: RHSInput,
        *,
        boundary_value: float = 0.0,
        return_laplacian: bool = False,
        left: float | np.ndarray | Callable[[np.ndarray], np.ndarray] | None = None,
        right: float | np.ndarray | Callable[[np.ndarray], np.ndarray] | None = None,
        bottom: float | np.ndarray | Callable[[np.ndarray], np.ndarray] | None = None,
        top: float | np.ndarray | Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        """Solve ``Delta u = rhs`` with constant or general Dirichlet boundary data."""
        if callable(rhs):
            load_full = self._load_matrix(rhs)
        else:
            rhs_arr = np.asarray(rhs, dtype=np.float64)
            if rhs_arr.ndim != 2:
                raise ValueError("rhs must be a 2D array with shape (len(y), len(x)).")
            if rhs_arr.shape != (self.y.size, self.x.size):
                raise ValueError(
                    f"rhs shape {rhs_arr.shape} must match (len(y), len(x))=({self.y.size}, {self.x.size})."
                )
            load_full = self._load_matrix(rhs_arr)
        if any(value is not None for value in (left, right, bottom, top)):
            if not all(value is not None for value in (left, right, bottom, top)):
                raise ValueError("Provide all four Dirichlet traces or none of them.")
            boundary_coefficients = self._build_boundary_coefficients(
                left=left,
                right=right,
                bottom=bottom,
                top=top,
            )
        else:
            boundary_coefficients = np.zeros(
                (self._basis_x.shape[0], self._basis_y.shape[0]),
                dtype=np.float64,
            )
            if boundary_value != 0.0:
                boundary_coefficients += self._build_boundary_coefficients(
                    left=boundary_value,
                    right=boundary_value,
                    bottom=boundary_value,
                    top=boundary_value,
                )

        if np.any(boundary_coefficients):
            load_full = load_full - (
                self._stiff_x @ boundary_coefficients @ self._mass_y
                + self._mass_x @ boundary_coefficients @ self._stiff_y
            )

        load_reduced = self.tx.T @ load_full @ self.ty

        transformed_rhs = self.eigvecs_x.T @ load_reduced @ self.eigvecs_y
        denom = self.eigvals_x[:, None] + self.eigvals_y[None, :]
        reduced_coefficients = transformed_rhs / denom
        reduced_coefficients = self.eigvecs_x @ reduced_coefficients @ self.eigvecs_y.T

        coefficients = boundary_coefficients + self.tx @ reduced_coefficients @ self.ty.T
        solution = self._evaluate_solution(coefficients)
        if not return_laplacian:
            return solution
        laplacian = self._evaluate_laplacian(coefficients)
        return solution, laplacian


__all__ = ["PODLayerBasis2D", "Poisson2DDirichletSolver"]
