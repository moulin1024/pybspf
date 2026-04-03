"""! @file solvers/poisson2d.py
@brief Direct tensor-product Poisson solver for homogeneous/constant Dirichlet data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.fft import dstn, idstn
from scipy import linalg as sla

from ..operators.bspf1d import BSPF1D

RHSInput = np.ndarray | Callable[[np.ndarray, np.ndarray], np.ndarray]


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

    def build_boundary_corrector_02(
        self,
        rhs: RHSInput,
        *,
        left: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        right: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        bottom: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        top: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
        laplacian_weight: float = 1.0e-6,
        ridge: float = 1.0e-10,
        n_strip: int = 0,
        strip_weight: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a spline boundary corrector using value and second-normal-derivative edge jets.

        If ``n_strip > 0``, a soft penalty is added that encourages the analytic
        Laplacian of the corrector to match ``rhs`` over the first ``n_strip``
        interior rows/columns adjacent to each boundary edge.
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
            np.kron(eye_y, second_x[[0], :]),
            np.kron(eye_y, second_x[[1], :]),
            np.kron(value_y[[0], :], eye_x),
            np.kron(value_y[[1], :], eye_x),
            np.kron(second_y[[0], :], eye_x),
            np.kron(second_y[[1], :], eye_x),
        ]
        targets = [
            left_coeffs,
            right_coeffs,
            left_second_coeffs,
            right_second_coeffs,
            bottom_coeffs,
            top_coeffs,
            bottom_second_coeffs,
            top_second_coeffs,
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

        # Strip Laplacian penalty: encourage Δw = rhs over interior rows/cols near each edge.
        # A_strip maps the coefficient vector (Fortran order) to Laplacian values in the strip:
        #   A_strip = kron(B_y[strip_y, :], B_x2[strip_x, :]) + kron(B_y2[strip_y, :], B_x[strip_x, :])
        # Penalty: strip_weight * ||A_strip @ c - rhs_strip||^2
        # This adds strip_weight * A_strip.T @ A_strip to H and a linear term g = -strip_weight * A_strip.T @ rhs_strip.
        g = np.zeros(nbx * nby, dtype=np.float64)
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


__all__ = ["Poisson2DDirichletSolver"]
