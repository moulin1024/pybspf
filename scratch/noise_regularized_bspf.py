"""Scratch joint spline/Fourier regularization; not a production API.

Run from the repository root with the corrected BLAS:
  OMP_NUM_THREADS=4 python docs/diagnostics/run_with_local_blas.py \
      scratch/noise_regularized_bspf.py

NumPy/SciPy prototype, real uniform odd grids, scalar or trailing batches.
No dense Fourier matrix, noisy endpoint constraints, or post-fit cutoff.
The demo compares the existing JAX operators using identical noisy samples.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import qr, solve_triangular


@dataclass(frozen=True)
class Geometry:
    x: np.ndarray
    basis: np.ndarray
    derivative: np.ndarray
    roughness: np.ndarray
    spectrum: np.ndarray
    omega: np.ndarray
    penalty_order: int


def geometry(x, degree=5, n_basis=18, penalty_order=2):
    """Spline roughness approximates (N/L)*integral |s^(p)|^2 exactly by Gauss quadrature."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size % 2 != 1 or x.size < 3:
        raise ValueError("Use an odd, one-dimensional grid with at least 3 samples.")
    if (
        not np.isfinite(x).all()
        or np.any(np.diff(x) <= 0)
        or not np.allclose(np.diff(x), x[1] - x[0])
    ):
        raise ValueError("Grid must be finite, increasing and uniform.")
    if not 1 <= penalty_order <= degree or n_basis < degree + 1:
        raise ValueError(
            "Require 1 <= penalty_order <= degree and n_basis >= degree+1."
        )
    breaks = np.linspace(x[0], x[-1], n_basis - degree + 1)
    knots = np.r_[np.repeat(x[0], degree), breaks, np.repeat(x[-1], degree)]
    spline = BSpline(knots, np.eye(n_basis), degree)
    nodes, weights = np.polynomial.legendre.leggauss(degree - penalty_order + 1)
    widths = np.diff(breaks)
    points = (
        (breaks[:-1, None] + breaks[1:, None]) / 2 + widths[:, None] * nodes / 2
    ).ravel()
    weights = (widths[:, None] * weights / 2 * x.size / (x[-1] - x[0])).ravel()
    B = spline(x)
    return Geometry(
        x,
        B,
        spline(x, nu=1),
        np.sqrt(weights[:, None]) * spline(points, nu=penalty_order),
        np.fft.fft(B, axis=0, norm="ortho"),
        2 * np.pi * np.fft.fftfreq(x.size, x[1] - x[0]),
        penalty_order,
    )


@dataclass(frozen=True)
class Plan:
    geometry: Geometry
    alpha: float
    transfer: np.ndarray
    projector: np.ndarray


def plan(g, alpha, spline_weight=1.0):
    """Minimize ||s+r-y||² + alpha*(spline_weight*||R c||² + ||D^p r||²).

    Fourier residual has zero mean. Eliminate r analytically; solve the reduced
    spline least-squares problem by augmented QR, without normal equations.
    Physical angular frequencies define the penalty, so alpha is dimensionful.
    """
    if (
        not np.isfinite(alpha)
        or alpha <= 0
        or not np.isfinite(spline_weight)
        or spline_weight <= 0
    ):
        raise ValueError(
            "Regularization and spline_weight must be finite and positive."
        )
    penalty = alpha * abs(g.omega) ** (2 * g.penalty_order)
    H = 1 / (1 + penalty)
    H[0] = 0.0  # Fix the constant-mode ambiguity: all mean belongs to the spline.
    S = penalty / (1 + penalty)  # Stable alternative to 1-H for very small alpha.
    S[0] = 1.0
    A = np.vstack(
        [np.sqrt(S[:, None]) * g.spectrum, np.sqrt(alpha * spline_weight) * g.roughness]
    )
    Q, R = qr(A, mode="economic")
    projector = solve_triangular(R, Q[: g.x.size].conj().T * np.sqrt(S)[None, :])
    return Plan(g, alpha, H, projector)


def apply(p, y):
    """Return fitted values, first derivative, and decomposition coefficients."""
    g = p.geometry
    y = np.asarray(y)
    if (
        y.ndim < 1
        or y.shape[0] != g.x.size
        or np.iscomplexobj(y)
        or not np.isfinite(y).all()
    ):
        raise ValueError("Expected finite real samples, grid axis first.")
    shape = y.shape
    values = y.reshape(g.x.size, -1)
    yhat = np.fft.fft(values, axis=0, norm="ortho")
    c = (p.projector @ yhat).real
    rhat = p.transfer[:, None] * (yhat - g.spectrum @ c)
    fitted = g.basis @ c + np.fft.ifft(rhat, axis=0, norm="ortho").real
    derivative = (
        g.derivative @ c
        + np.fft.ifft(1j * g.omega[:, None] * rhat, axis=0, norm="ortho").real
    )
    return fitted.reshape(shape), derivative.reshape(shape), c, rhat


def discrepancy_select(plans, y, noise_std):
    """Choose alpha per realization, closest to ||fit-y|| = sqrt(N)*noise_std.

    Known homoscedastic noise only. Search-grid edges are reported, not silently
    interpreted as a reliable optimum. No analytic derivative enters selection.
    """
    if noise_std <= 0:
        raise ValueError("Provide a positive, absolute noise standard deviation.")
    values = np.asarray(y)
    if values.ndim != 2:
        raise ValueError("Selection expects (samples, realizations).")
    residuals, derivatives, fits = [], [], []
    for p in plans:
        fit, derivative, _, _ = apply(p, values)
        residuals.append(np.linalg.norm(fit - values, axis=0))
        derivatives.append(derivative)
        fits.append(fit)
    residuals = np.array(residuals)
    target = np.sqrt(values.shape[0]) * noise_std
    selected = np.argmin(abs(residuals - target), axis=0)
    columns = np.arange(values.shape[1])
    return (
        np.array(fits)[selected, :, columns].T,
        np.array(derivatives)[selected, :, columns].T,
        selected,
        residuals[selected, columns] / target,
    )


def self_check(g):
    """Independent dense least-squares reference plus an unpenalized linear signal."""
    small = geometry(np.linspace(-0.3, 1.2, 33), degree=5, n_basis=10)
    p = plan(small, 1e-3)
    x = small.x
    fit, derivative, c, rhat = apply(p, np.exp(x) + np.sin(3 * x))
    n = x.size
    # Orthonormal real Fourier columns, omitting the constant; reference only.
    k = np.arange(1, (n + 1) // 2)
    phase = 2 * np.pi * np.arange(n)[:, None] * k[None, :] / n
    F = np.sqrt(2 / n) * np.column_stack([np.cos(phase), np.sin(phase)])
    w = small.omega[k]
    D = np.sqrt(2 / n) * np.column_stack([-np.sin(phase) * w, np.cos(phase) * w])
    A = np.column_stack([small.basis, F])
    regularizer = np.zeros((small.roughness.shape[0] + n - 1, A.shape[1]))
    regularizer[: small.roughness.shape[0], :10] = small.roughness
    regularizer[small.roughness.shape[0] :, 10:] = np.diag(
        np.tile(w**small.penalty_order, 2)
    )
    augmented = np.vstack([A, np.sqrt(p.alpha) * regularizer])
    coef = np.linalg.lstsq(
        augmented,
        np.r_[np.exp(x) + np.sin(3 * x), np.zeros(regularizer.shape[0])],
        rcond=None,
    )[0]
    np.testing.assert_allclose(fit, A @ coef, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(
        derivative,
        np.column_stack([small.derivative, D]) @ coef,
        atol=1e-10,
        rtol=1e-10,
    )
    assert abs(rhat[0]) == 0
    fitted, derivative, _, _ = apply(plan(g, 1.0), 2 + 3 * g.x)
    np.testing.assert_allclose(fitted, 2 + 3 * g.x, atol=1e-9)
    np.testing.assert_allclose(derivative, 3, atol=1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=513)
    parser.add_argument("--realizations", type=int, default=32)
    parser.add_argument(
        "--noise",
        type=float,
        default=1e-4,
        help="Noise standard deviation / RMS(clean signal)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("build/noise-regularized-bspf")
    )
    args = parser.parse_args()
    if args.realizations < 1 or args.noise <= 0:
        parser.error("realizations and noise must be positive")
    import jax

    jax.config.update("jax_enable_x64", True)
    import bspf_jax as bspf
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(0, 2 * np.pi, args.samples)
    clean = np.exp(0.2 * x) + np.sin(2.3 * x) + 0.2 * np.cos(5.1 * x)
    truth = 0.2 * np.exp(0.2 * x) + 2.3 * np.cos(2.3 * x) - 1.02 * np.sin(5.1 * x)
    sigma = args.noise * np.sqrt(np.mean(clean**2))
    y = clean[:, None] + sigma * np.random.default_rng(20260917).standard_normal(
        (x.size, args.realizations)
    )
    g = geometry(x)
    self_check(g)
    alphas = np.geomspace(1e-14, 1e2, 81)
    plans = [plan(g, a) for a in alphas]
    fitted, derivative, selected, ratios = discrepancy_select(plans, y, sigma)
    options = dict(degree=9, n_basis=18, constraint_order=8, lam=1e-6)
    fd = bspf.plan_1d(x, **options, boundary_points=9)
    ldc = bspf.plan_1d(
        x, **options, endpoint_method="chebyshev", chebyshev_modes=8, boundary_points=40
    )
    methods = {
        "FD9": np.asarray(jax.jit(bspf.differentiate)(fd, y)),
        "LDC M8/P40": np.asarray(jax.jit(bspf.differentiate)(ldc, y)),
        "Joint regularization": derivative,
    }
    boundary = np.r_[np.arange(40), np.arange(x.size - 40, x.size)]
    interior = np.arange(40, x.size - 40)
    if interior.size == 0:
        raise ValueError(
            "Demo needs more than 80 points for the boundary/interior comparison."
        )
    report = {
        "samples": x.size,
        "realizations": args.realizations,
        "relative_noise": args.noise,
        "absolute_noise_std": sigma,
        "self_checks": "passed",
        "alpha_median": float(np.median(alphas[selected])),
        "alpha_range": [float(alphas[selected].min()), float(alphas[selected].max())],
        "selection_at_grid_edge": int(
            np.sum((selected == 0) | (selected == len(plans) - 1))
        ),
        "residual_over_expected_noise_range": [
            float(ratios.min()),
            float(ratios.max()),
        ],
        "methods": {},
    }
    for name, result in methods.items():
        report["methods"][name] = {
            region: float(
                np.mean(
                    np.linalg.norm((result - truth[:, None])[ids], axis=0)
                    / np.linalg.norm(truth[ids])
                )
            )
            for region, ids in [
                ("all", np.arange(x.size)),
                ("boundary", boundary),
                ("interior", interior),
            ]
        }
        assert np.isfinite(result).all()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    np.savez(
        args.output / "results.npz",
        x=x,
        clean=clean,
        truth=truth,
        noisy=y,
        fitted=fitted,
        derivative=derivative,
        selected_alpha=alphas[selected],
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(x, truth, "k--", label="Exact")
    for name, result in methods.items():
        axes[0].plot(x, result[:, 0], label=name, alpha=0.8)
        axes[1].semilogy(
            x, np.sqrt(np.mean((result - truth[:, None]) ** 2, axis=1)), label=name
        )
    axes[0].set(xlabel="x", ylabel="First derivative, realization 0")
    axes[1].set(xlabel="x", ylabel="RMS derivative error across realizations")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(args.output / "comparison.png", dpi=160)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
