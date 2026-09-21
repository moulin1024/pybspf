"""Checks for the common metric and the two comparison spaces."""

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as la
from scipy.special import roots_legendre

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "pde"))
from compare_poisson_bases import (  # noqa: E402
    ProfileMMS,
    TrialBasis,
    exact_jets,
    field_jets,
    fit,
    pair,
    whiten,
)
from bspf_models.elliptic.convex_poisson import box_h2_root  # noqa: E402
from bspf_models.elliptic.random_wave_mms import RandomWaveMMS  # noqa: E402


@pytest.mark.parametrize("family", ["fourier", "bspline"])
def test_independent_integral_matches_h2_scaling(family):
    basis = TrialBasis(family, 17)
    # Separate composite quadrature, resolving both Fourier modes and spline knots.
    q, w = roots_legendre(32)
    edges = np.linspace(-1.2, 1.2, 17)
    points = np.concatenate(
        [(a + b) / 2 + (b - a) / 2 * q for a, b in zip(edges[:-1], edges[1:])]
    )
    weights = np.tile(w * (edges[1] - edges[0]) / 2, 16)
    b, g, h = basis.evaluate(points)
    np.testing.assert_allclose(b.T @ (weights[:, None] * b), np.eye(17), atol=2e-12)
    np.testing.assert_allclose(
        g.T @ (weights[:, None] * g), np.diag(basis.lam), atol=2e-10
    )
    c = np.random.default_rng(2).normal(size=(17, 17))
    direct = sum(
        np.sum(weights[:, None] * weights[None, :] * (a @ c @ z.T) ** 2)
        for a, z in ((b, b), (g, b), (b, g), (h, b), (np.sqrt(2) * g, g), (b, h))
    )
    np.testing.assert_allclose(
        np.linalg.norm(box_h2_root(basis) @ c.ravel()) ** 2, direct, rtol=2e-12
    )


@pytest.mark.parametrize("family", ["fourier", "bspline"])
def test_analytic_derivatives_and_tensor_order(family):
    basis = TrialBasis(family, 17)
    points = np.random.default_rng(4).uniform(-1.1, 1.1, (25, 2))
    v, g, h = basis.evaluate(points[:, 0])
    eps = 1e-6
    plus, minus = basis.evaluate(points[:, 0] + eps), basis.evaluate(points[:, 0] - eps)
    np.testing.assert_allclose(
        (plus[0] - minus[0]) / (2 * eps), g, atol=2e-7, rtol=2e-7
    )
    np.testing.assert_allclose(
        (plus[1] - minus[1]) / (2 * eps), h, atol=2e-5, rtol=2e-7
    )
    factors = basis.factors(points)
    c = np.random.default_rng(5).normal(size=17**2)
    (x, dx, xx), (y, dy, yy) = factors
    explicit = np.column_stack(
        [
            pair(a, b) @ c
            for a, b in (
                (x, y),
                (dx, y),
                (x, dy),
                (xx, y),
                (np.sqrt(2) * dx, dy),
                (x, yy),
            )
        ]
    )
    np.testing.assert_allclose(field_jets(factors, c), explicit, atol=2e-10)


def test_whitening_and_svd_solve_recover_known_coefficients():
    rng = np.random.default_rng(8)
    a = rng.normal(size=(50, 12))
    root = np.triu(rng.normal(size=(12, 12))) + 10 * np.eye(12)
    reference = la.solve_triangular(root.T, a.T, lower=True).T
    np.testing.assert_allclose(
        whiten(np.array(a, order="F"), root), reference, atol=1e-14
    )
    c = rng.normal(size=(12, 2))
    fits, _, _ = fit(np.array(a, order="F"), a @ c, root, [1e-13])
    np.testing.assert_allclose(fits[0]["coefficient"], c, atol=1e-13)


def test_mms_hessian_matches_forcing():
    mms = RandomWaveMMS.create()
    points = np.random.default_rng(9).uniform(-1, 1, (25, 2))
    jets = exact_jets(mms, points)
    np.testing.assert_allclose(
        -(jets[:, 3] + jets[:, 5]), mms.evaluate(points)[2], atol=1e-11
    )


@pytest.mark.parametrize("name", ["polynomial", "gaussian", "rational"])
def test_non_fourier_mms_jets(name):
    mms = ProfileMMS(name)
    points = np.random.default_rng(19).uniform(-0.6, 0.6, (50, 2))
    jets = mms.jets(points)
    for axis in (0, 1):
        offset = np.eye(2)[axis] * 1e-6
        difference = (mms.jets(points + offset) - mms.jets(points - offset)) / 2e-6
        np.testing.assert_allclose(
            difference[:, 0], jets[:, 1 + axis], atol=2e-8, rtol=2e-7
        )
        expected = jets[:, [3, 4]] if axis == 0 else jets[:, [4, 5]]
        expected = expected.copy()
        expected[:, 1 - axis] /= np.sqrt(2)
        np.testing.assert_allclose(difference[:, 1:3], expected, atol=2e-7, rtol=2e-7)
