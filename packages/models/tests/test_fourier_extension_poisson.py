"""Independent DFT/SVD, off-grid, PDE and boundary checks for FE + Poisson."""

import numpy as np
import pytest
import scipy.linalg as la

from pybspf.fourier_extension import FourierExtensionPlan
from bspf_models.elliptic.fourier_poisson import FourierPoissonPlan
from bspf_models.elliptic.fourier_poisson import _harmonic_matrix
from bspf_models.elliptic.embedded_poisson import benchmark_domains


@pytest.fixture(scope="module")
def extension():
    x = np.linspace(-1.2, 1.2, 26, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return FourierExtensionPlan(xx**2 + yy**2 < 0.8**2, 13, seed=23)


def test_fft_matches_independent_dense_dft_and_adjoint(extension):
    p = extension
    k = np.stack(np.meshgrid(p.frequencies, p.frequencies, indexing="ij"), axis=-1).reshape(-1, 2)
    a = np.exp(1j * np.pi / p.half_width * ((p.points + p.half_width) @ k.T)) / p.normalization
    rng = np.random.default_rng(91)
    c = rng.normal(size=(p.size, 3)) + 1j * rng.normal(size=(p.size, 3))
    b = rng.normal(size=(p.samples, 3)) + 1j * rng.normal(size=(p.samples, 3))
    np.testing.assert_allclose(p.forward(c), a @ c, atol=1e-14)
    np.testing.assert_allclose(p.adjoint(b), a.conj().T @ b, atol=1e-14)
    np.testing.assert_allclose(np.vdot(p.forward(c), b), np.vdot(c, p.adjoint(b)), atol=1e-13)
    np.testing.assert_allclose(p.evaluate(c[:, 0], p.points), p.normalization * p.forward(c[:, 0]), atol=4e-13)


def test_algorithm_one_agrees_with_dense_tsvd(extension):
    p = extension
    # Compare fitted functions/residuals, not unstable extension coefficients.
    a = p.forward(np.eye(p.size))
    u, s, vh = la.svd(a, full_matrices=False)
    f = np.exp(p.points[:, 0] + 0.4 * p.points[:, 1])
    keep = s > p.cutoff
    direct = vh[keep].conj().T @ ((u[:, keep].conj().T @ (f / p.normalization)) / s[keep])
    fast = p.solve(f)
    np.testing.assert_allclose(p.forward(fast.coefficients), a @ direct, atol=2e-10)
    check = np.random.default_rng(6).uniform(-0.45, 0.45, (71, 2))
    np.testing.assert_allclose(fast.evaluate(check), p.evaluate(direct, check), atol=2e-8)


def test_off_grid_complex_mode_derivatives(extension):
    p = extension
    wave = np.array([2, -1]) * np.pi / p.half_width
    c = np.zeros((p.modes, p.modes), complex)
    c[p.modes // 2 + 2, p.modes // 2 - 1] = 1
    q = np.random.default_rng(7).uniform(-1, 1, (40, 2))
    exact = np.exp(1j * ((q + p.half_width) @ wave))
    for d in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)):
        np.testing.assert_allclose(p.evaluate(c.ravel(), q, d), exact * (1j * wave[0])**d[0] * (1j * wave[1])**d[1], atol=2e-13)
    fit = p.solve(np.exp(1j * ((p.points + p.half_width) @ wave)))
    np.testing.assert_allclose(fit.evaluate(q[np.linalg.norm(q, axis=1) < 0.5]),
                               np.exp(1j * ((q[np.linalg.norm(q, axis=1) < 0.5] + p.half_width) @ wave)), atol=1e-9)


def test_full_box_needs_no_plunge_correction():
    p = FourierExtensionPlan(np.ones((18, 18), bool), 9)
    assert p.diagnostics["plunge_rank"] == 0
    f = np.cos(np.pi / 1.2 * p.points[:, 0])
    np.testing.assert_allclose(p.solve(f).evaluate(p.points), f, atol=2e-14)


def test_rank_budget_failure_is_explicit():
    x = np.linspace(-1, 1, 24, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    with pytest.raises(RuntimeError, match="Plunge range unresolved"):
        FourierExtensionPlan(xx**2 + yy**2 < 0.7, 11, max_rank=8)


@pytest.fixture(scope="module")
def poisson():
    return FourierPoissonPlan(benchmark_domains()[0], modes=33, source_count=128, boundary_count=256)


def test_nonperiodic_poisson_independent_values_gradient_hessian_boundary(poisson):
    p = poisson
    exact = lambda x: np.exp(x[:, 0] + x[:, 1])
    # Callbacks assert no exterior or exact interior solution data are requested.
    def forcing(x):
        np.testing.assert_array_equal(x, p.extension.points)
        return -2 * exact(x)

    def boundary(x):
        np.testing.assert_array_equal(x, p.boundary)
        return exact(x)

    solution = p.solve(forcing, boundary)
    q = np.random.default_rng(14).uniform(-0.45, 0.45, (83, 2))
    u, gradient, laplacian = solution.evaluate(q)
    np.testing.assert_allclose(u, exact(q), atol=3e-8)
    np.testing.assert_allclose(gradient, np.repeat(exact(q)[:, None], 2, axis=1), atol=2e-7)
    np.testing.assert_allclose(solution.hessian(q), np.tile(exact(q)[:, None, None], (1, 2, 2)), atol=2e-6)
    np.testing.assert_allclose(laplacian, 2 * exact(q), atol=2e-6)
    edge, _ = p.arc.sample(512, offset=0.371)
    np.testing.assert_allclose(solution.derivative(edge), exact(edge), atol=2e-8)
    # An algebraic PDE consistency check independent of the exact solution.
    np.testing.assert_allclose(-laplacian, solution.forcing_extension.evaluate(q), atol=5e-11)


def test_nonzero_mean_and_complex_data(poisson):
    p = poisson
    exact = lambda x: (1 + 2j) * (1 - np.sum(x**2, axis=1))
    s = p.solve(4 * (1 + 2j), exact)
    q = np.random.default_rng(11).uniform(-0.4, 0.4, (41, 2))
    u, gradient, laplacian = s.evaluate(q)
    np.testing.assert_allclose(u, exact(q), atol=2e-8)
    np.testing.assert_allclose(gradient, -2 * (1 + 2j) * q, atol=1e-7)
    np.testing.assert_allclose(laplacian, -4 * (1 + 2j), atol=2e-7)


def test_zero_forcing_nonzero_boundary_harmonic(poisson):
    p = poisson
    exact = lambda x: x[:, 0]**2 - x[:, 1]**2 + 0.7 * x[:, 0]
    s = p.solve(0, exact)
    q = np.random.default_rng(15).uniform(-0.4, 0.4, (47, 2))
    u, _, lap = s.evaluate(q)
    np.testing.assert_allclose(u, exact(q), atol=1e-9)
    np.testing.assert_allclose(lap, 0, atol=1e-10)
    hxx = _harmonic_matrix(q, p.sources, (2, 0))
    hyy = _harmonic_matrix(q, p.sources, (0, 2))
    np.testing.assert_allclose(hxx + hyy, 0, atol=2e-15)


def test_harmonic_derivatives_against_finite_differences(poisson):
    p = poisson
    q = np.random.default_rng(31).uniform(-0.4, 0.4, (13, 2))
    for axis in (0, 1):
        shift = np.eye(2)[axis] * 1e-5
        d = (1, 0) if axis == 0 else (0, 1)
        dd = (2, 0) if axis == 0 else (0, 2)
        numerical = (_harmonic_matrix(q + shift, p.sources) - _harmonic_matrix(q - shift, p.sources)) / 2e-5
        np.testing.assert_allclose(numerical, _harmonic_matrix(q, p.sources, d), atol=1e-9)
        numerical = (_harmonic_matrix(q + shift, p.sources, d) - _harmonic_matrix(q - shift, p.sources, d)) / 2e-5
        np.testing.assert_allclose(numerical, _harmonic_matrix(q, p.sources, dd), atol=2e-9)


def test_invalid_domain_and_inputs(poisson):
    with pytest.raises(ValueError, match="convex"):
        FourierPoissonPlan(benchmark_domains()[1], modes=9)
    with pytest.raises(ValueError, match="box"):
        FourierPoissonPlan(benchmark_domains()[0], modes=9, half_width=0.7)
    with pytest.raises(ValueError, match="finite"):
        poisson.solve(np.nan, 0)


def test_physical_residual_validator_and_plan_reuse(poisson):
    s = poisson.solve(4.0, lambda x: 1 - np.sum(x**2, axis=1))
    diagnostic = s.validate(lambda x: np.full(len(x), 4.0),
                            lambda x: 1 - np.sum(x**2, axis=1),
                            volume_order=12, boundary_count=512)
    assert diagnostic["forcing_relative_l2"] < 1e-6
    assert diagnostic["boundary_h32"] < 1e-6
    q = np.array([[0.2, -0.1], [-0.1, 0.3]])
    assert np.isrealobj(s.derivative(q))
    # A second RHS reuses the same geometry and factors; the prior solution is
    # not overwritten, and linearity also checks mean/boundary coupling.
    twice = poisson.solve(8.0, lambda x: 2 * (1 - np.sum(x**2, axis=1)))
    np.testing.assert_allclose(twice.derivative(q), 2*s.derivative(q), atol=1e-11)
