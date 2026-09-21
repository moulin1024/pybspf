"""Independent checks of random-wave Poisson manufacture."""

import numpy as np

from bspf_models.elliptic.random_wave_mms import RandomWaveMMS


def test_reproducible_continuous_wavevectors_and_variance():
    a = RandomWaveMMS.create()
    b = RandomWaveMMS.create()
    np.testing.assert_array_equal(a.wavevectors, b.wavevectors)
    np.testing.assert_array_equal(a.phases, b.phases)
    assert not np.array_equal(a.phases, RandomWaveMMS.create(seed=17).phases)
    k = np.linalg.norm(a.wavevectors, axis=1)
    assert np.all((k >= np.pi) & (k <= 8 * np.pi))
    assert len(k) == 64
    np.testing.assert_allclose(np.sum(a.amplitudes**2) / 2, 1, atol=1e-15)
    # The rectangular domain is not made periodic by choosing integer modes.
    assert np.max(abs(a.wavevectors / np.pi - np.rint(a.wavevectors / np.pi))) > 0.1


def test_analytic_gradient_and_nonconstant_poisson_forcing():
    mms = RandomWaveMMS.create(kmax=12 * np.pi)
    points = np.random.default_rng(48).uniform(-0.9, 0.9, (90, 2))
    value, gradient, forcing = mms.evaluate(points)
    h = 2e-5
    laplacian = np.zeros(len(points))
    for axis in range(2):
        step = np.eye(2)[axis] * h
        plus, minus = mms.evaluate(points + step)[0], mms.evaluate(points - step)[0]
        np.testing.assert_allclose(
            (plus - minus) / (2 * h), gradient[:, axis], rtol=3e-6, atol=3e-6
        )
        laplacian += (plus + minus - 2 * value) / h**2
    assert np.linalg.norm(laplacian + forcing) / np.linalg.norm(forcing) < 2e-6
    assert np.std(forcing) > 10


def test_single_wave_poisson_eigenvalue():
    wavevector = np.array([[2.3, -4.1]])
    mms = RandomWaveMMS(
        wavevector, np.array([0.7]), np.array([0.2]), np.array([1, 5]), np.array([0]), 0
    )
    p = np.array([[0.2, -0.4], [-0.3, 0.1]])
    value, gradient, forcing = mms.evaluate(p)
    phase = p @ wavevector[0] + 0.2
    np.testing.assert_allclose(value, 0.7 * np.cos(phase))
    np.testing.assert_allclose(gradient, -0.7 * np.sin(phase)[:, None] * wavevector)
    np.testing.assert_allclose(forcing, np.sum(wavevector**2) * value)
