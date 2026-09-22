"""Independent analytic-source checks for the random-wave stress test."""
import numpy as np
import pytest
from examples.pde.spline_annulus_turbulent_mms import RandomWaveMMS


@pytest.mark.parametrize('cutoff', [12., 24., 48.])
def test_analytic_poisson_helmholtz_source(cutoff):
    mms = RandomWaveMMS(cutoff)
    x = np.random.default_rng(871).uniform(-.8, .8, (25, 2))
    h = 2e-4
    lap = np.zeros(len(x))
    for d in (0, 1):
        dx = np.eye(2)[d]*h
        lap += (-mms.exact(x+2*dx)+16*mms.exact(x+dx)-30*mms.exact(x)
                +16*mms.exact(x-dx)-mms.exact(x-2*dx))/(12*h*h)
        derivative = (mms.exact(x+dx)-mms.exact(x-dx))/(2*h)
        np.testing.assert_allclose(derivative, mms.gradient(x)[:,d], atol=2e-4, rtol=2e-4)
    for sigma in (0., 64., -64.):
        actual = -lap+sigma*mms.exact(x)
        expected = mms.forcing(x,sigma)
        assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 2e-8
    assert np.sqrt(mms.norm2.max()) <= cutoff
    np.testing.assert_array_equal(mms.waves, RandomWaveMMS(cutoff).waves)
