"""Physical properties of derivative-closed pybspf trial spaces."""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from pybspf import ClosedBSPFLine


@pytest.mark.parametrize("n", [33, 65])
def test_closed_line_traces_derivative_and_mass(n):
    line = ClosedBSPFLine(n)
    wall = line.values(np.array([0.0, 1.0]), 2)
    np.testing.assert_array_equal(wall[0], 0.0)
    np.testing.assert_array_equal(wall[1], 0.0)
    x, w = leggauss(400)
    x, w = (x + 1) / 2, w / 2
    value, derivative = line.values(x, 1)
    np.testing.assert_allclose(
        derivative, line.tangent_values(x, 0)[0] @ line.derivative_map, atol=1e-12
    )
    np.testing.assert_allclose(value.T @ (w[:, None] * value), np.eye(n - 3), atol=2e-9)
    np.testing.assert_allclose(line.scalar_values(line.x, 0)[0], np.eye(n), atol=2e-10)
    # Independent central differences on arbitrary coefficients verify that the
    # analytic primitive and derivative represent the same physical function.
    c = np.random.default_rng(20).normal(size=n - 3)
    h = 1e-6
    points = np.linspace(0.1, 0.9, 31)
    fd = ((line.values(points + h, 0)[0] - line.values(points - h, 0)[0]) @ c) / (2 * h)
    np.testing.assert_allclose(fd, line.values(points, 1)[1] @ c, rtol=2e-6, atol=2e-6)


def test_closed_line_size_validation():
    for n in (True, 10, 33.5):
        with pytest.raises(ValueError):
            ClosedBSPFLine(n)
