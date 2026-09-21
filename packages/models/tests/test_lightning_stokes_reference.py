"""Independent boundary and differential checks for the rational reference."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "pde"))
from lightning_stokes_reference import LightningStokes  # noqa: E402


def test_rational_reference_boundary_and_stokes_equations():
    p = LightningStokes(72, 24, 36, 600)
    boundary = p.verify(811)
    assert boundary["velocity_boundary_max"] < 1e-8
    assert boundary["outlet_traction_over_nu_max"] < 1e-8
    assert boundary["hole_wall_max"] < 1e-10
    z = np.linspace(-0.6, 3, 31) + 0.64j
    f = p.evaluate(z)
    h = 1e-3
    c1 = np.array([1, -8, 0, 8, -1]) / 12
    c2 = np.array([-1 / 12, 4 / 3, -2.5, 4 / 3, -1 / 12])
    fx = np.array([p.evaluate(z + k * h) for k in range(-2, 3)])
    fy = np.array([p.evaluate(z + 1j * k * h) for k in range(-2, 3)])
    dx = np.einsum("k,kij->ij", c1, fx) / h
    dy = np.einsum("k,kij->ij", c1, fy) / h
    lap = np.einsum("k,kij->ij", c2, fx + fy) / h**2
    np.testing.assert_allclose(f[4], dx[0], atol=1e-8)
    np.testing.assert_allclose(f[5], dx[1], atol=1e-8)
    np.testing.assert_allclose(f[3], dx[1] - dy[0], atol=1e-8)
    np.testing.assert_allclose(dx[0] + dy[1], 0, atol=1e-8)
    np.testing.assert_allclose(lap[0] - dx[2], 0, atol=1e-7)
    np.testing.assert_allclose(lap[1] - dy[2], 0, atol=1e-7)
