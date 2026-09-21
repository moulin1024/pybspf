import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "pde"))
from boundary_fft_core import BoundaryFFTPlan, Circle, harmonic  # noqa: E402
from bspf_models.elliptic.embedded_poisson import benchmark_domains  # noqa: E402


def test_circle_symbol_and_harmonic_potential():
    plan = BoundaryFFTPlan(Circle(), 64)
    density = np.cos(5 * plan.theta)
    np.testing.assert_allclose(plan.single_layer @ density, density / 10, atol=2e-15)
    points = np.array([[0.5, 0.0], [0.0, 0.4], [-0.2, 0.3]])
    expected = np.real((points[:, 0] + 1j * points[:, 1]) ** 5) / 10
    np.testing.assert_allclose(
        plan.interior_evaluate(density, 0, points), expected, atol=2e-15
    )


@pytest.mark.parametrize(
    "domain", [Circle(), Circle((0.9, 0.76)), benchmark_domains()[0]]
)
def test_constant_mode(domain):
    plan = BoundaryFFTPlan(domain, 64)
    density, constant, diagnostics = plan.solve(np.full(64, 3.25))
    np.testing.assert_array_equal(density, np.zeros(64))
    assert constant == 3.25
    assert diagnostics["iterations"] == 0


@pytest.mark.parametrize("axes", [(1.0, 1.0), (0.9, 0.76)])
def test_analytic_geometry_independent_validation(axes):
    plan = BoundaryFFTPlan(Circle(axes), 128)
    density, constant, diagnostics = plan.solve(harmonic(plan.points))
    points, values = plan.boundary_evaluate(density, constant, 256)
    exact = harmonic(points)
    assert np.linalg.norm(values - exact) / np.linalg.norm(exact) < 1e-11
    assert diagnostics["iterations"] <= 10
    assert abs(density.mean()) < 1e-12
