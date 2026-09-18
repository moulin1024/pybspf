"""Independent manufactured vacuum solutions and periodic mapped BSPF traces."""

import jax
import numpy as np
import scipy.linalg as la
import pytest
from bspf_jax.tokamak_vacuum_bspf import mapped_bspf_vacuum

jax.config.update("jax_enable_x64", True)


class Ellipse:
    def evaluate(self, points):
        r, z = np.asarray(points).T
        return (
            1 - ((r - 2) / 0.5) ** 2 - (z / 0.8) ** 2,
            -2 * (r - 2) / 0.5**2,
            -2 * z / 0.8**2,
        )

    def surface(self, theta, bounds):
        direction = np.column_stack((np.cos(theta), np.sin(theta)))
        radius = 1 / np.sqrt(
            (direction[:, 0] / 0.5) ** 2 + (direction[:, 1] / 0.8) ** 2
        )
        p = np.array([2.0, 0.0]) + radius[:, None] * direction
        if bounds is None:
            outer_radius = 1.2 * radius
        else:
            outer_radius = np.minimum(
                1 / np.maximum(abs(direction[:, 0]), 1e-30),
                1.6 / np.maximum(abs(direction[:, 1]), 1e-30),
            )
        return p, np.array([2.0, 0.0]) + outer_radius[:, None] * direction, radius


@pytest.fixture(scope="module")
def responses():
    theta = np.arange(192) * 2 * np.pi / 192
    inner, outer, _ = Ellipse().surface(theta, None)
    return [
        mapped_bspf_vacuum(
            Ellipse(),
            [(1, 3), (-1.6, 1.6)],
            theta,
            inner,
            outer,
            wall_scale=1.2,
            radial_modes=nr,
            angular_modes=49,
            display_layers=12,
        )
        for nr in (12, 31)
    ]


def test_manufactured_vacuum_flux_and_strong_equation(responses):
    rng = np.random.default_rng(25)
    s = rng.uniform(0.1, 0.9, 80)
    t = rng.uniform(0, 2 * np.pi, 80)
    inner, outer, _ = Ellipse().surface(t, None)
    points = (1 - s[:, None]) * inner + s[:, None] * outer
    errors = []
    for vacuum in responses:
        ip = vacuum.points[vacuum.inner]
        op = vacuum.points[vacuum.outer]
        g, h = ip[:, 0] ** 2, op[:, 0] ** 2
        approx = vacuum.evaluate(s, t, g, h)
        errors.append(la.norm(approx - points[:, 0] ** 2) / la.norm(points[:, 0] ** 2))
        assert vacuum.residual < 1e-11
        assert la.eigvalsh(vacuum.boundary_energy)[0] > -1e-10
    assert errors[-1] < 2e-6
    assert errors[-1] < errors[0] * 0.2
    # Independently differentiate in PHYSICAL coordinates: Delta* R^2 = 0.
    vacuum = responses[-1]

    def physical(p):
        angle = np.mod(np.arctan2(p[:, 1], p[:, 0] - 2), 2 * np.pi)
        radius = 1 / np.sqrt((np.cos(angle) / 0.5) ** 2 + (np.sin(angle) / 0.8) ** 2)
        radial = (np.linalg.norm(p - [2, 0], axis=1) / radius - 1) / 0.2
        return vacuum.evaluate(radial, angle, g, h)

    step = 2e-4
    value = physical(points)
    rp, rm = physical(points + [step, 0]), physical(points - [step, 0])
    zp, zm = physical(points + [0, step]), physical(points - [0, step])
    star = (rp + rm + zp + zm - 4 * value) / step**2 - (rp - rm) / (
        2 * step * points[:, 0]
    )
    assert np.sqrt(np.mean(star**2)) < 0.02


def test_periodicity_and_manufactured_linear_flux(responses):
    vacuum = responses[-1]
    ip, op = vacuum.points[vacuum.inner], vacuum.points[vacuum.outer]
    g, h = ip[:, 1], op[:, 1]
    s = np.linspace(0, 1, 15)
    for order in (0, 1, 2):
        left = vacuum.evaluate(s, np.zeros_like(s), g, h, (0, order))
        right = vacuum.evaluate(s, np.full_like(s, 2 * np.pi - 1e-10), g, h, (0, order))
        np.testing.assert_allclose(left, right, atol=2e-8)
    theta = vacuum.angular_nodes
    np.testing.assert_allclose(
        vacuum.evaluate(np.zeros_like(theta), theta, g, h), g, atol=2e-6
    )
    np.testing.assert_allclose(
        vacuum.evaluate(np.ones_like(theta), theta, g, h), h, atol=2e-6
    )
    assert not hasattr(vacuum, "stiffness")  # no nodal finite-element stiffness


def test_rectangle_corner_patches_manufactured_solution(responses):
    bounds = [(1, 3), (-1.6, 1.6)]
    corners = np.mod(np.arctan2([1.6, 1.6, -1.6, -1.6], [-1, 1, 1, -1]), 2 * np.pi)
    theta = np.unique(np.r_[np.arange(192) * 2 * np.pi / 192, corners])
    inner, outer, _ = Ellipse().surface(theta, bounds)
    v = mapped_bspf_vacuum(
        Ellipse(),
        bounds,
        theta,
        inner,
        outer,
        radial_modes=31,
        angular_modes=96,
        display_layers=12,
    )
    assert v.diagnostics["vacuum_angular_patches"] == 4
    rng = np.random.default_rng(41)
    t = rng.uniform(0, 2 * np.pi, 100)
    s = rng.uniform(0.05, 0.95, 100)
    ip, op, _ = Ellipse().surface(t, bounds)
    physical = (1 - s[:, None]) * ip + s[:, None] * op
    actual = v.evaluate(s, t, inner[:, 0] ** 2, outer[:, 0] ** 2)
    assert la.norm(actual - physical[:, 0] ** 2) / la.norm(physical[:, 0] ** 2) < 3e-5
    # Trace is continuous across patch interfaces; reference derivative may jump.
    for angle in corners:
        left = v.evaluate(0.4, angle - 1e-9, inner[:, 0] ** 2, outer[:, 0] ** 2)
        right = v.evaluate(0.4, angle + 1e-9, inner[:, 0] ** 2, outer[:, 0] ** 2)
        np.testing.assert_allclose(left, right, atol=2e-7)
