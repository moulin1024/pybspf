"""Curved-domain geometry and independent Poisson accuracy checks."""

import jax
import numpy as np
import pytest

from bspf_models.elliptic.embedded_poisson import benchmark_domains
from bspf_models.elliptic.embedded_poisson import sample_domain
from bspf_models.elliptic.embedded_poisson import solve_poisson
from bspf_models.elliptic.embedded_poisson import error_metrics
from bspf_models.elliptic.embedded_poisson import manufactured

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("domain", benchmark_domains(), ids=lambda d: d.name)
def test_exact_curve_geometry_and_slice_moments(domain):
    for derivative in [0, 1, 2]:
        np.testing.assert_allclose(
            domain.curve(0, derivative),
            domain.curve(domain.period - 1e-10, derivative),
            atol=2e-9,
        )
    p, w, count = domain.volume_rule(20)
    edge, bw, n = domain.boundary_rule(24)
    # Independent Green-theorem integrals, including asymmetric first moments.
    reference = [
        np.sum(bw * np.sum(edge * n, axis=1)) / 2,
        np.sum(bw * edge[:, 0] ** 2 * n[:, 0]) / 2,
        np.sum(bw * edge[:, 1] ** 2 * n[:, 1]) / 2,
    ]
    actual = [w.sum(), w @ p[:, 0], w @ p[:, 1]]
    np.testing.assert_allclose(actual, reference, atol=3e-9, rtol=2e-9)
    geometry = domain.geometry_checks()
    if domain.name == "nonstar":
        assert geometry["nonstar_witness"]
        assert geometry["kernel_lp_status"] == 2
        assert count == 2
    else:
        assert geometry["curvature_min"] > 0
        assert not geometry["nonstar_witness"]
        assert count == 1


@pytest.fixture(scope="module", params=benchmark_domains(), ids=lambda d: d.name)
def sampled(request):
    return sample_domain(request.param, modes=10, order=12), sample_domain(
        request.param, modes=10, order=16
    )


def test_nitsche_convergence_on_independent_quadrature(sampled):
    assembly, check = sampled
    coarse, _ = solve_poisson(assembly, 6)
    fine, diagnostics = solve_poisson(assembly, 10)
    a, b = error_metrics(check, coarse), error_metrics(check, fine)
    assert b["relative_l2"] < 0.5 * a["relative_l2"]
    assert b["relative_h1_seminorm"] < 0.6 * a["relative_h1_seminorm"]
    # This inexpensive test uses only ten modes per direction; demand sub-1%
    # error on BOTH geometries and a separate, stronger convergence ratio.
    assert b["relative_l2"] < 0.01
    assert b["boundary_rms"] < 0.01
    assert b["boundary_rms"] < 0.5 * a["boundary_rms"]
    assert diagnostics["matrix_min_eigenvalue"] > 0
    assert diagnostics["linear_relative_residual"] < 1e-10
    # Reassembling at independent quadrature must preserve the physical field.
    finer, _ = solve_poisson(check, 10)
    difference = check.basis[0] @ (fine - finer)
    exact = manufactured(check.points)[0]
    relative = np.sqrt(
        np.sum(check.weights * difference**2) / np.sum(check.weights * exact**2)
    )
    assert relative < 2e-5


def test_affine_solution_and_poisson_sign(sampled):
    _, check = sampled

    def affine(points):
        x, y = points.T
        return (
            1 + 0.2 * x - 0.3 * y,
            np.tile([0.2, -0.3], (len(x), 1)),
            np.zeros_like(x),
        )

    coeff, _ = solve_poisson(check, 10, data=affine)
    assert error_metrics(check, coeff, data=affine)["relative_l2"] < 0.003
    # Green's identity for the original manufactured data tests the -Delta sign,
    # physical normals, and volume/boundary rules without calling the solver.
    _, grad, forcing = manufactured(check.points)
    x, y = check.points.T
    edge = check.boundary
    ep, ew, en = check.domain.boundary_rule(16)
    np.testing.assert_allclose(edge, ep)
    edge_grad = manufactured(edge)[1]
    v = 1 + x * x + y
    ev = 1 + edge[:, 0] ** 2 + edge[:, 1]
    volume = np.sum(check.weights * (2 * x * grad[:, 0] + grad[:, 1] - v * forcing))
    boundary = np.sum(ew * ev * np.sum(edge_grad * en, axis=1))
    assert abs(volume - boundary) < 3e-8
