"""Geometry/adjoint/normal-ray checks for the unknown-field extension prototype."""

import jax
import numpy as np
import pytest

from bspf_models.elliptic.embedded_poisson import benchmark_domains
from bspf_models.elliptic.smooth_extension import extension_line
from bspf_models.elliptic.smooth_extension import basis_operators
from bspf_models.elliptic.smooth_extension import tensor_inverse

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def line():
    return extension_line(33)


@pytest.mark.parametrize("domain", benchmark_domains(), ids=lambda d: d.name)
def test_analytic_spline_normal_and_bspf_normal_ray_derivatives(line, domain):
    p, w, n = domain.boundary_rule(3)
    # Test a selection spanning knot spans, curved arcs and straight segments.
    p, n, w = p[::3], n[::3], w[::3]
    np.testing.assert_allclose(np.sum(n * n, axis=1), 1, atol=5e-15)
    v, _, dn, dnn = basis_operators(line, p, n)
    step = 2e-5
    plus = basis_operators(line, p + step * n)[0]
    minus = basis_operators(line, p - step * n)[0]
    assert np.linalg.norm((plus - minus) / (2 * step) - dn) / np.linalg.norm(dn) < 2e-6
    assert (
        np.linalg.norm((plus + minus - 2 * v) / step**2 - dnn) / np.linalg.norm(dnn)
        < 2e-5
    )
    # Same analytic normals in interpolation and its weighted adjoint spread.
    trace = np.sqrt(w[:, None]) * dn
    rng = np.random.default_rng(83)
    u, force = rng.normal(size=trace.shape[1]), rng.normal(size=trace.shape[0])
    np.testing.assert_allclose(
        (trace @ u) @ force, u @ (trace.T @ force), rtol=2e-14, atol=1e-11
    )


def test_shared_tensor_inverse_multiple_rhs(line):
    lam = np.asarray(line.lam)
    denominator = lam[:, None] + lam[None, :]
    rng = np.random.default_rng(17)
    rhs = rng.normal(size=(denominator.size, 4))
    value = tensor_inverse(rhs, denominator)
    np.testing.assert_allclose(
        denominator.ravel()[:, None] * value, rhs, rtol=3e-15, atol=1e-15
    )
