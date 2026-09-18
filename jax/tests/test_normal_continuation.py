import numpy as np
import pytest

from bspf_jax.embedded_poisson import benchmark_domains
from bspf_jax.normal_continuation import NormalContinuation, chart


@pytest.mark.parametrize("domain", benchmark_domains(), ids=lambda d: d.name)
def test_chart_and_cartesian_derivatives(domain):
    t = np.arange(domain.period) + 0.371
    r = np.full_like(t, 0.003)
    p, jac, second = chart(domain, t, r)
    h = 1e-5
    plus, minus = chart(domain, t + h, r), chart(domain, t - h, r)
    np.testing.assert_allclose((plus[0] - minus[0]) / (2 * h), jac[..., 0], atol=1e-9)
    np.testing.assert_allclose(
        (plus[1][..., 0] - minus[1][..., 0]) / (2 * h), second[..., :, 0, 0], atol=1e-8
    )

    def polynomial(p):
        return p[:, 0] ** 2 + 2 * p[:, 0] * p[:, 1] + 3 * p[:, 1] ** 2

    ext = NormalContinuation(
        domain, polynomial, width=0.01, normal_degree=6, tangent_degree=36
    )
    u, grad, lap = ext.evaluate(t, r)
    np.testing.assert_allclose(u, polynomial(p), atol=2e-10)
    np.testing.assert_allclose(
        grad,
        np.column_stack((2 * p[:, 0] + 2 * p[:, 1], 2 * p[:, 0] + 6 * p[:, 1])),
        atol=2e-7,
    )
    np.testing.assert_allclose(lap, 8, atol=2e-4)
    wrapped = ext.evaluate(np.array([-1e-100, 0.0, domain.period]), 0.003)
    for field in wrapped:
        assert np.all(np.isfinite(field))
        np.testing.assert_allclose(field[0], field[1], atol=2e-4)
