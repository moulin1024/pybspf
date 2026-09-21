import jax
import numpy as np
import pytest

from bspf_models.elliptic.convex_poisson import ArcLengthBoundary
from bspf_models.elliptic.convex_poisson import ConvexPoissonPlan
from bspf_models.elliptic.convex_poisson import trace_transform
from bspf_models.elliptic.convex_poisson import validate_convex
from bspf_models.elliptic.embedded_poisson import benchmark_domains

jax.config.update("jax_enable_x64", True)


def test_convex_validation_and_arclength():
    domain, concave = benchmark_domains()
    assert abs(validate_convex(domain)["total_turning"] - 2 * np.pi) < 1e-10
    with pytest.raises(ValueError, match="convex"):
        validate_convex(concave)
    arc = ArcLengthBoundary(domain)
    fine = ArcLengthBoundary(domain, 48)
    points, t = arc.sample(64, offset=0.371)
    np.testing.assert_allclose(points, fine.sample(64, offset=0.371)[0], atol=3e-14)
    s = np.array([arc.offsets[int(a)] + arc.integral(int(a), a % 1) for a in t])
    np.testing.assert_allclose(s, (np.arange(64) + 0.371) * arc.length / 64, atol=3e-14)


def test_fractional_trace_norm():
    length = 3.7
    count = 128
    mode = 7
    x = np.arange(count) / count
    values = np.cos(2 * np.pi * mode * x)
    for order in (0.0, 1.5):
        norm = np.linalg.norm(trace_transform(values, length, order)) ** 2
        np.testing.assert_allclose(
            norm,
            length / 2 * (1 + (2 * np.pi * mode / length) ** 2) ** order,
            rtol=1e-13,
        )
    np.testing.assert_allclose(
        np.linalg.norm(trace_transform(np.ones(count), length)) ** 2, length, rtol=1e-14
    )


def test_poisson_nonzero_dirichlet_quadratic():
    domain = benchmark_domains()[0]
    plan = ConvexPoissonPlan(domain, nodes=33, volume_order=12, boundary_count=256)
    # Independent tensor quadrature checks the full metric, including xy.
    line = plan.line
    b, g, h = map(np.asarray, (line.b, line.g, line.h))
    w = np.asarray(line.weights)
    c = np.random.default_rng(4).normal(size=(b.shape[1], b.shape[1]))
    direct = sum(
        np.sum(w[:, None] * w[None, :] * (a @ c @ z.T) ** 2)
        for a, z in ((b, b), (g, b), (b, g), (h, b), (np.sqrt(2) * g, g), (b, h))
    )
    np.testing.assert_allclose(
        np.linalg.norm(plan.root @ c.ravel()) ** 2, direct, rtol=2e-10
    )

    def exact(p):
        return 1 + 0.2 * p[:, 0] + 0.3 * p[:, 1] + 0.1 * p[:, 0] ** 2

    solution = plan.solve(lambda p: np.full(len(p), -0.2), exact)
    p = np.array([[0.01, 0.03], [-0.21, 0.14], [0.36, -0.2], [0.6, 0.1]])
    u, g, lap = solution.evaluate(p)
    np.testing.assert_allclose(u, exact(p), atol=1e-8)
    np.testing.assert_allclose(
        g, np.column_stack((0.2 + 0.2 * p[:, 0], np.full(len(p), 0.3))), atol=1e-6
    )
    np.testing.assert_allclose(lap, 0.2, atol=1e-6)
    expected = np.zeros((len(p), 2, 2))
    expected[:, 0, 0] = 0.2
    np.testing.assert_allclose(solution.hessian(p), expected, atol=1e-6)
    assert solution.diagnostics["physical_data_only"]
