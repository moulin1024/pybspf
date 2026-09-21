import jax
import numpy as np
import pytest

from bspf_models.elliptic.convex_poisson_grid import ConvexPoissonGridPlan
from bspf_models.elliptic.convex_poisson import ConvexPoissonPlan
from bspf_models.elliptic.convex_poisson_grid import _closed_domain_mask
from bspf_models.elliptic.embedded_poisson import benchmark_domains

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def reference():
    return ConvexPoissonPlan(
        benchmark_domains()[0], nodes=17, volume_order=10, boundary_count=128
    )


def test_fixed_grid_matches_reference_and_reuses_factors(reference, monkeypatch):
    x = np.linspace(-1.2, 1.2, 29)
    y = np.linspace(-1.2, 1.2, 35)
    grid = ConvexPoissonGridPlan.from_plan(reference, x, y)
    # Owning immutable axis copies protects the cached basis from caller edits.
    x[:] = 0
    y[:] = 0

    def exact(p):
        return 1 + 0.2 * p[:, 0] + 0.3 * p[:, 1] + 0.1 * p[:, 0] ** 2

    def forcing(p):
        return np.full(len(p), -0.2)

    ref = reference.solve(forcing, exact)
    xx, yy = np.meshgrid(grid._output.x, grid._output.y)
    points = np.column_stack((xx[grid._output.inside], yy[grid._output.inside]))
    expected = ref.evaluate(points)[0]

    def unexpected(*args, **kwargs):
        pytest.fail("Repeated grid solves must not rebuild output basis factors")

    monkeypatch.setattr("bspf_models.elliptic.convex_poisson_grid.stream_evaluate_line", unexpected)
    result = grid.solve(forcing, exact)
    assert result.values.shape == (35, 29)
    assert np.isnan(result.values[~result.inside]).all()
    assert not hasattr(result, "coefficients")
    assert not result.x.flags.writeable and not result.inside.flags.writeable
    np.testing.assert_allclose(result.values[result.inside], expected, atol=3e-12)
    np.testing.assert_allclose(result.values[result.inside], exact(points), atol=1e-8)
    shifted = grid.solve(forcing, lambda p: exact(p) + 0.7)
    np.testing.assert_allclose(
        shifted.values[result.inside], exact(points) + 0.7, atol=1e-8
    )
    assert result.diagnostics["output_basis_storage_bytes"] == (29 + 35) * 17 * 8
    assert result.diagnostics["backend"] == "dense_reference_svd"


def test_grid_mask_includes_exact_tangencies():
    domain = benchmark_domains()[0]
    # This benchmark's four extremal points occur at integer parameters.
    p = domain.curve(np.arange(domain.period))
    xmin, xmax = p[:, 0].min(), p[:, 0].max()
    ymin, ymax = p[:, 1].min(), p[:, 1].max()
    x = np.array([-1.2, xmin, 0, xmax, 1.2])
    y = np.array([-1.2, ymin, 0, ymax, 1.2])
    mask = _closed_domain_mask(domain, x, y)
    expected = np.zeros((5, 5), bool)
    expected[2, 1:4] = True
    expected[1:4, 2] = True
    np.testing.assert_array_equal(mask, expected)


def test_grid_rejects_invalid_axes_before_solver_setup(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid grids must fail before expensive solver setup")

    monkeypatch.setattr("bspf_models.elliptic.convex_poisson_grid.ConvexPoissonPlan", unexpected)
    domain = benchmark_domains()[0]
    for x, y in (
        ([], [0]),
        ([0, 0], [0]),
        ([np.nan], [0]),
        ([1.3], [0]),
        ([[0]], [0]),
        ([1.1], [1.1]),
    ):
        with pytest.raises(ValueError):
            ConvexPoissonGridPlan(domain, x, y)


def test_nonsquare_contractions_and_single_sample(reference):
    # Exercise both contraction orders, nonuniform axes and one-point output.
    rng = np.random.default_rng(32)
    c = rng.normal(size=17**2)
    for x, y in (([-0.5, -0.12, 0.1, 0.6], [-0.2, 0.3]), ([0.1], [0.2])):
        grid = ConvexPoissonGridPlan.from_plan(reference, x, y)
        output = grid._output
        values = output.values(c)
        # Explicit Kronecker rows independently check orientation and ordering.
        expected = np.einsum("ai,bj,ij->ba", output.bx, output.by, c.reshape(17, 17))
        np.testing.assert_allclose(
            values[output.inside], expected[output.inside], atol=3e-13
        )
