"""Independent dense references and analytic checks for the masked projection."""

import numpy as np
import pytest
import scipy.linalg as la

from pybspf import PressurePoisson2D, PressurePoisson2DResult


def small_solver(**options):
    return PressurePoisson2D(
        np.linspace(-1, 2, 7),
        np.linspace(0, 2, 8),
        q=2,
        n_basis=5,
        degree=4,
        baseline_points=options.pop("baseline_points", 4),
        **options,
    )


def dense_operators(solver):
    nx, ny = solver.x.size, solver.y.size
    dx = np.kron(np.eye(ny), solver._x.D)
    dy = np.kron(solver._y.D, np.eye(nx))
    G = np.stack([dx, dy], axis=1).reshape(2 * nx * ny, nx * ny)
    mask = solver.mask.ravel()
    S = (dx * mask) @ dx + (dy * mask) @ dy
    return G, S


@pytest.mark.parametrize(
    "options",
    [{}, dict(endpoint_method="chebyshev", chebyshev_modes=5, baseline_points=6)],
)
def test_against_independent_dense_pseudoinverse_and_wall_completion(options):
    solver = small_solver(**options)
    G, S = dense_operators(solver)
    raw = np.random.default_rng(14).standard_normal(solver.shape + (2,))
    rhs = solver.divergence(solver.mask[..., None] * raw)
    p0 = la.lstsq(S, rhs.ravel(), cond=1e-11)[0]
    core = np.repeat(solver.mask.ravel().astype(bool), 2)
    Z = la.null_space(G[core], rcond=1e-11)
    assert Z.shape[1] == 8
    wall_gradient = G[~core]
    c = la.lstsq(
        wall_gradient @ Z, raw.ravel()[~core] - wall_gradient @ p0, cond=1e-11
    )[0]
    expected = solver.remove_mean((p0 + Z @ c).reshape(solver.shape))
    projected, result = solver.project(raw)
    assert isinstance(result, PressurePoisson2DResult)
    np.testing.assert_allclose(result.pressure, expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(S @ result.pressure.ravel(), rhs.ravel(), atol=1e-9)
    np.testing.assert_allclose(
        projected,
        solver.mask[..., None] * (raw - (G @ expected.ravel()).reshape(raw.shape)),
        atol=1e-9,
    )
    assert abs(solver.divergence(projected)).max() < 1e-9
    assert np.count_nonzero(projected[solver.walls]) == 0


@pytest.mark.parametrize("nx,ny", [(64, 64), (48, 65)])
def test_analytic_gradient_on_rectangles(nx, ny):
    solver = PressurePoisson2D(np.linspace(-0.5, 1.0, nx), np.linspace(0.0, 2.0, ny))
    x, y = np.meshgrid(solver.x, solver.y)
    p = np.exp(0.4 * x + 0.3 * y)
    raw = np.stack([0.4 * p, 0.3 * p], axis=-1)
    projected, result = solver.project(raw)
    expected = solver.remove_mean(p)
    assert la.norm(result.pressure - expected) / la.norm(expected) < 1e-8
    assert abs(projected).max() < 1e-8
    assert result.wall_gradient_fit_linf < 1e-8
    assert abs(np.sum(result.pressure * solver._weights)) < 1e-12


def test_projection_idempotence_and_null_completion():
    solver = small_solver()
    raw = np.random.default_rng(12).standard_normal(solver.shape + (2,))
    v, completed = solver.project(raw)
    uncompleted_v, uncompleted = solver.project(raw, completion=False)
    vv, _ = solver.project(v)
    np.testing.assert_allclose(vv, v, atol=1e-9)
    np.testing.assert_allclose(uncompleted_v, v, atol=1e-9)
    assert la.norm(completed.pressure - uncompleted.pressure) > 1e-3
    change = solver.mask[..., None] * solver.gradient(
        completed.pressure - uncompleted.pressure
    )
    assert abs(change).max() < 1e-10
    assert uncompleted.wall_gradient_fit_linf is None


def test_tensor_inverse_and_compatible_scalar_rhs():
    solver = small_solver()
    rng = np.random.default_rng(41)
    rhs = rng.standard_normal(solver.shape)
    p = solver._tensor_solve(rhs)
    np.testing.assert_allclose(solver.schur(p) + solver._lift(p), rhs, atol=1e-10)
    p = rng.standard_normal(solver.shape)
    result = solver.solve(solver.schur(p), wall_gradient=solver.gradient(p))
    np.testing.assert_allclose(result.pressure, solver.remove_mean(p), atol=1e-9)


def test_incompatible_rhs_is_rejected_including_noncorner_modes():
    solver = small_solver()
    rhs = np.zeros(solver.shape)
    rhs[0, 0] = 1
    with pytest.raises(ValueError, match="incompatible"):
        solver.solve(rhs)
    rhs[:] = 0
    rhs[2, 3] = 1
    with pytest.raises(ValueError, match="incompatible"):
        solver.solve(rhs)


def test_zero_data_and_input_validation():
    solver = small_solver()
    zero = np.zeros(solver.shape)
    np.testing.assert_array_equal(solver.solve(zero).pressure, zero)
    for rhs in [np.zeros((2, 2)), zero + np.nan, zero.astype(complex)]:
        with pytest.raises(ValueError):
            solver.solve(rhs)
    with pytest.raises(ValueError):
        solver.solve(zero, wall_gradient=zero)
    for tol in [-1, np.nan, np.inf]:
        with pytest.raises(ValueError):
            solver.solve(zero, rtol=tol)
    with pytest.raises(ValueError, match="uniform"):
        PressurePoisson2D(np.arange(40) ** 2, np.arange(40))
    with pytest.raises(ValueError, match="n_basis"):
        PressurePoisson2D(np.arange(16), np.arange(16))


def test_chebyshev_jets_polynomial_exactness_and_endpoint_values():
    from math import factorial
    from pybspf.solvers._pressure_bspf import chebyshev_jets

    x = np.linspace(-0.5, 1, 32)
    jets = chebyshev_jets(x, order=4, points=12, modes=8, alpha=0)
    for power in range(8):
        expected = np.array(
            [
                factorial(power) / factorial(power - k) * endpoint ** (power - k)
                if power >= k
                else 0
                for endpoint in [x[0], x[-1]]
                for k in range(4)
            ]
        )
        np.testing.assert_allclose(jets @ x**power, expected, atol=1e-8, rtol=1e-9)
    np.testing.assert_array_equal(jets[0], np.eye(x.size)[0])
    np.testing.assert_array_equal(jets[4], np.eye(x.size)[-1])


def test_chebyshev_improves_oscillatory_pressure():
    x = np.linspace(0, 1, 96)
    xx, yy = np.meshgrid(x, x)
    a, b = 5.3 * np.pi, 3.7 * np.pi
    p = np.sin(a * xx + 0.2) * np.cos(b * yy - 0.1)
    g = np.stack(
        [
            a * np.cos(a * xx + 0.2) * np.cos(b * yy - 0.1),
            -b * np.sin(a * xx + 0.2) * np.sin(b * yy - 0.1),
        ],
        axis=-1,
    )
    errors = []
    for opts in [
        {},
        dict(endpoint_method="chebyshev", chebyshev_modes=14, baseline_points=18),
    ]:
        solver = PressurePoisson2D(x, x, **opts)
        v, result = solver.project(g)
        errors.append(la.norm(result.pressure - solver.remove_mean(p)))
        assert abs(solver.divergence(v)).max() < 1e-7
    assert errors[1] < errors[0] / 1000


@pytest.mark.parametrize(
    "options",
    [
        dict(endpoint_method="unknown"),
        dict(endpoint_method="chebyshev", chebyshev_modes=20),
        dict(endpoint_method="chebyshev", endpoint_regularization=-1),
    ],
)
def test_endpoint_options_validation(options):
    with pytest.raises(ValueError):
        PressurePoisson2D(np.linspace(0, 1, 40), np.linspace(0, 1, 40), **options)
