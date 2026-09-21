"""Regression coverage for public NumPy/CuPy operator contracts."""

import numpy as np
import pytest

from pybspf import BSPF1D, BSPF2D, Grid1D, PiecewiseBSPF1D


@pytest.fixture
def op():
    return BSPF1D.from_grid(degree=5, x=np.linspace(0, 1, 65), n_basis=20)


@pytest.mark.parametrize("x, message", [
    (np.zeros((2, 2)), "1D"), ([0, 0, 0], "strictly increasing"),
    ([2, 1, 0], "strictly increasing"), ([0, np.nan], "finite"),
    ([0, np.inf], "finite"), ([0, 1j], "real"),
])
def test_invalid_grids(x, message):
    with pytest.raises(ValueError, match=message):
        Grid1D(x)


@pytest.mark.parametrize("f", [1, np.ones(64), np.ones((65, 1))])
def test_signal_shape_validation(op, f):
    with pytest.raises(ValueError, match="shape"):
        op.derivatives(f, orders=1)


@pytest.mark.parametrize("orders", [[], [1.5], 1.5, True, [0], [5]])
def test_derivative_orders_are_validated(op, orders):
    with pytest.raises(ValueError):
        op.derivatives(np.ones(65), orders=orders)


def test_complex_fit_and_batched_derivatives(op):
    x = op.grid.x
    f = np.sin(3*x) + 1j * np.cos(2*x)
    p, spline, residual = op.fit_spline(f, lam=0.01)
    assert np.iscomplexobj(p)
    np.testing.assert_allclose(spline + residual, f, atol=1e-14)
    batch = np.stack([f, 2j*f], axis=1)
    result = op.derivatives_batched(batch, orders=(1, 2), lam=0.01)
    single = op.derivatives(f, orders=(1, 2), lam=0.01)
    for k in (1, 2):
        np.testing.assert_allclose(result[k][:, 0], single[k], atol=1e-9)
        np.testing.assert_allclose(result[k][:, 1], 2j*single[k], atol=1e-9)


def test_empty_batch(op):
    result = op.derivatives_batched(np.empty((65, 0)), orders=(1, 2))
    assert result.spline.shape == result[1].shape == result[2].shape == (65, 0)


def test_repeated_refinement_uses_requested_grid(op):
    f = np.sin(3*op.grid.x)
    for factor in (2, 3, 1, 4, 2):
        x, values, spline, residual = op.interpolate_split_mesh(f, factor, lam=0.01)
        assert x.shape == values.shape == (factor*64+1,)
        np.testing.assert_allclose(values[::factor], f, atol=1e-12)
        np.testing.assert_allclose(spline + residual, values)


def test_basis_evaluation_does_not_cache_by_first_coordinate(op):
    for x in (np.array([0, .2, 1]), np.array([0, .4, .8])):
        actual = op.basis._evaluate_splines_vectorized(x)
        expected = np.stack([s(x) for s in op.basis._splines])
        np.testing.assert_allclose(actual, expected)


def test_partial_integral_is_additive_and_oriented(op):
    f = np.sin(13*op.grid.x) + op.grid.x
    left = op.definite_integral(f, 0, .413, lam=.1)
    right = op.definite_integral(f, .413, 1, lam=.1)
    total = op.definite_integral(f, lam=.1)
    np.testing.assert_allclose(left + right, total, atol=1e-13)
    np.testing.assert_allclose(op.definite_integral(f, .413, 0, lam=.1), -left, atol=1e-13)
    assert op.definite_integral(f, .413, .413, lam=.1) == 0


def test_no_correction_means_spline_derivative():
    op = BSPF1D.from_grid(degree=3, x=np.linspace(0, 1, 33), correction="none")
    f = np.sin(10*op.grid.x)
    p, _, _ = op.fit_spline(f, lam=.1)
    result = op.derivatives(f, orders=(1, 2), lam=.1)
    for k in (1, 2):
        np.testing.assert_allclose(result[k], op.basis.BkT(k) @ p)


def test_neumann_needs_first_derivative_constraint():
    op = BSPF1D.from_grid(degree=3, order=1, x=np.linspace(0, 1, 33))
    with pytest.raises(ValueError, match="order >= 2"):
        op.fit_spline(np.ones(33), neumann_bc=(0, 0))


@pytest.mark.parametrize("lam", [-1, np.nan, np.inf])
def test_regularization_is_validated(op, lam):
    with pytest.raises(ValueError, match="lam"):
        op.fit_spline(np.ones(65), lam=lam)


def test_piecewise_preserves_complex_data():
    x = np.linspace(0, 1, 65)
    op = PiecewiseBSPF1D(3, x, breakpoints=[.5])
    result = op.derivatives(x + 2j*x, orders=1, lam=1e-6)
    np.testing.assert_allclose(result[1], 1 + 2j, atol=1e-3)


@pytest.fixture
def cupy():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    return cp


@pytest.mark.gpu
@pytest.mark.parametrize("complex_signal", [False, True])
def test_gpu_parity(cupy, complex_signal):
    cp = cupy
    x = np.linspace(0, 1, 65)
    cpu = BSPF1D.from_grid(5, x, n_basis=20)
    gpu = BSPF1D.from_grid(5, cp.asarray(x), knots=cp.asarray(cpu.knots), use_gpu=True)
    f = np.sin(3*x)
    if complex_signal:
        f = f + 1j*np.cos(2*x)
    batch = np.stack([f, 2*f], axis=1)
    expected = cpu.derivatives_batched(batch, (1, 2), lam=.01)
    actual = gpu.derivatives_batched(cp.asarray(batch), (1, 2), lam=.01)
    for k in (1, 2):
        assert isinstance(actual[k], cp.ndarray)
        np.testing.assert_allclose(cp.asnumpy(actual[k]), expected[k], atol=1e-7, rtol=1e-7)
    with pytest.raises(ValueError, match="NumPy"):
        gpu.derivatives(f, 1)
    with pytest.raises(ValueError, match="CuPy"):
        cpu.derivatives(cp.asarray(f), 1)
    if not complex_signal:
        np.testing.assert_allclose(gpu.definite_integral(cp.asarray(f), .2, .8, lam=.01),
                                   cpu.definite_integral(f, .2, .8, lam=.01), atol=1e-9)
        actual_F, _ = gpu.antiderivative(cp.asarray(f), lam=.01)
        expected_F, _ = cpu.antiderivative(f, lam=.01)
        assert isinstance(actual_F, cp.ndarray)
        np.testing.assert_allclose(cp.asnumpy(actual_F), expected_F, atol=1e-9)


@pytest.mark.gpu
def test_gpu_2d(cupy):
    cp = cupy
    x, y = np.linspace(0, 1, 33), np.linspace(0, 1, 35)
    field = np.sin(x)[None, :] + y[:, None]**2
    cpu = BSPF2D.from_grids(x=x, y=y, degree_x=3)
    gpu = BSPF2D.from_grids(x=cp.asarray(x), y=cp.asarray(y), degree_x=3, use_gpu=True)
    actual = gpu.laplacian(cp.asarray(field), lam_x=.01, lam_y=.01)
    assert isinstance(actual, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(actual), cpu.laplacian(field, lam_x=.01, lam_y=.01), atol=1e-7)
