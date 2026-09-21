"""! @file test_backend.py
@brief Tests for the package backend abstraction and device-contract helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybspf.backend import _Backend, _HAS_CUPY, is_cupy_array, normalize_backend_array, validate_backend_array
from pybspf import BSPF1D


def test_normalize_backend_array_cpu_returns_numpy():
    """! @brief CPU normalization should produce a NumPy float64 array."""
    out = normalize_backend_array([1, 2, 3], use_gpu=False, name="test-array")

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0]))


def test_backend_to_device_cpu_mode_stays_numpy():
    """! @brief CPU-mode backend conversion should stay on NumPy arrays."""
    bk = _Backend(False)
    out = bk.to_device([0.0, 1.0])

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64


def test_is_cupy_array_false_for_numpy():
    """! @brief NumPy arrays must not be reported as CuPy arrays."""
    assert not is_cupy_array(np.arange(4.0))


def test_validate_backend_array_rejects_gpu_without_cupy():
    """! @brief GPU validation should fail clearly when CuPy is unavailable."""
    if _HAS_CUPY:
        pytest.skip("This environment has CuPy; the no-CuPy error path is not active.")

    with pytest.raises(RuntimeError, match="CuPy is not available"):
        validate_backend_array(np.arange(4.0), use_gpu=True, name="gpu-test")


def test_bspf1d_from_grid_rejects_gpu_without_cupy():
    """! @brief Operator construction should fail clearly when GPU mode is requested without CuPy."""
    if _HAS_CUPY:
        pytest.skip("This environment has CuPy; the no-CuPy error path is not active.")

    x = np.linspace(0.0, 1.0, 8)
    with pytest.raises(RuntimeError, match="CuPy is not available"):
        BSPF1D.from_grid(degree=3, x=x, use_gpu=True)
