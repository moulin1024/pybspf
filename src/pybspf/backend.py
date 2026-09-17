"""! @file backend.py
@brief Backend selection and explicit host/device conversion helpers.

This module centralizes optional CuPy support and the strict conversion rules
between CPU and GPU arrays.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg as sla

_HAS_CUPY = False
try:
    import cupy as cp
    import cupyx.scipy.interpolate as cp_interp
    import cupyx.scipy.linalg as cpla

    _HAS_CUPY = True
except (ImportError, OSError):
    cp = None
    cpla = None
    cp_interp = None


def get_array_module(*, use_gpu: bool = False):
    """Return NumPy or CuPy, failing explicitly if the GPU stack is missing."""
    if use_gpu:
        if not _HAS_CUPY:
            raise RuntimeError(
                "CuPy is not available. Install a CuPy build matching your CUDA "
                "runtime or set use_gpu=False."
            )
        return cp
    return np


def is_cupy_array(a) -> bool:
    """! @brief Return whether an object is a CuPy array.

    @param a Object to inspect.
    @return ``True`` when CuPy is available and ``a`` is a CuPy array.
    """
    return bool(_HAS_CUPY and isinstance(a, cp.ndarray))


def validate_backend_array(a, *, use_gpu: bool, name: str = "array") -> None:
    """! @brief Validate that an array matches the requested backend.

    @param a Array-like object to validate.
    @param use_gpu Whether GPU storage is required.
    @param name Human-readable name used in error messages.
    @throws RuntimeError If GPU mode is requested but CuPy is unavailable.
    @throws ValueError If the array type does not match the requested backend.
    """
    if use_gpu:
        if not _HAS_CUPY:
            raise RuntimeError(
                f"{name} requested GPU mode but CuPy is not available. "
                "Install cupy or set use_gpu=False."
            )
        if not is_cupy_array(a):
            raise ValueError(
                f"Cannot use NumPy array in {name} when use_gpu=True. "
                "Either: (1) convert input to CuPy array, or (2) set use_gpu=False."
            )
        return

    if is_cupy_array(a):
        raise ValueError(
            f"Cannot use CuPy array in {name} when use_gpu=False. "
            "Either: (1) convert input to NumPy array, or (2) set use_gpu=True."
        )


def normalize_backend_array(a, *, use_gpu: bool, dtype=np.float64, name: str = "array"):
    """! @brief Convert an array-like object onto the requested backend explicitly.

    @param a Input array-like object.
    @param use_gpu Whether the output should live on the GPU.
    @param dtype Target dtype used during conversion.
    @param name Human-readable name used in error messages.
    @return NumPy or CuPy array on the requested backend.
    @throws RuntimeError If GPU mode is requested but CuPy is unavailable.
    """
    if use_gpu:
        if not _HAS_CUPY:
            raise RuntimeError(
                f"{name} requested GPU mode but CuPy is not available. "
                "Install cupy or set use_gpu=False."
            )
        return cp.asarray(a, dtype=dtype)
    if is_cupy_array(a):
        raise ValueError(
            f"Cannot use CuPy array in {name} when use_gpu=False. "
            "Either: (1) convert input to NumPy array, or (2) set use_gpu=True."
        )
    return np.asarray(a, dtype=dtype)


class _Backend:
    """! @brief Switch between NumPy/SciPy and CuPy/CuPyX cleanly.

    @param use_gpu If ``True``, require the CuPy stack and expose GPU-backed
        array, FFT, and linear-algebra modules. Otherwise use NumPy/SciPy.
    """

    __slots__ = ("xp", "la", "fft", "is_gpu")

    def __init__(self, use_gpu: bool):
        if use_gpu:
            if not _HAS_CUPY:
                raise RuntimeError(
                    "use_gpu=True but CuPy is not available. "
                    "Install cupy (e.g. `pip install cupy-cuda12x`) or set use_gpu=False."
                )
            self.xp = cp
            self.la = cpla
            self.fft = cp.fft
            self.is_gpu = True
        else:
            self.xp = np
            self.la = sla
            self.fft = np.fft
            self.is_gpu = False

    def to_device(self, a):
        """! @brief Explicitly move an array-like object onto the GPU.

        @param a Input array-like object.
        @return CuPy array when GPU mode is active.
        @throws ValueError If a CuPy array is supplied while running in CPU mode.
        """
        if self.is_gpu:
            return normalize_backend_array(a, use_gpu=True, dtype=cp.float64, name="array")
        # In CPU mode, normalize inputs to NumPy and reject accidental CuPy use
        # so device transfers always remain explicit.
        return normalize_backend_array(a, use_gpu=False, dtype=np.float64, name="array")

    def to_host(self, a):
        """! @brief Explicitly move an array-like object onto the CPU.

        @param a Input array-like object.
        @return NumPy array or the original CPU-backed object.
        @throws ValueError If the array/backend combination is inconsistent.
        """
        if self.is_gpu:
            if is_cupy_array(a):
                return cp.asnumpy(a)
            raise ValueError(
                "Inconsistency detected: use_gpu=True but array is NumPy. "
                "Arrays must match the use_gpu setting. Use to_device() for explicit conversion."
            )
        if is_cupy_array(a):
            raise ValueError(
                "Inconsistency detected: use_gpu=False but array is CuPy. "
                "Arrays must match the use_gpu setting. Use to_host() for explicit conversion."
            )
        return a

    def ensure_like_input(self, out, input_was_numpy: bool):
        """! @brief Enforce the package rule that GPU mode must stay on-device.

        @param out Computed output array.
        @param input_was_numpy Whether the original user input came from NumPy.
        @return ``out`` unchanged when the device contract is satisfied.
        @throws ValueError If GPU output would need to be implicitly converted.
        """
        if self.is_gpu and input_was_numpy:
            raise ValueError(
                "Cannot convert GPU results back to NumPy. "
                "When use_gpu=True, provide CuPy arrays as input to avoid GPU↔CPU conversions. "
                "Either: (1) convert input to CuPy array before calling, or (2) use use_gpu=False."
            )
        return out

    def validate_device(self, a, name: str = "array"):
        """! @brief Validate that an array matches the configured backend.

        @param a Array object to validate.
        @param name Human-readable name included in error messages.
        @throws ValueError If the array does not match the configured device.
        """
        if self.is_gpu and not is_cupy_array(a):
            raise ValueError(
                f"Inconsistency detected: use_gpu=True but {name} is not a CuPy array. "
                f"Arrays must match the use_gpu setting. Use to_device() for explicit conversion."
            )
        if not self.is_gpu and is_cupy_array(a):
            raise ValueError(
                f"Inconsistency detected: use_gpu=False but {name} is a CuPy array. "
                f"Arrays must match the use_gpu setting. Use to_host() for explicit conversion."
            )

__all__ = [
    "get_array_module",
    "_Backend",
    "_HAS_CUPY",
    "cp",
    "cpla",
    "cp_interp",
    "is_cupy_array",
    "normalize_backend_array",
    "validate_backend_array",
]
