"""Array annotations for the real and complex CPU/GPU numerical API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import cupy

# A forward reference keeps CuPy optional at runtime.
Array = Union[npt.NDArray[np.float64], npt.NDArray[np.complex128], "cupy.ndarray"]

__all__ = ["Array"]
