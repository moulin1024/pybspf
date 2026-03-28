"""! @file types.py
@brief Shared typing helpers used across the package.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Canonical dense real-valued array type used in the current API surface.
Array = npt.NDArray[np.float64]

__all__ = ["Array"]
