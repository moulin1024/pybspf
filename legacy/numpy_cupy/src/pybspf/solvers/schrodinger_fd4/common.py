from __future__ import annotations

import math

import numpy as np

PINFO_INNER = 1
PINFO_BOUNDARY = 2
PINFO_GHOST = 3


def circular_theta(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.mod(np.arctan2(y, x), 2.0 * math.pi)


def default_circular_params() -> dict[str, float]:
    return {
        "rhomin": 0.2,
        "rhomax": 0.4,
    }
