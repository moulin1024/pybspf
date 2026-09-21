"""! @file time_integration.py
@brief Fixed-step time-integration helpers for BSPF examples and workflows.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy as np

Array = np.ndarray
State = TypeVar("State", bound=Array)


def integrate_rk4(
    rhs: Callable[[float, State], State],
    y0: State,
    t_eval: Array,
    *,
    dt: float,
    post_step: Callable[[float, State], State] | None = None,
) -> Array:
    """Integrate an ODE with classical RK4 and sample on ``t_eval``."""
    times = np.asarray(t_eval, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("t_eval must be a non-empty 1D array.")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("t_eval must be nondecreasing.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    state0 = np.array(y0, copy=True)
    history = np.empty((times.size,) + state0.shape, dtype=state0.dtype)
    history[0] = state0

    state = state0
    current_time = float(times[0])

    for i in range(times.size - 1):
        target_time = float(times[i + 1])
        while current_time < target_time - 1.0e-15:
            step = min(float(dt), target_time - current_time)
            k1 = rhs(current_time, state)
            k2 = rhs(current_time + 0.5 * step, state + 0.5 * step * k1)
            k3 = rhs(current_time + 0.5 * step, state + 0.5 * step * k2)
            k4 = rhs(current_time + step, state + step * k3)
            state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            current_time += step
            if post_step is not None:
                state = np.array(post_step(current_time, state), copy=False)

        history[i + 1] = state

    return history


integrate_fixed_step_rk4 = integrate_rk4

__all__ = ["integrate_rk4"]
