"""! @file test_time_integration.py
@brief Tests for the package fixed-step RK4 helper.
"""

from __future__ import annotations

import numpy as np

from pybspf import integrate_rk4


def test_integrate_rk4_matches_exponential_decay():
    """! @brief RK4 should reproduce a smooth scalar ODE accurately on sampled outputs."""
    t_eval = np.array([0.0, 0.1, 0.35, 0.7, 1.0], dtype=np.float64)
    y0 = np.array([1.0], dtype=np.float64)

    history = integrate_rk4(lambda t, y: -y, y0, t_eval, dt=0.05)

    expected = np.exp(-t_eval)[:, None]
    np.testing.assert_allclose(history, expected, rtol=1.0e-6, atol=1.0e-8)


def test_integrate_rk4_applies_post_step_projection():
    """! @brief The optional post-step hook should run after each internal RK4 step."""
    t_eval = np.array([0.0, 0.4, 1.0], dtype=np.float64)
    y0 = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    history = integrate_rk4(
        lambda t, y: np.ones_like(y),
        y0,
        t_eval,
        dt=0.25,
        post_step=lambda t, y: np.array([0.0, y[1], 0.0], dtype=y.dtype),
    )

    np.testing.assert_allclose(history[:, 0], 0.0)
    np.testing.assert_allclose(history[:, 2], 0.0)
    np.testing.assert_allclose(history[:, 1], t_eval + 1.0)
