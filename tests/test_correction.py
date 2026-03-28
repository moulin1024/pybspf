"""! @file test_correction.py
@brief Regression tests for the extracted residual correction strategies.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybspf.correction import ResidualCorrection
from bspf1d import ResidualCorrection as LegacyResidualCorrection


def test_none_correction_returns_zeros():
    """! @brief The no-op correction should always return a zero vector."""
    residual = np.linspace(-1.0, 1.0, 11)
    omega = 2.0 * np.pi * np.fft.rfftfreq(residual.size, d=0.2)

    out = ResidualCorrection.none(residual, omega, kind="diff", order=1, n=residual.size)

    np.testing.assert_allclose(out, np.zeros_like(residual))


def test_spectral_diff_matches_legacy():
    """! @brief Spectral differentiation should match the legacy implementation."""
    x = np.linspace(0.0, 2.0 * np.pi, 64)
    residual = np.sin(2.0 * x) + 0.25 * np.cos(5.0 * x)
    omega = 2.0 * np.pi * np.fft.rfftfreq(residual.size, d=x[1] - x[0])

    new_out = ResidualCorrection.spectral(residual, omega, kind="diff", order=2, n=residual.size, x=x)
    old_out = LegacyResidualCorrection.spectral(residual, omega, kind="diff", order=2, n=residual.size, x=x)

    np.testing.assert_allclose(new_out, old_out)


def test_spectral_integral_matches_legacy():
    """! @brief Spectral integration with nullspace handling should match legacy output."""
    x = np.linspace(-1.0, 1.0, 51)
    residual = np.cos(3.0 * x) + 0.1
    omega = 2.0 * np.pi * np.fft.rfftfreq(residual.size, d=x[1] - x[0])

    new_out = ResidualCorrection.spectral(residual, omega, kind="int", order=1, n=residual.size, x=x)
    old_out = LegacyResidualCorrection.spectral(residual, omega, kind="int", order=1, n=residual.size, x=x)

    np.testing.assert_allclose(new_out, old_out)


def test_spectral_integral_rejects_unsupported_order():
    """! @brief Only first and second antiderivatives are currently supported."""
    residual = np.linspace(0.0, 1.0, 9)
    omega = 2.0 * np.pi * np.fft.rfftfreq(residual.size, d=0.25)

    with pytest.raises(ValueError, match="Only int orders 1 and 2"):
        ResidualCorrection.spectral(residual, omega, kind="int", order=3, n=residual.size)
