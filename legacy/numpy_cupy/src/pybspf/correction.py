"""! @file correction.py
@brief Residual correction strategies.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .types import Array


class ResidualCorrection:
    """! @brief Pluggable residual correction strategies for spline residuals."""

    @staticmethod
    def none(
        residual: Array,
        omega: Array,
        *,
        kind: str,
        order: int,
        n: int,
        x: Optional[Array] = None,
    ) -> Array:
        """! @brief Return a zero correction.

        @param residual Residual signal.
        @param omega Frequency grid. Unused in this strategy.
        @param kind Correction kind. Unused in this strategy.
        @param order Differential or integral order. Unused in this strategy.
        @param n Output size.
        @param x Physical coordinates. Unused in this strategy.
        @return Zero-valued correction vector.
        """
        return np.zeros(n, dtype=np.float64)

    @staticmethod
    def spectral(
        residual: Array,
        omega: Array,
        *,
        kind: str,
        order: int,
        n: int,
        x: Optional[Array] = None,
    ) -> Array:
        """! @brief Apply spectral differentiation or integration to the residual.

        @param residual Residual signal sampled on a uniform grid.
        @param omega rFFT frequency grid.
        @param kind Either ``"diff"`` or ``"int"``.
        @param order Requested derivative or antiderivative order.
        @param n Output size.
        @param x Optional physical grid used for integral nullspace handling.
        @return Spectral correction evaluated in physical space.
        """
        # The current residual correction path is defined in NumPy space and is
        # kept identical to the legacy implementation while the package is
        # migrated module by module.
        R = np.fft.rfft(residual)

        if kind == "diff":
            return np.fft.irfft(R * (1j * omega) ** order, n=n).astype(np.float64)

        if kind == "int":
            out_hat = np.zeros_like(R, dtype=np.complex128)
            nz = omega != 0.0
            out_hat[nz] = R[nz] / ((1j * omega[nz]) ** order)
            out = np.fft.irfft(out_hat, n=n).astype(np.float64)

            # Integration leaves a nullspace, so match the legacy code's
            # polynomial corrections for the mean residual component.
            if x is None:
                x0, x1 = 0.0, 1.0
                xx = np.linspace(x0, x1, n)
            else:
                xx = x
                x0 = float(xx[0])
                x1 = float(xx[-1])

            if order == 1:
                mean_r = float(np.mean(residual))
                out = out + mean_r * (xx - x0)
                out -= out[0]
                return out

            if order == 2:
                mean_r = float(np.mean(residual))
                q = 0.5 * mean_r * (xx - x0) * (xx - x1)
                return out + q

            raise ValueError("Only int orders 1 and 2 are supported.")

        raise ValueError("kind must be 'diff' or 'int'.")


__all__ = ["ResidualCorrection"]
