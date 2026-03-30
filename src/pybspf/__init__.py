"""! @file __init__.py
@brief Public package exports for the pybspf package.

This module exposes the stable user-facing API while the internal codebase is
being migrated away from the legacy monolithic implementation.
"""

# Re-export the canonical grid type and the current operator wrappers so users
# can import from ``pybspf`` directly instead of depending on file layout.
from .grid import Grid1D
from .ops.differentiation import DerivativeResult
from .operators import BSPF1D, BSPF2D, PiecewiseBSPF1D, bspf1d, bspf2d
from .time_integration import integrate_rk4

__all__ = [
    "BSPF1D",
    "BSPF2D",
    "DerivativeResult",
    "Grid1D",
    "PiecewiseBSPF1D",
    "bspf1d",
    "bspf2d",
    "integrate_rk4",
]
