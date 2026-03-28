"""! @file compat.py
@brief Compatibility aliases for the legacy public API.
"""

# Re-export compatibility names so downstream callers can transition gradually
# from the legacy lowercase class name to the package-style ``BSPF1D``.
from .operators import BSPF1D, bspf1d

__all__ = ["BSPF1D", "bspf1d"]
