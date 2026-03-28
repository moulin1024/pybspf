"""! @file knots.py
@brief Knot generation helpers for the BSPF spline basis.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .grid import Grid1D
from .types import Array


class _Knot:
    """! @brief Internal helper for constructing or validating knot vectors."""

    @staticmethod
    def _generate(
        *,
        degree: int,
        domain: Tuple[float, float],
        n_basis: int,
        use_clustering: bool,
        clustering_factor: float,
    ) -> Array:
        """! @brief Generate a clamped knot vector over a physical domain.

        @param degree Polynomial degree of the B-spline basis.
        @param domain Physical interval ``(a, b)``.
        @param n_basis Number of basis functions requested.
        @param use_clustering Whether to cluster knots toward the boundaries.
        @param clustering_factor Strength of the tanh-based clustering.
        @return One-dimensional knot vector.
        """
        if n_basis <= degree:
            raise ValueError("n_basis must exceed degree.")
        n_knots = n_basis + degree + 1
        n_interior = n_knots - 2 * (degree + 1)

        if n_interior > 0:
            # Build a normalized knot coordinate in ``[-1, 1]`` first so the
            # optional clustering is independent of the physical domain size.
            u = np.linspace(-1.0, 1.0, n_interior + 2)
            if use_clustering:
                u = np.tanh(clustering_factor * u) / np.tanh(clustering_factor)
            uniq = degree * (u + 1.0) / 2.0
            ks = [float(uniq[0])] * (degree + 1)
            ks += list(map(float, uniq[1:-1]))
            ks += [float(uniq[-1])] * (degree + 1)
            k = np.array(ks, dtype=np.float64)
        else:
            k = np.concatenate(
                [np.zeros(degree + 1), np.full(degree + 1, degree)],
                dtype=np.float64,
            )

        a, b = domain
        return (k / degree) * (b - a) + a

    @staticmethod
    def resolve(
        *,
        degree: int,
        grid: Grid1D,
        knots: Optional[Array],
        n_basis: Optional[int],
        domain: Optional[Tuple[float, float]],
        use_clustering: bool,
        clustering_factor: float,
    ) -> Array:
        """! @brief Resolve explicit or generated knots for a given grid.

        @param degree Polynomial degree of the spline basis.
        @param grid Grid defining the default physical domain.
        @param knots Explicit knot vector, if provided.
        @param n_basis Requested number of basis functions when generating.
        @param domain Explicit physical domain used for generation.
        @param use_clustering Whether generated knots should be boundary-clustered.
        @param clustering_factor Strength of the clustering map.
        @return One-dimensional knot vector.
        """
        if knots is not None:
            k = np.asarray(knots, dtype=np.float64)
            if k.ndim != 1:
                raise ValueError("knots must be a 1D array.")
            return k
        # Match the legacy default until the package API defines a more explicit
        # constructor configuration object.
        if n_basis is None:
            n_basis = 2 * (degree + 1) * 2
        if domain is None:
            domain = (grid.a, grid.b)
        return _Knot._generate(
            degree=degree,
            domain=domain,
            n_basis=n_basis,
            use_clustering=use_clustering,
            clustering_factor=clustering_factor,
        )


__all__ = ["_Knot"]
