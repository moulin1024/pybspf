"""! @file operators/piecewise.py
@brief Piecewise operator wrapper for discontinuous signals.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..types import Array
from .bspf1d import BSPF1D


class PiecewiseBSPF1D:
    """! @brief Piecewise BSPF operator for functions with known discontinuities.

    @param degree B-spline degree for each segment.
    @param x Full uniform grid.
    @param breakpoints Physical coordinates of discontinuities.
    @param min_points_per_seg Minimum number of points retained per segment.
    @param bspf_kwargs Additional keyword arguments passed to ``BSPF1D.from_grid``.
    """

    def __init__(
        self,
        degree: int,
        x: Array,
        breakpoints: Optional[List[float]] = None,
        min_points_per_seg: int = 16,
        **bspf_kwargs,
    ):
        self.degree = int(degree)
        self.x = np.asarray(x, dtype=np.float64)
        self.breakpoints = sorted(breakpoints or [])
        self.min_points_per_seg = int(min_points_per_seg)

        N = self.x.size

        # Convert physical breakpoint coordinates into segment boundaries between
        # grid cells. Each boundary splits the data into independent BSPF solves.
        cut_indices = []
        for bp in self.breakpoints:
            idx = int(np.searchsorted(self.x, bp))
            if 1 <= idx <= N - 1:
                cut_indices.append(idx)
        cut_indices = sorted(set(cut_indices))

        self.segments = []

        i_start = 0
        for idx in cut_indices:
            i_end = idx - 1
            if i_end - i_start + 1 >= self.min_points_per_seg:
                x_seg = self.x[i_start : i_end + 1]
                op = BSPF1D.from_grid(degree=self.degree, x=x_seg, **bspf_kwargs)
                self.segments.append(dict(i0=i_start, i1=i_end, op=op))
            i_start = idx

        if N - i_start >= self.min_points_per_seg:
            x_seg = self.x[i_start:]
            op = BSPF1D.from_grid(degree=self.degree, x=x_seg, **bspf_kwargs)
            self.segments.append(dict(i0=i_start, i1=N - 1, op=op))

    def differentiate_1_2(
        self,
        f: Array,
        lam: float = 0.0,
        neumann_bc_global: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> Tuple[Array, Array, Array]:
        """! @brief Differentiate each segment independently and stitch results.

        @param f Function values on the full grid.
        @param lam Tikhonov regularization parameter.
        @param neumann_bc_global Optional Neumann data for the two global ends.
        @return Tuple ``(df1, df2, f_spline)`` on the full grid.
        """
        f = np.asarray(f, dtype=np.float64)
        if f.shape[0] != self.x.size:
            raise ValueError(f"f length {f.shape[0]} must match x length {self.x.size}")

        df1_full = np.zeros_like(f, dtype=np.float64)
        df2_full = np.zeros_like(f, dtype=np.float64)
        fs_full = np.zeros_like(f, dtype=np.float64)

        if neumann_bc_global is not None:
            left_flux_global, right_flux_global = neumann_bc_global
        else:
            left_flux_global = right_flux_global = None

        n_seg = len(self.segments)
        for k, seg in enumerate(self.segments):
            i0, i1, op = seg["i0"], seg["i1"], seg["op"]
            f_seg = f[i0 : i1 + 1]

            # Only the outermost segments inherit the global Neumann data.
            bc_left = left_flux_global if k == 0 else None
            bc_right = right_flux_global if k == n_seg - 1 else None
            neumann_bc_seg = (bc_left, bc_right)

            d1_seg, d2_seg, fs_seg = op.differentiate_1_2(
                f_seg,
                lam=lam,
                neumann_bc=neumann_bc_seg,
            )

            df1_full[i0 : i1 + 1] = d1_seg
            df2_full[i0 : i1 + 1] = d2_seg
            fs_full[i0 : i1 + 1] = fs_seg

        return df1_full, df2_full, fs_full


__all__ = ["PiecewiseBSPF1D"]
