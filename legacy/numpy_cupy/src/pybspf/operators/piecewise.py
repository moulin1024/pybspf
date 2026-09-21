"""! @file operators/piecewise.py
@brief Piecewise operator wrapper for discontinuous signals.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..backend import get_array_module, validate_backend_array
from ..grid import Grid1D
from ..ops.differentiation import DerivativeResult, _normalize_orders
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
        self.use_gpu = bool(bspf_kwargs.get("use_gpu", False))
        self._xp = get_array_module(use_gpu=self.use_gpu)
        validate_backend_array(x, use_gpu=self.use_gpu, name="x")
        self.x = Grid1D(x, use_gpu=self.use_gpu).x
        self.breakpoints = sorted([] if breakpoints is None else breakpoints)
        self.min_points_per_seg = int(min_points_per_seg)

        if self.min_points_per_seg < 2:
            raise ValueError("min_points_per_seg must be at least 2.")
        if any(not np.isfinite(bp) or not self.x[0] < bp < self.x[-1] for bp in self.breakpoints):
            raise ValueError("breakpoints must be finite and strictly inside the grid domain.")
        N = self.x.size

        # Convert physical breakpoint coordinates into segment boundaries between
        # grid cells. Each boundary splits the data into independent BSPF solves.
        cut_indices = []
        for bp in self.breakpoints:
            idx = int(self._xp.searchsorted(self.x, bp))
            if 1 <= idx <= N - 1:
                cut_indices.append(idx)
        cut_indices = sorted(set(cut_indices))

        self.segments = []

        boundaries = [0, *cut_indices, N]
        for i_start, i_stop in zip(boundaries[:-1], boundaries[1:]):
            if i_stop - i_start < self.min_points_per_seg:
                raise ValueError(
                    f"Segment [{i_start}:{i_stop}] has fewer than "
                    f"min_points_per_seg={self.min_points_per_seg} samples."
                )
            op = BSPF1D.from_grid(degree=self.degree, x=self.x[i_start:i_stop], **bspf_kwargs)
            self.segments.append(dict(i0=i_start, i1=i_stop - 1, op=op))

    def derivatives(
        self,
        f: Array,
        orders,
        lam: float = 0.0,
        neumann_bc_global: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ):
        """Compute requested derivative orders on each segment and stitch them."""
        validate_backend_array(f, use_gpu=self.use_gpu, name="f")
        xp = self._xp
        f = xp.asarray(f)
        if f.ndim != 1 or f.size != self.x.size:
            raise ValueError(f"f must have shape ({self.x.size},).")
        dtype = xp.complex128 if xp.iscomplexobj(f) else xp.float64
        f = f.astype(dtype, copy=False)
        normalized_orders = _normalize_orders(orders)
        derivative_full = {order: xp.empty_like(f) for order in normalized_orders}
        fs_full = xp.empty_like(f)

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

            seg_result = op.derivatives(
                f_seg,
                orders=normalized_orders,
                lam=lam,
                neumann_bc=neumann_bc_seg,
            )
            for order in normalized_orders:
                derivative_full[order][i0 : i1 + 1] = seg_result[order]
            fs_full[i0 : i1 + 1] = seg_result.spline

        return DerivativeResult(values=derivative_full, spline=fs_full)


__all__ = ["PiecewiseBSPF1D"]
