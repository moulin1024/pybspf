from __future__ import annotations
import os
import sys

# Set CUDA_PATH for NVHPC SDK before importing CuPy
# This ensures CuPy can find CUDA headers during JIT compilation
if 'CUDA_PATH' not in os.environ:
    nvhpc_cuda_path = '/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6'
    if os.path.exists(nvhpc_cuda_path):
        os.environ['CUDA_PATH'] = nvhpc_cuda_path
        os.environ['CUDA_HOME'] = nvhpc_cuda_path

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Tuple as TupType
import numpy as np
import numpy.typing as npt
from scipy import linalg as sla  # CPU fallback

# Optional GPU backend
_HAS_CUPY = False
try:
    import cupy as cp
    import cupyx.scipy.linalg as cpla
    _HAS_CUPY = True
except Exception:
    cp = None
    cpla = None

# Support running as a script or as part of the package
# Prefer optimized version, fall back to regular version
from bspf1d import bspf1d

Array = npt.NDArray[np.float64]


# -------------------------- backend helpers --------------------------
class _Backend:
    """Tiny dispatch wrapper to switch between NumPy/SciPy and CuPy/CuPyX."""
    __slots__ = ("xp", "la", "fft", "is_gpu")

    def __init__(self, use_gpu: bool):
        if use_gpu:
            if not _HAS_CUPY:
                raise RuntimeError(
                    "use_gpu=True but CuPy is not available. "
                    "Install cupy (e.g., `pip install cupy-cuda12x`) or set use_gpu=False."
                )
            self.xp = cp
            self.la = cpla
            self.fft = cp.fft
            self.is_gpu = True
        else:
            self.xp = np
            self.la = sla
            self.fft = np.fft
            self.is_gpu = False

    def to_device(self, a):
        return self.xp.asarray(a)

    def to_host(self, a):
        if self.is_gpu:
            return cp.asnumpy(a)
        return a

    def ensure_like_input(self, out, input_was_numpy: bool):
        if self.is_gpu and input_was_numpy:
            return self.to_host(out)
        return out


# =====================================================================
# Precomputed derivative plans (fast path for repeated evaluations)
# =====================================================================
@dataclass
class _AxisPlan:
    """Precomputed derivative plan along one axis for repeated calls."""
    model: bspf1d
    axis: int                  # 1 for x (columns), 0 for y (rows)
    order: int                 # derivative order k
    lam: float
    neumann: bool = False      # enforce first-derivative flux at the ends?

    # GPU/CPU selection
    use_gpu: bool = False

    # Uniform BC support: use a single boundary RHS, broadcast across batch
    uniform_bc: bool = False
    # bc: scalar or (m,) array of boundary RHS values. If None, defaults to zeros.
    bc: float | Array | None = None

    # cached handles / constants (filled in __post_init__)
    BW: any = field(init=False)
    BND: any = field(init=False)
    BT: any = field(init=False)
    BkT: any = field(init=False)
    omega: any = field(init=False)
    n_b: int = field(init=False)
    m: int = field(init=False)
    n: int = field(init=False)
    left_row: int = field(init=False)
    right_row: int = field(init=False)
    lu: any = field(init=False)
    piv: any = field(init=False)
    _bc_vec: Optional[any] = field(init=False, default=None)
    _bc_is_zero: bool = field(init=False, default=False)  # Cache whether BC is all zeros

    # backend
    _bk: _Backend = field(init=False, repr=False)

    def __post_init__(self):
        self._bk = _Backend(self.use_gpu)
        xp = self._bk.xp

        # Copy static matrices/constants to the selected device
        self.BW  = xp.asarray(self.model.BW)
        self.BND = xp.asarray(self.model.end.BND)
        self.BT  = xp.asarray(self.model.basis.BT0)
        self.BkT = xp.asarray(self.model.basis.BkT(self.order))
        self.omega = xp.asarray(self.model.grid.omega)
        # Pre-compute frequency multipliers for spectral correction (avoids (1j*omega)**order each call)
        self._iomega = 1j * self.omega
        self._iomega2 = self._iomega ** 2
        self._iomega3 = self._iomega ** 3
        self.n_b = int(self.BW.shape[0])
        self.m   = int(self.BND.shape[0])
        self.n   = int(self.model.grid.n)

        # Neumann rows
        ord_ = self.model.end.order
        if self.neumann:
            if ord_ < 2:
                raise ValueError("Model 'order' must be >= 2 to enforce Neumann flux.")
            self.left_row  = 1            # first-derivative @ left
            self.right_row = ord_ + 1     # first-derivative @ right
        else:
            self.left_row = self.right_row = -1  # unused

        # Precompute LU of KKT on CPU (cached in bspf1d), then move to device if needed
        lu_cpu, piv_cpu = self.model._kkt_lu(self.lam)
        if self._bk.is_gpu:
            self.lu  = xp.asarray(lu_cpu)
            self.piv = xp.asarray(piv_cpu)
        else:
            self.lu, self.piv = lu_cpu, piv_cpu

        # Prepare uniform boundary RHS vector if requested
        if self.uniform_bc:
            if self.bc is None:
                self._bc_vec = xp.zeros(self.m, dtype=xp.float64)
                self._bc_is_zero = True
            else:
                v = xp.asarray(self.bc, dtype=xp.float64)
                if v.ndim == 0:
                    val = float(v)
                    self._bc_vec = xp.full(self.m, val)
                    self._bc_is_zero = (val == 0.0)
                elif v.shape == (self.m,):
                    self._bc_vec = v
                    # Check if all zeros (convert to CPU for check if on GPU)
                    if self._bk.is_gpu:
                        self._bc_is_zero = bool(cp.all(v == 0.0))
                    else:
                        self._bc_is_zero = bool(np.all(v == 0.0))
                else:
                    raise ValueError(f"'bc' must be scalar or shape=({self.m},), got {tuple(v.shape)}.")
        else:
            self._bc_vec = None
            self._bc_is_zero = False

    def _broadcast_flux(self, val, batch: int):
        xp = self._bk.xp
        v = xp.asarray(val, dtype=xp.float64)
        if v.ndim == 0:
            return xp.full(batch, float(v))
        if v.shape == (batch,):
            return v
        raise ValueError(f"Flux must be scalar or shape=({batch},), got {tuple(v.shape)}.")

    def apply(self, F: Array, *, flux: Tuple[float | Array, float | Array] = (0.0, 0.0),
              return_spline: bool = False):
        """Compute ∂^order(F)/∂axis^order with optional Neumann flux enforcement (GPU-aware)."""
        xp, la, fft = self._bk.xp, self._bk.la, self._bk.fft
        input_was_numpy = (not self._bk.is_gpu) or isinstance(F, np.ndarray)
        
        # Validate device consistency - throw error instead of implicit conversion
        if self._bk.is_gpu:
            if _HAS_CUPY and isinstance(F, np.ndarray):
                raise ValueError(
                    "Cannot use NumPy array when use_gpu=True. "
                    "Either: (1) convert input to CuPy array before calling, or (2) use use_gpu=False."
                )
        else:
            if _HAS_CUPY and isinstance(F, cp.ndarray):
                raise ValueError(
                    "Cannot use CuPy array when use_gpu=False. "
                    "Either: (1) convert input to NumPy array, or (2) use use_gpu=True."
                )
        
        # Optimize: avoid unnecessary asarray if F is already the right type and on the right device
        if self._bk.is_gpu:
            if _HAS_CUPY and isinstance(F, cp.ndarray) and F.dtype == xp.float64:
                # GPU path: already a CuPy array with correct dtype - just moveaxis
                FT = xp.moveaxis(F, self.axis, 0)
            else:
                # GPU path: convert to CuPy with correct dtype
                FT = xp.moveaxis(xp.asarray(F, dtype=xp.float64), self.axis, 0)
        else:
            # CPU path: avoid extra copy if F is already a NumPy float64 array
            if isinstance(F, np.ndarray) and F.dtype == np.float64:
                FT = np.moveaxis(F, self.axis, 0)
            else:
                FT = np.moveaxis(np.asarray(F, dtype=np.float64), self.axis, 0)
        n, batch = FT.shape

        # Build RHS
        rhs_top = 2.0 * (self.BW @ FT)  # (n_b, batch)
        if self.uniform_bc:
            # Optimize for zero BCs: use zeros instead of repeat for better GPU performance
            # This avoids the expensive repeat operation when BCs are all zero
            if self._bc_is_zero:
                dY = xp.zeros((self.m, batch), dtype=xp.float64)
            else:
                dY = xp.repeat(self._bc_vec[:, None], batch, axis=1)
        else:
            dY = self.BND @ FT
        if self.neumann:
            lf = self._broadcast_flux(flux[0], batch)
            rf = self._broadcast_flux(flux[1], batch)
            dY[self.left_row,  :] = lf
            dY[self.right_row, :] = rf
        RHS = xp.vstack([rhs_top, dY])  # (n_b+m, batch)

        # Solve with precomputed LU (on device if GPU)
        if self._bk.is_gpu:
            SOL = la.lu_solve((self.lu, self.piv), RHS)
        else:
            # CPU path: RHS is already a NumPy array, no need for to_host/asarray round-trip
            SOL = la.lu_solve((self.lu, self.piv), RHS)

        P = SOL[: self.n_b, :]  # spline coeffs

        # Spline part + spectral correction
        spline = self.BT @ P
        deriv  = self.BkT @ P

        resid = FT - spline  # shape: (n, batch)
        # Batch FFT: fft.rfft along axis=0 performs 'batch' FFTs of length 'n' in one call
        # R shape: (n//2+1, batch), omega shape: (n//2+1,), broadcasting works correctly
        R = fft.rfft(resid, axis=0)
        # Use pre-computed frequency multipliers for common derivative orders (1, 2, 3)
        if self.order == 1:
            mult = self._iomega
        elif self.order == 2:
            mult = self._iomega2
        elif self.order == 3:
            mult = self._iomega3
        else:
            # Fallback for higher orders (rare in current use)
            mult = (1j * self.omega) ** self.order
        corr = fft.irfft(R * mult[:, None], n=self.n, axis=0)

        D = deriv + corr
        if self.neumann and self.order == 1:
            D[0,  :] = dY[self.left_row,  :]
            D[-1, :] = dY[self.right_row, :]

        if return_spline:
            Dout = xp.moveaxis(D, 0, self.axis)
            Sout = xp.moveaxis(spline, 0, self.axis)
            return (self._bk.ensure_like_input(Dout, input_was_numpy),
                    self._bk.ensure_like_input(Sout, input_was_numpy))
        Dout = xp.moveaxis(D, 0, self.axis)
        return self._bk.ensure_like_input(Dout, input_was_numpy)


@dataclass
class DiffPlan2D:
    """Two-axis plan for repeated derivatives with fixed (order, lam, BCs)."""
    x_plan: _AxisPlan
    y_plan: _AxisPlan

    def dx(self, F: Array, *, flux=(0.0, 0.0), return_spline=False):
        return self.x_plan.apply(F, flux=flux, return_spline=return_spline)

    def dy(self, F: Array, *, flux=(0.0, 0.0), return_spline=False):
        return self.y_plan.apply(F, flux=flux, return_spline=return_spline)


# =====================================================================
# 2D facade (GPU-aware)
# =====================================================================
@dataclass
class bspf2d:
    """
    Vectorized 2D facade composed from two bspf1d models.
    Can run on CPU (NumPy/SciPy) or GPU (CuPy/CuPyX) depending on `use_gpu`.
    """
    x: Array           # (nx,)
    y: Array           # (ny,)
    x_model: bspf1d    # acts along axis=1 (x)
    y_model: bspf1d    # acts along axis=0 (y)
    use_gpu: bool = False

    # ---------- construction ----------
    @classmethod
    def from_grids(
        cls,
        *,
        x: Array,
        y: Array,
        degree_x: int = 10,
        degree_y: Optional[int] = None,
        knots_x: Optional[Array] = None, knots_y: Optional[Array] = None,
        n_basis_x: Optional[int] = None, n_basis_y: Optional[int] = None,
        domain_x: Optional[Tuple[float, float]] = None, domain_y: Optional[Tuple[float, float]] = None,
        use_clustering_x: bool = False, use_clustering_y: bool = False,
        order_x: Optional[int] = None, order_y: Optional[int] = None,
        num_boundary_points_x: Optional[int] = None, num_boundary_points_y: Optional[int] = None,
        correction: str = "spectral",
        use_gpu: bool = False,
    ) -> "bspf2d":
        # Strict backend type checking: enforce that array backend matches use_gpu setting
        # If use_gpu=True, arrays must be CuPy arrays; if use_gpu=False, arrays must be NumPy arrays
        is_x_cupy = _HAS_CUPY and isinstance(x, cp.ndarray)
        is_y_cupy = _HAS_CUPY and isinstance(y, cp.ndarray)
        
        if use_gpu:
            if not _HAS_CUPY:
                raise ValueError("use_gpu=True requires CuPy to be available, but CuPy is not installed")
            if not is_x_cupy:
                raise TypeError(f"use_gpu=True requires CuPy arrays, but x is of type {type(x).__name__}")
            if not is_y_cupy:
                raise TypeError(f"use_gpu=True requires CuPy arrays, but y is of type {type(y).__name__}")
        else:
            if is_x_cupy:
                raise TypeError(f"use_gpu=False requires NumPy arrays, but x is a CuPy array")
            if is_y_cupy:
                raise TypeError(f"use_gpu=False requires NumPy arrays, but y is a CuPy array")
        
        # Pass arrays directly to bspf1d.from_grid() without conversion
        # bspf1d.from_grid() will handle backend checking internally
        # Since we've already validated the backend matches use_gpu, no conversion is needed
        if degree_y is None:
            degree_y = degree_x
        
        xm = bspf1d.from_grid(
            degree=degree_x, x=x, knots=knots_x, n_basis=n_basis_x, domain=domain_x,
            use_clustering=use_clustering_x, order=order_x, num_boundary_points=num_boundary_points_x,
            correction=correction, 
            use_gpu=use_gpu,
        )
        ym = bspf1d.from_grid(
            degree=degree_y, x=y, knots=knots_y, n_basis=n_basis_y, domain=domain_y,
            use_clustering=use_clustering_y, order=order_y, num_boundary_points=num_boundary_points_y,
            correction=correction,
            use_gpu=use_gpu,
        )
        
        # Store arrays in their original backend type (no conversion needed)
        # x and y are only used for .size attribute, which works for both NumPy and CuPy
        return cls(x=x, y=y, x_model=xm, y_model=ym, use_gpu=use_gpu)

    # ---------- init cache ----------
    def __post_init__(self):
        # cache for precomputed plans: key -> _AxisPlan
        self._plan_cache: Dict[
            Tuple[str, int, float, bool, bool, Tuple[float, ...] | None, bool], _AxisPlan
        ] = {}

    # ---------- shape guard ----------
    def _check_shape(self, F: Array) -> Tuple[int, int]:
        if F.ndim != 2:
            raise ValueError("F must be 2D (ny, nx).")
        ny, nx = F.shape
        if ny != self.y.size or nx != self.x.size:
            raise ValueError(f"F shape {F.shape} must match (len(y), len(x))=({self.y.size},{self.x.size}).")
        return ny, nx

    # ---------- helpers for uniform BC ----------
    @staticmethod
    def _prepare_bc_vector(model: bspf1d, bc: float | Array | None) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, ...]]]:
        """Return (bc_vec_cpu, bc_key) for caching. bc_vec_cpu is (m,) or None (stored on CPU; moved per-plan)."""
        if bc is None:
            return None, None
        m = model.end.BND.shape[0]
        # Error if bc is a CuPy array (this method prepares CPU arrays)
        if _HAS_CUPY and isinstance(bc, cp.ndarray):
            raise ValueError(
                "Cannot use CuPy array in _prepare_bc_vector. "
                "This method prepares boundary condition vectors on CPU. "
                "Either: (1) convert input to NumPy array, or (2) use a GPU-aware BC preparation method."
            )
        v = np.asarray(bc, dtype=np.float64)
        if v.ndim == 0:
            vec = np.full(m, float(v))
            key = tuple(vec.tolist())
            return vec, key
        if v.shape == (m,):
            return v, tuple(v.tolist())
        raise ValueError(f"'bc' must be scalar or shape=({m},), got {v.shape}.")

    # ---------- axis-generic kernels (on-the-fly path; will transfer each call) ----------
    @staticmethod
    def _diff_axis(
        F: Array,
        model: bspf1d,
        *,
        lam: float,
        k: int,
        axis: int,
        return_spline: bool,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
        use_gpu: bool = False,
    ):
        """
        Vectorized derivative of order k along `axis`.
        For best performance, prefer plans; this on-the-fly path copies matrices each call.
        """
        bk = _Backend(use_gpu)
        xp, la, fft = bk.xp, bk.la, bk.fft
        input_was_numpy = (not bk.is_gpu) or isinstance(F, np.ndarray)
        
        # Validate device consistency - throw error instead of implicit conversion
        if bk.is_gpu:
            if _HAS_CUPY and isinstance(F, np.ndarray):
                raise ValueError(
                    "Cannot use NumPy array when use_gpu=True in _diff_axis. "
                    "Either: (1) convert input to CuPy array before calling, or (2) use use_gpu=False."
                )
        else:
            if _HAS_CUPY and isinstance(F, cp.ndarray):
                raise ValueError(
                    "Cannot use CuPy array when use_gpu=False in _diff_axis. "
                    "Either: (1) convert input to NumPy array, or (2) use use_gpu=True."
                )

        FT = xp.moveaxis(xp.asarray(F, dtype=xp.float64), axis, 0)  # (n, batch)
        n, batch = FT.shape

        BW  = xp.asarray(model.BW)
        BND = xp.asarray(model.end.BND)
        BT  = xp.asarray(model.basis.BT0)
        BkT = xp.asarray(model.basis.BkT(k))
        om  = xp.asarray(model.grid.omega)
        n_b = BW.shape[0]
        m   = BND.shape[0]

        # Boundary RHS
        if uniform_bc:
            if bc is None:
                dY = xp.zeros((m, batch), dtype=xp.float64)
            else:
                v = xp.asarray(bc, dtype=xp.float64)
                if v.ndim == 0:
                    v = xp.full(m, float(v))
                elif v.shape != (m,):
                    raise ValueError(f"'bc' must be scalar or shape=({m},), got {tuple(v.shape)}.")
                dY = xp.repeat(v[:, None], batch, axis=1)
        else:
            dY = BND @ FT

        RHS = xp.vstack([2.0 * (BW @ FT), dY])
        lu_cpu, piv_cpu = model._kkt_lu(lam)
        if bk.is_gpu:
            # Validate LU factors are on correct device
            if _HAS_CUPY and isinstance(lu_cpu, np.ndarray):
                # This is OK - we'll convert to GPU
                lu_gpu = xp.asarray(lu_cpu)
                piv_gpu = xp.asarray(piv_cpu)
            elif _HAS_CUPY and isinstance(lu_cpu, cp.ndarray):
                # Already on GPU
                lu_gpu = lu_cpu
                piv_gpu = piv_cpu
            else:
                lu_gpu = xp.asarray(lu_cpu)
                piv_gpu = xp.asarray(piv_cpu)
            SOL = la.lu_solve((lu_gpu, piv_gpu), RHS)
        else:
            # CPU path: lu_cpu/piv_cpu are NumPy arrays; RHS is already on host
            SOL = la.lu_solve((lu_cpu, piv_cpu), RHS)

        P = SOL[:n_b, :]
        spline = BT @ P
        deriv  = BkT @ P
        resid  = FT - spline  # shape: (n, batch)
        # Batch FFT: fft.rfft along axis=0 performs 'batch' FFTs of length 'n' in one call
        # R shape: (n//2+1, batch), om shape: (n//2+1,), broadcasting works correctly
        R      = fft.rfft(resid, axis=0)
        corr   = fft.irfft(R * (1j * om)[:, None]**k, n=n, axis=0)

        D = deriv + corr
        if return_spline:
            return (
                bk.ensure_like_input(xp.moveaxis(D, 0, axis), input_was_numpy),
                bk.ensure_like_input(xp.moveaxis(spline, 0, axis), input_was_numpy),
            )
        return bk.ensure_like_input(xp.moveaxis(D, 0, axis), input_was_numpy)

    @staticmethod
    def _broadcast_flux_backend(bk: _Backend, val, batch: int):
        xp = bk.xp
        v = xp.asarray(val, dtype=xp.float64)
        if v.ndim == 0:
            return xp.full(batch, float(v))
        if v.shape == (batch,):
            return v
        raise ValueError(f"Flux must be scalar or shape=({batch},), got {tuple(v.shape)}.")

    @staticmethod
    def _diff_axis_neumann(
        F: Array,
        model: bspf1d,
        *,
        lam: float,
        k: int,
        axis: int,
        flux: Tuple[float | Array, float | Array],   # (left_flux, right_flux)
        return_spline: bool,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
        use_gpu: bool = False,
    ):
        """Neumann-enforced path; on-the-fly (matrices copied each call)."""
        bk = _Backend(use_gpu)
        xp, la, fft = bk.xp, bk.la, bk.fft
        input_was_numpy = (not bk.is_gpu) or isinstance(F, np.ndarray)
        
        # Validate device consistency - throw error instead of implicit conversion
        if bk.is_gpu:
            if _HAS_CUPY and isinstance(F, np.ndarray):
                raise ValueError(
                    "Cannot use NumPy array when use_gpu=True in _diff_axis_neumann. "
                    "Either: (1) convert input to CuPy array before calling, or (2) use use_gpu=False."
                )
        else:
            if _HAS_CUPY and isinstance(F, cp.ndarray):
                raise ValueError(
                    "Cannot use CuPy array when use_gpu=False in _diff_axis_neumann. "
                    "Either: (1) convert input to NumPy array, or (2) use use_gpu=True."
                )

        # Avoid unnecessary asarray: respect existing backend and dtype when possible
        if bk.is_gpu:
            FT = xp.moveaxis(xp.asarray(F, dtype=xp.float64), axis, 0)
        else:
            if isinstance(F, np.ndarray) and F.dtype == np.float64:
                FT = np.moveaxis(F, axis, 0)
            else:
                FT = np.moveaxis(np.asarray(F, dtype=np.float64), axis, 0)
        n, batch = FT.shape

        BW  = xp.asarray(model.BW)
        BND = xp.asarray(model.end.BND)
        BT  = xp.asarray(model.basis.BT0)
        BkT = xp.asarray(model.basis.BkT(k))
        om  = xp.asarray(model.grid.omega)
        n_b = BW.shape[0]
        m   = BND.shape[0]
        ord_ = model.end.order
        if ord_ < 2:
            raise ValueError("Model 'order' must be >=2 to enforce first-derivative Neumann flux.")

        if uniform_bc:
            if bc is None:
                dY = xp.zeros((m, batch), dtype=xp.float64)
            else:
                v = xp.asarray(bc, dtype=xp.float64)
                if v.ndim == 0:
                    v = xp.full(m, float(v))
                elif v.shape != (m,):
                    raise ValueError(f"'bc' must be scalar or shape=({m},), got {tuple(v.shape)}.")
                dY = xp.repeat(v[:, None], batch, axis=1)
        else:
            dY = BND @ FT

        left_row  = 1
        right_row = ord_ + 1
        left_flux  = bspf2d._broadcast_flux_backend(bk, flux[0], batch)
        right_flux = bspf2d._broadcast_flux_backend(bk, flux[1], batch)
        dY[left_row,  :] = left_flux
        dY[right_row, :] = right_flux

        RHS = xp.vstack([2.0 * (BW @ FT), dY])
        lu_cpu, piv_cpu = model._kkt_lu(lam)
        if bk.is_gpu:
            # Validate LU factors are on correct device
            if _HAS_CUPY and isinstance(lu_cpu, np.ndarray):
                # This is OK - we'll convert to GPU
                lu_gpu = xp.asarray(lu_cpu)
                piv_gpu = xp.asarray(piv_cpu)
            elif _HAS_CUPY and isinstance(lu_cpu, cp.ndarray):
                # Already on GPU
                lu_gpu = lu_cpu
                piv_gpu = piv_cpu
            else:
                lu_gpu = xp.asarray(lu_cpu)
                piv_gpu = xp.asarray(piv_cpu)
            SOL = la.lu_solve((lu_gpu, piv_gpu), RHS)
        else:
            # CPU path: lu_cpu/piv_cpu are NumPy arrays; RHS is already on host
            SOL = la.lu_solve((lu_cpu, piv_cpu), RHS)

        P = SOL[:n_b, :]
        spline = BT @ P
        deriv  = BkT @ P
        resid  = FT - spline  # shape: (n, batch)
        # Batch FFT: fft.rfft along axis=0 performs 'batch' FFTs of length 'n' in one call
        # R shape: (n//2+1, batch), om shape: (n//2+1,), broadcasting works correctly
        R      = fft.rfft(resid, axis=0)
        corr   = fft.irfft(R * (1j * om)[:, None]**k, n=n, axis=0)

        D = deriv + corr
        if k == 1:
            D[0,  :] = left_flux
            D[-1, :] = right_flux

        if return_spline:
            return (
                bk.ensure_like_input(xp.moveaxis(D, 0, axis), input_was_numpy),
                bk.ensure_like_input(xp.moveaxis(spline, 0, axis), input_was_numpy),
            )
        return bk.ensure_like_input(xp.moveaxis(D, 0, axis), input_was_numpy)

    # ---------- precomputed plan builders (move everything once) ----------
    def make_plan_dx(
        self,
        *,
        order: int = 1,
        lam: float = 0.0,
        neumann: bool = False,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
    ) -> _AxisPlan:
        bc_vec_cpu, bc_key = (None, None)
        if uniform_bc:
            bc_vec_cpu, bc_key = self._prepare_bc_vector(self.x_model, bc)
        key = ('x', order, float(lam), bool(neumann), bool(uniform_bc), bc_key, bool(self.use_gpu))
        plan = self._plan_cache.get(key)
        if plan is None:
            plan = _AxisPlan(
                model=self.x_model, axis=1, order=order, lam=lam,
                neumann=neumann, use_gpu=self.use_gpu,
                uniform_bc=uniform_bc, bc=bc_vec_cpu
            )
            self._plan_cache[key] = plan
        return plan

    def make_plan_dy(
        self,
        *,
        order: int = 1,
        lam: float = 0.0,
        neumann: bool = False,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
    ) -> _AxisPlan:
        bc_vec_cpu, bc_key = (None, None)
        if uniform_bc:
            bc_vec_cpu, bc_key = self._prepare_bc_vector(self.y_model, bc)
        key = ('y', order, float(lam), bool(neumann), bool(uniform_bc), bc_key, bool(self.use_gpu))
        plan = self._plan_cache.get(key)
        if plan is None:
            plan = _AxisPlan(
                model=self.y_model, axis=0, order=order, lam=lam,
                neumann=neumann, use_gpu=self.use_gpu,
                uniform_bc=uniform_bc, bc=bc_vec_cpu
            )
            self._plan_cache[key] = plan
        return plan

    def make_plan_pair(
        self,
        *,
        order_x: int = 1, lam_x: float = 0.0, neumann_x: bool = False, uniform_bc_x: bool = False, bc_x: float | Array | None = None,
        order_y: int = 1, lam_y: float = 0.0, neumann_y: bool = False, uniform_bc_y: bool = False, bc_y: float | Array | None = None,
    ) -> DiffPlan2D:
        return DiffPlan2D(
            x_plan=self.make_plan_dx(order=order_x, lam=lam_x, neumann=neumann_x, uniform_bc=uniform_bc_x, bc=bc_x),
            y_plan=self.make_plan_dy(order=order_y, lam=lam_y, neumann=neumann_y, uniform_bc=uniform_bc_y, bc=bc_y),
        )

    # ---------- public API (on-the-fly path) ----------
    def partial_dx(
        self,
        F: Array,
        *,
        order: int = 1,
        lam: float = 0.0,
        return_spline: bool = False,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
    ):
        self._check_shape(F)
        return self._diff_axis(
            F, self.x_model, lam=lam, k=order, axis=1, return_spline=return_spline,
            uniform_bc=uniform_bc, bc=bc, use_gpu=self.use_gpu
        )

    def partial_dy(
        self,
        F: Array,
        *,
        order: int = 1,
        lam: float = 0.0,
        return_spline: bool = False,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
    ):
        self._check_shape(F)
        return self._diff_axis(
            F, self.y_model, lam=lam, k=order, axis=0, return_spline=return_spline,
            uniform_bc=uniform_bc, bc=bc, use_gpu=self.use_gpu
        )

    # ---- second-order partials ----
    def partial_dxx(self, F: Array, *, lam: float = 0.0, uniform_bc: bool = False, bc: float | Array | None = None) -> Array:
        return self.partial_dx(F, order=2, lam=lam, uniform_bc=uniform_bc, bc=bc)

    def partial_dyy(self, F: Array, *, lam: float = 0.0, uniform_bc: bool = False, bc: float | Array | None = None) -> Array:
        return self.partial_dy(F, order=2, lam=lam, uniform_bc=uniform_bc, bc=bc)

    # ---- mixed partial ----
    def partial_dxy(self, F: Array, *, lam_x: float = 0.0, lam_y: float = 0.0, symmetrize: bool = True) -> Array:
        self._check_shape(F)
        dFx = self._diff_axis(F, self.x_model, lam=lam_x, k=1, axis=1, return_spline=False, use_gpu=self.use_gpu)
        dxy = self._diff_axis(dFx, self.y_model, lam=lam_y, k=1, axis=0, return_spline=False, use_gpu=self.use_gpu)
        if not symmetrize:
            return dxy.astype(np.float64)
        dFy = self._diff_axis(F, self.y_model, lam=lam_y, k=1, axis=0, return_spline=False, use_gpu=self.use_gpu)
        dyx = self._diff_axis(dFy, self.x_model, lam=lam_x, k=1, axis=1, return_spline=False, use_gpu=self.use_gpu)
        return (0.5 * (dxy + dyx)).astype(np.float64)

    # ---- compute first and second derivatives together (efficient) ----
    def differentiate_1_2(
        self,
        F: Array,
        *,
        lam_x: float = 0.0,
        lam_y: float = 0.0,
        uniform_bc_x: bool = False,
        uniform_bc_y: bool = False,
        bc_x: float | Array | None = None,
        bc_y: float | Array | None = None
    ) -> Tuple[Array, Array, Array, Array]:
        """
        Compute first and second derivatives in both x and y directions together.
        More efficient than calling partial_dx, partial_dy, partial_dxx, partial_dyy separately
        because it reuses intermediate computations.
        
        Uses batched operations for optimal performance on both CPU and GPU.
        
        Parameters
        ----------
        F : Array
            2D array of shape (ny, nx)
        lam_x : float, default 0.0
            Tikhonov regularization parameter for x-direction
        lam_y : float, default 0.0
            Tikhonov regularization parameter for y-direction
        uniform_bc_x : bool, default False
            Whether to use uniform boundary conditions in x-direction (currently unused)
        uniform_bc_y : bool, default False
            Whether to use uniform boundary conditions in y-direction (currently unused)
        bc_x : float | Array | None, default None
            Boundary condition values for x-direction (currently unused)
        bc_y : float | Array | None, default None
            Boundary condition values for y-direction (currently unused)
        
        Returns
        -------
        dF_dx : Array
            First derivative in x-direction: ∂F/∂x, shape (ny, nx)
        dF_dy : Array
            First derivative in y-direction: ∂F/∂y, shape (ny, nx)
        d2F_dx2 : Array
            Second derivative in x-direction: ∂²F/∂x², shape (ny, nx)
        d2F_dy2 : Array
            Second derivative in y-direction: ∂²F/∂y², shape (ny, nx)
        """
        self._check_shape(F)

        is_gpu = _HAS_CUPY and isinstance(F, cp.ndarray)
        xp = cp if is_gpu else np
        is_complex = (cp.iscomplexobj(F) if is_gpu else np.iscomplexobj(F))
        dtype = xp.complex128 if is_complex else xp.float64
        F_cast = xp.asarray(F, dtype=dtype)

        # X-direction: differentiate along columns (axis=1)
        F_T = F_cast.T  # (nx, ny)
        dF_dx_T, d2F_dx2_T, _ = self.x_model.differentiate_1_2_batched(
            F_T, lam=lam_x
        )
        dF_dx = dF_dx_T.T
        d2F_dx2 = d2F_dx2_T.T

        # Y-direction: differentiate along rows (axis=0)
        dF_dy, d2F_dy2, _ = self.y_model.differentiate_1_2_batched(
            F_cast, lam=lam_y
        )

        return (
            dF_dx.astype(dtype),
            dF_dy.astype(dtype),
            d2F_dx2.astype(dtype),
            d2F_dy2.astype(dtype),
        )

    # ---- Neumann-enforced variants ----
    def partial_dx_neumann(
        self,
        F: Array,
        *,
        order: int = 1,
        lam: float = 0.0,
        flux: Tuple[float | Array, float | Array] = (0.0, 0.0),
        return_spline: bool = False,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
    ):
        self._check_shape(F)
        return self._diff_axis_neumann(
            F, self.x_model, lam=lam, k=order, axis=1, flux=flux, return_spline=return_spline,
            uniform_bc=uniform_bc, bc=bc, use_gpu=self.use_gpu
        )

    def partial_dy_neumann(
        self,
        F: Array,
        *,
        order: int = 1,
        lam: float = 0.0,
        flux: Tuple[float | Array, float | Array] = (0.0, 0.0),
        return_spline: bool = False,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
    ):
        self._check_shape(F)
        return self._diff_axis_neumann(
            F, self.y_model, lam=lam, k=order, axis=0, flux=flux, return_spline=return_spline,
            uniform_bc=uniform_bc, bc=bc, use_gpu=self.use_gpu
        )

    def partial_dxx_neumann(
        self,
        F: Array,
        *,
        lam: float = 0.0,
        flux: Tuple[float | Array, float | Array] = (0.0, 0.0),
        return_spline: bool = False,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
    ):
        return self.partial_dx_neumann(F, order=2, lam=lam, flux=flux, return_spline=return_spline,
                                       uniform_bc=uniform_bc, bc=bc)

    def partial_dyy_neumann(
        self,
        F: Array,
        *,
        lam: float = 0.0,
        flux: Tuple[float | Array, float | Array] = (0.0, 0.0),
        return_spline: bool = False,
        uniform_bc: bool = False,
        bc: float | Array | None = None,
    ):
        return self.partial_dy_neumann(F, order=2, lam=lam, flux=flux, return_spline=return_spline,
                                       uniform_bc=uniform_bc, bc=bc)

    def laplacian_neumann(
        self,
        F: Array,
        *,
        lam_x: float = 0.0,
        lam_y: float = 0.0,
        flux_x: Tuple[float | Array, float | Array] = (0.0, 0.0),
        flux_y: Tuple[float | Array, float | Array] = (0.0, 0.0),
        uniform_bc_x: bool = False,
        uniform_bc_y: bool = False,
        bc_x: float | Array | None = None,
        bc_y: float | Array | None = None,
    ) -> Array:
        Fxx = self.partial_dxx_neumann(F, lam=lam_x, flux=flux_x, uniform_bc=uniform_bc_x, bc=bc_x)
        Fyy = self.partial_dyy_neumann(F, lam=lam_y, flux=flux_y, uniform_bc=uniform_bc_y, bc=bc_y)
        return (Fxx + Fyy).astype(np.float64)

    # ---- convenience: Hessian & Laplacian ----
    def hessian(
        self,
        F: Array,
        *,
        lam_x: float = 0.0,
        lam_y: float = 0.0,
        symmetrize_xy: bool = True,
        uniform_bc_x: bool = False,
        uniform_bc_y: bool = False,
        bc_x: float | Array | None = None,
        bc_y: float | Array | None = None,
    ):
        Fxx = self.partial_dxx(F, lam=lam_x, uniform_bc=uniform_bc_x, bc=bc_x)
        Fyy = self.partial_dyy(F, lam=lam_y, uniform_bc=uniform_bc_y, bc=bc_y)
        Fxy = self.partial_dxy(F, lam_x=lam_x, lam_y=lam_y, symmetrize=symmetrize_xy)
        return Fxx, Fxy, Fyy

    def laplacian(
        self,
        F: Array,
        *,
        lam_x: float = 0.0,
        lam_y: float = 0.0,
        uniform_bc_x: bool = False,
        uniform_bc_y: bool = False,
        bc_x: float | Array | None = None,
        bc_y: float | Array | None = None,
    ) -> Array:
        return (self.partial_dxx(F, lam=lam_x, uniform_bc=uniform_bc_x, bc=bc_x)
                + self.partial_dyy(F, lam=lam_y, uniform_bc=uniform_bc_y, bc=bc_y)).astype(np.float64)