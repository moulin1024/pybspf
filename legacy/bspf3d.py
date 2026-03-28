from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt

# Optional GPU backend
_HAS_CUPY = False
try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None

from bspf1d import bspf1d

Array = npt.NDArray[np.float64]


@dataclass
class bspf3d:
    """
    Vectorized 3D facade composed from three bspf1d models.
    Can run on CPU (NumPy/SciPy) or GPU (CuPy/CuPyX) depending on `use_gpu`.
    """
    x: Array           # (nx,)
    y: Array           # (ny,)
    z: Array           # (nz,)
    x_model: bspf1d    # acts along axis=2 (x)
    y_model: bspf1d    # acts along axis=1 (y)
    z_model: bspf1d    # acts along axis=0 (z)
    use_gpu: bool = False
    
    # Strict backend-aware constructor (mirrors 2D facade behavior)
    @classmethod
    def from_grids(
        cls,
        *,
        x: Array,
        y: Array,
        z: Array,
        degree_x: int = 10,
        degree_y: Optional[int] = None,
        degree_z: Optional[int] = None,
        knots_x: Optional[Array] = None, knots_y: Optional[Array] = None, knots_z: Optional[Array] = None,
        n_basis_x: Optional[int] = None, n_basis_y: Optional[int] = None, n_basis_z: Optional[int] = None,
        domain_x: Optional[Tuple[float, float]] = None, domain_y: Optional[Tuple[float, float]] = None, domain_z: Optional[Tuple[float, float]] = None,
        use_clustering_x: bool = False, use_clustering_y: bool = False, use_clustering_z: bool = False,
        order_x: Optional[int] = None, order_y: Optional[int] = None, order_z: Optional[int] = None,
        num_boundary_points_x: Optional[int] = None, num_boundary_points_y: Optional[int] = None, num_boundary_points_z: Optional[int] = None,
        correction: str = "spectral",
        use_gpu: bool = False,
    ) -> "bspf3d":
        # Enforce backend consistency
        is_x_cupy = _HAS_CUPY and isinstance(x, cp.ndarray)
        is_y_cupy = _HAS_CUPY and isinstance(y, cp.ndarray)
        is_z_cupy = _HAS_CUPY and isinstance(z, cp.ndarray)

        if use_gpu:
            if not _HAS_CUPY:
                raise ValueError("use_gpu=True requires CuPy to be available, but CuPy is not installed")
            if not is_x_cupy:
                raise TypeError(f"use_gpu=True requires CuPy arrays, but x is of type {type(x).__name__}")
            if not is_y_cupy:
                raise TypeError(f"use_gpu=True requires CuPy arrays, but y is of type {type(y).__name__}")
            if not is_z_cupy:
                raise TypeError(f"use_gpu=True requires CuPy arrays, but z is of type {type(z).__name__}")
        else:
            if is_x_cupy:
                raise TypeError("use_gpu=False requires NumPy arrays, but x is a CuPy array")
            if is_y_cupy:
                raise TypeError("use_gpu=False requires NumPy arrays, but y is a CuPy array")
            if is_z_cupy:
                raise TypeError("use_gpu=False requires NumPy arrays, but z is a CuPy array")

        if degree_y is None:
            degree_y = degree_x
        if degree_z is None:
            degree_z = degree_x

        # Build 1D models; bspf1d.from_grid handles backend internally
        xm = bspf1d.from_grid(
            degree=degree_x, x=x, knots=knots_x, n_basis=n_basis_x, domain=domain_x,
            use_clustering=use_clustering_x, order=order_x, num_boundary_points=num_boundary_points_x,
            correction=correction, use_gpu=use_gpu,
        )
        ym = bspf1d.from_grid(
            degree=degree_y, x=y, knots=knots_y, n_basis=n_basis_y, domain=domain_y,
            use_clustering=use_clustering_y, order=order_y, num_boundary_points=num_boundary_points_y,
            correction=correction, use_gpu=use_gpu,
        )
        zm = bspf1d.from_grid(
            degree=degree_z, x=z, knots=knots_z, n_basis=n_basis_z, domain=domain_z,
            use_clustering=use_clustering_z, order=order_z, num_boundary_points=num_boundary_points_z,
            correction=correction, use_gpu=use_gpu,
        )

        # Store arrays as provided (NumPy or CuPy)
        return cls(x=x, y=y, z=z, x_model=xm, y_model=ym, z_model=zm, use_gpu=use_gpu)
    
    def differentiate_1_2_batched(
        self,
        F,
        *,
        lam_x: float = 0.0,
        lam_y: float = 0.0,
        lam_z: float = 0.0,
    ):
        """
        Batched-only differentiate_1_2_batched with strict backend handling.
        """
        is_gpu_array = _HAS_CUPY and cp is not None and isinstance(F, cp.ndarray)
        has_batched = (
            hasattr(self.x_model, 'differentiate_1_2_batched')
            and hasattr(self.y_model, 'differentiate_1_2_batched')
            and hasattr(self.z_model, 'differentiate_1_2_batched')
        )
        if not has_batched:
            raise RuntimeError("Batched differentiate_1_2_batched not available on bspf1d models.")

        xp = cp if is_gpu_array else np
        
        # Detect if input is complex and use appropriate dtype
        if is_gpu_array:
            is_complex = cp.iscomplexobj(F) if _HAS_CUPY else False
        else:
            is_complex = np.iscomplexobj(F)
        
        if is_complex:
            F_arr = F if is_gpu_array else xp.asarray(F, dtype=xp.complex128)
        else:
            F_arr = F if is_gpu_array else xp.asarray(F, dtype=xp.float64)

        nz, ny, nx = F_arr.shape

        # X-direction: axis=2
        F_reshaped_x = F_arr.reshape(nz * ny, nx).T
        dF_dx_T, d2F_dx2_T, _ = self.x_model.differentiate_1_2_batched(F_reshaped_x, lam=lam_x)
        dF_dx = dF_dx_T.T.reshape(nz, ny, nx)
        d2F_dx2 = d2F_dx2_T.T.reshape(nz, ny, nx)

        # Y-direction: axis=1
        F_reshaped_y = F_arr.transpose(0, 2, 1).reshape(nz * nx, ny).T
        dF_dy_T, d2F_dy2_T, _ = self.y_model.differentiate_1_2_batched(F_reshaped_y, lam=lam_y)
        dF_dy = dF_dy_T.T.reshape(nz, nx, ny).transpose(0, 2, 1)
        d2F_dy2 = d2F_dy2_T.T.reshape(nz, nx, ny).transpose(0, 2, 1)

        # Z-direction: axis=0
        F_reshaped_z = F_arr.reshape(nz, ny * nx)
        dF_dz, d2F_dz2, _ = self.z_model.differentiate_1_2_batched(F_reshaped_z, lam=lam_z)
        dF_dz = dF_dz.reshape(nz, ny, nx)
        d2F_dz2 = d2F_dz2.reshape(nz, ny, nx)

        return (dF_dx, dF_dy, dF_dz, d2F_dx2, d2F_dy2, d2F_dz2)
