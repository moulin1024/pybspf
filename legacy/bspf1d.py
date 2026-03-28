from __future__ import annotations

import os

# Set CUDA_PATH for NVHPC SDK before importing CuPy
# This ensures CuPy can find CUDA headers during JIT compilation
if 'CUDA_PATH' not in os.environ:
    nvhpc_cuda_path = '/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6'
    if os.path.exists(nvhpc_cuda_path):
        os.environ['CUDA_PATH'] = nvhpc_cuda_path
        os.environ['CUDA_HOME'] = nvhpc_cuda_path

import math
import time
from typing import Dict, Optional, Tuple, Callable, List

import numpy as np
import numpy.typing as npt
from scipy import linalg as sla
from scipy.interpolate import BSpline, make_interp_spline

# Optional GPU backend
_HAS_CUPY = False
try:
    import cupy as cp
    import cupyx.scipy.linalg as cpla
    import cupyx.scipy.interpolate as cp_interp
    _HAS_CUPY = True
except Exception:
    cp = None
    cpla = None
    cp_interp = None

Array = npt.NDArray[np.float64]


# ============================== backend ==============================
class _Backend:
    """Switch between NumPy/SciPy and CuPy/CuPyX cleanly."""
    __slots__ = ("xp", "la", "fft", "is_gpu")

    def __init__(self, use_gpu: bool):
        if use_gpu:
            if not _HAS_CUPY:
                raise RuntimeError(
                    "use_gpu=True but CuPy is not available. "
                    "Install cupy (e.g. `pip install cupy-cuda12x`) or set use_gpu=False."
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
        """Explicit conversion to device (GPU). Use this for NumPy -> CuPy conversion."""
        if self.is_gpu:
            if _HAS_CUPY and isinstance(a, cp.ndarray):
                return a  # Already on device
            return cp.asarray(a, dtype=cp.float64)
        else:
            # CPU mode: return as-is (should be NumPy)
            if _HAS_CUPY and isinstance(a, cp.ndarray):
                raise ValueError(
                    "Cannot use CuPy array in CPU mode. "
                    "Use to_host() to convert CuPy array to NumPy, or set use_gpu=True."
                )
            return np.asarray(a, dtype=np.float64)

    def to_host(self, a):
        """Explicit conversion to host (CPU). Use this for CuPy -> NumPy conversion."""
        if self.is_gpu:
            if _HAS_CUPY and isinstance(a, cp.ndarray):
                return cp.asnumpy(a)
            # Already NumPy, but we're in GPU mode - this is an inconsistency
            raise ValueError(
                "Inconsistency detected: use_gpu=True but array is NumPy. "
                "Arrays must match the use_gpu setting. Use to_device() for explicit conversion."
            )
        else:
            # CPU mode: return as-is (should be NumPy)
            if _HAS_CUPY and isinstance(a, cp.ndarray):
                raise ValueError(
                    "Inconsistency detected: use_gpu=False but array is CuPy. "
                    "Arrays must match the use_gpu setting. Use to_host() for explicit conversion."
                )
            return a

    def ensure_like_input(self, out, input_was_numpy: bool):
        if self.is_gpu and input_was_numpy:
            raise ValueError(
                "Cannot convert GPU results back to NumPy. "
                "When use_gpu=True, provide CuPy arrays as input to avoid GPU↔CPU conversions. "
                "Either: (1) convert input to CuPy array before calling, or (2) use use_gpu=False."
            )
        return out
    
    def validate_device(self, a, name: str = "array"):
        """
        Validate that array is on the correct device for this backend.
        Throws error if there's a mismatch instead of implicitly converting.
        """
        if self.is_gpu:
            if _HAS_CUPY and not isinstance(a, cp.ndarray):
                raise ValueError(
                    f"Inconsistency detected: use_gpu=True but {name} is not a CuPy array. "
                    f"Arrays must match the use_gpu setting. Use to_device() for explicit conversion."
                )
        else:
            if _HAS_CUPY and isinstance(a, cp.ndarray):
                raise ValueError(
                    f"Inconsistency detected: use_gpu=False but {name} is a CuPy array. "
                    f"Arrays must match the use_gpu setting. Use to_host() for explicit conversion."
                )


# =============================================================================
# Low-level building blocks
# =============================================================================
class Grid1D:
    """Uniform 1D grid with rFFT frequencies and trapezoid weights."""
    def __init__(self, x: Array, *, atol: float = 1e-13, use_gpu: bool = False):
        # Detect if input is CuPy array
        is_gpu_array = _HAS_CUPY and isinstance(x, cp.ndarray)
        
        if is_gpu_array:
            if not use_gpu:
                raise ValueError(
                    "Cannot use CuPy array in Grid1D when use_gpu=False. "
                    "Either: (1) convert input to NumPy array, or (2) set use_gpu=True."
                )
            xp = cp
            x = cp.asarray(x, dtype=cp.float64)
        else:
            if use_gpu:
                raise ValueError(
                    "Cannot use NumPy array in Grid1D when use_gpu=True. "
                    "Either: (1) convert input to CuPy array, or (2) set use_gpu=False."
                )
            xp = np
            x = np.asarray(x, dtype=np.float64)
        
        if x.size < 2:
            raise ValueError("x must have at least 2 points.")
        
        # Compute dx (convert to Python float for compatibility)
        dx = float(x[1] - x[0])
        
        # Check uniform spacing using appropriate backend
        if is_gpu_array:
            if not cp.allclose(cp.diff(x), dx, rtol=0, atol=atol):
                raise ValueError("x must be uniformly spaced.")
        else:
            if not np.allclose(np.diff(x), dx, rtol=0, atol=atol):
                raise ValueError("x must be uniformly spaced.")
        
        self.x: Array = x
        self.dx: float = dx
        self.use_gpu: bool = use_gpu
        
        # Compute FFT frequencies using appropriate backend
        if is_gpu_array:
            self.omega: Array = 2.0 * cp.pi * cp.fft.rfftfreq(x.size, d=dx)
            w = cp.full(x.size, dx, dtype=cp.float64)
            w[0] = w[-1] = dx / 2.0
        else:
            self.omega: Array = 2.0 * np.pi * np.fft.rfftfreq(x.size, d=dx)
            w = np.full(x.size, dx, dtype=np.float64)
            w[0] = w[-1] = dx / 2.0
        self.trap: Array = w

    @property
    def a(self) -> float: 
        return float(self.x[0])

    @property
    def b(self) -> float: 
        return float(self.x[-1])

    @property
    def n(self) -> int: 
        return self.x.size


class _Knot:
    @staticmethod
    def _generate(
        *, degree: int, domain: Tuple[float, float], n_basis: int,
        use_clustering: bool, clustering_factor: float
    ) -> Array:
        if n_basis <= degree:
            raise ValueError("n_basis must exceed degree.")
        n_knots = n_basis + degree + 1
        n_interior = n_knots - 2 * (degree + 1)

        if n_interior > 0:
            u = np.linspace(-1.0, 1.0, n_interior + 2)
            if use_clustering:
                u = np.tanh(clustering_factor * u) / np.tanh(clustering_factor)
            uniq = degree * (u + 1.0) / 2.0  # in [0, degree]
            ks = [float(uniq[0])] * (degree + 1) + list(map(float, uniq[1:-1])) + [float(uniq[-1])] * (degree + 1)
            k = np.array(ks, dtype=np.float64)
        else:
            k = np.concatenate([np.zeros(degree + 1), np.full(degree + 1, degree)], dtype=np.float64)

        a, b = domain
        return (k / degree) * (b - a) + a

    @staticmethod
    def resolve(
        *, degree: int, grid: Grid1D,
        knots: Optional[Array],
        n_basis: Optional[int],
        domain: Optional[Tuple[float, float]],
        use_clustering: bool,
        clustering_factor: float,
    ) -> Array:
        if knots is not None:
            k = np.asarray(knots, dtype=np.float64)
            if k.ndim != 1:
                raise ValueError("knots must be a 1D array.")
            return k
        if n_basis is None:
            n_basis = 2 * (degree + 1) * 2
        if domain is None:
            domain = (grid.a, grid.b)
        return _Knot._generate(
            degree=degree, domain=domain, n_basis=n_basis,
            use_clustering=use_clustering, clustering_factor=clustering_factor
        )


class BSplineBasis1D:
    """B-spline basis on a uniform grid with lazy derivative matrices."""
    def __init__(self, degree: int, knots: Array, grid: Grid1D, use_gpu: bool = False):
        self.degree = int(degree)
        self.use_gpu = bool(use_gpu)
        
        # Store knots on appropriate device
        if use_gpu and _HAS_CUPY:
            if isinstance(knots, cp.ndarray):
                self.knots: Array = cp.asarray(knots, dtype=cp.float64)
            else:
                self.knots: Array = cp.asarray(knots, dtype=cp.float64)
        else:
            self.knots: Array = np.asarray(knots, dtype=np.float64)
        
        self.grid = grid

        self._splines = self._mk_splines()
        n_basis = len(self._splines)
        
        # Create B0 matrix on appropriate device
        # grid.x is already on the correct device (NumPy or CuPy) from Grid1D
        if use_gpu and _HAS_CUPY:
            xp = cp
            B0 = xp.empty((n_basis, grid.n), dtype=xp.float64)
            x_eval = grid.x
        else:
            xp = np
            B0 = xp.empty((n_basis, grid.n), dtype=xp.float64)
            x_eval = grid.x
        
        for i, s in enumerate(self._splines):
            B0[i, :] = s(x_eval)
        
        self._B0: Array = B0
        self._BT0: Array = B0.T.copy()

        self._BkT: Dict[int, Array] = {}
        self._eval_cache: Dict[Tuple[float, int], Array] = {}

    def _mk_splines(self):
        """Create BSpline objects (scipy or cupyx depending on use_gpu)."""
        n_basis = len(self.knots) - self.degree - 1
        
        if self.use_gpu and _HAS_CUPY:
            # Use cupyx.scipy.interpolate.BSpline for GPU
            xp = cp
            coeffs = xp.eye(n_basis, dtype=xp.float64)
            return [cp_interp.BSpline(self.knots, coeffs[i], self.degree) for i in range(n_basis)]
        else:
            # Use scipy.interpolate.BSpline for CPU
            if _HAS_CUPY and isinstance(self.knots, cp.ndarray):
                raise ValueError(
                    "Cannot convert CuPy knots to NumPy in CPU mode. "
                    "When use_gpu=False, provide NumPy arrays. "
                    "Either: (1) convert knots to NumPy before creating operator, or (2) use use_gpu=True."
                )
            knots_np = self.knots
            coeffs = np.eye(n_basis, dtype=np.float64)
            return [BSpline(knots_np, coeffs[i], self.degree) for i in range(n_basis)]

    def _evaluate_splines_vectorized(self, x: Array, deriv_order: int = 0) -> Array:
        cache_key = (float(x[0]), deriv_order)  # uniform-grid key
        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]
        n_basis = len(self._splines)
        
        if self.use_gpu and _HAS_CUPY:
            xp = cp
            result = xp.empty((n_basis, len(x)), dtype=xp.float64)
            x_eval = cp.asarray(x) if isinstance(x, np.ndarray) else x
        else:
            xp = np
            result = xp.empty((n_basis, len(x)), dtype=xp.float64)
            if _HAS_CUPY and isinstance(x, cp.ndarray):
                raise ValueError(
                    "Cannot convert CuPy array to NumPy in CPU mode. "
                    "When use_gpu=False, provide NumPy arrays. "
                    "Either: (1) convert input to NumPy before calling, or (2) use use_gpu=True."
                )
            x_eval = x
        
        for i, s in enumerate(self._splines):
            result[i, :] = (s.derivative(deriv_order) if deriv_order else s)(x_eval)
        self._eval_cache[cache_key] = result
        return result

    @property
    def B0(self) -> Array: return self._B0
    @property
    def BT0(self) -> Array: return self._BT0

    def BkT(self, k: int) -> Array:
        if k == 0:
            return self._BT0
        if k not in self._BkT:
            Bk = self._evaluate_splines_vectorized(self.grid.x, deriv_order=k)
            self._BkT[k] = Bk.T.copy()
        return self._BkT[k]

    def integrate_basis(self, a: float, b: float) -> Array:
        return np.array([s.integrate(a, b) for s in self._splines])


class EndpointOps1D:
    """Endpoint constraints and sample-to-endpoint operator for endpoint derivatives."""
    def __init__(self, basis: BSplineBasis1D, *, order: int, num_bd: int):
        self.order = int(order)
        self.num_bd = int(num_bd)
        self.grid = basis.grid
        self.use_gpu = basis.use_gpu
        
        # Detect backend from basis.B0
        if self.use_gpu and _HAS_CUPY and isinstance(basis.B0, cp.ndarray):
            xp = cp
            # Use cupy.linalg.solve for GPU (cupyx.scipy.linalg doesn't have solve)
            la_solve = cp.linalg.solve
        else:
            xp = np
            la_solve = sla.solve
        
        B0 = basis.B0
        Bk = {0: B0}
        for k in range(1, order + 1):
            Bk[k] = basis.BkT(k).T

        n_basis, n_points = B0.shape

        C = xp.zeros((2 * order, n_basis), dtype=xp.float64)
        for p in range(order):
            C[p, :] = Bk[p][:, 0]
            C[order + p, :] = Bk[p][:, -1]

        # meshgrid: CuPy uses np.meshgrid, NumPy uses np.meshgrid
        # For GPU, we need to use np.meshgrid and convert, or use cupy's equivalent
        if self.use_gpu and _HAS_CUPY:
            # CuPy doesn't have meshgrid, use NumPy and convert
            i_np, j_np = np.meshgrid(np.arange(num_bd), np.arange(num_bd), indexing="ij")
            i = cp.asarray(i_np)
            j = cp.asarray(j_np)
        else:
            i, j = np.meshgrid(np.arange(num_bd), np.arange(num_bd), indexing="ij")
        fact = xp.array([math.factorial(k) for k in range(num_bd)], dtype=xp.float64)
        A_left = (j**i) / fact[:, None]
        A_right = xp.flip(A_left * ((-1.0) ** i), axis=(0, 1))

        E_left = xp.eye(num_bd, dtype=xp.float64)[:order, :].T
        idx = xp.arange(num_bd - 1, num_bd - order - 1, -1)
        E_right = xp.eye(num_bd, dtype=xp.float64)[idx, :].T

        X_left = la_solve(A_left, E_left).T
        X_right = la_solve(A_right, E_right).T

        # Vectorized computation of dx powers (matches bfpsm1d)
        if self.use_gpu and _HAS_CUPY:
            dx_pows = xp.asarray(self.grid.dx ** np.arange(order, dtype=np.float64))
        else:
            dx_pows = self.grid.dx ** np.arange(order, dtype=np.float64)
        BND = xp.zeros((2 * order, n_points), dtype=xp.float64)
        BND[:order, :num_bd] = X_left / dx_pows[:, None]
        BND[order:, n_points - num_bd:] = X_right / dx_pows[:, None]

        self.C: Array = C.astype(xp.float64)
        self.BND: Array = BND.astype(xp.float64)
        self.X_left: Array = X_left.astype(xp.float64)
        self.X_right: Array = X_right.astype(xp.float64)


# =============================================================================
# Residual correction strategies
# =============================================================================
class ResidualCorrection:
    """Pluggable residual correction."""
    @staticmethod
    def none(residual: Array, omega: Array, *, kind: str, order: int, n: int, x: Optional[Array] = None) -> Array:
        return np.zeros(n, dtype=np.float64)

    @staticmethod
    def spectral(residual: Array, omega: Array, *, kind: str, order: int, n: int, x: Optional[Array] = None) -> Array:
        R = np.fft.rfft(residual)

        if kind == "diff":
            return np.fft.irfft(R * (1j * omega) ** order, n=n).astype(np.float64)

        if kind == "int":
            out_hat = np.zeros_like(R, dtype=np.complex128)
            nz = omega != 0.0
            out_hat[nz] = R[nz] / ((1j * omega[nz]) ** order)
            out = np.fft.irfft(out_hat, n=n).astype(np.float64)

            # Need x for correct nullspace handling
            if x is None:
                x0, x1 = 0.0, 1.0
                xx = np.linspace(x0, x1, n)
            else:
                xx = x
                x0 = float(xx[0]); x1 = float(xx[-1])

            if order == 1:
                mean_r = float(np.mean(residual))
                out = out + mean_r * (xx - x0)
                out -= out[0]
                return out

            if order == 2:
                mean_r = float(np.mean(residual))
                q = 0.5 * mean_r * (xx - x0) * (xx - x1)  # q'' = mean_r, q(x0)=q(x1)=0
                return out + q

            raise ValueError("Only int orders 1 and 2 are supported.")

        raise ValueError("kind must be 'diff' or 'int'.")


# =============================================================================
# Facade
# =============================================================================
class bspf1d:
    """Facade for 1D bspf: derivatives, definite integrals, antiderivatives."""
    def __init__(
        self,
        *,
        grid: Grid1D,
        degree: int,
        knots: Array,
        order: Optional[int] = None,
        num_boundary_points: Optional[int] = None,
        correction: str = "spectral",
        use_gpu: bool = False,      # <--- NEW
    ):
        # Set use_gpu FIRST (needed by basis creation)
        self.use_gpu = bool(use_gpu)
        self._bk = _Backend(self.use_gpu) if self.use_gpu else None
        
        self.grid = grid
        self.degree = int(degree)
        self.order = self.degree - 1 if order is None else int(order)
        self.num_bd = self.degree if num_boundary_points is None else int(num_boundary_points)
        # Detect backend (CuPy or NumPy) - knots should match grid's backend
        if _HAS_CUPY and isinstance(knots, cp.ndarray):
            self.knots = cp.asarray(knots, dtype=cp.float64)
        else:
            self.knots = np.asarray(knots, dtype=np.float64)

        self.basis = BSplineBasis1D(self.degree, self.knots, self.grid, use_gpu=self.use_gpu)
        
        # grid.trap is already on the correct device (NumPy or CuPy) from Grid1D
        trap = self.grid.trap
        
        self.BW = self.basis.B0 * trap
        self.Q = self.BW @ self.basis.B0.T

        self.end = EndpointOps1D(self.basis, order=self.order, num_bd=self.num_bd)

        # Precompute Fortran-order (column-major) copies for faster, more stable BLAS matvecs
        # This significantly reduces variability in rhs_build (see investigate_rhs_build.py)
        # Fortran-order is optimal for BLAS matrix-vector operations
        # Only create these for CPU - GPU arrays don't need this optimization
        if self.use_gpu and _HAS_CUPY:
            # For GPU, just store references (CuPy handles memory layout efficiently)
            self._BW_f = self.BW
            self._BND_f = self.end.BND
            self._BT0_f = self.basis.BT0
            self._B1T_f = self.basis.BkT(1)
            self._B2T_f = self.basis.BkT(2)
            self._B3T_f = self.basis.BkT(3)
            # Pre-allocate RHS buffer on GPU
            n_b = self.basis.B0.shape[0]
            self._rhs_buf = cp.empty(n_b + 2 * self.order, dtype=cp.float64)
            # Pre-compute FFT frequency multipliers on GPU
            # grid.omega is already a CuPy array from Grid1D
            omega = self.grid.omega
            self._iomega = 1j * omega
            self._iomega2 = self._iomega ** 2
            self._iomega3 = self._iomega ** 3
            # Pre-allocate buffer for residual computation on GPU
            self._residual_buf = cp.empty(self.grid.n, dtype=cp.float64)
        else:
            # CPU path: use Fortran-order for better BLAS performance
            self._BW_f = np.asfortranarray(self.BW)
            self._BND_f = np.asfortranarray(self.end.BND)
            self._BT0_f = np.asfortranarray(self.basis.BT0)
            # Cache BkT(1), BkT(2), and BkT(3) in Fortran order for differentiate_1_2 and differentiate_1_2_3
            self._B1T_f = np.asfortranarray(self.basis.BkT(1))
            self._B2T_f = np.asfortranarray(self.basis.BkT(2))
            self._B3T_f = np.asfortranarray(self.basis.BkT(3))
            # Pre-allocate RHS buffer to avoid concatenate overhead
            n_b = self.basis.B0.shape[0]
            self._rhs_buf = np.empty(n_b + 2 * self.order, dtype=np.float64)
            # Pre-compute FFT frequency multipliers for spectral correction
            # This avoids recomputing (1j * omega), (1j * omega)**2, and (1j * omega)**3 on every call
            omega = self.grid.omega
            self._iomega = 1j * omega  # Pre-computed for first derivative correction
            self._iomega2 = self._iomega ** 2  # Pre-computed for second derivative correction
            self._iomega3 = self._iomega ** 3  # Pre-computed for third derivative correction
            # Pre-allocate buffer for residual computation to avoid allocations
            self._residual_buf = np.empty(self.grid.n, dtype=np.float64)

        self.correction = correction  # Store for optimization checks
        if correction == "spectral":
            self._correct = lambda residual, omega, kind, order, n: ResidualCorrection.spectral(
                residual, omega, kind=kind, order=order, n=n, x=self.grid.x
            )
        else:
            self._correct = ResidualCorrection.none

        self._kkt_cache: Dict[float, Tuple[Array, Array]] = {}
        self._cached_arrays: Dict[str, Array] = {}

    @classmethod
    def from_grid(
        cls,
        degree: int,
        x: Array,
        *,
        knots: Optional[Array] = None,
        n_basis: Optional[int] = None,
        domain: Optional[Tuple[float, float]] = None,
        use_clustering: bool = False,
        clustering_factor: float = 2.0,
        order: Optional[int] = None,
        num_boundary_points: Optional[int] = None,
        correction: str = "spectral",
        use_gpu: bool = False,   # <--- NEW
    ) -> "bspf1d":
        # Grid1D now supports both NumPy and CuPy arrays directly
        # Convert x to appropriate backend if needed
        if use_gpu and _HAS_CUPY:
            if not isinstance(x, cp.ndarray):
                # Convert NumPy to CuPy if use_gpu=True
                x = cp.asarray(x, dtype=cp.float64)
        else:
            if _HAS_CUPY and isinstance(x, cp.ndarray):
                raise ValueError(
                    "Cannot use CuPy array when use_gpu=False. "
                    "Either: (1) convert input to NumPy array, or (2) set use_gpu=True."
                )
            x = np.asarray(x, dtype=np.float64)
        
        grid = Grid1D(x, use_gpu=use_gpu)
        k = _Knot.resolve(
            degree=degree, grid=grid, knots=knots, n_basis=n_basis, domain=domain,
            use_clustering=use_clustering, clustering_factor=clustering_factor
        )
        
        # Convert knots to GPU if use_gpu=True (knots come from _Knot.resolve which returns NumPy)
        if use_gpu and _HAS_CUPY:
            if not isinstance(k, cp.ndarray):
                k = cp.asarray(k, dtype=cp.float64)
        else:
            if _HAS_CUPY and isinstance(k, cp.ndarray):
                raise ValueError(
                    "Cannot use CuPy knots when use_gpu=False. "
                    "This indicates an internal inconsistency."
                )
            k = np.asarray(k, dtype=np.float64)
        
        return cls(
            grid=grid, degree=degree, knots=k,
            order=order, num_boundary_points=num_boundary_points, correction=correction,
            use_gpu=use_gpu,
        )

    # ---------- private solvers ----------
    def _kkt_lu(self, lam: float) -> Tuple[Array, Array]:
        """LU factorization cached by lambda (CPU/GPU aware)."""
        lam = float(lam)
        if lam in self._kkt_cache:
            return self._kkt_cache[lam]
        
        # Detect backend from self.Q
        if self.use_gpu and _HAS_CUPY and isinstance(self.Q, cp.ndarray):
            xp = cp
            la_factor = cpla.lu_factor
        else:
            xp = np
            la_factor = sla.lu_factor
        
        n_b = self.basis.B0.shape[0]
        m = 2 * self.order
        KKT = xp.zeros((n_b + m, n_b + m), dtype=xp.float64)
        KKT[:n_b, :n_b] = 2.0 * (self.Q + lam * xp.eye(n_b, dtype=xp.float64))
        KKT[:n_b, n_b:] = -self.end.C.T
        KKT[n_b:, :n_b] = self.end.C
        
        # LU factorization
        if self.use_gpu and _HAS_CUPY and isinstance(KKT, cp.ndarray):
            lu, piv = cpla.lu_factor(KKT)
        else:
            # CPU LU factorization - error if KKT is CuPy array
            if _HAS_CUPY and isinstance(KKT, cp.ndarray):
                raise ValueError(
                    "Cannot convert CuPy array to NumPy in CPU mode. "
                    "When use_gpu=False, internal arrays should be NumPy. "
                    "This indicates an internal inconsistency in the operator setup."
                )
            KKT_np = KKT
            lu, piv = sla.lu_factor(KKT_np)
        
        self._kkt_cache[lam] = (lu, piv)
        return lu, piv

    def _get_or_compute_array(self, key: str, compute_func: Callable[[], Array], *, no_cache: bool = False) -> Array:
        """
        Cache arrays (for compatibility with bfpsm1d API).
        
        Parameters
        ----------
        key : str
            Cache key
        compute_func : callable
            Function to compute the array if not cached
        no_cache : bool, optional
            If True, always recompute (don't cache). Use this for computations
            that depend on changing inputs like `f`. Default: False
        """
        if no_cache:
            return compute_func()
        if key not in self._cached_arrays:
            self._cached_arrays[key] = compute_func()
        return self._cached_arrays[key]

    # ---------- public operations ----------
    def differentiate(self, f: Array, k: int = 1, lam: float = 0.0, *,
        neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None) -> Tuple[Array, Array]:
        """
        GPU-aware derivative: if use_gpu=True, multiplies/FFTs/solves on device.
        Supports both real (float64) and complex (complex128) input arrays.
        For CPU real case, matches bfpsm1d performance exactly.
        """
        if k not in (1, 2, 3):
            raise ValueError("Only 1st/2nd/3rd derivatives are supported.")
        
        # Timing dictionary stored on the instance
        timings: dict = {}
        t_total_start = time.perf_counter()

        # Fast path: CPU + real input (most common case, no overhead)
        if not self.use_gpu:
            # Check if complex first, then convert to appropriate dtype
            is_complex = np.iscomplexobj(f)
            if is_complex:
                f = np.asarray(f, dtype=np.complex128)
            else:
                f = np.asarray(f, dtype=np.float64)
            
            # Real case: use exact bfpsm1d implementation for performance
            if not is_complex:
                if f.shape[0] != self.grid.n:
                    raise ValueError("Length of f must match grid size.")

                t_rhs_start = time.perf_counter()
                # Compute RHS using Fortran-order matrices for better BLAS performance
                # This reduces variability significantly (see investigate_rhs_build.py)
                # Use pre-allocated buffer to avoid concatenate overhead
                rhs = self._rhs_buf
                n_b = self._BW_f.shape[0]
                rhs[:n_b] = 2.0 * (self._BW_f @ f)
                rhs[n_b:] = self._BND_f @ f
                # Neumann BC: overwrite first-derivative rows
                if neumann_bc is not None:
                    if self.order < 1:
                        raise ValueError("Neumann BC requires self.order ≥ 1.")
                    left_flux, right_flux = neumann_bc
                    if left_flux is not None:
                        rhs[n_b + 1] = float(left_flux)
                    if right_flux is not None:
                        rhs[n_b + self.order + 1] = float(right_flux)

                lu, piv = self._kkt_lu(lam)
                sol = sla.lu_solve((lu, piv), rhs, overwrite_b=True)
                P = sol[: self.basis.B0.shape[0]]

                # Use Fortran-order matrices for spline evaluation
                f_spline = self._BT0_f @ P
                # For k-th derivative, use Fortran-order if available
                if k == 1:
                    df = self._B1T_f @ P
                elif k == 2:
                    df = self._B2T_f @ P
                else:  # k == 3
                    df = self._B3T_f @ P

                # FFT correction using pre-allocated buffer and pre-computed frequency multipliers
                residual = self._residual_buf
                residual[:] = f - f_spline  # In-place assignment to pre-allocated buffer
                
                # Use pre-computed frequency multipliers for k <= 3 (avoids recomputing (1j*omega)^k)
                if self.correction == "spectral" and k <= 3:
                    R = np.fft.rfft(residual)
                    if k == 1:
                        corr = np.fft.irfft(R * self._iomega, n=self.grid.n)
                    elif k == 2:
                        corr = np.fft.irfft(R * self._iomega2, n=self.grid.n)
                    else:  # k == 3
                        corr = np.fft.irfft(R * self._iomega3, n=self.grid.n)
                else:
                    # Fallback to general correction method for k > 3 or non-spectral correction
                    corr = self._correct(residual, self.grid.omega, kind="diff", order=k, n=self.grid.n)
                
                df = df + corr
                return df, f_spline
            
            # Complex case: CPU path with complex support
            f = f.astype(np.complex128, copy=False)
            if f.shape[0] != self.grid.n:
                raise ValueError("Length of f must match grid size.")

            # For complex, we need to handle real and imaginary parts
            # Use full FFT instead of rFFT
            rhs_2bw = 2.0 * (self.BW @ f)
            dY = self.end.BND @ f
            if neumann_bc is not None:
                if self.order < 1:
                    raise ValueError("Neumann BC requires self.order ≥ 1.")
                left_flux, right_flux = neumann_bc
                if left_flux is not None:
                    dY[1] = complex(left_flux)
                if right_flux is not None:
                    dY[self.order + 1] = complex(right_flux)
            
            rhs = np.concatenate((rhs_2bw, dY))

            lu, piv = self._kkt_lu(lam)
            sol = sla.lu_solve((lu, piv), rhs, overwrite_b=True)
            P = sol[: self.basis.B0.shape[0]]

            f_spline = self.basis.BT0 @ P
            df = self.basis.BkT(k) @ P

            residual = f - f_spline
            # For complex, use full FFT
            R = np.fft.fft(residual)
            omega = 2.0 * np.pi * np.fft.fftfreq(self.grid.n, d=self.grid.dx)
            corr = np.fft.ifft(R * (1j * omega) ** k)
            
            return (df + corr).astype(np.complex128), f_spline.astype(np.complex128)
        
        # GPU path (with complex support)
        bk = self._bk
        xp, la, fft = bk.xp, bk.la, bk.fft
        
        # Detect input type and convert to backend array (don't use np.asarray on CuPy arrays)
        input_was_numpy = isinstance(f, np.ndarray) or (_HAS_CUPY and not isinstance(f, cp.ndarray))
        if _HAS_CUPY and isinstance(f, cp.ndarray):
            # Input is CuPy array - use CuPy operations
            f_x = xp.asarray(f)
        else:
            # Input is NumPy array - convert to backend
            f_x = xp.asarray(f)
        
        if f_x.shape[0] != self.grid.n:
            raise ValueError("Length of f must match grid size.")
        
        # Detect if input is complex (use backend's iscomplexobj)
        if _HAS_CUPY and isinstance(f_x, cp.ndarray):
            is_complex = cp.iscomplexobj(f_x)
            if is_complex:
                # Ensure complex128 for complex inputs
                f_x = f_x.astype(xp.complex128)
            else:
                # Ensure float64 for real inputs
                f_x = f_x.astype(xp.float64)
        else:
            is_complex = np.iscomplexobj(f_x)
            if is_complex:
                # Ensure complex128 for complex inputs
                f_x = f_x.astype(np.complex128)
            else:
                # Ensure float64 for real inputs
                f_x = f_x.astype(np.float64)
        BW  = xp.asarray(self.BW)
        BND = xp.asarray(self.end.BND)
        BT0 = xp.asarray(self.basis.BT0)
        BkT = xp.asarray(self.basis.BkT(k))
        
        # Use full FFT frequencies for complex, rFFT frequencies for real
        if is_complex:
            # Use backend's fftfreq (CuPy or NumPy)
            if bk.is_gpu and _HAS_CUPY:
                om = xp.asarray(2.0 * cp.pi * cp.fft.fftfreq(self.grid.n, d=self.grid.dx))
            else:
                om = xp.asarray(2.0 * np.pi * np.fft.fftfreq(self.grid.n, d=self.grid.dx))
        else:
            om = xp.asarray(self.grid.omega)

        # Build RHS
        rhs_2bw = 2.0 * (BW @ f_x)
        dY = BND @ f_x

        # Neumann BC: overwrite first-derivative rows
        if neumann_bc is not None:
            if self.order < 1:
                raise ValueError("Neumann BC requires self.order ≥ 1.")
            left_flux, right_flux = neumann_bc
            # rows: 1 (left d/dx), order+1 (right d/dx)
            if left_flux is not None:
                dY[1] = complex(left_flux) if is_complex else float(left_flux)
            if right_flux is not None:
                dY[self.order + 1] = complex(right_flux) if is_complex else float(right_flux)

        rhs = xp.concatenate((rhs_2bw, dY), axis=0)

        # Solve KKT with cached CPU LU (copied to device if needed)
        lu_cpu, piv_cpu = self._kkt_lu(lam)
        if bk.is_gpu:
            # Convert LU factors to GPU arrays
            lu_gpu = xp.asarray(lu_cpu)
            piv_gpu = xp.asarray(piv_cpu)
            # If RHS is complex, convert LU factors to complex as well
            # (KKT matrix is real, so LU factors can be treated as complex with zero imaginary part)
            if is_complex and _HAS_CUPY and not cp.iscomplexobj(lu_gpu):
                lu_gpu = lu_gpu.astype(xp.complex128)
            SOL = la.lu_solve((lu_gpu, piv_gpu), rhs, overwrite_b=True)
        else:
            SOL = la.lu_solve((lu_cpu, piv_cpu), rhs, overwrite_b=True)

        n_b = self.basis.B0.shape[0]
        P = SOL[:n_b]

        f_spline = BT0 @ P
        df = BkT @ P

        residual = f_x - f_spline
        
        # Use appropriate FFT based on input type
        if is_complex:
            R = fft.fft(residual)
            corr = fft.ifft(R * (1j * om) ** k)
        else:
            R = fft.rfft(residual)
            corr = fft.irfft(R * (1j * om) ** k, n=self.grid.n)

        df_final = (df + corr)
        f_spline_out = f_spline

        # Return appropriate dtype (use backend's dtype)
        if bk.is_gpu and _HAS_CUPY:
            out_dtype = xp.complex128 if is_complex else xp.float64
        else:
            out_dtype = np.complex128 if is_complex else np.float64
        return (bk.ensure_like_input(df_final, input_was_numpy).astype(out_dtype),
                bk.ensure_like_input(f_spline_out, input_was_numpy).astype(out_dtype))

    def differentiate_1_2(self, f: Array, lam: float = 0.0, *,
                          neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None
                          ) -> Tuple[Array, Array, Array]:
        """
        Compute first & second derivatives together (GPU-aware).
        Supports both real (float64) and complex (complex128) input arrays.
        For CPU real case, matches bfpsm1d performance exactly.
        """
        # Initialize timing dictionary
        timings: dict = {}
        t_total_start = time.perf_counter()
        
        # Fast path: CPU + real input (most common case, no overhead)
        if not self.use_gpu:
            # Check if complex first, then convert to appropriate dtype
            is_complex = np.iscomplexobj(f)
            if is_complex:
                f = np.asarray(f, dtype=np.complex128)
            else:
                f = np.asarray(f, dtype=np.float64)
            
            # Real case: use exact bfpsm1d implementation for performance
            if not is_complex:
                if f.shape[0] != self.grid.n:
                    raise ValueError("Length of f must match grid size.")

                t_rhs_start = time.perf_counter()
                # Compute RHS using Fortran-order matrices for better BLAS performance
                # This reduces variability significantly (see investigate_rhs_build.py)
                # Use pre-allocated buffer to avoid concatenate overhead
                rhs = self._rhs_buf
                n_b = self._BW_f.shape[0]
                rhs[:n_b] = 2.0 * (self._BW_f @ f)
                rhs[n_b:] = self._BND_f @ f

                if neumann_bc is not None:
                    if self.order < 1:
                        raise ValueError("Neumann BC requires self.order ≥ 1.")
                    left_flux, right_flux = neumann_bc
                    if left_flux is not None:
                        rhs[n_b + 1] = float(left_flux)
                    if right_flux is not None:
                        rhs[n_b + self.order + 1] = float(right_flux)
                
                t_rhs_end = time.perf_counter()

                lu, piv = self._kkt_lu(lam)
                sol = sla.lu_solve((lu, piv), rhs, overwrite_b=True)
                P = sol[: self.basis.B0.shape[0]]
                t_solve_end = time.perf_counter()

                # Use Fortran-order matrices for spline evaluation
                f_spline = self._BT0_f @ P
                df1_spline = self._B1T_f @ P
                df2_spline = self._B2T_f @ P
                t_spline_end = time.perf_counter()

                # FFT correction using pre-computed frequency multipliers and buffers
                residual = self._residual_buf
                residual[:] = f - f_spline  # In-place assignment to pre-allocated buffer
                R = np.fft.rfft(residual)  # FFT returns new array, but residual buffer is reused
                
                # Use pre-computed frequency multipliers (avoids recomputing 1j*omega each time)
                corr1 = np.fft.irfft(R * self._iomega, n=self.grid.n)
                corr2 = np.fft.irfft(R * self._iomega2, n=self.grid.n)
                t_fft_end = time.perf_counter()

                df1 = (df1_spline + corr1).astype(np.float64)
                df2 = (df2_spline + corr2).astype(np.float64)

                t_total_end = time.perf_counter()
                timings["rhs_build"] = t_rhs_end - t_rhs_start
                timings["kkt_solve"] = t_solve_end - t_rhs_end
                timings["spline_eval"] = t_spline_end - t_solve_end
                timings["fft_corr"] = t_fft_end - t_spline_end
                timings["total"] = t_total_end - t_total_start
                self.last_timing_d12 = timings

                return df1, df2, f_spline.astype(np.float64)
            
            # Complex case: CPU path with complex support
            f = f.astype(np.complex128, copy=False)
            if f.shape[0] != self.grid.n:
                raise ValueError("Length of f must match grid size.")

            t_rhs_start = time.perf_counter()
            rhs_2bw = 2.0 * (self.BW @ f)
            dY = self.end.BND @ f

            if neumann_bc is not None:
                if self.order < 1:
                    raise ValueError("Neumann BC requires self.order ≥ 1.")
                left_flux, right_flux = neumann_bc
                if left_flux is not None:
                    dY[1] = complex(left_flux)
                if right_flux is not None:
                    dY[self.order + 1] = complex(right_flux)

            rhs = np.concatenate((rhs_2bw, dY))
            t_rhs_end = time.perf_counter()

            lu, piv = self._kkt_lu(lam)
            sol = sla.lu_solve((lu, piv), rhs, overwrite_b=True)
            P = sol[: self.basis.B0.shape[0]]
            t_solve_end = time.perf_counter()

            f_spline = self.basis.BT0 @ P
            df1_spline = self.basis.BkT(1) @ P
            df2_spline = self.basis.BkT(2) @ P
            t_spline_end = time.perf_counter()

            residual = f - f_spline
            # For complex, use full FFT
            R = np.fft.fft(residual)
            omega = 2.0 * np.pi * np.fft.fftfreq(self.grid.n, d=self.grid.dx)
            
            corr1 = np.fft.ifft(R * (1j * omega))
            corr2 = np.fft.ifft(R * (1j * omega) ** 2)
            t_fft_end = time.perf_counter()

            df1 = (df1_spline + corr1).astype(np.complex128)
            df2 = (df2_spline + corr2).astype(np.complex128)

            t_total_end = time.perf_counter()
            timings["rhs_build"] = t_rhs_end - t_rhs_start
            timings["kkt_solve"] = t_solve_end - t_rhs_end
            timings["spline_eval"] = t_spline_end - t_solve_end
            timings["fft_corr"] = t_fft_end - t_spline_end
            timings["total"] = t_total_end - t_total_start
            self.last_timing_d12 = timings

            return df1, df2, f_spline.astype(np.complex128)
        
        # GPU path (with complex support)
        bk = self._bk
        xp, la, fft = bk.xp, bk.la, bk.fft
        
        # Detect input type and convert to backend array (don't use np.asarray on CuPy arrays)
        input_was_numpy = isinstance(f, np.ndarray) or (_HAS_CUPY and not isinstance(f, cp.ndarray))
        if _HAS_CUPY and isinstance(f, cp.ndarray):
            # Input is CuPy array - use CuPy operations
            f_x = xp.asarray(f)
        else:
            # Input is NumPy array - convert to backend
            f_x = xp.asarray(f)
        
        if f_x.shape[0] != self.grid.n:
            raise ValueError("Length of f must match grid size.")
        
        # Detect if input is complex (use backend's iscomplexobj)
        if _HAS_CUPY and isinstance(f_x, cp.ndarray):
            is_complex = cp.iscomplexobj(f_x)
            if is_complex:
                # Ensure complex128 for complex inputs
                f_x = f_x.astype(xp.complex128)
            else:
                # Ensure float64 for real inputs
                f_x = f_x.astype(xp.float64)
        else:
            is_complex = np.iscomplexobj(f_x)
            if is_complex:
                # Ensure complex128 for complex inputs
                f_x = f_x.astype(np.complex128)
            else:
                # Ensure float64 for real inputs
                f_x = f_x.astype(np.float64)

        BW  = xp.asarray(self.BW)
        BND = xp.asarray(self.end.BND)
        BT0 = xp.asarray(self.basis.BT0)
        B1T = xp.asarray(self.basis.BkT(1))
        B2T = xp.asarray(self.basis.BkT(2))
        
        # Use full FFT frequencies for complex, rFFT frequencies for real
        if is_complex:
            # Use backend's fftfreq (CuPy or NumPy)
            if bk.is_gpu and _HAS_CUPY:
                om = xp.asarray(2.0 * cp.pi * cp.fft.fftfreq(self.grid.n, d=self.grid.dx))
            else:
                om = xp.asarray(2.0 * np.pi * np.fft.fftfreq(self.grid.n, d=self.grid.dx))
        else:
            om = xp.asarray(self.grid.omega)

        t_rhs_start = time.perf_counter()
        rhs_2bw = 2.0 * (BW @ f_x)
        dY = BND @ f_x

        if neumann_bc is not None:
            if self.order < 1:
                raise ValueError("Neumann BC requires self.order ≥ 1.")
            left_flux, right_flux = neumann_bc
            if left_flux is not None:
                dY[1] = complex(left_flux) if is_complex else float(left_flux)
            if right_flux is not None:
                dY[self.order + 1] = complex(right_flux) if is_complex else float(right_flux)

        rhs = xp.concatenate((rhs_2bw, dY), axis=0)
        t_rhs_end = time.perf_counter()

        lu_cpu, piv_cpu = self._kkt_lu(lam)
        if bk.is_gpu:
            # Convert LU factors to GPU arrays
            lu_gpu = xp.asarray(lu_cpu)
            piv_gpu = xp.asarray(piv_cpu)
            # If RHS is complex, convert LU factors to complex as well
            # (KKT matrix is real, so LU factors can be treated as complex with zero imaginary part)
            if is_complex and not cp.iscomplexobj(lu_gpu):
                lu_gpu = lu_gpu.astype(xp.complex128)
            SOL = la.lu_solve((lu_gpu, piv_gpu), rhs, overwrite_b=True)
        else:
            SOL = la.lu_solve((lu_cpu, piv_cpu), bk.to_host(rhs))
            SOL = xp.asarray(SOL)
        t_solve_end = time.perf_counter()

        n_b = self.basis.B0.shape[0]
        P = SOL[:n_b]

        f_spline = BT0 @ P
        df1_spline = B1T @ P
        df2_spline = B2T @ P
        t_spline_end = time.perf_counter()

        residual = f_x - f_spline
        
        # Use appropriate FFT based on input type
        if is_complex:
            R = fft.fft(residual)
            corr1 = fft.ifft(R * (1j * om))
            corr2 = fft.ifft(R * (1j * om) ** 2)
        else:
            R = fft.rfft(residual)
            corr1 = fft.irfft(R * (1j * om), n=self.grid.n)
            corr2 = fft.irfft(R * (1j * om) ** 2, n=self.grid.n)
        t_fft_end = time.perf_counter()

        df1 = df1_spline + corr1
        df2 = df2_spline + corr2

        # Return appropriate dtype (use backend's dtype)
        if bk.is_gpu and _HAS_CUPY:
            out_dtype = xp.complex128 if is_complex else xp.float64
        else:
            out_dtype = np.complex128 if is_complex else np.float64
        t_total_end = time.perf_counter()
        timings["rhs_build"] = t_rhs_end - t_rhs_start
        timings["kkt_solve"] = t_solve_end - t_rhs_end
        timings["spline_eval"] = t_spline_end - t_solve_end
        timings["fft_corr"] = t_fft_end - t_spline_end
        timings["total"] = t_total_end - t_total_start
        self.last_timing_d12 = timings

        return (bk.ensure_like_input(df1, input_was_numpy).astype(out_dtype),
                bk.ensure_like_input(df2, input_was_numpy).astype(out_dtype),
                bk.ensure_like_input(f_spline, input_was_numpy).astype(out_dtype))

    def differentiate_1_2_3(self, f: Array, lam: float = 0.0, *,
                            neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None
                            ) -> Tuple[Array, Array, Array, Array]:
        """
        Compute first, second & third derivatives together (GPU-aware).
        Supports both real (float64) and complex (complex128) input arrays.
        For CPU real case, matches bfpsm1d performance exactly.
        
        Returns
        -------
        df1 : Array
            First derivative
        df2 : Array
            Second derivative
        df3 : Array
            Third derivative
        f_spline : Array
            Spline approximation
        """
        # Initialize timing dictionary
        timings: dict = {}
        t_total_start = time.perf_counter()
        
        # Fast path: CPU + real input (most common case, no overhead)
        if not self.use_gpu:
            # Check if complex first, then convert to appropriate dtype
            is_complex = np.iscomplexobj(f)
            if is_complex:
                f = np.asarray(f, dtype=np.complex128)
            else:
                f = np.asarray(f, dtype=np.float64)
            
            # Real case: use exact bfpsm1d implementation for performance
            if not is_complex:
                if f.shape[0] != self.grid.n:
                    raise ValueError("Length of f must match grid size.")

                t_rhs_start = time.perf_counter()
                # Compute RHS using Fortran-order matrices for better BLAS performance
                rhs = self._rhs_buf
                n_b = self._BW_f.shape[0]
                rhs[:n_b] = 2.0 * (self._BW_f @ f)
                rhs[n_b:] = self._BND_f @ f

                if neumann_bc is not None:
                    if self.order < 1:
                        raise ValueError("Neumann BC requires self.order ≥ 1.")
                    left_flux, right_flux = neumann_bc
                    if left_flux is not None:
                        rhs[n_b + 1] = float(left_flux)
                    if right_flux is not None:
                        rhs[n_b + self.order + 1] = float(right_flux)
                
                t_rhs_end = time.perf_counter()

                lu, piv = self._kkt_lu(lam)
                sol = sla.lu_solve((lu, piv), rhs, overwrite_b=False)
                P = sol[: self.basis.B0.shape[0]]
                t_solve_end = time.perf_counter()

                # Use Fortran-order matrices for spline evaluation
                f_spline = self._BT0_f @ P
                df1_spline = self._B1T_f @ P
                df2_spline = self._B2T_f @ P
                df3_spline = self._B3T_f @ P
                t_spline_end = time.perf_counter()

                # FFT correction using pre-computed frequency multipliers and buffers
                residual = self._residual_buf
                residual[:] = f - f_spline  # In-place assignment to pre-allocated buffer
                R = np.fft.rfft(residual)  # FFT returns new array, but residual buffer is reused
                
                # Use pre-computed frequency multipliers
                corr1 = np.fft.irfft(R * self._iomega, n=self.grid.n)
                corr2 = np.fft.irfft(R * self._iomega2, n=self.grid.n)
                corr3 = np.fft.irfft(R * self._iomega3, n=self.grid.n)
                t_fft_end = time.perf_counter()

                df1 = (df1_spline + corr1).astype(np.float64)
                df2 = (df2_spline + corr2).astype(np.float64)
                df3 = (df3_spline + corr3).astype(np.float64)

                t_total_end = time.perf_counter()
                timings["rhs_build"] = t_rhs_end - t_rhs_start
                timings["kkt_solve"] = t_solve_end - t_rhs_end
                timings["spline_eval"] = t_spline_end - t_solve_end
                timings["fft_corr"] = t_fft_end - t_spline_end
                timings["total"] = t_total_end - t_total_start
                self.last_timing_d123 = timings

                return df1, df2, df3, f_spline.astype(np.float64)
            
            # Complex case: CPU path with complex support
            f = f.astype(np.complex128, copy=False)
            if f.shape[0] != self.grid.n:
                raise ValueError("Length of f must match grid size.")

            t_rhs_start = time.perf_counter()
            rhs_2bw = 2.0 * (self.BW @ f)
            dY = self.end.BND @ f

            if neumann_bc is not None:
                if self.order < 1:
                    raise ValueError("Neumann BC requires self.order ≥ 1.")
                left_flux, right_flux = neumann_bc
                if left_flux is not None:
                    dY[1] = complex(left_flux)
                if right_flux is not None:
                    dY[self.order + 1] = complex(right_flux)

            rhs = np.concatenate((rhs_2bw, dY))
            t_rhs_end = time.perf_counter()

            lu, piv = self._kkt_lu(lam)
            sol = sla.lu_solve((lu, piv), rhs, overwrite_b=False)
            P = sol[: self.basis.B0.shape[0]]
            t_solve_end = time.perf_counter()

            f_spline = self.basis.BT0 @ P
            df1_spline = self.basis.BkT(1) @ P
            df2_spline = self.basis.BkT(2) @ P
            df3_spline = self.basis.BkT(3) @ P
            t_spline_end = time.perf_counter()

            residual = f - f_spline
            # For complex, use full FFT
            R = np.fft.fft(residual)
            omega = 2.0 * np.pi * np.fft.fftfreq(self.grid.n, d=self.grid.dx)
            
            corr1 = np.fft.ifft(R * (1j * omega))
            corr2 = np.fft.ifft(R * (1j * omega) ** 2)
            corr3 = np.fft.ifft(R * (1j * omega) ** 3)
            t_fft_end = time.perf_counter()

            df1 = (df1_spline + corr1).astype(np.complex128)
            df2 = (df2_spline + corr2).astype(np.complex128)
            df3 = (df3_spline + corr3).astype(np.complex128)

            t_total_end = time.perf_counter()
            timings["rhs_build"] = t_rhs_end - t_rhs_start
            timings["kkt_solve"] = t_solve_end - t_rhs_end
            timings["spline_eval"] = t_spline_end - t_solve_end
            timings["fft_corr"] = t_fft_end - t_spline_end
            timings["total"] = t_total_end - t_total_start
            self.last_timing_d123 = timings

            return df1, df2, df3, f_spline.astype(np.complex128)
        
        # GPU path (with complex support)
        bk = self._bk
        xp, la, fft = bk.xp, bk.la, bk.fft
        
        # Detect input type and convert to backend array
        input_was_numpy = isinstance(f, np.ndarray) or (_HAS_CUPY and not isinstance(f, cp.ndarray))
        if _HAS_CUPY and isinstance(f, cp.ndarray):
            f_x = xp.asarray(f)
        else:
            f_x = xp.asarray(f)
        
        if f_x.shape[0] != self.grid.n:
            raise ValueError("Length of f must match grid size.")
        
        # Detect if input is complex
        if _HAS_CUPY and isinstance(f_x, cp.ndarray):
            is_complex = cp.iscomplexobj(f_x)
            if is_complex:
                f_x = f_x.astype(xp.complex128)
            else:
                f_x = f_x.astype(xp.float64)
        else:
            is_complex = np.iscomplexobj(f_x)
            if is_complex:
                f_x = f_x.astype(np.complex128)
            else:
                f_x = f_x.astype(np.float64)

        BW  = xp.asarray(self.BW)
        BND = xp.asarray(self.end.BND)
        BT0 = xp.asarray(self.basis.BT0)
        B1T = xp.asarray(self.basis.BkT(1))
        B2T = xp.asarray(self.basis.BkT(2))
        B3T = xp.asarray(self.basis.BkT(3))
        
        # Use full FFT frequencies for complex, rFFT frequencies for real
        if is_complex:
            if bk.is_gpu and _HAS_CUPY:
                om = xp.asarray(2.0 * cp.pi * cp.fft.fftfreq(self.grid.n, d=self.grid.dx))
            else:
                om = xp.asarray(2.0 * np.pi * np.fft.fftfreq(self.grid.n, d=self.grid.dx))
        else:
            om = xp.asarray(self.grid.omega)

        t_rhs_start = time.perf_counter()
        rhs_2bw = 2.0 * (BW @ f_x)
        dY = BND @ f_x

        if neumann_bc is not None:
            if self.order < 1:
                raise ValueError("Neumann BC requires self.order ≥ 1.")
            left_flux, right_flux = neumann_bc
            if left_flux is not None:
                dY[1] = complex(left_flux) if is_complex else float(left_flux)
            if right_flux is not None:
                dY[self.order + 1] = complex(right_flux) if is_complex else float(right_flux)

        rhs = xp.concatenate((rhs_2bw, dY), axis=0)
        t_rhs_end = time.perf_counter()

        lu_cpu, piv_cpu = self._kkt_lu(lam)
        if bk.is_gpu:
            lu_gpu = xp.asarray(lu_cpu)
            piv_gpu = xp.asarray(piv_cpu)
            if is_complex and _HAS_CUPY and not cp.iscomplexobj(lu_gpu):
                lu_gpu = lu_gpu.astype(xp.complex128)
            SOL = la.lu_solve((lu_gpu, piv_gpu), rhs)
        else:
            SOL = la.lu_solve((lu_cpu, piv_cpu), bk.to_host(rhs), overwrite_b=True)
            SOL = xp.asarray(SOL)
        t_solve_end = time.perf_counter()

        n_b = self.basis.B0.shape[0]
        P = SOL[:n_b]

        f_spline = BT0 @ P
        df1_spline = B1T @ P
        df2_spline = B2T @ P
        df3_spline = B3T @ P
        t_spline_end = time.perf_counter()

        residual = f_x - f_spline
        
        # Use appropriate FFT based on input type
        if is_complex:
            R = fft.fft(residual)
            corr1 = fft.ifft(R * (1j * om))
            corr2 = fft.ifft(R * (1j * om) ** 2)
            corr3 = fft.ifft(R * (1j * om) ** 3)
        else:
            R = fft.rfft(residual)
            corr1 = fft.irfft(R * (1j * om), n=self.grid.n)
            corr2 = fft.irfft(R * (1j * om) ** 2, n=self.grid.n)
            corr3 = fft.irfft(R * (1j * om) ** 3, n=self.grid.n)
        t_fft_end = time.perf_counter()

        df1 = df1_spline + corr1
        df2 = df2_spline + corr2
        df3 = df3_spline + corr3

        # Return appropriate dtype
        if bk.is_gpu and _HAS_CUPY:
            out_dtype = xp.complex128 if is_complex else xp.float64
        else:
            out_dtype = np.complex128 if is_complex else np.float64
        t_total_end = time.perf_counter()
        timings["rhs_build"] = t_rhs_end - t_rhs_start
        timings["kkt_solve"] = t_solve_end - t_rhs_end
        timings["spline_eval"] = t_spline_end - t_solve_end
        timings["fft_corr"] = t_fft_end - t_spline_end
        timings["total"] = t_total_end - t_total_start
        self.last_timing_d123 = timings

        return (bk.ensure_like_input(df1, input_was_numpy).astype(out_dtype),
                bk.ensure_like_input(df2, input_was_numpy).astype(out_dtype),
                bk.ensure_like_input(df3, input_was_numpy).astype(out_dtype),
                bk.ensure_like_input(f_spline, input_was_numpy).astype(out_dtype))

    def differentiate_1_2_batched(self, f: Array, lam: float = 0.0, *,
                                  neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None
                                  ) -> Tuple[Array, Array, Array]:
        """
        Batched version of differentiate_1_2: compute first & second derivatives together
        for multiple columns/rows at once.
        
        Parameters
        ----------
        f : Array
            Input array of shape (n, batch) where n is the grid size and batch is the number
            of columns/rows to process together
        lam : float, default 0.0
            Tikhonov regularization parameter
        neumann_bc : tuple, optional
            Neumann boundary conditions (left_flux, right_flux) for each batch.
            If provided, should be (left_flux, right_flux) where each can be:
            - scalar: same BC for all batches
            - array of shape (batch,): different BC for each batch
        
        Returns
        -------
        df1 : Array
            First derivative, shape (n, batch)
        df2 : Array
            Second derivative, shape (n, batch)
        f_spline : Array
            Spline approximation, shape (n, batch)
        """
        # CPU path (supports real and complex)
        if not self.use_gpu:
            f = np.asarray(f)
            if f.ndim != 2:
                raise ValueError("f must be 2D with shape (n, batch)")
            n, batch = f.shape
            if n != self.grid.n:
                raise ValueError(f"First dimension of f ({n}) must match grid size ({self.grid.n})")

            is_complex = np.iscomplexobj(f)
            dtype = np.complex128 if is_complex else np.float64
            f = f.astype(dtype, copy=False)

            BW = self._BW_f.astype(dtype, copy=False)
            BND = self._BND_f.astype(dtype, copy=False)
            BT0 = self._BT0_f.astype(dtype, copy=False)
            B1T = self._B1T_f.astype(dtype, copy=False)
            B2T = self._B2T_f.astype(dtype, copy=False)
            
            rhs_top = 2.0 * (BW @ f)
            dY = BND @ f
            
            if neumann_bc is not None:
                if self.order < 1:
                    raise ValueError("Neumann BC requires self.order ≥ 1.")
                left_flux, right_flux = neumann_bc
                if left_flux is not None:
                    dY[1, :] = np.asarray(left_flux, dtype=dtype)
                if right_flux is not None:
                    dY[self.order + 1, :] = np.asarray(right_flux, dtype=dtype)
            
            rhs = np.vstack([rhs_top, dY]).astype(dtype, copy=False)
            
            lu, piv = self._kkt_lu(lam)
            sol = sla.lu_solve((lu.astype(dtype, copy=False), piv), rhs, overwrite_b=False)
            n_b = BW.shape[0]
            P = sol[:n_b, :]
            
            f_spline = BT0 @ P
            df1_spline = B1T @ P
            df2_spline = B2T @ P
            
            residual = f - f_spline
            if is_complex:
                R = np.fft.fft(residual, axis=0)
                omega_full = 2.0 * np.pi * np.fft.fftfreq(self.grid.n, d=self.grid.dx)
                corr1 = np.fft.ifft(R * (1j * omega_full)[:, None], n=self.grid.n, axis=0)
                corr2 = np.fft.ifft(R * (1j * omega_full)[:, None]**2, n=self.grid.n, axis=0)
                out_dtype = np.complex128
            else:
                R = np.fft.rfft(residual, axis=0)
                corr1 = np.fft.irfft(R * self._iomega[:, None], n=self.grid.n, axis=0)
                corr2 = np.fft.irfft(R * self._iomega2[:, None], n=self.grid.n, axis=0)
                out_dtype = np.float64
            
            df1 = (df1_spline + corr1).astype(out_dtype, copy=False)
            df2 = (df2_spline + corr2).astype(out_dtype, copy=False)
            f_spline = f_spline.astype(out_dtype, copy=False)
            
            return df1, df2, f_spline
        
        # GPU path (supports real and complex)
        bk = self._bk
        xp, la, fft = bk.xp, bk.la, bk.fft
        
        input_was_numpy = isinstance(f, np.ndarray) or (_HAS_CUPY and not isinstance(f, cp.ndarray))
        f_x = xp.asarray(f)
        
        if f_x.ndim != 2:
            raise ValueError("f must be 2D with shape (n, batch)")
        n, batch = f_x.shape
        if n != self.grid.n:
            raise ValueError(f"First dimension of f ({n}) must match grid size ({self.grid.n})")
        
        is_complex = cp.iscomplexobj(f_x) if _HAS_CUPY and isinstance(f_x, cp.ndarray) else np.iscomplexobj(f_x)
        dtype = xp.complex128 if is_complex else xp.float64
        f_x = f_x.astype(dtype)
        
        BW = xp.asarray(self.BW).astype(dtype)
        BND = xp.asarray(self.end.BND).astype(dtype)
        BT0 = xp.asarray(self.basis.BT0).astype(dtype)
        B1T = xp.asarray(self.basis.BkT(1)).astype(dtype)
        B2T = xp.asarray(self.basis.BkT(2)).astype(dtype)
        om_real = xp.asarray(self.grid.omega)
        
        # Build RHS for all batches
        rhs_top = 2.0 * (BW @ f_x)  # (n_b, batch)
        dY = BND @ f_x  # (m, batch)
        
        if neumann_bc is not None:
            if self.order < 1:
                raise ValueError("Neumann BC requires self.order ≥ 1.")
            left_flux, right_flux = neumann_bc
            if left_flux is not None:
                lf = xp.asarray(left_flux, dtype=dtype)
                dY[1, :] = lf if lf.ndim > 0 else float(lf)
            if right_flux is not None:
                rf = xp.asarray(right_flux, dtype=dtype)
                dY[self.order + 1, :] = rf if rf.ndim > 0 else float(rf)
        
        rhs = xp.vstack([rhs_top, dY]).astype(dtype)
        
        # Solve KKT system
        lu_cpu, piv_cpu = self._kkt_lu(lam)
        if bk.is_gpu:
            lu_gpu = xp.asarray(lu_cpu).astype(dtype, copy=False)
            piv_gpu = xp.asarray(piv_cpu)
            SOL = la.lu_solve((lu_gpu, piv_gpu), rhs, overwrite_b=True)
        else:
            SOL = la.lu_solve((lu_cpu.astype(dtype, copy=False), piv_cpu), bk.to_host(rhs), overwrite_b=True)
            SOL = xp.asarray(SOL)
        
        n_b = self.basis.B0.shape[0]
        P = SOL[:n_b, :]
        
        # Evaluate splines
        f_spline = BT0 @ P
        df1_spline = B1T @ P
        df2_spline = B2T @ P
        
        # FFT correction
        residual = f_x - f_spline
        om_real = xp.asarray(self.grid.omega)
        if is_complex:
            R = fft.fft(residual, axis=0)
            omega_full = 2.0 * xp.pi * fft.fftfreq(self.grid.n, d=self.grid.dx)
            corr1 = fft.ifft(R * (1j * omega_full)[:, None], n=self.grid.n, axis=0)
            corr2 = fft.ifft(R * (1j * omega_full)[:, None]**2, n=self.grid.n, axis=0)
            out_dtype = xp.complex128
        else:
            R = fft.rfft(residual, axis=0)
            corr1 = fft.irfft(R * (1j * om_real)[:, None], n=self.grid.n, axis=0)
            corr2 = fft.irfft(R * (1j * om_real)[:, None]**2, n=self.grid.n, axis=0)
            out_dtype = xp.float64
        
        df1 = (df1_spline + corr1).astype(out_dtype, copy=False)
        df2 = (df2_spline + corr2).astype(out_dtype, copy=False)
        f_spline = f_spline.astype(out_dtype, copy=False)
        
        return (bk.ensure_like_input(df1, input_was_numpy),
                bk.ensure_like_input(df2, input_was_numpy),
                bk.ensure_like_input(f_spline, input_was_numpy))

    def definite_integral(self, f: Array, a: Optional[float] = None, b: Optional[float] = None, lam: float = 0.0) -> float:
        """
        Integral remains CPU-based (splines & trapezoid on host); this is fine since
        it's typically not the performance bottleneck.
        """
        f = np.asarray(f, dtype=np.float64)
        if f.shape[0] != self.grid.n:
            raise ValueError("Length of f must match grid size.")
        a = self.grid.a if a is None else float(a)
        b = self.grid.b if b is None else float(b)

        rhs_2bw = 2.0 * (self.BW @ f)
        dY = self.end.BND @ f
        rhs = np.concatenate((rhs_2bw, dY))

        lu, piv = self._kkt_lu(lam)
        sol = sla.lu_solve((lu, piv), rhs, overwrite_b=True)
        P = sol[: self.basis.B0.shape[0]]

        basis_integrals = self.basis.integrate_basis(a, b)
        spline_integral = basis_integrals @ P

        residual = f - (self.basis.BT0 @ P)
        residual_integral = np.sum(residual * self.grid.trap)
        return float(spline_integral + residual_integral)

    def antiderivative(
        self,
        f: Array,
        order: int = 1,
        *,
        left_value: float = 0.0,
        match_right: Optional[float] = None,
        lam: float = 0.0,
    ) -> Array:
        """
        GPU-aware spectral correction; spline antiderivative is evaluated on CPU (BSpline API),
        then combined on the selected backend.
        """
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2.")

        f = np.asarray(f, dtype=np.float64)
        if f.shape[0] != self.grid.n:
            raise ValueError("Length of f must match grid size.")

        # CPU spline solve (same as before)
        rhs_2bw = 2.0 * (self.BW @ f)
        dY = self.end.BND @ f
        rhs = np.concatenate((rhs_2bw, dY))
        lu, piv = self._kkt_lu(lam)
        sol = sla.lu_solve((lu, piv), rhs, overwrite_b=True)
        P = sol[: self.basis.B0.shape[0]]

        x = self.grid.x
        f_spline = self.basis.BT0 @ P

        # Antiderivative of the spline part (CPU)
        F_spline_host = np.zeros_like(x)
        for i, s in enumerate(self.basis._splines):
            s_int = s.antiderivative(order)
            F_spline_host += P[i] * s_int(x)

        # Residual correction via spectral integration (GPU-aware)
        bk = self._bk
        if bk is None:
            # CPU mode: use NumPy directly
            xp, fft = np, np.fft
            input_was_numpy = True
        else:
            xp, fft = bk.xp, bk.fft
            input_was_numpy = True  # this method returns NumPy anyway

        residual = xp.asarray(f - f_spline)
        om = xp.asarray(self.grid.omega)

        R = fft.rfft(residual)
        if order == 1:
            # Avoid divide-by-zero warning by handling DC component separately
            # Use mask to avoid division when om == 0
            mask = om != 0.0
            denom = 1j * om
            out_hat = xp.zeros_like(R, dtype=xp.complex128)
            out_hat[mask] = R[mask] / denom[mask]
            # DC component (om == 0) is already zero from initialization
            F_corr = fft.irfft(out_hat, n=self.grid.n)
            xx = xp.asarray(x)
            mean_r = float(xp.mean(residual))
            F_corr = F_corr + mean_r * (xx - float(xx[0]))
            F_corr = F_corr - F_corr[0]
        else:
            # Avoid divide-by-zero warning by handling DC component separately
            # Use mask to avoid division when om == 0
            mask = om != 0.0
            denom = (1j * om) ** 2
            out_hat = xp.zeros_like(R, dtype=xp.complex128)
            out_hat[mask] = R[mask] / denom[mask]
            # DC component (om == 0) is already zero from initialization
            F_corr = fft.irfft(out_hat, n=self.grid.n)
            xx = xp.asarray(x)
            x0 = float(xx[0]); x1 = float(xx[-1])
            mean_r = float(xp.mean(residual))
            F_corr = F_corr + 0.5 * mean_r * (xx - x0) * (xx - x1)

        F = xp.asarray(F_spline_host) + F_corr

        # Enforce boundary constraints using correct nullspace
        xx = xp.asarray(x)
        x0 = float(xx[0]); x1 = float(xx[-1])
        F = F - (F[0] - float(left_value))
        if match_right is not None:
            if order == 1:
                F = F + (float(match_right) - F[-1])  # constant shift
            else:
                F = F + (float(match_right) - F[-1]) * (xx - x0) / (x1 - x0)

        # Convert to NumPy if needed
        if bk is None:
            F_result = np.asarray(F, dtype=np.float64)
        else:
            F_result = bk.ensure_like_input(F, input_was_numpy).astype(np.float64)
        
        return (F_result, f_spline.astype(np.float64))

    def enforced_zero_flux(self, f: Array) -> Tuple[float, float]:
        """
        Enforce zero-flux boundary conditions using ghost points and B-spline interpolation.
        
        This method adjusts the boundary values of the function to satisfy zero-flux
        (Neumann) boundary conditions by:
        1. Creating ghost points by mirroring internal points (excluding boundaries)
        2. Fitting a B-spline to the extended points (excluding original boundary points)
        3. Evaluating the B-spline at the boundaries to get corrected boundary values
        
        The number of ghost points is set to `degree - 1`.
        
        Parameters
        ----------
        f : Array
            Function values on the grid (must match grid size)
        
        Returns
        -------
        f_left_corrected : float
            Corrected left boundary value that satisfies zero flux
        f_right_corrected : float
            Corrected right boundary value that satisfies zero flux
        """
        f = np.asarray(f, dtype=np.float64)
        if f.shape[0] != self.grid.n:
            raise ValueError("Length of f must match grid size.")
        
        # Number of ghost points is degree - 1
        n_ghost = self.degree - 1
        if n_ghost < 1:
            raise ValueError(f"degree must be at least 2 for enforced_zero_flux (got {self.degree})")
        
        x = self.grid.x
        dx = self.grid.dx
        
        # Create ghost points by mirroring internal points
        # Left ghost points: mirror from interior (excluding boundary point x[0])
        # For zero-flux at x[0], we want symmetric reflection: f(x[0] - i*dx) = f(x[0] + i*dx)
        x_left = x[0] - dx * np.arange(1, n_ghost + 1)
        x_left = x_left[::-1]  # Reverse to make strictly increasing
        f_left = f[1:1+n_ghost][::-1]  # Mirror values
        
        # Right ghost points: mirror from interior (excluding boundary point x[-1])
        x_right = x[-1] + dx * np.arange(1, n_ghost + 1)
        f_right = f[-1-n_ghost:-1][::-1]  # Mirror values
        
        # Combine to create extended arrays
        x_extended = np.concatenate([x_left, x, x_right])
        f_extended = np.concatenate([f_left, f, f_right])
        
        # Verify strictly increasing
        if not np.all(np.diff(x_extended) > 0):
            raise ValueError("Extended x array is not strictly increasing!")
        
        # Fit B-spline and evaluate at left boundary
        # Left boundary is at x_extended[n_ghost] (which equals x[0])
        boundary_idx_left = n_ghost
        x_boundary_left = x_extended[boundary_idx_left]
        
        # Exclude left boundary point from fitting
        mask_left = np.ones(len(x_extended), dtype=bool)
        mask_left[boundary_idx_left] = False
        x_fit_left = x_extended[mask_left]
        f_fit_left = f_extended[mask_left]
        
        # Create B-spline interpolant
        try:
            spline_left = make_interp_spline(x_fit_left, f_fit_left, k=self.degree, bc_type='natural')
        except:
            spline_left = make_interp_spline(x_fit_left, f_fit_left, k=self.degree)
        
        # Evaluate at left boundary to get corrected value
        f_left_corrected = float(spline_left(x_boundary_left))
        
        # Fit B-spline and evaluate at right boundary
        # Right boundary is at x_extended[-(n_ghost+1)] (which equals x[-1])
        boundary_idx_right = -(n_ghost + 1)
        x_boundary_right = x_extended[boundary_idx_right]
        
        # Exclude right boundary point from fitting
        mask_right = np.ones(len(x_extended), dtype=bool)
        mask_right[boundary_idx_right] = False
        x_fit_right = x_extended[mask_right]
        f_fit_right = f_extended[mask_right]
        
        # Create B-spline interpolant
        try:
            spline_right = make_interp_spline(x_fit_right, f_fit_right, k=self.degree, bc_type='natural')
        except:
            spline_right = make_interp_spline(x_fit_right, f_fit_right, k=self.degree)
        
        # Evaluate at right boundary to get corrected value
        f_right_corrected = float(spline_right(x_boundary_right))
        
        return f_left_corrected, f_right_corrected

    def interpolate(self, f: Array, lam: float = 0.0, use_fft: bool = False) -> Tuple[Array, Array]:
        """
        High-order interpolation that doubles the resolution.
        
        Takes an input function f of size N and returns interpolated values
        on a finer grid of size 2*N - 1 by inserting midpoints between
        existing grid points.
        
        The interpolation uses the B-spline representation to achieve high-order
        accuracy. The method:
        1. Fits a B-spline to the input data
        2. Creates a new grid with 2*N - 1 points (midpoints inserted)
        3. Evaluates the spline at the new grid points
        
        Parameters
        ----------
        f : Array
            Function values on the current grid (must match grid size N)
        lam : float, default 0.0
            Tikhonov regularization parameter for spline fitting
        use_fft : bool, default False
            If True, use pure FFT interpolation instead of B-spline.
            This forces the B-spline part to 0 and uses FFT for the entire
            interpolation. Useful for testing/debugging.
        
        Returns
        -------
        x_new : Array
            New grid points of size 2*N - 1
        f_new : Array
            Interpolated function values on the new grid
        
        Example
        -------
        >>> x = np.linspace(0, 2*np.pi, 64)
        >>> model = bspf1d.from_grid(degree=5, x=x)
        >>> f = np.sin(x)
        >>> x_new, f_new = model.interpolate(f)
        >>> print(f"Original size: {len(f)}, New size: {len(f_new)}")
        Original size: 64, New size: 127
        """
        # Convert input to appropriate backend
        if self.use_gpu and _HAS_CUPY:
            xp = cp
            input_was_numpy = isinstance(f, np.ndarray) or not isinstance(f, cp.ndarray)
            f_x = xp.asarray(f, dtype=xp.float64)
        else:
            xp = np
            input_was_numpy = True
            f_x = np.asarray(f, dtype=np.float64)
        
        if f_x.shape[0] != self.grid.n:
            raise ValueError(f"Length of f ({f_x.shape[0]}) must match grid size ({self.grid.n})")
        
        # Create new grid with 2*N - 1 points
        # Insert midpoints between existing grid points
        x_old = self.grid.x  # On correct device (NumPy or CuPy) from Grid1D
        N_old = len(x_old)
        N_new = 2 * N_old - 1
        
        # Create new grid: original points + midpoints
        # Use same backend as x_old
        if self.use_gpu and _HAS_CUPY and isinstance(x_old, cp.ndarray):
            xp = cp
            x_new = xp.empty(N_new, dtype=xp.float64)
        else:
            xp = np
            x_new = xp.empty(N_new, dtype=xp.float64)
        
        # Fill in original points at even indices
        x_new[::2] = x_old
        
        # Fill in midpoints at odd indices
        if N_old > 1:
            midpoints = 0.5 * (x_old[:-1] + x_old[1:])
            x_new[1::2] = midpoints
        
        # If use_fft is True, use pure FFT interpolation (skip B-spline)
        # This forces the B-spline part to 0 and uses FFT for the entire interpolation
        # Useful for testing/debugging to verify FFT interpolation correctness
        if use_fft:
            # use_fft path requires NumPy (uses np.interp which doesn't have CuPy equivalent)
            if self.use_gpu and _HAS_CUPY:
                raise ValueError(
                    "use_fft=True is not supported when use_gpu=True. "
                    "The FFT interpolation path uses NumPy-specific functions (np.interp). "
                    "Either: (1) set use_fft=False, or (2) use use_gpu=False."
                )
            f_fft = np.asarray(f_x, dtype=np.float64)
            fft_module = np.fft
            
            # For machine precision FFT interpolation, we need true periodicity
            # The grid x_old is endpoint=True, but for FFT we need endpoint=False
            # For periodic functions: f(0) = f(L), so we exclude the last point
            # and treat the grid as [0, L) with period L
            
            # Use N-1 points for FFT (exclude last point since f(L) = f(0) for periodic functions)
            f_periodic = f_fft[:-1].copy()
            N_periodic = len(f_periodic)
            
            # Determine refine factor (approximately 2 for 2*N-1 grid)
            refine_factor = N_new / N_old
            
            # Use integer refine factor closest to actual
            refine_factor_int = int(round(refine_factor))
            N_fft_fine = refine_factor_int * N_periodic
            
            # Forward real FFT on periodic grid (endpoint=False equivalent)
            F = fft_module.rfft(f_periodic)
            
            # Zero-pad in frequency space to interpolate
            f_fft_fine = fft_module.irfft(F, n=N_fft_fine) * (N_fft_fine / N_periodic)
            
            # Replace values at original coarse grid points with known values
            coarse_indices = np.arange(0, N_fft_fine, refine_factor_int)
            if len(coarse_indices) == N_periodic:
                f_fft_fine[coarse_indices] = f_periodic
            
            # Create FFT grid coordinates (periodic, endpoint=False)
            domain = (float(x_old[0]), float(x_old[-1]))
            L = domain[1] - domain[0]  # Period length
            x_fft_fine = np.linspace(domain[0], domain[0] + L, N_fft_fine, endpoint=False)
            
            # Interpolate from FFT grid to target grid (x_new)
            # Use periodic extension: map all x_new to [domain[0], domain[0] + L)
            x_new_wrapped = ((x_new - domain[0]) % L) + domain[0]
            
            # Use np.interp for efficient linear interpolation
            f_new = np.interp(x_new_wrapped, x_fft_fine, f_fft_fine, 
                             left=f_fft_fine[-1], right=f_fft_fine[0])
            
            # For points at the exact boundary
            boundary_mask = np.abs(x_new_wrapped - (domain[0] + L)) < 1e-12
            if np.any(boundary_mask):
                f_new[boundary_mask] = f_fft_fine[0]
            
            # Error if input was NumPy but operator uses GPU (avoid implicit conversions)
            if self.use_gpu and _HAS_CUPY and input_was_numpy:
                raise ValueError(
                    "Cannot convert GPU results back to NumPy in interpolate. "
                    "When use_gpu=True, provide CuPy arrays as input to avoid GPU↔CPU conversions. "
                    "Either: (1) convert input to CuPy array before calling, or (2) use use_gpu=False."
                )
            
            return x_new, f_new
        
        # Fit B-spline to the input data
        # Use the same approach as differentiate to get spline coefficients
        if self.use_gpu and _HAS_CUPY:
            BW = xp.asarray(self.BW)
            BND = xp.asarray(self.end.BND)
            BT0 = xp.asarray(self.basis.BT0)
            la = cpla
        else:
            BW = self.BW
            BND = self.end.BND
            BT0 = self.basis.BT0
            la = sla
        
        # Build RHS for spline fitting
        rhs_2bw = 2.0 * (BW @ f_x)
        dY = BND @ f_x
        rhs = xp.concatenate((rhs_2bw, dY), axis=0)
        
        # Solve for spline coefficients
        lu_cpu, piv_cpu = self._kkt_lu(lam)
        if self.use_gpu and _HAS_CUPY:
            SOL = la.lu_solve((xp.asarray(lu_cpu), xp.asarray(piv_cpu)), rh, overwrite_b=True)
        else:
            SOL = la.lu_solve((lu_cpu, piv_cpu), rhs, overwrite_b=True)
        
        n_b = self.basis.B0.shape[0]
        P = SOL[:n_b]
        
        # Evaluate spline at new grid points
        # Use the basis functions to evaluate
        # Note: scipy BSpline accepts NumPy arrays, cupyx BSpline accepts CuPy arrays
        # Strict rule: no implicit conversions - arrays must match use_gpu setting
        if self.use_gpu and _HAS_CUPY:
            if not isinstance(x_new, cp.ndarray):
                raise ValueError(
                    "Inconsistency detected: use_gpu=True but x_new is not a CuPy array. "
                    "This indicates an internal error. Arrays must match the use_gpu setting. "
                    "Use to_device() or to_host() for explicit conversions."
                )
            # cupyx BSpline accepts CuPy arrays
            x_eval = x_new
            B0_new = cp.empty((n_b, N_new), dtype=cp.float64)
        else:
            if _HAS_CUPY and isinstance(x_new, cp.ndarray):
                raise ValueError(
                    "Inconsistency detected: use_gpu=False but x_new is a CuPy array. "
                    "This indicates an internal error. Arrays must match the use_gpu setting. "
                    "Use to_device() or to_host() for explicit conversions."
                )
            # scipy BSpline accepts NumPy arrays
            x_eval = x_new
            B0_new = np.empty((n_b, N_new), dtype=np.float64)
        
        # Evaluate each basis function at the new grid points
        for i, s in enumerate(self.basis._splines):
            B0_new[i, :] = s(x_eval)
        
        # Convert to backend if needed and compute interpolated values
        # Strict rule: validate arrays are on correct device, no implicit conversions
        if self.use_gpu and _HAS_CUPY:
            # Validate B0_new and P are already on GPU (they should be from spline evaluation)
            if _HAS_CUPY and not isinstance(B0_new, cp.ndarray):
                raise ValueError(
                    "Inconsistency detected: use_gpu=True but B0_new is not a CuPy array. "
                    "This indicates an internal error. Arrays must match the use_gpu setting."
                )
            if _HAS_CUPY and not isinstance(P, cp.ndarray):
                raise ValueError(
                    "Inconsistency detected: use_gpu=True but P is not a CuPy array. "
                    "This indicates an internal error. Arrays must match the use_gpu setting."
                )
            f_new = (B0_new.T @ P).astype(cp.float64)
            # Error if input was NumPy but operator uses GPU (avoid implicit conversions)
            if input_was_numpy:
                raise ValueError(
                    "Cannot convert GPU results back to NumPy in interpolate. "
                    "When use_gpu=True, provide CuPy arrays as input to avoid GPU↔CPU conversions. "
                    "Either: (1) convert input to CuPy array before calling, or (2) use use_gpu=False."
                )
        else:
            # Validate B0_new and P are already on CPU
            if _HAS_CUPY and isinstance(B0_new, cp.ndarray):
                raise ValueError(
                    "Inconsistency detected: use_gpu=False but B0_new is a CuPy array. "
                    "This indicates an internal error. Arrays must match the use_gpu setting."
                )
            if _HAS_CUPY and isinstance(P, cp.ndarray):
                raise ValueError(
                    "Inconsistency detected: use_gpu=False but P is a CuPy array. "
                    "This indicates an internal error. Arrays must match the use_gpu setting."
                )
            f_new = (B0_new.T @ P).astype(np.float64)
        
        return x_new, f_new

    def fit_spline(
        self,
        f: Array,
        lam: float = 0.0,
        neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> Tuple[Array, Array, Array]:
        """
        Fit the B-spline coefficients P to samples f, enforcing endpoint
        derivatives up to order-1. Optionally override first derivative
        (Neumann) values via neumann_bc=(left_flux, right_flux).

        Parameters
        ----------
        f : Array
            Function values on the grid (must match grid size)
        lam : float, default 0.0
            Tikhonov regularization parameter
        neumann_bc : tuple, optional
            Neumann boundary conditions (left_flux, right_flux). If provided,
            overrides the computed first derivative values at boundaries.

        Returns
        -------
        P : Array
            Spline coefficients (shape: n_basis)
        f_spline : Array
            Spline approximation on the original grid (shape: n_points)
        residual : Array
            residual = f - f_spline (shape: n_points)
        """
        # Convert input to appropriate backend
        if self.use_gpu and _HAS_CUPY:
            xp = cp
            input_was_numpy = isinstance(f, np.ndarray) or not isinstance(f, cp.ndarray)
            f_x = xp.asarray(f, dtype=xp.float64)
        else:
            xp = np
            input_was_numpy = True
            f_x = np.asarray(f, dtype=np.float64)
        
        if f_x.shape[0] != self.grid.n:
            raise ValueError(f"Length of f ({f_x.shape[0]}) must match grid size ({self.grid.n})")
        
        # Build RHS
        if self.use_gpu and _HAS_CUPY:
            BW = xp.asarray(self.BW)
            BND = xp.asarray(self.end.BND)
            BT0 = xp.asarray(self.basis.BT0)
            la = cpla
        else:
            BW = self.BW
            BND = self.end.BND
            BT0 = self.basis.BT0
            la = sla
        
        rhs_2bw = 2.0 * (BW @ f_x)
        dY = BND @ f_x
        
        # Optional Neumann override on first derivatives
        if neumann_bc is not None:
            if self.order < 1:
                raise ValueError("Neumann BC requires order ≥ 1 (function + first derivative).")
            left_flux, right_flux = neumann_bc
            # rows: 1 (left d/dx), order+1 (right d/dx)
            if left_flux is not None:
                dY[1] = float(left_flux)
            if right_flux is not None:
                dY[self.order + 1] = float(right_flux)
        
        rhs = xp.concatenate((rhs_2bw, dY), axis=0)
        
        # Solve KKT system
        lu_cpu, piv_cpu = self._kkt_lu(lam)
        if self.use_gpu and _HAS_CUPY:
            SOL = la.lu_solve((xp.asarray(lu_cpu), xp.asarray(piv_cpu)), rhs, overwrite_b=True)
        else:
            SOL = la.lu_solve((lu_cpu, piv_cpu), rhs, overwrite_b=True)
        
        n_b = self.basis.B0.shape[0]
        P = SOL[:n_b]
        
        # Evaluate spline
        f_spline = BT0 @ P
        residual = f_x - f_spline
        
        # Error if input was NumPy but operator uses GPU (avoid implicit conversions)
        if self.use_gpu and _HAS_CUPY and input_was_numpy:
            raise ValueError(
                "Cannot convert GPU results back to NumPy. "
                "When use_gpu=True, provide CuPy arrays as input to avoid GPU↔CPU conversions. "
                "Either: (1) convert input to CuPy array before calling, or (2) use use_gpu=False."
            )
        
        return P, f_spline, residual

    def interpolate_split_mesh(
        self,
        f: Array,
        refine_factor: int,
        lam: float = 0.0,
        neumann_bc: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> Tuple[Array, Array, Array, Array]:
        """
        Interpolate a *non-periodic* function f on a refined grid using
        B-splines + FFT split-mesh interpolation of the periodic residual.

        This method provides more control than `interpolate()` by allowing
        arbitrary refine factors and explicitly returning the spline and
        residual components.

        Steps:
        1. Fit spline P, obtain f_spline and residual r = f - f_spline.
        2. Use FFT split-mesh interpolation on r (treated as periodic) to
           obtain r_fine on a periodic grid of length N_per = refine_factor * N.
        3. Restrict r_fine to the physical interval [a, b] so that the refined
           grid contains the original endpoints and all original points.
        4. Evaluate spline on the refined grid and add r_fine (restricted).

        Parameters
        ----------
        f : Array
            Function values on the current grid (must match grid size N)
        refine_factor : int
            How many fine cells per coarse cell (M).
            The fine grid size will be N_fine = M*(N-1) + 1.
        lam : float, default 0.0
            Tikhonov regularization parameter for spline fitting
        neumann_bc : tuple, optional
            Neumann boundary conditions (left_flux, right_flux) for spline fitting

        Returns
        -------
        x_fine : Array
            Refined grid points of size M*(N-1) + 1
        f_fine : Array
            Interpolated function values on the refined grid
        f_spline_fine : Array
            Spline component on the refined grid
        r_fine : Array
            Residual component on the refined grid

        Example
        -------
        >>> x = np.linspace(0, 2*np.pi, 100, endpoint=True)
        >>> model = bspf1d.from_grid(degree=5, x=x)
        >>> f = np.sin(x / (1.01 + np.cos(x)))
        >>> x_fine, f_fine, f_spline_fine, r_fine = model.interpolate_split_mesh(f, refine_factor=2)
        >>> print(f"Original: {len(f)}, Refined: {len(f_fine)}")
        Original: 100, Refined: 199
        """
        # Convert input to appropriate backend
        if self.use_gpu and _HAS_CUPY:
            if not isinstance(f, cp.ndarray):
                raise ValueError(
                    "Cannot use NumPy array in interpolate_split_mesh when use_gpu=True. "
                    "Either: (1) convert input to CuPy array, or (2) use use_gpu=False."
                )
            f = cp.asarray(f, dtype=cp.float64)
        else:
            if _HAS_CUPY and isinstance(f, cp.ndarray):
                raise ValueError(
                    "Cannot use CuPy array in interpolate_split_mesh when use_gpu=False. "
                    "Either: (1) convert input to NumPy array, or (2) use use_gpu=True."
                )
            f = np.asarray(f, dtype=np.float64)
        
        N = self.grid.n
        dx = self.grid.dx
        x0 = self.grid.a

        if f.shape[0] != N:
            raise ValueError(f"Length of f ({f.shape[0]}) must match grid size ({N})")

        # 1) B-spline fit + residual
        P, f_spline, residual = self.fit_spline(f, lam=lam, neumann_bc=neumann_bc)

        # 2) FFT split-mesh interpolation of periodic residual
        # This method uses NumPy-specific functions (np.fft.rfft, np.interp)
        # For GPU support, we would need to implement GPU equivalents
        if self.use_gpu and _HAS_CUPY:
            raise ValueError(
                "interpolate_split_mesh is not supported when use_gpu=True. "
                "The FFT split-mesh interpolation uses NumPy-specific functions (np.fft.rfft, np.interp). "
                "Either: (1) use use_gpu=False, or (2) use the interpolate() method instead."
            )
        
        M = int(refine_factor)
        if M < 1:
            raise ValueError("refine_factor must be a positive integer")
        
        # 3) Define physical refined grid on [a, b]
        #    Use N_fine = M*(N-1) + 1 so original points are included exactly.
        N_fine = M * (N - 1) + 1
        x_fine = x0 + dx * (np.arange(N_fine, dtype=np.float64) / M)
        
        # Make residual periodic for FFT (enforce periodicity by averaging endpoints)
        residual_periodic = residual.copy()
        periodicity_error = abs(residual_periodic[0] - residual_periodic[-1])
        if periodicity_error > 1e-12:
            avg_endpoint = 0.5 * (residual_periodic[0] + residual_periodic[-1])
            residual_periodic[0] = avg_endpoint
            residual_periodic[-1] = avg_endpoint
        
        # Use FFT split-mesh interpolation (treat residual as periodic)
        # For periodic FFT, use N-1 points (exclude last point since f(0) = f(L))
        residual_fft = residual_periodic[:-1].copy()
        N_periodic = len(residual_fft)
        
        # We need N_fine points on the physical grid, but FFT gives us M*N_periodic points
        # We need to interpolate from the FFT grid to the physical grid
        # The FFT grid has M*N_periodic points, but we only need N_fine = M*(N-1) + 1
        # For M=2, N=100: N_fine = 199, N_periodic = 99, M*N_periodic = 198
        # So we need to handle the endpoint separately
        
        # Forward real FFT on periodic grid
        F = np.fft.rfft(residual_fft)
        
        # Create FFT fine grid with M*N_periodic points (periodic, endpoint=False)
        N_fft_fine = M * N_periodic
        domain = (x0, x0 + dx * (N - 1))  # Physical domain length
        L = domain[1] - domain[0]  # Period length
        
        # Zero-pad in frequency space to interpolate
        r_fine_fft = np.fft.irfft(F, n=N_fft_fine) * (N_fft_fine / N_periodic)
        
        # Enforce exact agreement at coarse nodes
        coarse_indices = np.arange(0, N_fft_fine, M)
        if len(coarse_indices) == N_periodic:
            r_fine_fft[coarse_indices] = residual_fft
        
        # Create FFT grid coordinates (periodic, endpoint=False)
        x_fft_fine = np.linspace(domain[0], domain[0] + L, N_fft_fine, endpoint=False)
        
        # Interpolate from FFT grid to physical grid (x_fine)
        # The physical grid x_fine has N_fine = M*(N-1)+1 points
        # The FFT grid has N_fft_fine = M*N_periodic = M*(N-1) points (one less)
        # We need to handle the endpoint separately
        
        # For most points, use periodic interpolation
        x_fine_wrapped = ((x_fine - domain[0]) % L) + domain[0]
        
        # Use np.interp for efficient linear interpolation with periodic boundaries
        r_fine = np.interp(x_fine_wrapped, x_fft_fine, r_fine_fft,
                          left=r_fine_fft[-1], right=r_fine_fft[0])
        
        # For the last point (endpoint), use the periodic value (same as first point)
        # Since the residual is periodic, r_fine[-1] should equal r_fine[0]
        # But we want to use the value from the FFT grid that corresponds to the endpoint
        # The endpoint x_fine[-1] = domain[0] + L, which in periodic sense equals domain[0]
        # So we use r_fine_fft[0] for the endpoint
        if len(r_fine) > 0:
            # The last point of x_fine is at domain[0] + L, which is periodic with domain[0]
            # So use the first point of the FFT result
            r_fine[-1] = r_fine_fft[0]

        # 4) Evaluate spline on refined grid
        # Note: We need to bypass the cache or use a unique cache key since x_fine
        # has a different length than the original grid. We'll evaluate directly.
        n_basis = len(self.basis._splines)
        B_fine = np.empty((n_basis, N_fine), dtype=np.float64)
        for i, s in enumerate(self.basis._splines):
            B_fine[i, :] = s(x_fine)
        
        # P and residual are already NumPy arrays (since use_gpu=False at this point)
        f_spline_fine = (B_fine.T @ P).astype(np.float64)

        # 5) Combine
        f_fine = f_spline_fine + r_fine

        return x_fine, f_fine, f_spline_fine, r_fine


# =============================================================================
# Piecewise BSPF for functions with discontinuities
# =============================================================================
class PiecewiseBSPF1D:
    """
    Piecewise BSPF operator for functions with known discontinuities.
    
    Segments the domain at breakpoints and applies bspf1d to each segment
    independently. This improves accuracy for functions with jumps or
    discontinuities.
    
    Breakpoints are interpreted as physical coordinates that can fall between
    grid cells. Each breakpoint is interpreted as: discontinuity lies between
    x[idx-1] and x[idx]. Left segment uses indices 0..idx-1, right segment
    uses idx..N-1.
    
    Parameters
    ----------
    degree : int
        B-spline degree for each segment
    x : Array
        Full grid points (must be uniformly spaced)
    breakpoints : List[float], optional
        List of x-coordinates where discontinuities occur. Default: []
    min_points_per_seg : int, default 16
        Minimum number of points required per segment. Segments with fewer
        points are skipped.
    **bspf_kwargs
        Additional arguments passed to bspf1d.from_grid for each segment
        (e.g., order, correction, use_gpu)
    
    Example
    -------
    >>> x = np.linspace(0, 2*np.pi, 512)
    >>> pw = PiecewiseBSPF1D(degree=5, x=x, breakpoints=[np.pi/2, 3*np.pi/2])
    >>> df1, df2, f_spline = pw.differentiate_1_2(f)
    """
    
    def __init__(self, degree: int, x: Array, breakpoints: Optional[List[float]] = None,
                 min_points_per_seg: int = 16, **bspf_kwargs):
        self.degree = int(degree)
        self.x = np.asarray(x, dtype=np.float64)
        self.breakpoints = sorted(breakpoints or [])
        self.min_points_per_seg = int(min_points_per_seg)
        
        N = self.x.size
        
        # 1. Convert physical coordinates to cell interface indices
        cut_indices = []
        for bp in self.breakpoints:
            idx = int(np.searchsorted(self.x, bp))  # x[idx-1] < bp <= x[idx]
            if 1 <= idx <= N - 1:
                cut_indices.append(idx)
        cut_indices = sorted(set(cut_indices))
        
        self.segments = []  # Each segment: {i0, i1, op}
        
        i_start = 0
        for idx in cut_indices:
            i_end = idx - 1  # Left segment goes to idx-1
            if i_end - i_start + 1 >= self.min_points_per_seg:
                x_seg = self.x[i_start:i_end + 1]
                op = bspf1d.from_grid(degree=self.degree, x=x_seg, **bspf_kwargs)
                self.segments.append(dict(i0=i_start, i1=i_end, op=op))
            # Otherwise segment is too short, skip it
            i_start = idx  # Right segment starts from idx
        
        # Last segment
        if N - 1 - i_start + 1 >= self.min_points_per_seg:
            x_seg = self.x[i_start:]
            op = bspf1d.from_grid(degree=self.degree, x=x_seg, **bspf_kwargs)
            self.segments.append(dict(i0=i_start, i1=N - 1, op=op))
    
    def differentiate_1_2(self, f: Array, lam: float = 0.0,
                          neumann_bc_global: Optional[Tuple[Optional[float], Optional[float]]] = None
                          ) -> Tuple[Array, Array, Array]:
        """
        Compute first and second derivatives using piecewise BSPF.
        
        Calls bspf1d.differentiate_1_2 on each segment, then concatenates results.
        
        Parameters
        ----------
        f : Array
            Function values on the full grid
        lam : float, default 0.0
            Tikhonov regularization parameter
        neumann_bc_global : tuple, optional
            Neumann boundary conditions (left_flux, right_flux) for the global domain.
            Interior interfaces do not apply Neumann BCs, determined by physics.
        
        Returns
        -------
        df1 : Array
            First derivative on full grid
        df2 : Array
            Second derivative on full grid
        f_spline : Array
            Spline approximation on full grid
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
            f_seg = f[i0:i1 + 1]
            
            # Only apply global Neumann BC at the two ends of the entire domain
            if k == 0:
                bc_left = left_flux_global
            else:
                bc_left = None
            if k == n_seg - 1:
                bc_right = right_flux_global
            else:
                bc_right = None
            neumann_bc_seg = (bc_left, bc_right)
            
            d1_seg, d2_seg, fs_seg = op.differentiate_1_2(
                f_seg, lam=lam, neumann_bc=neumann_bc_seg
            )
            
            df1_full[i0:i1 + 1] = d1_seg
            df2_full[i0:i1 + 1] = d2_seg
            fs_full[i0:i1 + 1] = fs_seg
        
        return df1_full, df2_full, fs_full
