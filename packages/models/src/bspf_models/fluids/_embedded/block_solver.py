"""Block-triangular preconditioner for the unchanged mixed Stokes system."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, gmres, splu


class BlockStokesFactor:
    def __init__(self, a, k, b, c, mean, rtol=1e-13, viscosity=1.0):
        self.a, self.b, self.mean = a, b, mean
        self.nv = k.shape[0]
        self.np = c.shape[0]
        self.gauge = a.shape[0] == 2 * self.nv + self.np + 1
        self.rtol = rtol
        self.velocity = splu(k, permc_spec="MMD_AT_PLUS_A")
        # Pressure bases are orthonormal in physical L2. I/nu approximates the
        # velocity Schur complement; C retains the actual jump stabilization.
        self.pressure = splu(
            c + sp.eye(c.shape[0], format="csc") / viscosity, permc_spec="MMD_AT_PLUS_A"
        )
        if self.gauge:
            self.hm = self.pressure.solve(mean)
            self.den = float(mean @ self.hm)
        self.preconditioner = LinearOperator(a.shape, matvec=self.apply, dtype=float)
        self.factor_nnz = sum(f.L.nnz + f.U.nnz for f in (self.velocity, self.pressure))
        self.stats = {}

    def apply(self, rhs):
        n = self.nv
        v = self.velocity.solve(np.column_stack((rhs[:n], rhs[n : 2 * n]))).T.ravel()
        q = self.pressure.solve(rhs[2 * n : 2 * n + self.np] - self.b @ v)
        if not self.gauge:
            return np.r_[v, -q]
        multiplier = (rhs[-1] + self.mean @ q) / self.den
        return np.r_[v, -q + self.hm * multiplier, multiplier]

    def solve(self, rhs):
        history = []
        x, flag = gmres(
            self.a,
            rhs,
            M=self.preconditioner,
            rtol=self.rtol,
            atol=0,
            restart=80,
            maxiter=10,
            callback=lambda r: history.append(float(r)),
            callback_type="pr_norm",
        )
        residual = float(np.linalg.norm(self.a @ x - rhs) / max(np.linalg.norm(rhs), 1))
        self.stats = dict(
            iterations=len(history),
            gmres_info=int(flag),
            residual=residual,
            factor_nnz=self.factor_nnz,
        )
        if flag or not np.all(np.isfinite(x)):
            raise RuntimeError(f"Block Stokes GMRES failed: {self.stats}")
        return x
