"""Mapped 2D no-slip Boussinesq model assembled from pybspf trial spaces.

Geometry and background stratification are caller-supplied functions. Setup is
on the host; evolution uses the shared JAX tensor, PCG and RK4 kernels.
"""

import time
import numpy as np
import scipy.linalg as la
from numpy.polynomial.legendre import leggauss
import jax
import jax.numpy as jnp
from pybspf.trial_spaces import ClosedBSPFLine
import bspf_models.fluids.isw_slope as backend
from bspf_models.fluids.isw_slope import tensor


class MappedBoussinesq:
    backend_name = "bspf_models.fluids.isw_slope"

    def __init__(self, nx, nz, *, geometry, background, quad=2.0, nu=1e-4, kappa=3e-5):
        if not jax.config.x64_enabled:
            raise ValueError(
                "MappedBoussinesq requires JAX float64; enable jax_enable_x64."
            )
        tic = time.perf_counter()
        self.nx = nx
        self.nz = nz
        self.nu = nu
        self.kappa = kappa
        self.shape = (nx - 3, nz - 3)
        self.bshape = (nx, nz)
        self.lx = ClosedBSPFLine(nx)
        self.lz = ClosedBSPFLine(nz)
        q, w = leggauss(int(np.ceil(nx * quad)))
        self.sx = (q + 1) / 2
        self.wx = w / 2
        s, w = leggauss(int(np.ceil(nz * quad)))
        self.sz = (s + 1) / 2
        self.wz = w / 2
        self.geo = geometry(self.sx, self.sz)
        for key in ["d", "dx", "dxx", "k", "g"]:
            setattr(self, key, self.geo[key])
        self.Z = self.geo["z"]
        self.S = np.broadcast_to(self.sz[None, :], self.Z.shape)
        self.W = self.wx[:, None] * self.wz[None, :] * self.g * self.d
        X = self.lx.values(self.sx, 2)
        Z = self.lz.values(self.sz, 2)
        X = [X[0], X[1] / self.g, X[2] / self.g**2 - X[1] * self.geo["gq"] / self.g**3]
        mx = X[0].T @ ((self.wx * self.g[:, 0])[:, None] * X[0])
        kx = X[1].T @ ((self.wx * self.g[:, 0])[:, None] * X[1])
        mz = Z[0].T @ (self.wz[:, None] * Z[0])
        kz = Z[1].T @ (self.wz[:, None] * Z[1])
        self.ex, self.Qx = la.eigh(kx, mx)
        self.ez, self.Qz = la.eigh(kz, mz)
        self.Px = [v @ self.Qx for v in X]
        self.Pz = [v @ self.Qz for v in Z]
        tx = self.lx.scalar_values(self.sx, 1)
        tx[1] /= self.g
        tz = self.lz.scalar_values(self.sz, 1)
        bmx = tx[0].T @ ((self.wx * self.g[:, 0] * self.d[:, 0])[:, None] * tx[0])
        bmz = tz[0].T @ (self.wz[:, None] * tz[0])
        self.BQx = la.solve_triangular(
            la.cholesky(bmx, lower=True).T, np.eye(nx), lower=False
        )
        self.BQz = la.solve_triangular(
            la.cholesky(bmz, lower=True).T, np.eye(nz), lower=False
        )
        self.Tx = [v @ self.BQx for v in tx]
        self.Tz = [v @ self.BQz for v in tz]
        self.metric = self.metric_coeffs(self.d, self.dx, self.dxx, self.k, self.S)

        def xm(i, j, v):
            return self.Px[i].T @ ((self.wx * v)[:, None] * self.Px[j])

        def zm(i, j, v):
            return self.Pz[i].T @ ((self.wz * v)[:, None] * self.Pz[j])

        d = self.d[:, 0]
        dx = self.dx[:, 0]
        g = self.g[:, 0]
        ss = self.sz - 1
        self.mass_terms = [
            (xm(0, 0, g / d), zm(1, 1, np.ones_like(ss))),
            (xm(0, 0, g * dx * dx / d), zm(1, 1, ss * ss)),
            (xm(1, 1, g * d), zm(0, 0, np.ones_like(ss))),
            (-xm(0, 1, g * dx), zm(1, 0, ss)),
            (-xm(1, 0, g * dx), zm(0, 1, ss)),
        ]
        self.b0, self.n2 = background(self.Z)
        self.iterations = []
        self.residual_max = 0.0
        self.quad_factor = quad
        keys = (
            "Px",
            "Pz",
            "Tx",
            "Tz",
            "metric",
            "mass_terms",
            "W",
            "d",
            "k",
            "n2",
            "nu",
            "kappa",
        )
        self.plan = jax.tree_util.tree_map(
            jnp.asarray, {k: getattr(self, k) for k in keys}
        )
        self.plan["mass_preconditioner"] = backend.plan_mass_preconditioner(
            self.mass_terms
        )
        jax.block_until_ready(self.plan)
        self.build_seconds = time.perf_counter() - tic

    def _record(self, info):
        counts, residuals, ok = (np.asarray(v) for v in jax.device_get(info))
        self._record_host(counts, residuals, ok)

    def _record_host(self, counts, residuals, ok):
        if not np.all(ok):
            raise RuntimeError(
                f"BSPF JAX mass CG failed: residual={residuals}, iterations={counts}"
            )
        self.iterations.extend(counts.astype(np.int64).reshape(-1).tolist())
        self.residual_max = max(self.residual_max, float(residuals.max()))

    def mass(self, a):
        return backend.mass(self.plan, a)

    def fields(self, a, px=None, pz=None, metric=None):
        if px is None and pz is None and metric is None:
            return backend.fields(self.plan, a)
        plan = dict(self.plan)
        for key, value in (("Px", px), ("Pz", pz), ("metric", metric)):
            if value is not None:
                plan[key] = jax.tree_util.tree_map(jnp.asarray, value)
        return backend.fields(plan, a)

    def adjoint(self, values):
        return backend.adjoint(self.plan, values)

    def solve(self, r):
        a, info = backend.solve_mass(self.plan, r)
        self._record(info)
        return a

    def project_initial(self, initial):
        u, w = initial.velocity(self.sx, self.sz)
        zero = np.zeros_like(u)
        a = self.solve(self.adjoint([u, w, zero, zero, zero, zero]))
        b = self.Tx[0].T @ (self.W * initial.b(self.sx, self.sz)) @ self.Tz[0]
        return a, jnp.asarray(b)

    def rhs(self, a, bc, diagnostic=False):
        state, info = backend.rhs(self.plan, a, bc)
        self._record(info)
        if diagnostic:
            u, w, ux, uz, wx, wz = backend.fields(self.plan, a)
            p = self.plan
            b = backend.tensor(p["Tx"][0], bc, p["Tz"][0])
            bw = jnp.sum(p["W"] * w * b)
            vis = -p["nu"] * jnp.sum(p["W"] * (ux * ux + uz * uz + wx * wx + wz * wz))
            edot = jnp.sum(a * backend.mass(p, state[0]))
            self.last_budget = dict(
                mass_residual=float(info[1]),
                kinetic_budget_rel=float(
                    jnp.abs(edot - bw - vis)
                    / jnp.maximum(jnp.abs(bw) + jnp.abs(vis), 1e-30)
                ),
            )
        return state

    def rk4(self, a, b, dt):
        state, report = backend.rk4_checked(self.plan, a, b, dt)
        # One 104-byte transfer/synchronization, never a transfer of the fields.
        report = np.asarray(jax.device_get(report))
        self._record_host(report[:4], report[4:8], report[8:12])
        if not report[12]:
            raise FloatingPointError("Non-finite RK4 state; step rejected.")
        return state

    @staticmethod
    def metric_coeffs(d, dx, dxx, k, s):
        # Physical derivatives of the Piola/curl velocity; no numerical metric D.
        return [
            {(0, 1): 1 / d},
            {(1, 0): -1.0, (0, 1): k / d},
            {(1, 1): 1 / d, (0, 1): -dx / d**2, (0, 2): -k / d**2},
            {(0, 2): 1 / d**2},
            {
                (2, 0): -1.0,
                (1, 1): 2 * k / d,
                (0, 1): dxx * (s - 1) / d - 2 * k * dx / d**2,
                (0, 2): -k * k / d**2,
            },
            {(1, 1): -1 / d, (0, 1): dx / d**2, (0, 2): k / d**2},
        ]

    def diagnostics(self, t, a, bc):
        self.rhs(a, bc, True)
        u, w, *_ = self.fields(a)
        b = tensor(self.Tx[0], bc, self.Tz[0])
        B = b + self.b0
        N2 = self.n2 + tensor(self.Tx[0], bc, self.Tz[1]) / self.d
        return dict(
            time_s=t * 100,
            kinetic=float(0.5 * np.sum(self.W * (u * u + w * w))),
            min_N2=float(N2.min() / 10000),
            Bmin=float(B.min() / 100),
            Bmax=float(B.max() / 100),
            umax=float(abs(u).max()),
            wmax=float(abs(w).max()),
            scalar_integral=float(np.sum(self.W * B)),
            mass_CG_mean=float(np.mean(self.iterations)),
            **self.last_budget,
        )
