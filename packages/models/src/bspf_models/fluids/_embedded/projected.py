"""Implicit sparse-QR constrained CPU solver for large fixed-obstacle problems.

Setup uses implicit Q and a bounded dense rank core. An optional bounded array
cache materializes the null basis in local divergence-free coordinates.
The public state/diagnostics match the JAX solver, but this explicitly selected
host backend is not JIT-compatible or differentiable.
"""

from time import perf_counter
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, cg, LinearOperator

from .convection import Convection


class SparseProjectedNavierStokes2D:
    def __init__(self, spatial, boundary, dt, convection_order, residual_tolerance,
                 projector_backend="implicit_qr"):
        from ..embedded_navier_stokes import EmbeddedNSState, EmbeddedNSDiagnostics

        self._State, self._Diagnostics = EmbeddedNSState, EmbeddedNSDiagnostics
        self.spatial = spatial
        self.nv = spatial.nv
        self.size = 4 * self.nv
        self.dt = float(dt)
        self.residual_tolerance = residual_tolerance
        self.structural = True
        self.linear_backend = "host_sparse"
        self.info = dict(
            spatial.info,
            linear_backend="host_sparse",
            time_solver="implicit-QR projected CG",
            dt=self.dt,
        )
        self.boundary = boundary
        self.projector = spatial.projector
        if projector_backend == "array":
            from .sparse_qr import ArrayNullSpace
            start = perf_counter()
            self.projector = ArrayNullSpace(spatial.projector)
            self.info["array_projector_setup_seconds"] = perf_counter() - start
            self.info["array_projector_bytes"] = self.projector.bytes
        elif projector_backend != "implicit_qr":
            raise ValueError("Unknown projector_backend")
        self.info["projector_backend"] = projector_backend
        self.info["time_solver"] = f"{projector_backend} projected CG"
        self.lift = spatial.affine_velocity
        self.scalar_mass = spatial.base.mass.matrix()
        self.mass = sp.block_diag((self.scalar_mass, self.scalar_mass), format="csr")
        self.stiffness = sp.block_diag((spatial.K, spatial.K), format="csr")
        self.boundary_load = spatial.base.rhs(boundary)[: 2 * self.nv]
        self.convection_operator = Convection(spatial, boundary, convection_order)
        self.force_points = np.vstack([p for p, w in spatial.volume.values()])
        self.traction_points = (
            np.vstack([c[1] for c in spatial.outflow_curves])
            if spatial.outflow_curves
            else np.empty((0, 2))
        )
        self.traction_normals = (
            np.vstack([c[3] for c in spatial.outflow_curves])
            if spatial.outflow_curves
            else np.empty((0, 2))
        )
        self._factor = splu(
            (spatial.K + 1.5 / self.dt * self.scalar_mass).tocsc(),
            permc_spec="MMD_AT_PLUS_A",
        )
        self.last_iterations = 0
        self.initial_seconds = 0.0
        # Fixed dt means only BE and BDF2 Helmholtz matrices are needed.
        self._helmholtz = {0.: self.stiffness}

    def load(self, force=None, traction=None):
        result = self.boundary_load.copy()
        if force is not None:
            force = np.asarray(force)
            if force.shape != self.force_points.shape:
                raise ValueError("Force sample shape mismatch")
            start = 0
            for r, (p, w) in self.spatial.volume.items():
                v = self.spatial.vbasis[r].evaluate(p, 0)[0]
                local = v.T @ (w[:, None] * force[start : start + len(p)])
                for d in range(2):
                    result[self.spatial.vi[r] + d * self.nv] += local[:, d]
                start += len(p)
        if traction is not None:
            traction = np.asarray(traction)
            if traction.shape != self.traction_points.shape:
                raise ValueError("Traction sample shape mismatch")
            start = 0
            for r, p, w, n, tag, v, dn, q in self.spatial.outflow_curves:
                local = v.T @ (w[:, None] * traction[start : start + len(p)])
                for d in range(2):
                    result[self.spatial.vi[r] + d * self.nv] += local[:, d]
                start += len(p)
        return result

    def convection(self, coefficients):
        return self.convection_operator.residual(coefficients)

    def incompressibility_errors(self, coefficients):
        s = self.spatial
        v = np.asarray(coefficients)[: 2 * self.nv]
        divergence = np.linalg.norm(s.audit_divergence @ v)
        jump = np.linalg.norm(s.audit_jumps @ v)
        boundary = np.linalg.norm(s.audit_boundary @ v - s.audit_boundary_data)
        scale = max(np.linalg.norm(s.audit_gradient @ v), 1)
        speed = max(np.sqrt(max(v @ (self.mass @ v), 0)), 1)
        target = max(np.linalg.norm(s.audit_boundary_data), 1)
        valid = (
            divergence <= s.tolerance * scale
            and jump <= s.tolerance * speed
            and boundary <= s.tolerance * target
        )
        if s.wall_enforcement == "constraint":
            valid = valid and self.wall_slip_error(coefficients) <= s.tolerance * max(
                np.linalg.norm(s.audit_wall_tangent_data), 1.)
        return divergence, jump, boundary, bool(valid)

    def wall_slip_error(self, coefficients):
        """Independently integrated obstacle tangential Dirichlet mismatch."""
        s = self.spatial
        return float(np.linalg.norm(s.audit_wall_tangent @ np.asarray(coefficients)[:2*self.nv]
                                    - s.audit_wall_tangent_data))

    def initialize(self, coefficients, *, time=0.0):
        c = np.asarray(coefficients, dtype=float)
        if (
            c.shape != (self.size,)
            or not np.all(np.isfinite(c))
            or not np.isfinite(time)
        ):
            raise ValueError("Invalid structural coefficients or time")
        errors = self.incompressibility_errors(c)
        if not errors[-1]:
            raise ValueError(
                f"Initial velocity violates structural incompressibility or wall constraints: {errors[:3]}, tangential={self.wall_slip_error(c)}"
            )
        return self._State(c.copy(), c.copy(), np.zeros(2 * self.nv), 0, float(time))

    def _solve(self, rhs, alpha, predictor=None):
        if alpha not in self._helmholtz:
            self._helmholtz[alpha] = self.stiffness + alpha * self.mass
        h = self._helmholtz[alpha]
        factor = (
            self._factor
            if alpha
            else splu(self.spatial.K.tocsc(), permc_spec="MMD_AT_PLUS_A")
        )
        q = self.projector

        def apply(z):
            return q.restrict(h @ q.lift(z))

        def pre(z):
            v = q.lift(z).reshape(2, self.nv).T
            return q.restrict(factor.solve(v).T.ravel())

        operator = LinearOperator((q.free, q.free), matvec=apply, dtype=float)
        preconditioner = LinearOperator(operator.shape, matvec=pre, dtype=float)
        u = (
            self.lift.copy()
            if predictor is None
            else self.lift + q.lift(q.restrict(predictor - self.lift))
        )
        residual = np.inf
        iterations = []
        for _ in range(2):
            reduced = q.restrict(rhs - h @ u)
            delta, flag = cg(
                operator,
                reduced,
                M=preconditioner,
                rtol=1e-12,
                atol=1e-13,
                maxiter=600,
                callback=lambda z: iterations.append(1),
            )
            u += q.lift(delta)
            reaction = rhs - h @ u
            residual = np.linalg.norm(q.restrict(reaction)) / max(
                np.linalg.norm(rhs), 1
            )
            if residual <= self.residual_tolerance:
                break
        self.last_iterations = len(iterations)
        if not np.all(np.isfinite(u)) or residual > self.residual_tolerance:
            raise RuntimeError(
                f"Projected momentum solve failed: flag={flag}, residual={residual}, iterations={len(iterations)}"
            )
        return np.r_[u, reaction], residual

    def stokes_initial_state(self, force=None, traction=None, *, time=0.0):
        start = perf_counter()
        coefficients, _ = self._solve(self.load(force, traction), 0.0)
        self.initial_seconds = perf_counter() - start
        return self.initialize(coefficients, time=time)

    def step(self, state, load=None):
        current, older, old_conv, count, time = state
        n = 2 * self.nv
        startup = int(count) == 0
        rhs = (
            self.boundary_load.copy()
            if load is None
            else np.asarray(load, dtype=float).copy()
        )
        if rhs.shape != (n,):
            raise ValueError("Implicit QR backend expects a velocity momentum load")
        conv = self.convection(current)
        rhs += (
            self.mass
            @ (current[:n] if startup else 2 * current[:n] - 0.5 * older[:n])
            / self.dt
        )
        rhs -= conv if startup else 2 * conv - old_conv
        candidate, residual = self._solve(
            rhs,
            (1.0 if startup else 1.5) / self.dt,
            current[:n] if startup else 2 * current[:n] - older[:n],
        )
        divergence, jump, boundary, valid = self.incompressibility_errors(candidate)
        energy = 0.5 * candidate[:n] @ (self.mass @ candidate[:n])
        valid = bool(valid and np.isfinite(energy) and np.all(np.isfinite(candidate)))
        return self._State(
            candidate, current, conv, int(count) + 1, float(time) + self.dt
        ), self._Diagnostics(residual, energy, valid, divergence, jump, boundary)

    def advance(self, state, steps, *, load=None, callback=None):
        if not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        for _ in range(steps):
            rhs = None if load is None else load(float(state.time) + self.dt)
            candidate, d = self.step(state, rhs)
            if not d.valid:
                raise RuntimeError(
                    f"Structural audit failed at t={candidate.time}: {d}"
                )
            state = candidate
            if callback is not None:
                callback(state, d)
        return state

    def evaluate(self, points, state):
        return self.spatial.evaluate(points, np.asarray(state.coefficients))

    def close(self):
        self.projector.close()
        if self.projector is not self.spatial.projector:
            self.spatial.projector.close()
