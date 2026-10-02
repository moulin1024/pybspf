"""Unsteady 2D BSPF Navier--Stokes outside a fixed elliptical obstacle.

Geometry, quadrature, basis design and factorization are host setup. The dense
and sparse runtime backends use JAX float64 without callbacks. The explicitly
selected host_sparse backend executes native CPU operations; implicit_qr is
a CPU-only projected solver sharing the state and diagnostics API.
The default constrained SIPDG formulation enforces divergence and normal traces
independently of pressure recovery. The former stabilized mixed formulation is
explicitly opt-in. Natural right outflow fixes the recovered pressure level;
closed Dirichlet domains use zero-mean recovered pressure.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres
import numpy as np
import scipy.sparse as sp

from ._embedded.geometry import ObstacleGrid
from ._embedded.solver import StokesPlan, Blocks
from ._embedded.runtime import sparse, sample_matrix, factor_solve

__all__ = [
    "EmbeddedNSState",
    "EmbeddedNSDiagnostics",
    "EmbeddedNavierStokes2D",
    "plan_embedded_navier_stokes2d",
    "geometry_prior_options",
]


class EmbeddedNSState(NamedTuple):
    """Complete restart state; keep every field for the same plan and dt."""

    coefficients: jax.Array
    previous_coefficients: jax.Array
    previous_convection: jax.Array
    step: jax.Array
    time: jax.Array


class EmbeddedNSDiagnostics(NamedTuple):
    linear_residual: jax.Array
    kinetic_energy: jax.Array
    valid: jax.Array
    divergence_l2: jax.Array
    normal_jump_l2: jax.Array
    boundary_normal_l2: jax.Array


class EmbeddedNavierStokes2D:
    """Fixed spatial plan and time step. Construct with the factory below.

    ``step(state, load=None)`` is JIT-compatible; load is the assembled mixed
    RHS at the *new* time. ``advance`` adds host-side failure checks. Fixed
    Dirichlet data enter both Nitsche terms and incoming convective traces.
    """

    def __init__(
        self,
        spatial,
        boundary,
        dt,
        convection_order,
        linear_backend,
        residual_tolerance,
    ):
        self.spatial = spatial
        self.linear_backend = linear_backend
        self.dt = float(dt)
        self.nv = spatial.nv
        self.size = spatial.A.shape[0]
        self.info = dict(spatial.info, dt=self.dt, linear_backend=linear_backend)
        self.residual_tolerance = residual_tolerance
        self.boundary_load = jnp.asarray(spatial.rhs(boundary))
        mass = Blocks((self.nv, self.nv))
        for cell, (p, w) in spatial.grid.volume.items():
            r = spatial.owner[cell]
            v = spatial.vbasis[r].evaluate(p, 0)[0]
            mass.add(spatial.vi[r], spatial.vi[r], v.T @ (w[:, None] * v))
        scalar_mass = mass.matrix()
        full_mass = sp.block_diag(
            (scalar_mass, scalar_mass, sp.csc_matrix((self.size - 2 * self.nv,) * 2)),
            format="csc",
        )
        self.mass = sparse(full_mass)
        self.operator = sparse(spatial.A)
        be = spatial.A + full_mass / dt
        bdf = spatial.A + 1.5 * full_mass / dt
        self._be_operator, self._bdf_operator = sparse(be), sparse(bdf)
        self._bdf_solve = factor_solve(bdf, linear_backend)
        # Reuse the BDF2 factor for the single BE startup, as a linear
        # preconditioner. No nonlinear iteration or second large LU is needed.
        self._be_solve = lambda rhs: gmres(
            lambda x: self._be_operator @ x,
            rhs,
            M=self._bdf_solve,
            tol=2e-14,
            atol=0.0,
            restart=40,
            maxiter=3,
        )[0]
        self._prepare_convection(boundary, convection_order)
        self._prepare_loads()
        self.structural = spatial.info.get("incompressibility") == "structural"
        if self.structural:
            self._audit_divergence = sparse(spatial.audit_divergence)
            self._audit_gradient = sparse(spatial.audit_gradient)
            self._audit_jumps = sparse(spatial.audit_jumps)
            self._audit_boundary = sparse(spatial.audit_boundary)
            self._audit_boundary_data = jnp.asarray(spatial.audit_boundary_data)
            self._audit_wall_tangent = sparse(spatial.audit_wall_tangent)
            self._audit_wall_tangent_data = jnp.asarray(spatial.audit_wall_tangent_data)
        self._compiled_step = jax.jit(self.step)

    def _prepare_convection(self, boundary, order):
        s = self.spatial
        if self.linear_backend == "host_sparse":
            from ._embedded.convection import Convection

            self._host_convection = Convection(s, boundary, order)
            return
        dense = [r for r in s.grid.full if getattr(s.vbasis[s.owner[r]], "added", 0)]
        grid = ObstacleGrid(
            s.grid.cells,
            order,
            s.grid.center,
            s.grid.axes,
            edges=s.grid.edges,
            full_order=max(20, s.info["degree"] + 6),
            dense_full=dense,
        )
        values, dx, dy, weights = [], [], [], []
        for cell, (p, w) in grid.volume.items():
            r = s.owner[cell]
            v, x, y = s.vbasis[r].evaluate(p)
            for records, table in ((values, v), (dx, x), (dy, y)):
                records.append((s.vi[r], table))
            weights.append(w)
        self._v, self._dx, self._dy = [
            sparse(sample_matrix(a, self.nv)) for a in (values, dx, dy)
        ]
        self._weights = jnp.asarray(np.concatenate(weights))
        left, right, weights, normals = [], [], [], []
        for aa, bb, p, w, n in grid.faces:
            a, b = s.owner[aa], s.owner[bb]
            if a == b:
                continue
            left.append((s.vi[a], s.vbasis[a].evaluate(p, 0)[0]))
            right.append((s.vi[b], s.vbasis[b].evaluate(p, 0)[0]))
            weights.append(w)
            normals.append(n)
        self._left, self._right = [
            sparse(sample_matrix(a, self.nv)) for a in (left, right)
        ]
        self._face_weights = jnp.asarray(
            np.concatenate(weights) if weights else np.empty(0)
        )
        self._normals = jnp.asarray(np.vstack(normals) if normals else np.empty((0, 2)))
        traces, weights, prescribed = [], [], []
        for r, p, w, n, tag, v, dn, q in s.curves:
            g = np.asarray(boundary(p, tag))
            speed = np.maximum(-np.sum(g * n, axis=1), 0)
            if np.any(speed):
                traces.append((s.vi[r], v))
                weights.append(w * speed)
                prescribed.append(g)
        self._inflow = sparse(sample_matrix(traces, self.nv))
        self._incoming_weights = jnp.asarray(
            np.concatenate(weights) if weights else np.empty(0)
        )
        self._incoming_values = jnp.asarray(
            np.vstack(prescribed) if prescribed else np.empty((0, 2))
        )

    def _prepare_loads(self):
        s = self.spatial
        if self.linear_backend == "host_sparse":
            self.force_points = jnp.asarray(
                np.vstack([p for p, w in s.volume.values()])
            )
            self.traction_points = jnp.asarray(
                np.vstack([c[1] for c in s.outflow_curves])
                if s.outflow_curves
                else np.empty((0, 2))
            )
            self.traction_normals = jnp.asarray(
                np.vstack([c[3] for c in s.outflow_curves])
                if s.outflow_curves
                else np.empty((0, 2))
            )
            return
        records, points = [], []
        for r, (p, w) in s.volume.items():
            records.append((s.vi[r], w[:, None] * s.vbasis[r].evaluate(p, 0)[0]))
            points.append(p)
        self.force_points = jnp.asarray(np.vstack(points))
        self._force_load = sparse(sample_matrix(records, self.nv).T)
        records, points, normals = [], [], []
        for r, p, w, n, tag, v, dn, q in s.outflow_curves:
            records.append((s.vi[r], w[:, None] * v))
            points.append(p)
            normals.append(n)
        self.traction_points = jnp.asarray(
            np.vstack(points) if points else np.empty((0, 2))
        )
        self.traction_normals = jnp.asarray(
            np.vstack(normals) if normals else np.empty((0, 2))
        )
        self._traction_load = sparse(sample_matrix(records, self.nv).T)

    def load(self, force=None, traction=None):
        """JAX RHS from sampled body force and vector-Laplacian outlet traction.

        Samples have shape (len(force_points), 2) / (len(traction_points), 2).
        Evaluate time-dependent data at state.time + dt before stepping.
        """
        result = self.boundary_load
        if self.linear_backend == "host_sparse":
            if force is None and traction is None:
                return result
            force = (
                jnp.zeros_like(self.force_points)
                if force is None
                else jnp.asarray(force)
            )
            traction = (
                jnp.zeros_like(self.traction_points)
                if traction is None
                else jnp.asarray(traction)
            )
            if (
                force.shape != self.force_points.shape
                or traction.shape != self.traction_points.shape
            ):
                raise ValueError("Load samples must match quadrature point shapes")

            def assemble(f, tr):
                load = np.array(self.boundary_load)
                offset = 0
                for r, (p, w) in self.spatial.volume.items():
                    v = self.spatial.vbasis[r].evaluate(p, 0)[0]
                    local = v.T @ (w[:, None] * f[offset : offset + len(p)])
                    for d in range(2):
                        load[self.spatial.vi[r] + d * self.nv] += local[:, d]
                    offset += len(p)
                offset = 0
                for r, p, w, n, tag, v, dn, q in self.spatial.outflow_curves:
                    local = v.T @ (w[:, None] * tr[offset : offset + len(p)])
                    for d in range(2):
                        load[self.spatial.vi[r] + d * self.nv] += local[:, d]
                    offset += len(p)
                return load

            return jax.pure_callback(
                assemble,
                jax.ShapeDtypeStruct((self.size,), jnp.float64),
                force,
                traction,
            )
        for values, operator, points in (
            (force, self._force_load, self.force_points),
            (traction, self._traction_load, self.traction_points),
        ):
            if values is not None:
                values = jnp.asarray(values)
                if values.shape != points.shape:
                    raise ValueError(
                        "Load samples must have shape (number of quadrature points, 2)"
                    )
                result = result.at[: 2 * self.nv].add((operator @ values).T.reshape(-1))
        return result

    def convection(self, coefficients):
        """Advective volume residual plus upwind interior/incoming traces."""
        if self.linear_backend == "host_sparse":
            return jax.pure_callback(
                self._host_convection.residual,
                jax.ShapeDtypeStruct((2 * self.nv,), jnp.float64),
                coefficients,
            )
        velocity = coefficients[: 2 * self.nv].reshape(2, self.nv).T
        wind = self._v @ velocity
        adv = wind[:, 0, None] * (self._dx @ velocity) + wind[:, 1, None] * (
            self._dy @ velocity
        )
        result = self._v.T @ (self._weights[:, None] * adv)
        ua, ub = self._left @ velocity, self._right @ velocity
        wn = jnp.sum(0.5 * (ua + ub) * self._normals, axis=1)
        jump = ua - ub
        result += self._left.T @ (
            (self._face_weights * jnp.maximum(-wn, 0))[:, None] * jump
        )
        result -= self._right.T @ (
            (self._face_weights * jnp.maximum(wn, 0))[:, None] * jump
        )
        result += self._inflow.T @ (
            self._incoming_weights[:, None]
            * (self._inflow @ velocity - self._incoming_values)
        )
        return result.T.reshape(-1)

    def stokes_initial_state(self, force=None, traction=None, *, time=0.0):
        """Host Stokes solve for a compatible starting field at this viscosity.

        Force/traction use the same quadrature samples as :meth:`load`. This
        is an initial condition, not a converged Navier--Stokes solution.
        """
        from scipy.sparse.linalg import spsolve

        rhs = np.asarray(self.load(force, traction))
        coefficients = spsolve(self.spatial.A, rhs)
        residual = np.linalg.norm(self.spatial.A @ coefficients - rhs) / max(
            np.linalg.norm(rhs), 1
        )
        if not np.isfinite(residual) or residual > self.residual_tolerance:
            raise RuntimeError(f"Stokes initialization failed: residual={residual}")
        return self.initialize(coefficients, time=time)

    def initialize(self, coefficients, *, time=0.0):
        """Start with a finite, discretely compatible mixed velocity/pressure field.

        Structural mode also checks divergence and normal-flux compatibility
        on independent quadrature. Stabilized mode checks only shape/finiteness.
        """
        c = np.asarray(coefficients, dtype=float)
        if (
            c.shape != (self.size,)
            or not np.all(np.isfinite(c))
            or not np.isfinite(time)
        ):
            raise ValueError("Expected finite mixed coefficients and initial time")
        c = jnp.asarray(c)
        if self.structural:
            _, _, _, valid = self.incompressibility_errors(c)
            if not bool(valid):
                raise ValueError(
                    "Initial velocity violates structural incompressibility or prescribed normal flux"
                )
        return EmbeddedNSState(
            c,
            c,
            jnp.zeros(2 * self.nv, c.dtype),
            jnp.asarray(0),
            jnp.asarray(float(time)),
        )

    def wall_slip_error(self, coefficients):
        """Independent obstacle tangential mismatch in structural mode."""
        if not self.structural:
            return jnp.asarray(jnp.nan)
        return jnp.linalg.norm(self._audit_wall_tangent @ coefficients[:2*self.nv]
                               - self._audit_wall_tangent_data)

    def incompressibility_errors(self, coefficients):
        """Independent-quadrature norms and acceptance flag; JIT-compatible.

        Structural mode checks volume divergence, internal normal jumps and
        boundary-normal mismatch. Legacy mode returns NaN (not certified).
        """
        if not self.structural:
            nan = jnp.asarray(jnp.nan)
            return nan, nan, nan, jnp.asarray(True)
        velocity = coefficients[: 2 * self.nv]
        divergence = jnp.linalg.norm(self._audit_divergence @ velocity)
        jumps = jnp.linalg.norm(self._audit_jumps @ velocity)
        boundary = jnp.linalg.norm(
            self._audit_boundary @ velocity - self._audit_boundary_data
        )
        gradient_scale = jnp.maximum(
            jnp.linalg.norm(self._audit_gradient @ velocity), 1
        )
        velocity_scale = jnp.maximum(
            jnp.sqrt(jnp.maximum(coefficients @ (self.mass @ coefficients), 0)), 1
        )
        boundary_scale = jnp.maximum(jnp.linalg.norm(self._audit_boundary_data), 1)
        tolerance = self.spatial.tolerance
        valid = (
            (divergence <= tolerance * gradient_scale)
            & (jumps <= tolerance * velocity_scale)
            & (boundary <= tolerance * boundary_scale)
        )
        if self.spatial.wall_enforcement == "constraint":
            valid = valid & (self.wall_slip_error(coefficients) <= tolerance *
                jnp.maximum(jnp.linalg.norm(self._audit_wall_tangent_data), 1))
        return divergence, jumps, boundary, valid

    def step(self, state, load=None):
        """One BE startup or BDF2/AB2 step, with a true mixed residual check.

        Pure JAX: invalid results are flagged, not raised inside JIT. Callers
        using this in lax.scan must inspect diagnostics.valid for every step.
        """
        current, older, old_conv, count, time = state
        rhs = self.boundary_load if load is None else jnp.asarray(load)
        if rhs.shape != (self.size,):
            raise ValueError("Expected assembled mixed RHS")
        conv = self.convection(current)
        startup = count == 0
        temporal = jnp.where(startup, current, 2 * current - 0.5 * older)
        explicit = jnp.where(startup, conv, 2 * conv - old_conv)
        rhs = rhs + (self.mass @ temporal) / self.dt
        rhs = rhs.at[: 2 * self.nv].add(-explicit)
        predictor = jnp.where(startup, current, 2 * current - older)

        def solve(args, operator, inverse):
            rhs, predictor = args
            candidate = predictor + inverse(rhs - operator @ predictor)
            defect = rhs - operator @ candidate
            candidate = jax.lax.cond(
                jnp.linalg.norm(defect) > 1e-12 * jnp.maximum(jnp.linalg.norm(rhs), 1),
                lambda c: c + inverse(defect),
                lambda c: c,
                candidate,
            )
            residual = jnp.linalg.norm(rhs - operator @ candidate) / jnp.maximum(
                jnp.linalg.norm(rhs), 1
            )
            return candidate, residual

        candidate, residual = jax.lax.cond(
            startup,
            lambda a: solve(a, self._be_operator, self._be_solve),
            lambda a: solve(a, self._bdf_operator, self._bdf_solve),
            (rhs, predictor),
        )
        divergence, jumps, boundary, compatible = self.incompressibility_errors(
            candidate
        )
        energy = 0.5 * jnp.dot(candidate, self.mass @ candidate)
        valid = (
            jnp.all(jnp.isfinite(candidate))
            & jnp.isfinite(energy)
            & (residual <= self.residual_tolerance)
            & compatible
        )
        return EmbeddedNSState(
            candidate, current, conv, count + 1, time + self.dt
        ), EmbeddedNSDiagnostics(residual, energy, valid, divergence, jumps, boundary)

    def advance(self, state, steps, *, load=None, callback=None):
        """Checked host loop; load(t) optionally returns the new-time mixed RHS."""
        if not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        for _ in range(steps):
            rhs = None if load is None else load(float(state.time) + self.dt)
            candidate, diagnostics = self._compiled_step(state, rhs)
            if not bool(diagnostics.valid):
                raise RuntimeError(
                    f"Invalid NS step at t={float(candidate.time)}: residual={float(diagnostics.linear_residual)}, divergence={float(diagnostics.divergence_l2)}, normal jump={float(diagnostics.normal_jump_l2)}, boundary normal={float(diagnostics.boundary_normal_l2)}"
                )
            state = candidate
            if callback is not None:
                callback(state, diagnostics)
        return state

    def evaluate(self, points, state):
        """Host reconstruction of u,v,p,ux,uy,vx,vy at physical fluid points."""
        return self.spatial.evaluate(points, np.asarray(state.coefficients))


def plan_embedded_navier_stokes2d(
    boundary,
    *,
    dt,
    viscosity,
    cells=4,
    degree=3,
    order=28,
    convection_order=32,
    linear_backend="sparse",
    residual_tolerance=1e-9,
    incompressibility="structural",
    incompressibility_tolerance=1e-9,
    constraint_rank_tolerance=1e-8,
    constraint_backend="dense_reference",
    wall_enforcement="nitsche",
    projector_backend="implicit_qr",
    **spatial_options,
):
    """Build a fixed-obstacle BSPF plan (requires jax_enable_x64=True).

    boundary(points, tag) returns fixed velocities; tag is 'outer' or 'hole'.
    Geometry defaults to [-1,1]^2 minus the ellipse (center .13,-.07; axes .27,.19).
    Pass edges=[x_edges,y_edges], center, axes, outflow=True for a channel.
    Structural mode is the default. Its dense_reference setup is capped at
    2048 velocity DOFs; implicit_qr with host_sparse uses sparse implicit Q
    and a bounded dense SVD of its range core (not JIT-compatible).
    It enforces independent divergence/normal-trace constraints to the specified
    numerical tolerance and checks them at independent quadrature. The previous
    pressure-stabilized mixed method requires incompressibility="stabilized".
    wall_enforcement="constraint" additionally enforces the obstacle tangential
    velocity and includes its independent audit in the step acceptance flag.
    wall_slip_error(coefficients) returns this L2 mismatch. The outer boundary
    retains weak tangential data. The default "nitsche" imposes wall tangential
    data weakly; wall_penalty_factor >= 1 scales only obstacle Nitsche terms.
    projector_backend="array" caches the local-coordinate null basis (256 MiB
    cap) for repeated GEMV actions. This requires implicit_qr/host_sparse setup;
    the full time integrator still runs on the CPU. Pure JAX projection actions
    are available through plan.projector.jax_actions(), without host callbacks.
    Structural coefficient tails contain generalized constraint multipliers;
    use evaluate() for recovered scalar pressure.
    Geometry-prior enrichment is available through prior_levels, prior_modulation,
    prior_surface_samples and the cut/prior tolerances; none uses a PDE solution.
    linear_backend='dense' is limited to 4096 unknowns. 'sparse' transfers host LU
    factors for JAX triangular substitution; large-case speed is not yet tuned.
    """
    if incompressibility not in ("structural", "stabilized"):
        raise ValueError("incompressibility must be 'structural' or 'stabilized'")
    if wall_enforcement not in ("nitsche", "constraint"):
        raise ValueError("wall_enforcement must be 'nitsche' or 'constraint'")
    if projector_backend not in ("implicit_qr", "array"):
        raise ValueError("Unknown projector_backend")
    if projector_backend == "array" and (constraint_backend != "implicit_qr" or incompressibility != "structural"):
        raise ValueError("Array projection requires the implicit_qr structural setup")
    if wall_enforcement == "constraint" and incompressibility != "structural":
        raise ValueError("Wall constraints require structural incompressibility")
    if constraint_backend not in ("dense_reference", "implicit_qr"):
        raise ValueError("Unknown constraint backend")
    if (
        incompressibility == "structural"
        and constraint_backend == "implicit_qr"
        and linear_backend != "host_sparse"
    ):
        raise ValueError("implicit_qr requires the explicit host_sparse backend")
    for value in (incompressibility_tolerance, constraint_rank_tolerance):
        if not np.isfinite(value) or not 0 < value < 1:
            raise ValueError("Constraint tolerances must be finite and in (0,1)")
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX float64 before constructing an embedded NS plan")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0:
        raise ValueError("residual_tolerance must be finite and positive")
    if linear_backend not in ("dense", "sparse", "host_sparse"):
        raise ValueError("linear_backend must be 'dense', 'sparse', or 'host_sparse'")
    if not isinstance(convection_order, (int, np.integer)) or convection_order < 4:
        raise ValueError("convection_order must be an integer >= 4")
    for name, value, minimum in (
        ("cells", cells, 2),
        ("degree", degree, 2),
        ("order", order, 4),
    ):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name in ("center", "axes"):
        if name in spatial_options:
            value = np.asarray(spatial_options[name])
            if value.shape != (2,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite pair")
    if "edges" in spatial_options and spatial_options["edges"] is not None:
        edges = spatial_options["edges"]
        if len(edges) != 2 or any(
            np.asarray(e).ndim != 1 or len(e) != cells + 1 for e in edges
        ):
            raise ValueError("edges must contain two arrays of length cells + 1")
    options = dict(
        cut_fraction=0.15,
        aggregate_frame=True,
        cut_tolerance=1e-3,
        pressure_degree=degree + 1,
        divergence_form="gradient",
        pressure_gradient_penalty=0.1,
    )
    options.update(spatial_options)
    options["factorize"] = False
    spatial = StokesPlan(
        cells=cells, degree=degree, order=order, viscosity=viscosity, **options
    )
    if incompressibility == "structural":
        from ._embedded.incompressible import ConstrainedSpace

        spatial = ConstrainedSpace(
            spatial,
            boundary,
            rank_tolerance=constraint_rank_tolerance,
            tolerance=incompressibility_tolerance,
            backend=constraint_backend,
            wall_enforcement=wall_enforcement,
        )
    else:
        spatial.info["incompressibility"] = "stabilized"
        spatial.info["pressure_stabilization_active"] = True
    if incompressibility == "structural" and constraint_backend == "implicit_qr":
        from ._embedded.projected import SparseProjectedNavierStokes2D

        return SparseProjectedNavierStokes2D(
            spatial, boundary, dt, convection_order, residual_tolerance, projector_backend
        )
    return EmbeddedNavierStokes2D(
        spatial, boundary, dt, convection_order, linear_backend, residual_tolerance
    )


def geometry_prior_options(degree):
    """Geometry-only enrichment settings used in the p=5,7,9,11 sequence.

    Combine these with a geometrically graded ``edges`` array. These settings
    preserve polynomial modes and filter only cut/enrichment residual modes;
    they are a reproducible configuration, not an accuracy guarantee.
    """
    if degree not in (5, 7, 9, 11):
        raise ValueError("Geometry-prior sequence supports degree 5, 7, 9, 11")
    stage = (degree - 5) // 2
    return dict(
        prior_levels=3 + stage,
        prior_modulation=2,
        prior_surface_samples=16 * 2**stage,
        prior_tolerance=1e-5,
        cut_velocity_tolerance=1e-5,
        prior_velocity_tolerance=1e-6,
        pressure_gradient_penalty=0.1,
        prior_velocity_halo=0.5,
        full_order=32,
    )
