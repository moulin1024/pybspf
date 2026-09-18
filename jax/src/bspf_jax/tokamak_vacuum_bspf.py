"""Mapped tensor BSPF vacuum: no finite-element solve or mesh stiffness.

The annulus is parameterized by x(s,theta)=center+rho(s,theta)e(theta).
Both reference directions use the existing BSPF space; angular functions are
restricted to periodic C2 traces. A polynomial lift and homogeneous BSPF modes
enforce the inner/outer flux data. Triangles are ONLY visualization connectivity.
"""

from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import scipy.linalg as la
from scipy.special import roots_legendre
from .stream_navier_stokes import _stream_line
from .tokamak_linear import tensor_form


@lru_cache(maxsize=4)
def reference_lines(radial_nodes, angular_nodes):
    from .tokamak_vacuum import line_interpolant

    r = _stream_line(np.linspace(0, 1, radial_nodes), clamped=False, dirichlet=True)
    t = _stream_line(np.linspace(0, 2 * np.pi, angular_nodes), clamped=False)
    constraints = np.array(
        [np.asarray(v)[0] - np.asarray(v)[-1] for v in (t.bn, t.gn, t.hn)]
    )
    constraints /= la.norm(constraints, axis=1)[:, None]
    periodic = la.null_space(constraints)
    _, rotation = la.eigh(periodic.T @ np.diag(np.asarray(t.lam)) @ periodic)
    return r, t, line_interpolant(r), line_interpolant(t), periodic @ rotation


@dataclass
class PatchAngular:
    """C0 BSPF sectors joined on the rays ending at rectangle corners.

    A rectangular wall makes the annulus map only piecewise smooth. Keeping a
    single C2 angular space there would converge slowly to derivative jumps.
    Each sector uses BSPF homogeneous functions plus its polynomial trace lift.
    """

    evaluator: object
    breaks: np.ndarray
    interior_modes: int

    def __call__(self, theta, nu=0):
        theta = np.asarray(theta)
        mapped = np.mod(theta - self.breaks[0], 2 * np.pi) + self.breaks[0]
        sector = np.minimum(np.searchsorted(self.breaks, mapped, side="right") - 1, 3)
        result = np.zeros((len(theta), 4 * (self.interior_modes + 1)))
        for k in range(4):
            pick = np.where(sector == k)[0]
            length = self.breaks[k + 1] - self.breaks[k]
            x = (mapped[pick] - self.breaks[k]) / length
            if nu == 0:
                left, right = 1 - x, x
            elif nu == 1:
                left, right = -np.ones_like(x) / length, np.ones_like(x) / length
            else:
                left = right = np.zeros_like(x)
            result[pick, k] = left
            result[pick, (k + 1) % 4] = right
            start = 4 + k * self.interior_modes
            result[pick, start : start + self.interior_modes] = (
                self.evaluator(x, nu)[:, : self.interior_modes] / length**nu
            )
        return result


@dataclass
class MappedBSPFVacuum:
    points: np.ndarray
    triangles: np.ndarray
    inner: np.ndarray
    outer: np.ndarray
    free: np.ndarray
    extension: np.ndarray
    boundary_energy: np.ndarray
    residual: float
    homogeneous_stiffness: np.ndarray
    coupling: np.ndarray
    response: np.ndarray
    trace_projection: np.ndarray
    radial_evaluator: object
    angular_evaluator: object
    radial_transform: np.ndarray
    angular_transform: np.ndarray
    angular_nodes: np.ndarray
    diagnostics: dict

    def harmonic_residual(self, trace):
        g = self.trace_projection @ trace
        rhs = self.coupling[:, : len(g)] @ g
        return self.homogeneous_stiffness @ (self.response[:, : len(g)] @ g) + rhs

    def evaluate(self, s, theta, inner_trace, outer_trace=None, derivative=(0, 0)):
        """Evaluate reference derivatives at paired s/theta points."""
        s, theta = np.broadcast_arrays(s, theta)
        shape = s.shape
        s = s.ravel()
        theta = np.mod(theta.ravel(), 2 * np.pi)
        zero = np.zeros_like(inner_trace)
        outer_trace = zero if outer_trace is None else outer_trace
        g = np.r_[
            self.trace_projection @ inner_trace, self.trace_projection @ outer_trace
        ]
        ns, nt = self.radial_transform.shape[1], self.angular_transform.shape[1]
        coeff = (self.response @ g).reshape(ns, nt)
        ds, dt = derivative
        rb = self.radial_evaluator(s, ds) @ self.radial_transform
        tb = self.angular_evaluator(theta, dt) @ self.angular_transform
        value = np.einsum("qi,ij,qj->q", rb, coeff, tb, optimize=True)
        if ds == 0:
            lo, hi = 1 - s, s
        elif ds == 1:
            lo, hi = -np.ones_like(s), np.ones_like(s)
        else:
            lo = hi = np.zeros_like(s)
        value += lo * (tb @ g[:nt]) + hi * (tb @ g[nt:])
        return value.reshape(shape)


def mapped_bspf_vacuum(
    evaluator,
    bounds,
    mesh_theta,
    inner_points,
    outer_points,
    *,
    wall_scale=None,
    radial_modes=12,
    angular_modes=48,
    radial_nodes=33,
    angular_nodes=65,
    quadrature_order=12,
    display_layers=24,
    elliptic_solver="tensor_pcg",
):
    if radial_modes < 2 or angular_modes < 4 or quadrature_order < 4:
        raise ValueError("Insufficient mapped BSPF resolution")
    if wall_scale is not None:
        angular_nodes = max(angular_nodes, angular_modes + 4)
    r, t, rb_eval, tb_eval, periodic = reference_lines(radial_nodes, angular_nodes)
    rt = np.eye(r.b.shape[1])[:, :radial_modes]
    tt = periodic[:, :angular_modes]
    if rt.shape[1] != radial_modes or (
        wall_scale is not None and tt.shape[1] != angular_modes
    ):
        raise ValueError("Requested more modes than the BSPF space contains")
    # Integrate the mapped rectangle corners as separate smooth angular sectors.
    corners = np.array([(x, z) for x in bounds[0] for z in bounds[1]])
    corner_angles = np.mod(np.arctan2(corners[:, 1], corners[:, 0] - 2), 2 * np.pi)
    if wall_scale is None:
        sectors = np.sort(corner_angles)
        sectors = np.r_[sectors, sectors[0] + 2 * np.pi]
        count = int(np.ceil(angular_modes / 4)) - 1
        if count > r.b.shape[1]:
            raise ValueError("Requested too many BSPF modes per angular patch")
        tb_eval = PatchAngular(rb_eval, sectors, count)
        angular_modes = 4 * (count + 1)
        tt = np.eye(angular_modes)
    breaks = np.unique(np.r_[np.linspace(0, 2 * np.pi, 65), corner_angles])
    gauss, weight = roots_legendre(quadrature_order)
    theta = np.concatenate(
        [(a + b) / 2 + (b - a) / 2 * gauss for a, b in zip(breaks[:-1], breaks[1:])]
    )
    wt = np.concatenate([(b - a) / 2 * weight for a, b in zip(breaks[:-1], breaks[1:])])
    s = np.asarray(r.points)
    ws = np.asarray(r.weights)
    surface, wall, a = evaluator.surface(theta, bounds)
    _, pr, pz = evaluator.evaluate(surface)
    c, sn = np.cos(theta), np.sin(theta)
    ap = -a * (-pr * sn + pz * c) / (pr * c + pz * sn)
    if wall_scale is not None:
        b = wall_scale * a
        bp = wall_scale * ap
    else:
        b = np.linalg.norm(wall - [2, 0], axis=1)
        radial_side = (
            np.minimum(abs(wall[:, 0] - bounds[0][0]), abs(wall[:, 0] - bounds[0][1]))
            < 1e-10
        )
        bp = np.empty_like(b)
        bp[radial_side] = b[radial_side] * sn[radial_side] / c[radial_side]
        bp[~radial_side] = -b[~radial_side] * c[~radial_side] / sn[~radial_side]
    h = b - a
    rho = a[None, :] + s[:, None] * h[None, :]
    rhot = ap[None, :] + s[:, None] * (bp - ap)[None, :]
    major = 2 + rho * c[None, :]
    weights = ws[:, None] * wt[None, :]
    aa = weights * (rho * rho + rhot * rhot) / (h[None, :] * rho * major)
    bb = -weights * rhot / (rho * major)
    cc = weights * h[None, :] / (rho * major)
    rb = np.asarray(r.b) @ rt
    rg = np.asarray(r.g) @ rt
    tb = tb_eval(theta) @ tt
    tg = tb_eval(theta, 1) @ tt
    lift = np.column_stack((1 - s, s))
    liftg = np.column_stack((-np.ones_like(s), np.ones_like(s)))

    def form(x, g, y, d):
        return (
            tensor_form(g, d, tb, tb, aa)
            + tensor_form(g, y, tb, tg, bb)
            + tensor_form(x, d, tg, tb, bb)
            + tensor_form(x, y, tg, tg, cc)
        )

    stiffness = form(rb, rg, rb, rg)
    stiffness = (stiffness + stiffness.T) / 2
    coupling = form(rb, rg, lift, liftg)
    lift_energy = form(lift, liftg, lift, liftg)
    if elliptic_solver == "tensor_pcg":
        from ._tensor_pcg import plan_tensor_preconditioner, tensor_pcg

        preconditioner = plan_tensor_preconditioner(
            rb.T @ (ws[:, None] * rb),
            rg.T @ (ws[:, None] * rg),
            tb.T @ (wt[:, None] * tb),
            tg.T @ (wt[:, None] * tg),
            float(aa.sum() / weights.sum()),
            float(cc.sum() / weights.sum()),
        )
        response, elliptic_diagnostics = tensor_pcg(
            stiffness, -coupling, preconditioner
        )
    elif elliptic_solver == "cholesky":
        response = -la.cho_solve(la.cho_factor(stiffness), coupling)
        elliptic_diagnostics = dict(elliptic_solver="cholesky")
    else:
        raise ValueError("elliptic_solver must be tensor_pcg or cholesky")
    residual = la.norm(stiffness @ response + coupling) / la.norm(coupling)
    schur = lift_energy + coupling.T @ response
    schur = (schur + schur.T) / 2
    # Periodic trapezoidal weights for the nonuniform boundary sampling grid.
    mesh_theta = np.asarray(mesh_theta)
    spacing = np.diff(np.r_[mesh_theta, mesh_theta[0] + 2 * np.pi])
    wm = (spacing + np.roll(spacing, 1)) / 2
    bn = tb_eval(mesh_theta) @ tt
    projection = la.lstsq(np.sqrt(wm[:, None]) * bn, np.diag(np.sqrt(wm)))[0]
    boundary_energy = projection.T @ schur[:angular_modes, :angular_modes] @ projection
    n = len(mesh_theta)
    sd = np.linspace(0, 1, display_layers + 1)
    points = (
        (1 - sd[:, None, None]) * inner_points + sd[:, None, None] * outer_points
    ).reshape(-1, 2)
    triangles = []
    for k in range(display_layers):
        for i in range(n):
            u, v = k * n + i, k * n + (i + 1) % n
            triangles.extend(((u, u + n, v + n), (u, v + n, v)))
    rr = rb_eval(sd) @ rt
    modes = response[:, :angular_modes].reshape(
        radial_modes, angular_modes, angular_modes
    )
    nodal_response = np.einsum("si,ijk,tj->stk", rr, modes, bn, optimize=True)
    nodal_response += (1 - sd[:, None, None]) * bn[None, :, :]
    extension = nodal_response.reshape(-1, angular_modes) @ projection
    # Exactly zero outer data in representation, avoiding harmless endpoint noise.
    extension[-n:] = 0
    diagnostics = dict(
        vacuum_method="mapped_bspf",
        vacuum_radial_modes=radial_modes,
        vacuum_angular_modes=angular_modes,
        vacuum_dofs=radial_modes * angular_modes,
        vacuum_boundary_projection_rank=int(np.linalg.matrix_rank(projection)),
        vacuum_angular_patches=4 if wall_scale is None else 1,
        vacuum_quadrature_order=quadrature_order,
        **elliptic_diagnostics,
    )
    return MappedBSPFVacuum(
        points,
        np.asarray(triangles),
        np.arange(n),
        np.arange(display_layers * n, (display_layers + 1) * n),
        np.arange(n, display_layers * n),
        extension,
        (boundary_energy + boundary_energy.T) / 2,
        float(residual),
        stiffness,
        coupling,
        response,
        projection,
        rb_eval,
        tb_eval,
        rt,
        tt,
        mesh_theta,
        diagnostics,
    )
