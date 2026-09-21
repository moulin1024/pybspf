"""Linear, incompressible n=0 plasma--vacuum free-boundary energy formulation.

Plasma displacement uses restricted tensor BSPF functions. Vacuum has NO
velocity, density or resistivity: mapped BSPF solves div(grad(f)/R)=0.
The historical fitted P1 backend remains an explicitly selected comparator.
The interface trace f=-xi.grad(psi0) couples its Dirichlet-to-Neumann energy
to the plasma. M*qddot + K*q = 0. Only the vertical parity sector is included.

Assumes the supplied equilibrium has constant F, p=j=0 at its smooth closed
surface, no equilibrium sheet current, and is star shaped about (2,0).
The equilibrium surface is fixed for assembly; x+xi is its LINEAR displacement.
"""

from dataclasses import dataclass
import numpy as np
import scipy.linalg as la
from scipy.interpolate import BPoly
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from scipy.special import roots_legendre
from bspf_models.plasma.tokamak_equilibrium import external_field
from bspf_models.plasma.tokamak_linear import parity_basis
from pybspf.tensor import tensor_product
from bspf_models._numerics._flow_kernels import curl_from_gradient


def line_interpolant(line):
    """C2 quintic Hermite evaluation of densely sampled BSPF factors.

    Values/first/second derivatives share ONE interpolant, preserving curl/div
    identities. Interpolation accuracy is checked against multiprecision BSPF.
    """
    x = np.r_[float(line.x[0]), np.asarray(line.points), float(line.x[-1])]
    arrays = [
        np.concatenate((np.asarray(n)[[0]], np.asarray(q), np.asarray(n)[[-1]]))
        for n, q in ((line.bn, line.b), (line.gn, line.g), (line.hn, line.h))
    ]
    return BPoly.from_derivatives(x, np.stack(arrays, axis=1))


class EquilibriumEvaluator:
    def __init__(self, plan, equilibrium, coils, offset):
        self.r = line_interpolant(plan.radial)
        self.z = line_interpolant(plan.vertical)
        self.a, self.alpha = equilibrium["a"], equilibrium["alpha"]
        self.coils, self.offset = coils, offset

    def evaluate(self, points, hessian=False):
        r, z = np.asarray(points).T
        b, g = self.r(r), self.z(z)
        contract = lambda x, y: np.einsum("qi,ij,qj->q", x, self.a, y, optimize=True)
        psi, br, bz = external_field(r, z, self.coils, self.offset)
        psi += contract(b, g)
        pr = contract(self.r(r, 1), g) + r * bz
        pz = contract(b, self.z(z, 1)) - r * br
        if not hessian:
            return psi, pr, pz
        # Only smooth analytic coil fields are differentiated numerically;
        # plasma derivatives are the same BSPF interpolant used above.
        h = 2e-5
        _, brp, bzp = external_field(r + h, z, self.coils, self.offset)
        _, brm, bzm = external_field(r - h, z, self.coils, self.offset)
        _, brz_p, _ = external_field(r, z + h, self.coils, self.offset)
        _, brz_m, _ = external_field(r, z - h, self.coils, self.offset)
        prr = contract(self.r(r, 2), g) + bz + r * (bzp - bzm) / (2 * h)
        prz = contract(self.r(r, 1), self.z(z, 1)) - br - r * (brp - brm) / (2 * h)
        pzz = contract(b, self.z(z, 2)) - r * (brz_p - brz_m) / (2 * h)
        return psi, pr, pz, prr, prz, pzz

    def surface(self, theta, bounds, center=(2.0, 0.0), level=0.0):
        theta = np.asarray(theta)
        direction = np.column_stack((np.cos(theta), np.sin(theta)))
        center = np.asarray(center)
        distances = np.full_like(direction, np.inf)
        for k, (lo, hi) in enumerate(bounds):
            positive = direction[:, k] > 1e-14
            negative = direction[:, k] < -1e-14
            distances[positive, k] = (hi - center[k]) / direction[positive, k]
            distances[negative, k] = (lo - center[k]) / direction[negative, k]
        wall = distances.min(axis=1)
        # First sign crossing, not a remote positive vacuum flux branch.
        scan = np.linspace(0.01, 0.999, 100)
        grid = center + (wall[:, None] * scan)[..., None] * direction[:, None, :]
        values = self.evaluate(grid.reshape(-1, 2))[0].reshape(len(theta), -1) - level
        if np.any(values[:, 0] <= 0) or np.any(~np.any(values < 0, axis=1)):
            raise ValueError("Expected closed star-shaped central plasma")
        index = np.argmax(values < 0, axis=1)
        low = wall * scan[np.maximum(index - 1, 0)]
        high = wall * scan[index]
        for _ in range(40):
            mid = (low + high) / 2
            positive = self.evaluate(center + mid[:, None] * direction)[0] > level
            low = np.where(positive, mid, low)
            high = np.where(positive, high, mid)
        radius = (low + high) / 2
        return (
            center + radius[:, None] * direction,
            center + wall[:, None] * direction,
            radius,
        )


@dataclass
class VacuumResponse:
    points: np.ndarray
    triangles: np.ndarray
    stiffness: object
    inner: np.ndarray
    outer: np.ndarray
    free: np.ndarray
    extension: np.ndarray
    boundary_energy: np.ndarray
    residual: float


def vacuum_response(inner_points, outer_points, layers=24):
    """Discrete harmonic extension; outer delta-psi=0, no changing coil current."""
    if layers < 2:
        raise ValueError("At least two vacuum layers are required")
    n = len(inner_points)
    s = np.linspace(0, 1, layers + 1)
    points = (
        (1 - s[:, None, None]) * inner_points + s[:, None, None] * outer_points
    ).reshape(-1, 2)
    triangles = []
    for k in range(layers):
        for i in range(n):
            a, b = k * n + i, k * n + (i + 1) % n
            c, d = a + n, b + n
            triangles.extend(((a, c, d), (a, d, b)))
    triangles = np.asarray(triangles)
    v = points[triangles]
    det = np.linalg.det(np.stack((v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=-1))
    area = abs(det) / 2
    if np.any(area < 1e-14):
        raise ValueError("Degenerate vacuum mesh")
    grad = (
        np.stack(
            (
                v[:, [1, 2, 0], 1] - v[:, [2, 0, 1], 1],
                v[:, [2, 0, 1], 0] - v[:, [1, 2, 0], 0],
            ),
            axis=-1,
        )
        / det[:, None, None]
    )
    bary = np.array(
        [[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]]
    )
    rq = np.einsum("qi,tij->tqj", bary, v)[..., 0]
    local = (
        np.einsum("tik,tjk->tij", grad, grad)
        * (area * np.mean(1 / rq, axis=1))[:, None, None]
    )
    rows = np.broadcast_to(triangles[:, :, None], local.shape).ravel()
    cols = np.broadcast_to(triangles[:, None, :], local.shape).ravel()
    stiffness = coo_matrix(
        (local.ravel(), (rows, cols)), shape=(len(points),) * 2
    ).tocsr()
    inner, outer = np.arange(n), np.arange(layers * n, (layers + 1) * n)
    free = np.arange(n, layers * n)
    extension = np.zeros((len(points), n))
    extension[inner] = np.eye(n)
    rhs = -stiffness[free][:, inner].toarray()
    extension[free] = splu(stiffness[free][:, free].tocsc()).solve(rhs)
    residual = la.norm((stiffness @ extension)[free]) / max(la.norm(rhs), 1e-30)
    energy = extension.T @ (stiffness @ extension)
    return VacuumResponse(
        points,
        triangles,
        stiffness,
        inner,
        outer,
        free,
        extension,
        (energy + energy.T) / 2,
        float(residual),
    )


class DisplacementBasis:
    def __init__(self, plan, modes):
        self.r, self.z = line_interpolant(plan.radial), line_interpolant(plan.vertical)
        # Low radial eigenfunctions and low even vertical eigenfunctions.
        # Neither velocity nor displacement is constrained on the plasma surface.
        radial_values, radial_vectors = la.eigh(plan.stiffness_r, plan.mass_r)
        self.rt = radial_vectors[:, :modes]
        even = parity_basis(plan.vertical, 1)
        _, rotation = la.eigh(even.T @ np.diag(np.asarray(plan.vertical.lam)) @ even)
        self.zt = even @ rotation[:, : max(2, modes // 2)]
        self.size = self.rt.shape[1] * self.zt.shape[1]

    def evaluate(self, points):
        r, z = np.asarray(points).T
        a = [self.r(r, k) @ self.rt for k in range(3)]
        b = [self.z(z, k) @ self.zt for k in range(3)]
        tensor = lambda i, j: tensor_product(a[i], b[j], paired=True)
        c, cr, cz, crr, crz, czz = [
            tensor(i, j) for i, j in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
        ]
        rr = r[:, None]
        xr, xz = curl_from_gradient(cr, cz, radius=rr)
        xrr, xrz = -crz / rr + cz / rr**2, -czz / rr
        xzr, xzz = crr / rr - cr / rr**2, crz / rr
        zero = np.zeros_like(c)
        join = lambda x, y: np.column_stack((x, y))
        return dict(
            xr=join(xr, zero),
            xz=join(xz, zero),
            xp=join(zero, c),
            xrr=join(xrr, zero),
            xrz=join(xrz, zero),
            xzr=join(xzr, zero),
            xzz=join(xzz, zero),
            xpr=join(zero, cr),
            xpz=join(zero, cz),
        )


def magnetic_displacement(fields, equilibrium_values, points, toroidal_f):
    psi, pr, pz, prr, prz, pzz = equilibrium_values
    r = points[:, 0, None]
    xr, xz, xp = [fields[k] for k in ("xr", "xz", "xp")]
    flux = -xr * pr[:, None] - xz * pz[:, None]
    fr = -(
        fields["xrr"] * pr[:, None]
        + xr * prr[:, None]
        + fields["xzr"] * pz[:, None]
        + xz * prz[:, None]
    )
    fz = -(
        fields["xrz"] * pr[:, None]
        + xr * prz[:, None]
        + fields["xzz"] * pz[:, None]
        + xz * pzz[:, None]
    )
    br, bz = -pz[:, None] / r, pr[:, None] / r
    qp = br * (fields["xpr"] - xp / r) + bz * fields["xpz"] + 2 * toroidal_f * xr / r**2
    return flux, -fz / r, fr / r, qp


@dataclass
class PlasmaVacuumModel:
    basis: object
    evaluator: object
    vacuum: VacuumResponse
    transform: np.ndarray
    stiffness: np.ndarray
    plasma_stiffness: np.ndarray
    vacuum_stiffness: np.ndarray
    trace: np.ndarray
    quadrature_points: np.ndarray
    quadrature_weights: np.ndarray
    boundary: np.ndarray
    theta: np.ndarray
    radius: np.ndarray
    mass_eigenvalues: np.ndarray
    diagnostics: dict
    toroidal_f: float

    def displacement(self, coefficients, points):
        """Only call on plasma points; no vacuum displacement is defined."""
        f = self.basis.evaluate(points)
        raw = self.transform @ coefficients
        return np.column_stack([f[k] @ raw for k in ("xr", "xz", "xp")])

    def vacuum_flux(self, coefficients):
        return self.vacuum.extension @ (self.trace @ coefficients)

    def modes(self):
        # Mass is identity after an explicitly documented restricted-mass cutoff.
        values, vectors = la.eigh(self.stiffness)
        return values, vectors


def assemble_plasma_vacuum(
    plan,
    equilibrium,
    coils,
    offset,
    *,
    modes=12,
    angles=192,
    radial_quadrature=48,
    vacuum_layers=24,
    mass_cutoff=1e-10,
    toroidal_f=6.0,
    wall_scale=None,
    vacuum_method="bspf",
    vacuum_radial_modes=None,
    vacuum_angular_modes=None,
    vacuum_quadrature_order=12,
    vacuum_solver="tensor_pcg",
):
    if modes < 4 or modes > min(plan.shape) or angles < 32 or radial_quadrature < 8:
        raise ValueError("Invalid spatial resolution")
    if not 0 < mass_cutoff < 1:
        raise ValueError("mass_cutoff must lie between zero and one")
    ev = EquilibriumEvaluator(plan, equilibrium, coils, offset)
    bounds = [
        (float(plan.radial.x[0]), float(plan.radial.x[-1])),
        (float(plan.vertical.x[0]), float(plan.vertical.x[-1])),
    ]
    theta = np.arange(angles) * 2 * np.pi / angles
    # Include exact rectangle corners in the vacuum outer boundary.
    corners = np.array([(r, z) for r in bounds[0] for z in bounds[1]])
    corner_angles = np.mod(np.arctan2(corners[:, 1], corners[:, 0] - 2), 2 * np.pi)
    mesh_theta = np.unique(np.r_[theta, corner_angles])
    boundary, wall, _ = ev.surface(mesh_theta, bounds)
    if wall_scale is not None:
        if not np.isfinite(wall_scale) or wall_scale <= 1:
            raise ValueError("wall_scale must exceed one")
        proposed = np.array([2.0, 0.0]) + wall_scale * (boundary - [2.0, 0.0])
        if np.any(
            np.linalg.norm(proposed - [2.0, 0.0], axis=1)
            >= np.linalg.norm(wall - [2.0, 0.0], axis=1)
        ):
            raise ValueError(
                "Shaped conducting wall must remain inside equilibrium rectangle"
            )
        wall = proposed
    _, _, radius = ev.surface(theta, bounds)
    q, w = roots_legendre(radial_quadrature)
    s, sw = (q + 1) / 2, w / 2
    direction = np.column_stack((np.cos(theta), np.sin(theta)))
    points = (
        np.array([2.0, 0.0])
        + radius[:, None, None] * s[None, :, None] * direction[:, None, :]
    ).reshape(-1, 2)
    weights = (
        radius[:, None] ** 2 * s[None, :] * sw[None, :] * (2 * np.pi / angles)
    ).ravel() * points[:, 0]
    basis = DisplacementBasis(plan, modes)
    f = basis.evaluate(points)
    eq = ev.evaluate(points, hessian=True)
    flux, qr, qz, qp = magnetic_displacement(f, eq, points, toroidal_f)
    gram = lambda a, b: a.T @ (weights[:, None] * b)
    weighted = np.vstack([np.sqrt(weights[:, None]) * f[k] for k in ("xr", "xz", "xp")])
    _, singular, vt = la.svd(weighted, full_matrices=False)
    d = singular**2
    keep = d > mass_cutoff * d[0]
    transform = vt[keep].T / singular[keep]
    # Form energies AFTER transforming sampled fields; whitening an assembled
    # ill-conditioned Gram matrix would amplify its cancellation errors.
    f = {key: value @ transform for key, value in f.items()}
    qr, qz, qp = [a @ transform for a in (qr, qz, qp)]
    mass = sum(gram(f[k], f[k]) for k in ("xr", "xz", "xp"))
    # J_phi = alpha R psi^2; no current or pressure sheet at psi=0.
    j = equilibrium["alpha"] * points[:, 0] * np.maximum(eq[0], 0) ** 2
    source = gram(f["xr"], j[:, None] * qz) - gram(f["xz"], j[:, None] * qr)
    kp = sum(gram(a, a) for a in (qr, qz, qp)) - (source + source.T) / 2
    if vacuum_method == "fem":
        vacuum = vacuum_response(boundary, wall, vacuum_layers)
        vacuum_diagnostics = dict(vacuum_method="fem")
    elif vacuum_method == "bspf":
        from bspf_models.plasma.tokamak_vacuum_bspf import mapped_bspf_vacuum

        if vacuum_radial_modes is None:
            vacuum_radial_modes = 31 if wall_scale is None else 24
        if vacuum_angular_modes is None:
            vacuum_angular_modes = 96 if wall_scale is None else 121
        vacuum = mapped_bspf_vacuum(
            ev,
            bounds,
            mesh_theta,
            boundary,
            wall,
            wall_scale=wall_scale,
            radial_modes=vacuum_radial_modes,
            angular_modes=vacuum_angular_modes,
            quadrature_order=vacuum_quadrature_order,
            display_layers=vacuum_layers,
            elliptic_solver=vacuum_solver,
        )
        vacuum_diagnostics = vacuum.diagnostics
    else:
        raise ValueError("vacuum_method must be bspf or fem")
    bf = basis.evaluate(boundary)
    _, pr, pz = ev.evaluate(boundary)
    trace = (-bf["xr"] * pr[:, None] - bf["xz"] * pz[:, None]) @ transform
    trace_error = vacuum.extension[vacuum.inner] @ trace - trace
    kv = trace.T @ vacuum.boundary_energy @ trace
    stiffness = kp + kv
    diagnostics = dict(
        raw_dofs=len(d),
        retained_dofs=int(keep.sum()),
        mass_cutoff=mass_cutoff,
        mass_identity_error=float(la.norm(mass - np.eye(keep.sum()))),
        vacuum_weak_residual=vacuum.residual,
        boundary_flux_equilibrium_linf=float(np.max(abs(ev.evaluate(boundary)[0]))),
        interface_operator_relative_error=float(
            la.norm(trace_error) / max(la.norm(trace), 1e-30)
        ),
        plasma_quadrature_points=len(points),
        vacuum_nodes=len(vacuum.points),
        modes=modes,
        angles=angles,
        radial_quadrature=radial_quadrature,
        vacuum_layers=vacuum_layers,
        wall_scale=wall_scale,
        **vacuum_diagnostics,
    )
    return PlasmaVacuumModel(
        basis,
        ev,
        vacuum,
        transform,
        (stiffness + stiffness.T) / 2,
        (kp + kp.T) / 2,
        (kv + kv.T) / 2,
        trace,
        points,
        weights,
        boundary,
        theta,
        radius,
        d,
        diagnostics,
        toroidal_f,
    )
