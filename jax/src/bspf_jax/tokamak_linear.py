"""Axisymmetric incompressible linear MHD about a BSPF tokamak equilibrium.

Four evolved 2D fields: poloidal velocity streamfunction, toroidal velocity,
poloidal magnetic flux, toroidal magnetic field. Toroidal background F/R is
retained. Up/down parities select the vertical n=0 family, not a rigid motion.

The exterior is a low-density resistive halo APPROXIMATION to vacuum, with a
stationary fixed-magnetic outer boundary. It is not an exact plasma/vacuum interface solver
and does not include a resistive vessel, feedback, thermodynamics, or VDE impact.
"""

from dataclasses import dataclass, replace
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, eigs
from .stream_navier_stokes import _stream_line
from .tokamak_equilibrium import external_field


def tensor_form(xt, xs, yt, ys, weight):
    """Weighted tensor Galerkin bilinear form, C-order modal flattening."""
    nx, ns = xt.shape[1], xs.shape[1]
    ny, nt = yt.shape[1], ys.shape[1]
    left = (xt[:, :, None] * xs[:, None, :]).reshape(len(xt), nx * ns)
    right = (yt[:, :, None] * ys[:, None, :]).reshape(len(yt), ny * nt)
    return (
        (left.T @ weight @ right)
        .reshape(nx, ns, ny, nt)
        .transpose(0, 2, 1, 3)
        .reshape(nx * ny, ns * nt)
    )


def parity_basis(line, sign):
    b, w = np.asarray(line.b), np.asarray(line.weights)
    reflection = b.T @ (w[:, None] * b[::-1])
    values, rotation = la.eigh((reflection + reflection.T) / 2)
    return rotation[:, sign * values > 0.5]


@dataclass
class LinearTokamak:
    equilibrium_plan: object
    velocity_r: object
    velocity_z: object
    parity_velocity: np.ndarray
    parity_flux: np.ndarray
    parity_toroidal_velocity: np.ndarray
    matrices: dict
    shapes: tuple
    density: np.ndarray
    resistivity: np.ndarray
    equilibrium: dict
    coils: np.ndarray
    offset: float
    toroidal_f: float

    def split(self, state):
        result = []
        start = 0
        for shape in self.shapes:
            size = int(np.prod(shape))
            result.append(state[start : start + size].reshape(shape))
            start += size
        return result

    def fields(self, state, nodes=False):
        a, c, b, d = self.split(state)
        r, z = self.equilibrium_plan.radial, self.equilibrium_plan.vertical
        vr, vz = self.velocity_r, self.velocity_z
        rb, rg = (r.bn, r.gn) if nodes else (r.b, r.g)
        zb, zg = (z.bn, z.gn) if nodes else (z.b, z.g)
        vb, vg = (vr.bn, vr.gn) if nodes else (vr.b, vr.g)
        yb, yg = (vz.bn, vz.gn) if nodes else (vz.b, vz.g)
        rr = np.asarray(r.x if nodes else r.points)[:, None]
        a = a @ self.parity_velocity.T
        c = c @ self.parity_toroidal_velocity.T
        b = b @ self.parity_flux.T
        d = d @ self.parity_flux.T
        ur = -np.asarray(vb) @ a @ np.asarray(yg).T / rr
        uz = np.asarray(vg) @ a @ np.asarray(yb).T / rr
        up = np.asarray(rb) @ c @ np.asarray(zb).T
        flux = np.asarray(rb) @ b @ np.asarray(zb).T
        br = -np.asarray(rb) @ b @ np.asarray(zg).T / rr
        bz = np.asarray(rg) @ b @ np.asarray(zb).T / rr
        bp = np.asarray(rb) @ d @ np.asarray(zb).T
        return flux, np.stack((ur, uz, up), axis=-1), np.stack((br, bz, bp), axis=-1)


def assemble_linear_tokamak(
    p,
    eq,
    coils,
    offset,
    *,
    toroidal_f=6.0,
    halo_density=0.01,
    vacuum_resistivity=1.0,
    viscosity=1e-4,
):
    if not 0 < halo_density <= 1 or vacuum_resistivity < 0 or viscosity < 0:
        raise ValueError("Invalid density, resistivity or viscosity")
    r, z = p.radial, p.vertical
    vr = _stream_line(np.asarray(r.x), clamped=True)
    vz = _stream_line(np.asarray(z.x), clamped=True)
    even = parity_basis(vz, 1)
    odd = parity_basis(z, -1)
    tor_even = parity_basis(z, 1)
    br, gr, hr = map(np.asarray, (vr.b, vr.g, vr.h))
    by, gy, hy = [np.asarray(a) @ even for a in (vz.b, vz.g, vz.h)]
    bm, gm = map(np.asarray, (r.b, r.g))
    zm, gzm = [np.asarray(a) @ odd for a in (z.b, z.g)]
    ze, gze = [np.asarray(a) @ tor_even for a in (z.b, z.g)]
    rr = np.asarray(r.points)[:, None]
    zz = np.asarray(z.points)[None, :]
    w = np.asarray(r.weights)[:, None] * np.asarray(z.weights)[None, :]
    core = eq["core"]
    density = halo_density + (1 - halo_density) * core.astype(float)
    eta = vacuum_resistivity * (~core).astype(float)
    _, brp, bzp = p.evaluate(eq["a"])
    _, brc, bzc = external_field(rr, zz, coils, offset)
    Br, Bz = brp + brc, bzp + bzc
    Bphi = toroidal_f / rr
    j0 = eq["current"]
    form = tensor_form
    print("Assembling axisymmetric mass and dissipation...", flush=True)
    mv = form(br, br, gy, gy, w * density / rr) + form(gr, gr, by, by, w * density / rr)
    mt = form(bm, bm, ze, ze, w * density * rr)
    mb = form(bm, bm, zm, zm, w * rr)
    kz = odd.T @ np.diag(np.asarray(z.lam)) @ odd
    k = np.kron(p.stiffness_r, np.eye(zm.shape[1])) + np.kron(p.mass_r, kz)
    geta = form(bm, bm, zm, zm, w * eta * rr)
    db = form(gm + bm / rr, gm + bm / rr, zm, zm, w * eta * rr) + form(
        bm, bm, gzm, gzm, w * eta * rr
    )
    dt = viscosity * (
        form(gm - bm / rr, gm - bm / rr, ze, ze, w * rr)
        + form(bm, bm, gze, gze, w * rr)
    )
    # 2*mu*epsilon(u):epsilon(v), including cylindrical hoop strain.
    dv = (
        2
        * viscosity
        * (
            form(-gr / rr + br / rr**2, -gr / rr + br / rr**2, gy, gy, w * rr)
            + form(gr / rr, gr / rr, gy, gy, w * rr)
            + form(br / rr**2, br / rr**2, gy, gy, w * rr)
        )
    )
    ar = hr / rr - gr / rr**2
    cr = -br / rr
    cross = form(ar, cr, by, hy, w * rr)
    dv += viscosity * (
        form(ar, ar, by, by, w * rr) + form(cr, cr, hy, hy, w * rr) + cross + cross.T
    )
    print("Assembling coupled full-vector Lorentz/induction terms...", flush=True)
    ba = form(bm, br, zm, gy, w * Bz) + form(bm, gr, zm, by, w * Br)
    # Equilibrium-current term is the source of free energy, not added driving.
    current_force = -form(br, gm, gy, zm, w * j0 / rr) + form(
        gr, bm, by, gzm, w * j0 / rr
    )
    da = -2 * form(bm, br, zm, gy, w * Bphi / rr)
    dc = form(bm, gm - bm / rr, zm, ze, w * rr * Br) + form(
        bm, bm, zm, gze, w * rr * Bz
    )
    nv, nt, nb = mv.shape[0], mt.shape[0], mb.shape[0]
    mass = la.block_diag(mv, mt, np.eye(nb), mb)
    avb = -ba.T @ k + current_force
    matrix = np.block(
        [
            [-dv, np.zeros((nv, nt)), avb, -da.T],
            [np.zeros((nt, nv)), -dt, np.zeros((nt, nb)), -dc.T],
            [ba, np.zeros((nb, nt)), -geta @ k, np.zeros((nb, nb))],
            [da, dc, np.zeros((nb, nb)), -db],
        ]
    )
    energy = la.block_diag(mv, mt, k, mb)
    # Independent baseline force-balance residual in the velocity test space.
    force = -br.T @ (w * j0 * Bz) @ gy + gr.T @ (w * (-j0 * Br)) @ by
    baseline_accel = la.solve(mv, force.ravel(), assume_a="pos")
    matrices = dict(
        A=matrix,
        M=mass,
        energy=energy,
        K=k,
        velocity_mass=mv,
        toroidal_mass=mt,
        magnetic_mass=mb,
        viscous=dv,
        toroidal_viscous=dt,
        resistive_flux=geta,
        resistive_toroidal=db,
        current_force=current_force,
        baseline_accel=baseline_accel,
    )
    return LinearTokamak(
        p,
        vr,
        vz,
        even,
        odd,
        tor_even,
        matrices,
        (
            (br.shape[1], by.shape[1]),
            (bm.shape[1], ze.shape[1]),
            (bm.shape[1], zm.shape[1]),
            (bm.shape[1], zm.shape[1]),
        ),
        density,
        eta,
        eq,
        np.asarray(coils),
        offset,
        toroidal_f,
    )


def growing_modes(model, shift=0.1, count=8):
    """Solve A*x=gamma*M*x near shift; verify eigenpairs in original equations."""
    a, m = model.matrices["A"], model.matrices["M"]
    lu = la.lu_factor(a - shift * m)
    op = LinearOperator(a.shape, matvec=lambda x: la.lu_solve(lu, m @ x), dtype=float)
    values, vectors = eigs(
        op,
        k=count,
        which="LM",
        tol=1e-10,
        maxiter=2000,
        v0=np.random.default_rng(0).normal(size=a.shape[0]),
    )
    gammas = shift + 1 / values
    order = np.argsort(gammas.real)[::-1]
    gammas, vectors = gammas[order], vectors[:, order]
    residuals = np.array(
        [
            la.norm(a @ v - g * m @ v)
            / (la.norm(a @ v) + abs(g) * la.norm(m @ v) + 1e-30)
            for g, v in zip(gammas, vectors.T)
        ]
    )
    return gammas, vectors, residuals


def with_exterior(model, *, halo_density=None, vacuum_resistivity=None):
    """Change only the exterior approximation, holding equilibrium/coils fixed."""
    if halo_density is not None and not 0 < halo_density < 1:
        raise ValueError("Require 0<halo_density<1")
    if vacuum_resistivity is not None and vacuum_resistivity <= 0:
        raise ValueError("Require positive exterior resistivity")
    matrices = {key: value.copy() for key, value in model.matrices.items()}
    sizes = [int(np.prod(shape)) for shape in model.shapes]
    nv, nt, nb, _ = sizes
    density = model.density.copy()
    eta = model.resistivity.copy()
    if halo_density is not None:
        vr, vz = model.velocity_r, model.velocity_z
        r = model.equilibrium_plan.radial
        b, g, w = map(np.asarray, (vr.b, vr.g, vr.weights))
        w = w / np.asarray(vr.points)
        gy = np.asarray(vz.g) @ model.parity_velocity
        ky = gy.T @ (np.asarray(vz.weights)[:, None] * gy)
        uniform_v = np.kron(b.T @ (w[:, None] * b), ky) + np.kron(
            g.T @ (w[:, None] * g), np.eye(gy.shape[1])
        )
        bm = np.asarray(r.b)
        wr = np.asarray(r.weights) * np.asarray(r.points)
        uniform_t = np.kron(bm.T @ (wr[:, None] * bm), np.eye(model.shapes[1][1]))
        old = float(np.min(density))
        for key, uniform, start, end in (
            ("velocity_mass", uniform_v, 0, nv),
            ("toroidal_mass", uniform_t, nv, nv + nt),
        ):
            original = matrices[key]
            updated = original + (halo_density - old) * (uniform - original) / (1 - old)
            matrices[key] = updated
            matrices["M"][start:end, start:end] = updated
            matrices["energy"][start:end, start:end] = updated
        density = np.where(model.equilibrium["core"], 1.0, halo_density)
    if vacuum_resistivity is not None:
        ratio = vacuum_resistivity / float(np.max(eta))
        eta *= ratio
        matrices["resistive_flux"] *= ratio
        matrices["resistive_toroidal"] *= ratio
        start = nv + nt
        matrices["A"][start : start + nb, start : start + nb] *= ratio
        matrices["A"][start + nb :, start + nb :] *= ratio
    return replace(model, matrices=matrices, density=density, resistivity=eta)
