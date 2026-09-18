"""Independent Goursat/AAA-lightning reference for the immersed channel.

Follows Xue, Waters & Trefethen, SISC 46 (2024), doi:10.1137/23M1576876.
The author's LARS MATLAB/Python formulas were consulted for cross-checks:
https://github.com/YidanXue/LARS . This is a benchmark implementation, not a
claim to be the official LARS package. No volume discretization is used.

Compare *unforced steady Stokes*, with sponge disabled in both methods:
rectangle [-1,5] x [-1,1], eccentric elliptical no-slip hole, parabolic inlet,
no-slip horizontal walls, and Laplacian traction (u_x-p/nu,v_x)=0 at x=5.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.linalg as la
from scipy.interpolate import AAA


class RationalBasis:
    """Block Arnoldi with differentiated recurrences through second order."""

    def __init__(self, z, degree, poles, laurent):
        self.blocks = [(None, degree), (np.zeros(laurent, complex), laurent)]
        self.blocks += [(np.asarray(p), len(p)) for p in poles]
        self.hessenberg = []
        for p, n in self.blocks:
            q = np.ones((len(z), n + 1), complex)
            h = np.zeros((n + 1, n), complex)
            for k in range(n):
                value = z * q[:, k] if p is None else q[:, k] / (z - p[k])
                # Twice modified Gram-Schmidt; the stored recurrence contains
                # both passes, so evaluation describes exactly the same basis.
                for _ in range(2):
                    for j in range(k + 1):
                        v = np.vdot(q[:, j], value) / len(z)
                        h[j, k] += v
                        value -= v * q[:, j]
                h[k + 1, k] = la.norm(value) / np.sqrt(len(z))
                q[:, k + 1] = value / h[k + 1, k]
            self.hessenberg.append(h)

    def evaluate(self, z):
        output = [[], [], []]
        for ib, ((p, n), h) in enumerate(zip(self.blocks, self.hessenberg)):
            q = np.ones((len(z), n + 1), complex)
            d = np.zeros_like(q)
            dd = np.zeros_like(q)
            for k in range(n):
                if p is None:
                    r, dr, ddr = z, 1, 0
                else:
                    r = 1 / (z - p[k])
                    dr, ddr = -(r**2), 2 * r**3
                hk, scale = h[: k + 1, k], h[k + 1, k]
                q[:, k + 1] = (r * q[:, k] - q[:, : k + 1] @ hk) / scale
                d[:, k + 1] = (r * d[:, k] + dr * q[:, k] - d[:, : k + 1] @ hk) / scale
                dd[:, k + 1] = (
                    r * dd[:, k]
                    + 2 * dr * d[:, k]
                    + ddr * q[:, k]
                    - dd[:, : k + 1] @ hk
                ) / scale
            for out, val in zip(output, (q, d, dd)):
                out.append(val[:, int(ib != 0) :])
        return tuple(np.column_stack(x) for x in output)


def boundary_samples(count, *, independent=False, span=7):
    """Four exterior edges, then ellipse; distinct verification abscissae."""
    if independent:
        t = np.cos(np.pi * (np.arange(count) + 0.381) / count)
    else:
        t = np.tanh(np.linspace(-span, span, count))
    theta = 2 * np.pi * (np.arange(count) + (0.371 if independent else 0)) / count
    z = np.concatenate(
        (
            -1 + 1j * t,
            2 + 3 * t + 1j,
            2 + 3 * t - 1j,
            5 + 1j * t,
            0.19 + 0.31 * np.cos(theta) + 1j * (-0.13 + 0.23 * np.sin(theta)),
        )
    )
    return z


class LightningStokes:
    def __init__(self, degree=64, corner_poles=24, laurent=32, samples=500):
        start = perf_counter()
        self.center = 0.19 - 0.13j
        zphysical = boundary_samples(samples, span=2 * np.sqrt(corner_poles) + 1)
        z = zphysical - self.center
        theta = 2 * np.pi * np.arange(1000) / 1000
        ellipse = 0.31 * np.cos(theta) + 1j * 0.23 * np.sin(theta)
        aaa = AAA(ellipse, ellipse.conj(), rtol=1e-13, max_terms=80)
        poles = aaa.poles()
        poles = poles[
            np.isfinite(poles)
            & ((poles.real / 0.31) ** 2 + (poles.imag / 0.23) ** 2 < 1 - 1e-10)
        ]
        corners = np.array([-1 - 1j, -1 + 1j, 5 - 1j, 5 + 1j]) - self.center
        directions = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j]) / np.sqrt(2)
        distances = 2 * np.exp(
            4 * (np.sqrt(np.arange(corner_poles, 0, -1)) - np.sqrt(corner_poles))
        )
        self.poles = [c + d * distances for c, d in zip(corners, directions)] + [poles]
        self.basis = RationalBasis(z, degree, self.poles, laurent)
        u, v, pressure, omega, ux, vx = self.rows(zphysical)
        a1, a2 = u.copy(), v.copy()
        outlet = slice(3 * samples, 4 * samples)
        a1[outlet] = ux[outlet] - pressure[outlet]
        a2[outlet] = vx[outlet]
        rhs1 = np.zeros(len(z))
        rhs1[:samples] = 1 - zphysical[:samples].imag ** 2
        rhs = np.r_[rhs1, np.zeros(len(z))]
        # Mixed traction/velocity rows: unweighted collocation avoids hiding
        # corner defects. Clustering resolves the smallest lightning pole.
        a = np.vstack((a1, a2))
        scale = la.norm(a, axis=0)
        scale[scale == 0] = 1
        c, _, rank, s = la.lstsq(a / scale, rhs, cond=1e-13, lapack_driver="gelsd")
        self.coefficients = c / scale
        self.info = dict(
            degree=degree,
            corner_poles=corner_poles,
            laurent=laurent,
            aaa_poles=len(poles),
            boundary_samples=len(z),
            unknowns=len(c),
            numerical_rank=rank,
            setup_seconds=perf_counter() - start,
        )

    def rows(self, points):
        z = np.asarray(points).ravel() - self.center
        r, dr, ddr = self.basis.evaluate(z)
        cz = z.conj()[:, None]
        o = 1 / z
        logz = np.log(z)
        zero = np.zeros_like(r)
        # Complex coefficient order: [f rational, g rational, f log, g log].
        # g contains -conj(a)*(z log z-z), making velocities single-valued.
        ub = np.column_stack((cz * dr - r, dr))
        vb = np.column_stack((-cz * dr - r, -dr))
        pb = np.column_stack((4 * dr, zero))
        wb = np.column_stack((-4 * dr, zero))
        u = np.column_stack(
            (
                ub.real,
                (z.conj() * o - 2 * logz).real,
                o.real,
                -ub.imag,
                -(z.conj() * o).imag,
                -o.imag,
            )
        )
        v = np.column_stack(
            (
                vb.imag,
                (-z.conj() * o).imag,
                -o.imag,
                vb.real,
                (-z.conj() * o - 2 * logz).real,
                -o.real,
            )
        )
        pressure = np.column_stack(
            (
                pb.real,
                (4 * o).real,
                np.zeros(len(z)),
                -pb.imag,
                -(4 * o).imag,
                np.zeros(len(z)),
            )
        )
        omega = np.column_stack(
            (
                wb.imag,
                (-4 * o).imag,
                np.zeros(len(z)),
                wb.real,
                (-4 * o).real,
                np.zeros(len(z)),
            )
        )
        xb = np.column_stack((cz * ddr, ddr))
        yb = np.column_stack((-2 * dr - cz * ddr, -ddr))
        ux = np.column_stack(
            (
                xb.real,
                (-z.conj() * o**2 - o).real,
                (-(o**2)).real,
                -xb.imag,
                -(-z.conj() * o**2 + o).imag,
                -(-(o**2)).imag,
            )
        )
        vx = np.column_stack(
            (
                yb.imag,
                (-o + z.conj() * o**2).imag,
                (o**2).imag,
                yb.real,
                (-3 * o + z.conj() * o**2).real,
                (o**2).real,
            )
        )
        return u, v, pressure, omega, ux, vx

    def evaluate(self, points):
        points = np.asarray(points)
        values = []
        for batch in np.array_split(
            points.ravel(), max(1, int(np.ceil(points.size / 2048)))
        ):
            values.append(np.array([r @ self.coefficients for r in self.rows(batch)]))
        return np.concatenate(values, axis=1).reshape((6,) + points.shape)

    def verify(self, count=997):
        z = boundary_samples(count, independent=True)
        u, v, p, w, ux, vx = self.evaluate(z)
        desired = np.zeros_like(u)
        desired[:count] = 1 - z[:count].imag ** 2
        mask = np.ones(len(z), bool)
        mask[3 * count : 4 * count] = False
        out = slice(3 * count, 4 * count)
        return dict(
            velocity_boundary_max=float(
                np.max(np.hypot(u[mask] - desired[mask], v[mask]))
            ),
            hole_wall_max=float(np.max(np.hypot(u[4 * count :], v[4 * count :]))),
            outlet_traction_over_nu_max=float(
                np.max(np.hypot(ux[out] - p[out], vx[out]))
            ),
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("build/immersed_flow/lightning"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    x, y = np.linspace(-1, 5, 401), np.linspace(-1, 1, 161)
    xx, yy = np.meshgrid(x, y)
    z = xx + 1j * yy
    fluid = ((xx - 0.19) / 0.31) ** 2 + ((yy + 0.13) / 0.23) ** 2 > 1
    # Vorticity has no prescribed single-valued trace at mixed-BC corners.
    fluid[[0, 0, -1, -1], [0, -1, 0, -1]] = False
    rows = []
    previous = None
    for degree, npol, nl, nb in [
        (48, 16, 24, 400),
        (72, 24, 36, 600),
        (96, 32, 48, 800),
        (120, 40, 60, 1000),
    ]:
        p = LightningStokes(degree, npol, nl, nb)
        fields = np.full((6,) + z.shape, np.nan)
        fields[:, fluid] = p.evaluate(z[fluid])
        row = {**p.info, **p.verify()}
        if previous is not None:
            d = fields - previous
            row["successive_velocity_max"] = float(np.nanmax(np.hypot(d[0], d[1])))
            row["successive_vorticity_max"] = float(np.nanmax(abs(d[3])))
        rows.append(row)
        print(json.dumps(row), flush=True)
        np.savez(
            args.out / f"reference_{degree}.npz",
            x=x,
            y=y,
            fields=fields,
            coefficients=p.coefficients,
            aaa_poles=p.poles[-1],
            all_poles=np.concatenate(p.poles) + p.center,
        )
        previous = fields
    (args.out / "convergence.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
