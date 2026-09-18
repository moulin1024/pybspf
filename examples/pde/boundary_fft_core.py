"""Arclength-Fourier single-layer core for interior Laplace Dirichlet problems.

No rectangular extension. This validates the boundary solver only: a general
Poisson forcing requires a separate, accurately evaluated volume potential.
Interior tests stay away from the boundary; close evaluation is not implemented.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import circulant
from scipy.signal import resample
from scipy.sparse.linalg import LinearOperator, gmres

from bspf_jax.convex_poisson import ArcLengthBoundary
from bspf_jax.embedded_poisson import benchmark_domains


class Circle:
    period = 4

    def __init__(self, axes=(1.0, 1.0)):
        self.axes = np.asarray(axes)

    def curve(self, t, nu=0):
        phase = np.asarray(t) * np.pi / 2 + nu * np.pi / 2
        return (
            self.axes
            * (np.pi / 2) ** nu
            * np.stack((np.cos(phase), np.sin(phase)), axis=-1)
        )


def geometry_remainder(targets, theta, sources, phi, radius):
    distance = np.linalg.norm(targets[:, None, :] - sources[None, :, :], axis=2)
    reference = 2 * radius * abs(np.sin((theta[:, None] - phi[None, :]) / 2))
    ratio = np.ones_like(distance)
    np.divide(distance, reference, out=ratio, where=reference > 1e-12 * radius)
    return -np.log(ratio) / (2 * np.pi)


class BoundaryFFTPlan:
    def __init__(self, domain, count):
        start = perf_counter()
        self.arc = ArcLengthBoundary(domain)
        self.points, _ = self.arc.sample(count)
        self.count = count
        self.length = self.arc.length
        self.radius = self.length / (2 * np.pi)
        self.theta = 2 * np.pi * np.arange(count) / count
        self.modes = np.fft.fftfreq(count, d=1 / count)
        self.inverse_symbol = 4 * np.pi * abs(self.modes) / self.length
        self.symbol = np.zeros(count)
        self.symbol[1:] = 1 / self.inverse_symbol[1:]
        self.remainder = (
            self.length
            / count
            * geometry_remainder(
                self.points, self.theta, self.points, self.theta, self.radius
            )
        )
        self.single_layer = circulant(np.fft.ifft(self.symbol).real) + self.remainder
        self.setup_seconds = perf_counter() - start

    def inverse_principal(self, values):
        return np.fft.ifft(self.inverse_symbol * np.fft.fft(values)).real

    def solve(self, boundary_data, tolerance=2e-13):
        start = perf_counter()
        h = np.asarray(boundary_data)
        rhs = h - h.mean()
        history = []

        def apply(v):
            correction = self.remainder @ self.inverse_principal(v)
            return v + correction - correction.mean()

        operator = LinearOperator((self.count, self.count), matvec=apply, dtype=float)
        v, status = gmres(
            operator,
            rhs,
            rtol=tolerance,
            atol=0,
            restart=80,
            maxiter=20,
            callback=history.append,
            callback_type="pr_norm",
        )
        if status:
            raise RuntimeError(f"Boundary GMRES failed: {status}")
        density = self.inverse_principal(v)
        constant = float(np.mean(h - self.single_layer @ density))
        residual = self.single_layer @ density + constant - h
        return (
            density,
            constant,
            dict(
                iterations=len(history),
                solve_seconds=perf_counter() - start,
                relative_boundary_residual=float(
                    np.linalg.norm(residual) / np.linalg.norm(h)
                ),
                density_mean=float(density.mean()),
            ),
        )

    def boundary_evaluate(self, density, constant, count, offset=0.371):
        points, _ = self.arc.sample(count, offset=offset)
        theta = 2 * np.pi * (np.arange(count) + offset) / count
        coefficients = self.symbol * np.fft.fft(density) / self.count
        principal = (np.exp(1j * theta[:, None] * self.modes) @ coefficients).real
        remainder = geometry_remainder(
            points, theta, self.points, self.theta, self.radius
        )
        values = principal + self.length / self.count * (remainder @ density) + constant
        return points, values

    def interior_evaluate(self, density, constant, points, oversample=4):
        # Ordinary oversampled quadrature: suitable for well-separated targets only.
        count = self.count * oversample
        sources, _ = self.arc.sample(count)
        rho = resample(density, count)
        distance = np.linalg.norm(points[:, None, :] - sources[None, :, :], axis=2)
        return constant - self.length / (2 * np.pi * count) * (
            np.log(distance / self.radius) @ rho
        )


def harmonic(points):
    x, y = points.T
    return np.exp(3 * x) * np.cos(3 * y) + 0.1 * np.real((x + 1j * y) ** 7)


def main():
    out = Path("build/boundary_fft_core")
    out.mkdir(parents=True, exist_ok=True)
    theta = 2 * np.pi * np.arange(48) / 48
    interior = np.vstack(
        [
            r * np.column_stack((np.cos(theta), np.sin(theta)))
            for r in (0.0, 0.15, 0.3, 0.45)
        ]
    )
    exact = harmonic(interior)
    rows = []
    for name, domain in (
        ("circle", Circle()),
        ("ellipse", Circle((0.9, 0.76))),
        ("convex_bspline", benchmark_domains()[0]),
    ):
        for count in (32, 64, 128, 256, 512):
            plan = BoundaryFFTPlan(domain, count)
            density, constant, diagnostics = plan.solve(harmonic(plan.points))
            points, boundary = plan.boundary_evaluate(density, constant, 2 * count)
            expected = harmonic(points)
            values = plan.interior_evaluate(density, constant, interior)
            row = dict(
                geometry=name,
                boundary_points=count,
                setup_seconds=plan.setup_seconds,
                remainder_max=float(abs(plan.remainder).max()),
                independent_boundary_relative=float(
                    np.linalg.norm(boundary - expected) / np.linalg.norm(expected)
                ),
                interior_relative=float(
                    np.linalg.norm(values - exact) / np.linalg.norm(exact)
                ),
                **diagnostics,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
            if name == "circle":
                assert row["remainder_max"] < 1e-13
                assert row["iterations"] <= 2
                if count >= 64:
                    assert row["independent_boundary_relative"] < 1e-11
                    assert row["interior_relative"] < 1e-11
            if name == "ellipse" and count >= 128:
                assert row["independent_boundary_relative"] < 1e-11
                assert row["interior_relative"] < 1e-11
            np.savez(
                out / f"{name}_{count}.npz",
                points=plan.points,
                density=density,
                constant=constant,
            )
    (out / "results.json").write_text(
        json.dumps(
            dict(
                scope="Laplace boundary core; no nonzero forcing or close evaluation",
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
