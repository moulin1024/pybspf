"""Random-wave Poisson MMS on an analytic ellipse, without field extension.

The volume potential is a source-specific independent reference: invert each
analytic forcing mode, then use Green's identity and high-order parametric
ellipse quadrature to evaluate its free-space volume potential. This is not a
general-purpose volume quadrature implementation. The tested unknown boundary
density is nonzero. All convergence runs use the same reference resolution.
"""

import json
from pathlib import Path

import numpy as np
from scipy.signal import resample

from boundary_fft_core import BoundaryFFTPlan, Circle, geometry_remainder
from bspf_models.elliptic.convex_poisson import ArcLengthBoundary
from bspf_models.elliptic.random_wave_mms import RandomWaveMMS


class ForcingModes:
    """An analytic particular solution reconstructed from forcing coefficients."""

    def __init__(self, waves):
        self.modes = []
        for wave in waves:
            k2 = np.sum(wave.wavevectors**2, axis=1)
            forcing_amplitude = wave.amplitudes * k2
            self.modes.append(
                (wave.wavevectors.copy(), forcing_amplitude / k2, wave.phases.copy())
            )

    def particular(self, points):
        values, gradients = [], []
        for vectors, amplitudes, phases in self.modes:
            phase = points @ vectors.T + phases
            values.append(np.cos(phase) @ amplitudes)
            gradients.append(-(np.sin(phase) * amplitudes) @ vectors)
        return np.stack(values, axis=1), np.stack(gradients, axis=1)


class EllipseVolumeReference:
    """Free-space volume potential v=p+D[p]-S[dn p], where -Delta p=f.

    On the boundary, v=p/2+K[p]-S[dn p]. Analytic ellipse kernel splitting is
    performed in eccentric angle, independently of the arclength solver. The
    regular kernels depend on the sum of the two angles (Hankel convolution).
    """

    def __init__(self, domain, source, count, radius):
        self.source, self.count, self.radius = source, count, radius
        self.axes = domain.axes
        a, b = self.axes
        theta = 2 * np.pi * np.arange(count) / count
        self.points = np.column_stack((a * np.cos(theta), b * np.sin(theta)))
        self.normal_measure = np.column_stack((b * np.cos(theta), a * np.sin(theta)))
        self.p, gradient = source.particular(self.points)
        self.sigma = np.einsum("nci,ni->nc", gradient, self.normal_measure)
        modes = np.fft.fftfreq(count, d=1 / count)
        symbol = np.zeros(count)
        symbol[1:] = 1 / (2 * abs(modes[1:]))
        denominator = a**2 * np.sin(theta / 2) ** 2 + b**2 * np.cos(theta / 2) ** 2
        remainder = -np.log(np.sqrt(denominator) / radius) / (2 * np.pi)
        double_kernel = -a * b / (4 * np.pi * denominator)

        def hankel(kernel, values):
            return (
                2
                * np.pi
                / count
                * np.fft.ifft(
                    np.fft.fft(kernel)[:, None] * np.conj(np.fft.fft(values, axis=0)),
                    axis=0,
                ).real
            )

        single = np.fft.ifft(
            symbol[:, None] * np.fft.fft(self.sigma, axis=0), axis=0
        ).real + hankel(remainder, self.sigma)
        trace = self.p / 2 + hankel(double_kernel, self.p) - single
        self.coefficients = np.fft.fft(trace, axis=0) / count
        self.modes = modes

    def boundary(self, points):
        theta = np.arctan2(points[:, 1] / self.axes[1], points[:, 0] / self.axes[0])
        return np.vstack(
            [
                (np.exp(1j * t[:, None] * self.modes) @ self.coefficients).real
                for t in np.array_split(theta, max(1, (len(theta) + 127) // 128))
            ]
        )

    def interior(self, points):
        values = []
        for target in np.array_split(points, max(1, (len(points) + 127) // 128)):
            difference = target[:, None, :] - self.points[None, :, :]
            r2 = np.sum(difference**2, axis=2)
            single = -np.log(np.sqrt(r2) / self.radius) @ self.sigma / self.count
            double = (
                (np.einsum("tni,ni->tn", difference, self.normal_measure) / r2)
                @ self.p
                / self.count
            )
            values.append(self.source.particular(target)[0] + double - single)
        return np.vstack(values)


def boundary_values(plan, densities, constants, points, arclength_angles):
    coefficients = plan.symbol[:, None] * np.fft.fft(densities, axis=0) / plan.count
    out = []
    for indices in np.array_split(np.arange(len(points)), (len(points) + 127) // 128):
        angles = arclength_angles[indices]
        principal = (np.exp(1j * angles[:, None] * plan.modes) @ coefficients).real
        remainder = geometry_remainder(
            points[indices], angles, plan.points, plan.theta, plan.radius
        )
        out.append(
            principal + plan.length / plan.count * remainder @ densities + constants
        )
    return np.vstack(out)


def interior_values(plan, densities, constants, points):
    # A fixed minimum source resolution prevents close-evaluation error from
    # masquerading as boundary discretization error. Targets satisfy rho<=0.92.
    count = max(4096, 4 * plan.count)
    sources, _ = plan.arc.sample(count)
    rho = resample(densities, count, axis=0)
    out = []
    for target in np.array_split(points, (len(points) + 127) // 128):
        distance = np.linalg.norm(target[:, None, :] - sources[None, :, :], axis=2)
        out.append(
            constants
            - plan.length / (2 * np.pi * count) * np.log(distance / plan.radius) @ rho
        )
    return np.vstack(out)


def relative(error, reference):
    return float(np.linalg.norm(error) / np.linalg.norm(reference))


def main():
    out = Path("build/boundary_fft_random_mms")
    out.mkdir(parents=True, exist_ok=True)
    domain = Circle((0.9, 0.76))
    arc = ArcLengthBoundary(domain)
    bands = [8, 32, 64, 128]
    waves = [RandomWaveMMS.create(kmax=k * np.pi) for k in bands]
    source = ForcingModes(waves)
    for k, wave in zip(bands, waves):
        wave.save(out / f"waves_{k}pi.npz")
    references = [
        EllipseVolumeReference(domain, source, n, arc.length / (2 * np.pi))
        for n in (2048, 4096)
    ]
    boundary_count, offset = 4096, 0.371
    boundary, _ = arc.sample(boundary_count, offset=offset)
    angles = 2 * np.pi * (np.arange(boundary_count) + offset) / boundary_count
    line = np.linspace(-0.92, 0.92, 55)
    xx, yy = np.meshgrid(line * domain.axes[0], line * domain.axes[1])
    mask = (xx / domain.axes[0]) ** 2 + (yy / domain.axes[1]) ** 2 <= 0.92**2
    interior = np.column_stack((xx[mask], yy[mask]))
    exact_boundary = np.column_stack([w.evaluate(boundary)[0] for w in waves])
    exact_interior = np.column_stack([w.evaluate(interior)[0] for w in waves])
    vb = [r.boundary(boundary) for r in references]
    vi = [r.interior(interior) for r in references]
    checks = [
        dict(
            kmax_pi=k,
            boundary_relative_change=relative(
                vb[1][:, j] - vb[0][:, j], exact_boundary[:, j]
            ),
            interior_relative_change=relative(
                vi[1][:, j] - vi[0][:, j], exact_interior[:, j]
            ),
        )
        for j, k in enumerate(bands)
    ]
    print(json.dumps(dict(reference_checks=checks)), flush=True)
    assert (
        max(
            max(c["boundary_relative_change"], c["interior_relative_change"])
            for c in checks
        )
        < 2e-11
    )
    rows = []
    for count in (
        32,
        48,
        64,
        80,
        96,
        112,
        128,
        144,
        160,
        176,
        192,
        208,
        224,
        240,
        256,
        288,
        320,
        352,
        384,
        448,
        512,
        544,
        576,
        608,
        640,
        672,
        704,
        736,
        768,
        1024,
        1536,
    ):
        plan = BoundaryFFTPlan(domain, count)
        g = np.column_stack([w.evaluate(plan.points)[0] for w in waves])
        h = g - references[1].boundary(plan.points)
        solved = [plan.solve(h[:, j], tolerance=2e-14) for j in range(len(waves))]
        densities = np.column_stack([s[0] for s in solved])
        constants = np.array([s[1] for s in solved])
        predicted_boundary = vb[1] + boundary_values(
            plan, densities, constants, boundary, angles
        )
        predicted_interior = vi[1] + interior_values(
            plan, densities, constants, interior
        )
        for j, k in enumerate(bands):
            row = dict(
                kmax_pi=k,
                boundary_points=count,
                boundary_relative=relative(
                    predicted_boundary[:, j] - exact_boundary[:, j],
                    exact_boundary[:, j],
                ),
                interior_relative=relative(
                    predicted_interior[:, j] - exact_interior[:, j],
                    exact_interior[:, j],
                ),
                interior_max_scaled=float(
                    np.max(abs(predicted_interior[:, j] - exact_interior[:, j]))
                    / np.max(abs(exact_interior[:, j]))
                ),
                correction_to_data=relative(h[:, j], g[:, j]),
                setup_seconds=plan.setup_seconds,
                **solved[j][2],
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
        if count in (128, 256, 512, 1024, 1536):
            np.savez(
                out / f"fields_{count}.npz",
                x=xx,
                y=yy,
                mask=mask,
                exact=exact_interior,
                predicted=predicted_interior,
                boundary=boundary,
                boundary_exact=exact_boundary,
                boundary_predicted=predicted_boundary,
            )
    result = dict(
        scope="Random-wave nonzero-source Poisson; source-specific free-space volume reference, not general volume quadrature",
        axes=domain.axes.tolist(),
        shell_variance="k^(-5/3)",
        seed=waves[0].seed,
        modes_per_signal=len(waves[0].amplitudes),
        boundary_validation_count=boundary_count,
        interior_validation_count=len(interior),
        interior_max_elliptic_radius=0.92,
        volume_reference_counts=[2048, 4096],
        reference_checks=checks,
        rows=rows,
    )
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
