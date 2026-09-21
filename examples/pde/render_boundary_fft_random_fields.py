"""Densely sample high-frequency MMS fields; do not interpolate coarse errors."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.signal import resample

from boundary_fft_core import BoundaryFFTPlan, Circle
from boundary_fft_random_mms import EllipseVolumeReference, ForcingModes
from bspf_models.elliptic.random_wave_mms import RandomWaveMMS


def main():
    out = Path("build/boundary_fft_random_mms")
    saved = np.load(out / "waves_128pi.npz")
    wave = RandomWaveMMS(**{key: saved[key] for key in saved.files})
    domain = Circle((0.9, 0.76))
    plans = [BoundaryFFTPlan(domain, count) for count in (512, 1024)]
    reference = EllipseVolumeReference(
        domain, ForcingModes([wave]), 4096, plans[0].radius
    )
    solved = [
        plan.solve(
            wave.evaluate(plan.points)[0] - reference.boundary(plan.points)[:, 0],
            tolerance=2e-14,
        )
        for plan in plans
    ]
    sources, _ = plans[0].arc.sample(4096)
    density = np.column_stack([resample(result[0], 4096) for result in solved])
    constants = np.array([result[1] for result in solved])

    # Check the outermost displayed radius before evaluating the dense image.
    angle = 2 * np.pi * (np.arange(257) + 0.37) / 257
    probes = 0.98 * np.column_stack((0.9 * np.cos(angle), 0.76 * np.sin(angle)))
    refined = EllipseVolumeReference(
        domain, ForcingModes([wave]), 8192, plans[0].radius
    )
    reference_change = float(
        np.max(abs(reference.interior(probes) - refined.interior(probes)))
    )

    def layer(points, source_points, source_density):
        distance = np.linalg.norm(
            points[:, None, :] - source_points[None, :, :], axis=2
        )
        return constants - plans[0].length / (2 * np.pi * len(source_points)) * (
            np.log(distance / plans[0].radius) @ source_density
        )

    refined_sources, _ = plans[0].arc.sample(8192)
    layer_change = float(
        np.max(
            abs(
                layer(probes, sources, density)
                - layer(probes, refined_sources, resample(density, 8192, axis=0))
            )
        )
    )
    print(
        json.dumps(
            dict(
                volume_probe_max_change=reference_change,
                layer_probe_max_change=layer_change,
            )
        ),
        flush=True,
    )
    assert max(reference_change, layer_change) < 2e-12

    size = 513
    x = np.linspace(-0.9, 0.9, size)
    y = np.linspace(-0.76, 0.76, size)
    xx, yy = np.meshgrid(x, y)
    mask = (xx / 0.9) ** 2 + (yy / 0.76) ** 2 <= 0.98**2
    points = np.column_stack((xx[mask], yy[mask]))
    exact = np.empty(len(points))
    numerical = np.empty((len(points), 2))
    for start in range(0, len(points), 256):
        stop = min(start + 256, len(points))
        target = points[start:stop]
        exact[start:stop] = wave.evaluate(target)[0]
        numerical[start:stop] = reference.interior(target) + layer(
            target, sources, density
        )
        if start % (256 * 100) == 0:
            print(f"Evaluated {stop}/{len(points)} display points", flush=True)
    error = numerical - exact[:, None]
    stats = dict(
        kmax_pi=128,
        display_grid=size,
        interior_points=len(points),
        max_elliptic_radius=0.98,
        volume_probe_max_change=reference_change,
        layer_probe_max_change=layer_change,
        rows=[
            dict(
                boundary_points=plan.count,
                relative_l2=float(np.linalg.norm(error[:, j]) / np.linalg.norm(exact)),
                absolute_max=float(np.max(abs(error[:, j]))),
            )
            for j, plan in enumerate(plans)
        ],
    )
    print(json.dumps(stats, indent=2), flush=True)
    (out / "dense_field_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    np.savez_compressed(
        out / "dense_fields_128pi.npz",
        x=x,
        y=y,
        mask=mask,
        exact=exact,
        numerical=numerical,
        boundary_counts=[512, 1024],
    )

    def field(values):
        grid = np.full(xx.shape, np.nan)
        grid[mask] = values
        return grid

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), constrained_layout=True)
    maximum = np.max(abs(exact))
    extent = (-0.9, 0.9, -0.76, 0.76)
    for ax, values, title in zip(
        axes[0],
        (exact, numerical[:, 1]),
        ("Exact random-wave MMS", "Numerical solution: M=1024"),
    ):
        artist = ax.imshow(
            field(values),
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-maximum,
            vmax=maximum,
            interpolation="none",
        )
        ax.set_title(title)
        fig.colorbar(artist, ax=ax, shrink=0.83, label="u")
    upper = 10.0 ** np.ceil(np.log10(np.max(abs(error))))
    for j, ax in enumerate(axes[1]):
        artist = ax.imshow(
            field(np.maximum(abs(error[:, j]), 1e-15)),
            origin="lower",
            extent=extent,
            cmap="magma",
            norm=LogNorm(1e-15, upper),
            interpolation="none",
        )
        ax.set_title(
            f"Absolute error: M={plans[j].count}\n"
            f"relative L2={stats['rows'][j]['relative_l2']:.2e}"
        )
        fig.colorbar(artist, ax=ax, shrink=0.83, label="|u_num - u_exact|")
    theta = np.linspace(0, 2 * np.pi, 1000)
    for ax in axes.flat:
        ax.plot(0.9 * np.cos(theta), 0.76 * np.sin(theta), color="0.4", lw=0.6)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
    fig.suptitle(
        "Analytic ellipse: high-wavenumber MMS, k_max = 128 pi\n"
        "513 x 513 display grid; rho <= 0.98 (thin outer band excluded)",
        fontsize=13,
    )
    fig.savefig(out / "high_wavenumber_fields.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
