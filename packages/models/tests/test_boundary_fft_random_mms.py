"""Check the independent volume reference before using it in a PDE benchmark."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "pde"))
from boundary_fft_core import Circle  # noqa: E402
from boundary_fft_random_mms import EllipseVolumeReference, ForcingModes  # noqa: E402
from bspf_models.elliptic.random_wave_mms import RandomWaveMMS  # noqa: E402


class UnitSource:
    def particular(self, points):
        return -np.sum(points**2, axis=1)[:, None] / 4, -points[:, None, :] / 2


def test_disk_constant_source_free_space_volume_potential():
    reference = EllipseVolumeReference(Circle(), UnitSource(), 128, 1.0)
    theta = np.linspace(0.13, 6.2, 37)
    boundary = np.column_stack((np.cos(theta), np.sin(theta)))
    np.testing.assert_allclose(reference.boundary(boundary), 0, atol=2e-15)
    points = np.array([[0.0, 0.0], [0.25, 0.13], [-0.3, 0.61]])
    np.testing.assert_allclose(
        reference.interior(points)[:, 0],
        (1 - np.sum(points**2, axis=1)) / 4,
        atol=2e-15,
    )


def test_free_space_potential_is_invariant_to_particular_harmonic_gauge():
    source = ForcingModes([RandomWaveMMS.create(kmax=8 * np.pi)])

    class ChangedGauge:
        def particular(self, points):
            value, gradient = source.particular(points)
            x, y = points.T
            value[:, 0] += x**2 - y**2 + 0.7 * x + 1.2
            gradient[:, 0, 0] += 2 * x + 0.7
            gradient[:, 0, 1] -= 2 * y
            return value, gradient

    ellipse = Circle((0.9, 0.76))
    references = [
        EllipseVolumeReference(ellipse, s, 512, 0.83) for s in (source, ChangedGauge())
    ]
    boundary = ellipse.curve(np.linspace(0.071, 3.93, 137))
    points = np.array([[0.1, 0.2], [-0.3, 0.4], [0.7, 0.0]])
    np.testing.assert_allclose(
        references[0].boundary(boundary), references[1].boundary(boundary), atol=5e-14
    )
    np.testing.assert_allclose(
        references[0].interior(points), references[1].interior(points), atol=5e-14
    )
