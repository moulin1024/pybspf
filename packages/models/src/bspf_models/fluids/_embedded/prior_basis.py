"""Geometry-only scalar multiscale enrichment; no Stokes or reference modes.

The radial dictionary uses Euclidean distance in any dimension. The source
placement below specializes geometry, not the PDE, to the square/ellipse test.
"""

import numpy as np
import scipy.linalg as la


class DistanceDictionary:
    def __init__(self, centers, scales, modulation=0, origin=None, length=None):
        self.centers = np.asarray(centers)
        self.scales = np.asarray(scales)
        if self.centers.ndim != 2 or self.scales.shape != (len(self.centers),):
            raise ValueError("Expected one positive scale per source point")
        if np.any(self.scales <= 0) or modulation not in (0, 1, 2):
            raise ValueError("Invalid dictionary scale or modulation degree")
        self.modulation = modulation
        self.origin = (
            np.zeros(self.centers.shape[1]) if origin is None else np.asarray(origin)
        )
        self.length = (
            np.ones(self.centers.shape[1]) if length is None else np.asarray(length)
        )

    def evaluate(self, points, order=1):
        if order not in (0, 1):
            raise ValueError(
                "Distance dictionary supports values and first derivatives"
            )
        delta = np.asarray(points)[:, None, :] - self.centers[None, :, :]
        z = np.sum(delta**2, axis=2) / self.scales**2
        # Centers are outside the fluid. No singularity lies in an integration cell.
        values = np.concatenate((np.log(z), z**-0.5, z**-1), axis=1)
        if order == 0 and not self.modulation:
            return (values,)
        deriv = []
        for axis in range(delta.shape[2]):
            dz = 2 * delta[:, :, axis] / self.scales**2
            deriv.append(
                np.concatenate((dz / z, -0.5 * dz * z**-1.5, -dz * z**-2), axis=1)
            )
        if not self.modulation:
            return (values, *deriv)
        zlocal = (np.asarray(points) - self.origin) / self.length
        polynomials = [np.ones(len(points))]
        gradients = [np.zeros_like(points)]
        for i in range(zlocal.shape[1]):
            polynomials.append(zlocal[:, i])
            gradient = np.zeros_like(points)
            gradient[:, i] = 1 / self.length[i]
            gradients.append(gradient)
        if self.modulation == 2:
            for i in range(zlocal.shape[1]):
                for j in range(i, zlocal.shape[1]):
                    polynomials.append(zlocal[:, i] * zlocal[:, j])
                    gradient = np.zeros_like(points)
                    gradient[:, i] += zlocal[:, j] / self.length[i]
                    gradient[:, j] += zlocal[:, i] / self.length[j]
                    gradients.append(gradient)
        value = np.concatenate([values * v[:, None] for v in polynomials], axis=1)
        if order == 0:
            return (value,)
        return (
            value,
            *[
                np.concatenate(
                    [
                        dv * v[:, None] + values * g[:, axis, None]
                        for v, g in zip(polynomials, gradients)
                    ],
                    axis=1,
                )
                for axis, dv in enumerate(deriv)
            ],
        )


def obstacle_near_box(origin, scale, center, axes, halo=0.0):
    """Geometry-only collar: intersection of ellipse and an expanded cell box."""
    lo, scale, center, axes = map(np.asarray, (origin, scale, center, axes))
    pad = halo * float(min(scale))
    nearest = np.clip(center, lo - pad, lo + scale + pad)
    return np.sum(((nearest - center) / axes) ** 2) < 1


def geometry_dictionary(
    origin,
    scale,
    grid,
    levels=3,
    corners=True,
    obstacle=True,
    modulation=0,
    surface_samples=24,
    obstacle_halo=0.0,
):
    """Fixed geometric scales/centers, independent of solution or reference."""
    lo, hi = np.asarray(origin), np.asarray(origin) + scale
    h = float(min(scale))
    centers, scales = [], []
    if corners:
        for sx in (-1, 1):
            for sy in (-1, 1):
                corner = np.array(
                    [
                        grid.edges[0][0 if sx < 0 else -1],
                        grid.edges[1][0 if sy < 0 else -1],
                    ]
                )
                if np.linalg.norm(corner - np.clip(corner, lo, hi)) > 1e-12:
                    continue
                for level in range(levels):
                    ell = h * 0.5 ** (level + 1)
                    for direction in ([sx, sy], [sx, 0.25 * sy], [0.25 * sx, sy]):
                        direction = np.asarray(direction)
                        centers.append(corner + ell * direction / la.norm(direction))
                        scales.append(h)
    if obstacle:
        if obstacle_near_box(lo, scale, grid.center, grid.axes, obstacle_halo):
            theta = 2 * np.pi * np.arange(surface_samples) / surface_samples
            surface, _, _ = grid.ellipse(theta)
            distance = la.norm(surface - np.clip(surface, lo, hi), axis=1)
            # Cover every sampled arc contained in a coarse cut cell, rather
            # than selecting an arbitrary subset when several distances vanish.
            indices = np.union1d(
                np.flatnonzero(distance < 1e-12), np.argsort(distance)[:4]
            )
            # Retain the nearest generators from coarser dyadic angle sets.
            # Local candidate replacement must not silently remove old sources.
            stride = 2
            while surface_samples % stride == 0 and surface_samples // stride >= 16:
                coarse = np.arange(0, surface_samples, stride)
                indices = np.union1d(indices, coarse[np.argsort(distance[coarse])[:4]])
                stride *= 2
            for index in indices:
                t = theta[index]
                normal = (surface[index] - grid.center) / grid.axes**2
                normal /= la.norm(normal)
                a, b = grid.axes
                curvature_radius = (
                    (a * np.sin(t)) ** 2 + (b * np.cos(t)) ** 2
                ) ** 1.5 / (a * b)
                # The inward normal can exit the opposite side of a thin
                # ellipse before reaching the curvature-based depth. Compute
                # its exact second intersection with the ellipse.
                normal_chord = 2 * np.sum((surface[index] - grid.center) * normal / grid.axes**2) / np.sum((normal / grid.axes)**2)
                for level in range(min(levels, 2)):
                    depth = min(min(h, curvature_radius, min(grid.axes)) * 0.8,
                                0.4 * normal_chord) * 0.5**level
                    source = surface[index] - depth * normal
                    if np.sum(((source - grid.center) / grid.axes) ** 2) >= 1:
                        raise ValueError("Prior source must be inside the solid")
                    centers.append(source)
                    scales.append(h)
    return (
        DistanceDictionary(centers, scales, modulation, (lo + hi) / 2, scale)
        if centers
        else None
    )


class PriorBasis:
    """Preserve the entire base span; filter only independent added functions."""

    def __init__(self, base, dictionary, points, weights, tolerance=1e-5):
        self.base, self.dictionary = base, dictionary
        self.origin, self.scale = base.origin, base.scale
        self.transform = None
        self.discarded = getattr(base, "discarded", 0)
        rootw = np.sqrt(weights[:, None])
        v = base.evaluate(points, 0)[0]
        raw = dictionary.evaluate(points, 0)[0]
        self.normalization = np.maximum(la.norm(rootw * raw, axis=0), 1e-300)
        raw /= self.normalization
        # Base has already been whitened in physical volume.
        self.projection = (rootw * v).T @ (rootw * raw)
        residual = rootw * (raw - v @ self.projection)
        # Only right singular vectors are used. Reduce the tall matrix first,
        # avoiding construction of a large, unused left-singular-vector array.
        triangular = la.qr(residual, mode="r")[0][: min(residual.shape)]
        _, singular, vh = la.svd(triangular, full_matrices=False)
        keep = singular > tolerance
        self.mapping = vh[keep].T / singular[keep]
        self.added = int(keep.sum())
        self.size = base.size + self.added

    def evaluate(self, points, order=1, cell=None):
        base = self.base.evaluate(points, order)
        raw = self.dictionary.evaluate(points, order)
        result = tuple(
            np.column_stack(
                (v, (r / self.normalization - v @ self.projection) @ self.mapping)
            )
            for v, r in zip(base, raw)
        )
        return (
            result
            if self.transform is None
            else tuple(v @ self.transform for v in result)
        )
