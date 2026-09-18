"""Independent value/gradient/Laplacian tests of spline-normal continuation."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from bspf_jax.embedded_poisson import benchmark_domains
from bspf_jax.normal_continuation import NormalContinuation, chart
from bspf_jax.random_wave_mms import RandomWaveMMS


def main():
    out = Path("build/normal_continuation")
    out.mkdir(parents=True, exist_ok=True)
    rows, geometry, seams = [], [], []
    for domain in benchmark_domains():
        # Offset independent tangential points, none at spline knots.
        t = (
            np.arange(domain.period)[:, None] + (np.arange(43)[None, :] + 0.371) / 43
        ).ravel()
        dense = domain.curve(np.linspace(0, domain.period, 40001, endpoint=False))
        tree = cKDTree(dense)
        for width in (0.02, 0.01, 0.005):
            tt, rr = np.meshgrid(
                t, width * np.array([-1.0, -0.51, 0.0, 0.25, 0.5, 1.0]), indexing="ij"
            )
            p, jac, _ = chart(domain, tt, rr)
            base = np.linalg.norm(domain.curve(tt, 1), axis=-1)
            ratio = -np.linalg.det(jac) / base
            nearest = tree.query(p.reshape(-1, 2))[0].reshape(tt.shape)
            # A closer nonlocal boundary is evidence of collar overlap.
            overlap = int(np.sum(nearest < abs(rr) - 1e-5))
            geometry.append(
                dict(
                    domain=domain.name,
                    width=width,
                    minimum_jacobian_ratio=float(ratio.min()),
                    sampled_closer_boundary_count=overlap,
                )
            )
            if ratio.min() <= 0 or overlap:
                raise ValueError("Unsafe normal collar")
            for band in (4, 12):
                mms = RandomWaveMMS.create(kmax=band * np.pi)
                for degree in (6, 8, 10, 14):
                    for tangent_degree in (24, 36):
                        ext = NormalContinuation(
                            domain,
                            lambda p: mms.evaluate(p)[0],
                            width=width,
                            normal_degree=degree,
                            tangent_degree=tangent_degree,
                        )
                        for location, radii in (
                            ("interior", [-0.873, -0.413, -0.071]),
                            ("boundary", [0.0]),
                            ("exterior_half", [0.113, 0.287, 0.479]),
                            ("exterior_full", [0.613, 0.797, 0.979]),
                        ):
                            tq, rq = np.meshgrid(
                                t, width * np.array(radii), indexing="ij"
                            )
                            points = chart(domain, tq, rq)[0]
                            u, g, lap = ext.evaluate(tq, rq)
                            eu, eg, ef = mms.evaluate(points.reshape(-1, 2))
                            eu, eg, el = (
                                eu.reshape(u.shape),
                                eg.reshape(g.shape),
                                -ef.reshape(lap.shape),
                            )
                            errors = [
                                float(np.linalg.norm(a - b) / np.linalg.norm(b))
                                for a, b in ((u, eu), (g, eg), (lap, el))
                            ]
                            rows.append(
                                dict(
                                    domain=domain.name,
                                    band=band,
                                    width=width,
                                    normal_degree=degree,
                                    tangent_degree=tangent_degree,
                                    location=location,
                                    value=errors[0],
                                    gradient=errors[1],
                                    laplacian=errors[2],
                                )
                            )
                print(domain.name, width, band, "done", flush=True)
        mms = RandomWaveMMS.create(kmax=12 * np.pi)
        ext = NormalContinuation(
            domain,
            lambda p: mms.evaluate(p)[0],
            width=0.01,
            normal_degree=8,
            tangent_degree=36,
        )
        knots = np.arange(domain.period, dtype=float)
        left = ext.evaluate(knots - 1e-10, np.full_like(knots, 0.00479))
        right = ext.evaluate(knots + 1e-10, np.full_like(knots, 0.00479))
        seams.append(
            dict(
                domain=domain.name,
                seam_nearby_difference=[
                    float(np.max(abs(a - b))) for a, b in zip(left, right)
                ],
            )
        )
    (out / "seams.json").write_text(json.dumps(seams, indent=2) + "\n")
    (out / "results.json").write_text(
        json.dumps(dict(geometry=geometry, results=rows), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
