"""Benchmark cached tensor output on previously solved MMS coefficients.

This reconstructs the same internal basis but does NOT rerun the PDE solver.
Timings compare only cached value evaluation, excluding factor construction,
the PDE solve, derivatives, and independent exact-solution verification.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from bspf_jax.convex_poisson_grid import _TensorGridOutput
from bspf_jax.embedded_poisson import benchmark_domains
from bspf_jax.random_wave_mms import RandomWaveMMS
from bspf_jax.stream_navier_stokes import _stream_line


def median_time(callback, repeat=21):
    callback()
    times = []
    for _ in range(repeat):
        start = perf_counter()
        callback()
        times.append(perf_counter() - start)
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("build/convex_poisson_n65"))
    parser.add_argument("--grids", type=int, nargs="+", default=[65, 129, 257])
    parser.add_argument(
        "--out", type=Path, default=Path("build/convex_poisson_fixed_grid")
    )
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    metadata = json.loads((args.input / "results.json").read_text())
    rows = [r for r in metadata if r["name"] == "h32_h2"]
    n = rows[0]["nodes"]
    if any(r["nodes"] != n or r["cached_dirichlet"] for r in rows):
        raise ValueError("Expected saved unrestricted solutions at one resolution")
    line = _stream_line(
        np.linspace(-1.2, 1.2, n),
        clamped=False,
        dirichlet=False,
        endpoint_points=12,
        chebyshev_modes=12,
    )
    domain = benchmark_domains()[0]
    report = []
    for count in args.grids:
        start = perf_counter()
        grid = _TensorGridOutput(
            line, domain, np.linspace(-1.2, 1.2, count), np.linspace(-1.2, 1.2, count)
        )
        setup = perf_counter() - start
        iy, ix = np.nonzero(grid.inside)
        points = np.column_stack((grid.x[ix], grid.y[iy]))
        # Cache paired factors too, so neither timed path includes basis setup.
        bx, by = grid.bx[ix], grid.by[iy]
        for row in rows:
            band = row["band"]
            saved = np.load(args.input / f"h32_h2_{band}pi.npz")
            c = saved["coefficient"].reshape(n, n)

            def paired():
                return np.sum((bx @ c) * by, axis=1)

            tensor = grid.values(c)[grid.inside]
            previous = paired()
            delta = float(np.max(abs(tensor - previous)))
            np.testing.assert_allclose(tensor, previous, atol=5e-11, rtol=2e-12)
            exact = RandomWaveMMS.create(kmax=band * np.pi).evaluate(points)[0]
            tensor_time = median_time(lambda: grid.values(c))
            paired_time = median_time(paired)
            item = dict(
                nodes=n,
                grid=count,
                band=band,
                inside_points=len(points),
                cached_tensor_seconds=tensor_time,
                cached_paired_seconds=paired_time,
                evaluation_speedup=paired_time / tensor_time,
                output_factor_setup_seconds=setup,
                tensor_basis_bytes=grid.basis_storage_bytes,
                paired_basis_bytes=bx.nbytes + by.nbytes,
                max_difference_from_paired=delta,
                relative_sample_l2=float(
                    np.linalg.norm(tensor - exact) / np.linalg.norm(exact)
                ),
                max_sample_error=float(np.max(abs(tensor - exact))),
            )
            report.append(item)
            print(json.dumps(item), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "output_benchmark.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
