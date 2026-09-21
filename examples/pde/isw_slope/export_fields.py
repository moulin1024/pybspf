#!/usr/bin/env python3
"""Export saved fields to NetCDF. No time interpolation; not a flow solver."""

from pathlib import Path
import argparse
import os
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
import numpy as np
from scipy.io import netcdf_file
from evaluation import SnapshotReader


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--times", nargs="*", type=float)
    p.add_argument(
        "--nx",
        type=int,
        default=401,
        help="Postprocessing sample count, not integration points.",
    )
    p.add_argument("--nz", type=int, default=193)
    a = p.parse_args()
    if min(a.nx, a.nz) < 3:
        raise SystemExit("Need at least 3 sample points per axis.")
    if a.out.exists():
        raise SystemExit(f"Refusing to overwrite {a.out}")
    q = np.linspace(0, 1, a.nx)
    s = np.linspace(0, 1, a.nz)
    reader = SnapshotReader(a.run, q, s)
    times = reader.times if a.times is None else a.times
    if not times:
        raise SystemExit("No times requested.")
    factors = {
        "u": 1.0,
        "w": 1.0,
        "B": 0.01,
        "bprime": 0.01,
        "N2": 1e-4,
        "N2prime": 1e-4,
        "uz": 0.01,
        "vorticity": 0.01,
        "psi": 100.0,
    }
    values = {k: [] for k in factors}
    for t in times:
        fields = reader.get(t)
        for k, factor in factors.items():
            values[k].append(np.asarray(fields[k] * factor, dtype=np.float64))
        print(f"Export actual saved state t={t:g}s", flush=True)
    units = {
        "u": "m s-1",
        "w": "m s-1",
        "B": "m s-2",
        "bprime": "m s-2",
        "N2": "s-2",
        "N2prime": "s-2",
        "uz": "s-1",
        "vorticity": "s-1",
        "psi": "m2 s-1",
        "z": "m",
        "bottom": "m",
        "time": "s",
        "x": "m",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".tmp")
    with netcdf_file(tmp, "w", version=2) as ds:
        for name, size in (("time", len(times)), ("x", a.nx), ("sigma", a.nz)):
            ds.createDimension(name, size)
        data = {k: (("time", "x", "sigma"), np.stack(v)) for k, v in values.items()}
        data.update(
            time=(("time",), times),
            x=(("x",), reader.geo["x"] * 100),
            sigma=(("sigma",), s),
            z=(("x", "sigma"), reader.geo["z"] * 100),
            bottom=(("x",), -reader.geo["d"][:, 0] * 100),
        )
        for name, (dimensions, value) in data.items():
            var = ds.createVariable(name, "d", dimensions)
            var[:] = value
            if name in units:
                var.units = units[name]
        ds.title = "BSPF internal solitary wave on a slope"
        ds.initial_warning = "Frozen reconstructed initial; not original DJL."
        ds.grid_warning = (
            "Export grid is postprocessing sampling, not finer simulation."
        )
        ds.simulation_nx = int(reader.config["nx"])
        ds.simulation_nz = int(reader.config["nz"])
        ds.precision = "float64 evaluation from saved coefficients"
        ds.source_run = str(a.run.resolve())
    os.replace(tmp, a.out)
    print(a.out)


if __name__ == "__main__":
    main()
