"""Compatibility CLI for the audited 2-D research platform.

New experiments may instead use: bspf-air-sea run --config FILE --out DIRECTORY
Legacy lagged experiments use the explicitly named legacy driver.
"""

import argparse
from dataclasses import replace
from pathlib import Path
import json
import runpy
import sys

from bspf_jax.air_sea_platform import RunConfig, run
from bspf_jax.surface_exchange import SurfaceExchangeConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=81)
    ap.add_argument("--quadrature-order", type=int, default=32)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--dt-air", type=float, default=60.0)
    ap.add_argument("--dt-ocean", type=float, default=300.0)
    ap.add_argument("--window", type=float, default=600.0)
    ap.add_argument("--output-hours", type=float, default=0.5)
    ap.add_argument("--method", choices=("mri-gark4", "lagged"), default="mri-gark4")
    ap.add_argument("--exchange", choices=("constant", "coare35"), default="constant")
    ap.add_argument("--out", type=Path, default=Path("build/air_sea_research"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if args.method == "lagged":
        if args.exchange != "constant" or args.quadrature_order != 32:
            ap.error(
                "Legacy comparison supports constant exchange and its original quadrature only"
            )
        sys.argv = [
            str(Path(__file__).with_name("air_sea_double_gyre_legacy.py")),
            "--n",
            str(args.n),
            "--hours",
            str(args.hours),
            "--dt-air",
            str(args.dt_air),
            "--dt-ocean",
            str(args.dt_ocean),
            "--window",
            str(args.window),
            "--output-hours",
            str(args.output_hours),
            "--method",
            "lagged",
            "--out",
            str(args.out),
        ]
        if args.no_plot:
            sys.argv.append("--no-plot")
        runpy.run_path(
            str(Path(__file__).with_name("air_sea_double_gyre_legacy.py")),
            run_name="__main__",
        )
        return
    base = RunConfig()
    config = replace(
        base,
        surface=SurfaceExchangeConfig(method=args.exchange),
        spatial=replace(base.spatial, n=args.n, quadrature_order=args.quadrature_order),
        time=replace(
            base.time, dt_air=args.dt_air, dt_ocean=args.dt_ocean, window=args.window
        ),
        output=replace(
            base.output,
            duration_seconds=args.hours * 3600,
            fields_seconds=args.output_hours * 3600,
        ),
    )
    print(json.dumps(run(config, args.out), indent=2))


if __name__ == "__main__":
    main()
