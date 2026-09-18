"""Compare final KH animation fields on their common central region.

python scratch/compare_kh_grids.py COARSE_FRAMES.npz FINE_FRAMES.npz OUTPUT.json
The interpolation is cubic, not part of either NS time integration.
"""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import RectBivariateSpline


def main():
    coarse_path, fine_path, output = map(Path, sys.argv[1:])
    with np.load(coarse_path) as coarse, np.load(fine_path) as fine:
        if not np.isclose(coarse["times"][-1], fine["times"][-1]):
            raise ValueError("Final times must agree")
        x, y = np.meshgrid(coarse["x"], coarse["y"], indexing="ij")
        core = (abs(x) < 2) & (abs(y) < 0.5)
        record = dict(
            coarse_grid=[coarse["x"].size, coarse["y"].size],
            fine_grid=[fine["x"].size, fine["y"].size],
            time=float(coarse["times"][-1]),
            comparison="Final central fields, |x|<2 and |y|<0.5; fine field cubically interpolated to coarse grid.",
        )
        for field in ["vorticity", "transverse_velocity"]:
            interpolated = RectBivariateSpline(fine["x"], fine["y"], fine[field][-1])(
                coarse["x"], coarse["y"]
            )
            difference = coarse[field][-1] - interpolated
            record[field + "_central_relative_l2"] = float(
                np.linalg.norm(difference[core]) / np.linalg.norm(interpolated[core])
            )
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
