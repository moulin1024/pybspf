"""Independent 1D bandwidth screen for the existing Chebyshev endpoint jets."""

import json
from pathlib import Path

import jax
import numpy as np
import scipy.linalg as la

from bspf_jax.embedded_poisson import background_line
from bspf_jax.stream_navier_stokes import stream_evaluate_line

jax.config.update("jax_enable_x64", True)


def main():
    out = Path("build/embedded_poisson_endpoint")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for modes, points in [
        (12, 16),
        (14, 18),
        (16, 20),
        (20, 24),
        (12, 12),
        (14, 14),
        (16, 16),
        (18, 18),
    ]:
        line = background_line(49, chebyshev_modes=modes, endpoint_points=points)
        check = np.linspace(-1, 1, 801)
        b, _, h = stream_evaluate_line(line, check)
        k = np.linspace(np.pi, 12 * np.pi, 49)
        x = np.asarray(line.x)
        nodal = np.column_stack((np.cos(x[:, None] * k), np.sin(x[:, None] * k)))
        coefficient = la.solve(np.asarray(line.bn), nodal)
        exact = np.column_stack(
            (np.cos(check[:, None] * k), np.sin(check[:, None] * k))
        )
        second = -np.tile(k * k, 2) * exact
        error = b @ coefficient - exact
        second_error = h @ coefficient - second
        row = dict(
            modes=modes,
            points=points,
            value_rms=float(la.norm(error) / la.norm(exact)),
            second_rms=float(la.norm(second_error) / la.norm(second)),
            max_error=float(np.max(abs(error))),
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
        (out / "screen.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
