"""Record accuracy sensitivity separately from structural audit residuals."""

import json
from pathlib import Path
import jax
import numpy as np
from bspf_models.fluids.embedded_navier_stokes import plan_embedded_navier_stokes2d


def exact(p):
    x, y = p.T
    return np.column_stack(
        (
            0.02 * x * x,
            -0.04 * x * y,
            0.3 + x + y,
            0.04 * x,
            0 * x,
            -0.04 * y,
            -0.04 * x,
        )
    )


def boundary(p, tag):
    return exact(p)[:, :2]


def traction(p, n):
    f = exact(p)
    return (
        0.38
        * np.column_stack(
            (
                f[:, 3] * n[:, 0] + f[:, 4] * n[:, 1],
                f[:, 5] * n[:, 0] + f[:, 6] * n[:, 1],
            )
        )
        - f[:, 2, None] * n
    )


def main():
    jax.config.update("jax_enable_x64", True)
    points = np.array([[-0.8, 0.4], [0.7, 0.6], [0.6, -0.7]])
    records = []
    for tolerance in (1e-5, 1e-9):
        p = plan_embedded_navier_stokes2d(
            boundary,
            dt=0.001,
            viscosity=0.38,
            cells=3,
            degree=3,
            order=40,
            convection_order=40,
            outflow=True,
            linear_backend="host_sparse",
            constraint_backend="implicit_qr",
            constraint_rank_tolerance=tolerance,
        )
        try:
            state = p.stokes_initial_state(
                np.tile([1 - 0.04 * 0.38, 1.0], (len(p.force_points), 1)),
                traction(p.traction_points, p.traction_normals),
            )
            difference = p.evaluate(points, state) - exact(points)
            records.append(
                dict(
                    rank_tolerance=tolerance,
                    rank=p.info["constraint_rank"],
                    sample_velocity_max_error=float(np.max(abs(difference[:, :2]))),
                    sample_pressure_max_error=float(np.max(abs(difference[:, 2]))),
                    sample_gradient_max_error=float(np.max(abs(difference[:, 3:]))),
                    structural_norms=[
                        float(x)
                        for x in p.incompressibility_errors(state.coefficients)[:3]
                    ],
                )
            )
        finally:
            p.close()
    output = Path("build/obstacle_stokes/ns/structural_rank_sensitivity.json")
    output.write_text(
        json.dumps(
            dict(
                points=points.tolist(),
                results=records,
                interpretation="Three-point manufactured-field diagnostic, not a convergence study. Strict numerical rank can harm polynomial reproduction despite small incompressibility norms.",
            ),
            indent=2,
        )
    )
    print(records)


if __name__ == "__main__":
    main()
