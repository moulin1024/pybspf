"""Small fixed-ellipse channel example using only the installed library.

Run: python examples/pde/embedded_navier_stokes.py --steps 10
Use the documented graded p=9 setup for the research case, not this smoke mesh.
"""

import argparse
import time

import jax
import numpy as np

from bspf_models.fluids.embedded_navier_stokes import plan_embedded_navier_stokes2d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reynolds", type=float, default=100)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--backend", choices=("dense", "sparse"), default="dense")
    args = parser.parse_args()
    if not np.isfinite(args.reynolds) or args.reynolds <= 0:
        parser.error("reynolds must be finite and positive")
    jax.config.update("jax_enable_x64", True)

    def boundary(points, tag):
        result = np.zeros_like(points)
        if tag == "outer":
            inlet = np.isclose(points[:, 0], -2)
            result[inlet, 0] = 1.5 * (1 - (points[inlet, 1] / 1.5) ** 2)
        return result

    plan = plan_embedded_navier_stokes2d(
        boundary,
        dt=args.dt,
        viscosity=0.38 / args.reynolds,
        cells=3,
        degree=3,
        order=40,
        convection_order=40,
        edges=[np.linspace(-2, 4, 4), np.linspace(-1.5, 1.5, 4)],
        outflow=True,
        linear_backend=args.backend,
    )
    state = plan.stokes_initial_state()
    start = time.perf_counter()
    state = plan.advance(
        state,
        args.steps,
        callback=lambda s, d: print(
            f"t={float(s.time):.6f} residual={float(d.linear_residual):.3e} energy={float(d.kinetic_energy):.8f} div={float(d.divergence_l2):.3e} jump={float(d.normal_jump_l2):.3e}"
        ),
    )
    print(
        f"{plan.size} unknowns; {args.steps} steps including JIT: {time.perf_counter() - start:.3f}s"
    )
    print(
        "This coarse example checks execution; it does not establish high-Re accuracy."
    )


if __name__ == "__main__":
    main()
