"""Cut-cell thin ellipse: body coordinates, uniform incidence, short NS audit.

This is a migration/smoke benchmark, not a resolved Re=1000 reference solution.
The rectangular far boundary is [-60,60] x [-40,40].
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from bspf_models.fluids.embedded_navier_stokes import (
    plan_embedded_navier_stokes2d, geometry_prior_options,
)
from bspf_models.fluids._embedded.geometry import ObstacleGrid


def boundary(points, tag):
    direction = np.array([np.cos(np.deg2rad(5)), np.sin(np.deg2rad(5))])
    return np.zeros_like(points) if tag == "hole" else np.tile(direction, (len(points), 1))


def wall_errors(plan, state):
    s = plan.spatial
    s.base.coefficients = np.r_[state.coefficients[:2 * plan.nv], np.zeros(s.base.np)]
    grid = ObstacleGrid(s.grid.cells, s.grid.order + 11, s.grid.center, s.grid.axes,
                        edges=s.grid.edges, full_order=24)
    result = {tag: dict(normal_L2=0., tangential_L2=0., tangential_max=0.)
              for tag in ("outer", "hole")}
    for segments in grid.boundary.values():
        for x, w, n, tag in segments:
            error = s.base.evaluate(x)[:, :2] - boundary(x, tag)
            normal = np.sum(error * n, axis=1)
            tangent = np.sum(error * np.column_stack((-n[:, 1], n[:, 0])), axis=1)
            result[tag]["normal_L2"] += float(w @ normal**2)
            result[tag]["tangential_L2"] += float(w @ tangent**2)
            result[tag]["tangential_max"] = max(result[tag]["tangential_max"], float(max(abs(tangent))))
    for values in result.values():
        for key in ("normal_L2", "tangential_L2"):
            values[key] = float(np.sqrt(values[key]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--wall-penalty-factor", type=float, default=1.)
    parser.add_argument("--wall-enforcement", choices=("nitsche", "constraint"), default="constraint")
    parser.add_argument("--projector-backend", choices=("implicit_qr", "array"), default="array")
    args = parser.parse_args()
    output = Path(f"build/obstacle_stokes/ns/thin_airfoil_structural_p{args.degree}")
    if args.wall_penalty_factor != 1:
        output = output.with_name(output.name + f"_wall{args.wall_penalty_factor:g}")
    if args.wall_enforcement == "constraint":
        output = output.with_name(output.name + "_noslip")
    if args.projector_backend == "array":
        output = output.with_name(output.name + "_array")
    output.parent.mkdir(parents=True, exist_ok=True)
    edges = [np.array([-60, -3, -1.2, -.4, .4, 1.2, 3, 60]),
             np.array([-40, -2, -.2, -.05, .05, .2, 2, 40])]
    start = perf_counter()
    plan = plan_embedded_navier_stokes2d(
        boundary, dt=args.dt, viscosity=.002, cells=7, degree=args.degree,
        order=64, convection_order=48, center=(0., 0.), axes=(1., .1),
        edges=edges, outflow=False, constraint_backend="implicit_qr",
        linear_backend="host_sparse", constraint_rank_tolerance=1e-9,
        wall_penalty_factor=args.wall_penalty_factor,
        wall_enforcement=args.wall_enforcement,
        projector_backend=args.projector_backend,
        **geometry_prior_options(args.degree),
    )
    print("SETUP", plan.info, perf_counter() - start, flush=True)
    try:
        state = plan.stokes_initial_state()
        initial = wall_errors(plan, state)
        print("INITIAL", initial, plan.incompressibility_errors(state.coefficients), flush=True)
        history = []
        for _ in range(args.steps):
            tick = perf_counter()
            state, diagnostics = plan.step(state)
            record = dict(time=float(state.time), seconds=perf_counter()-tick,
                          iterations=plan.last_iterations,
                          wall_slip_L2=plan.wall_slip_error(state.coefficients),
                          **{k: bool(v) if k == "valid" else float(v)
                             for k, v in diagnostics._asdict().items()})
            history.append(record)
            print("STEP", record, flush=True)
            if not diagnostics.valid:
                raise RuntimeError(f"Independent structural audit failed: {record}")
        result = dict(info=plan.info, reynolds_chord=1000, chord=2., axes=[1., .1],
                      incidence_degrees=5., dt=args.dt, time=float(state.time),
                      farfield="rectangular [-60,60] x [-40,40], uniform Dirichlet",
                      frame="body axes; freestream (cos(5deg),sin(5deg))",
                      initial_boundary=initial, final_boundary=wall_errors(plan, state),
                      history=history, wall_seconds=perf_counter()-start,
                      scope="Short evolution from compatible Stokes initialization; no spatial/time convergence claim")
        output.with_suffix(".json").write_text(json.dumps(result, indent=2))
        np.savez_compressed(output.with_suffix(".npz"), **state._asdict())
        print("DONE", result["final_boundary"], result["wall_seconds"], flush=True)
    finally:
        plan.close()


if __name__ == "__main__":
    main()
