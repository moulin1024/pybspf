from __future__ import annotations

import numpy as np

from examples.navier_stokes.mixed_rt0 import (
    RT0State,
    build_projection_cache,
    divergence,
    exact_state,
    make_rt0_grid,
    project_velocity,
    run_rt0_solver,
)
from examples.navier_stokes.options import SolverOptions


def test_rt0_projection_removes_cellwise_divergence():
    opts = SolverOptions.from_dict({"Nx": 7, "Ny": 6, "nSteps": 0, "makeFigures": False})
    grid = make_rt0_grid(opts)
    cache = build_projection_cache(grid)
    rng = np.random.default_rng(123)
    state = RT0State(
        u=rng.standard_normal((grid.Ny, grid.Nx + 1)),
        v=rng.standard_normal((grid.Ny + 1, grid.Nx)),
    )
    state.u[:, 0] = 0.0
    state.u[:, -1] = 0.0
    state.v[0, :] = 0.0
    state.v[-1, :] = 0.0

    projected, div_linf = project_velocity(state, grid, cache)

    assert div_linf < 1.0e-11
    assert np.max(np.abs(divergence(projected, grid))) < 1.0e-11
    assert np.max(np.abs(projected.u[:, [0, -1]])) == 0.0
    assert np.max(np.abs(projected.v[[0, -1], :])) == 0.0


def test_rt0_exact_mms_face_samples_are_discretely_divergence_free():
    opts = SolverOptions.from_dict({"Nx": 10, "Ny": 9, "nSteps": 0, "makeFigures": False})
    grid = make_rt0_grid(opts)
    state = exact_state(grid, opts, 0.0)

    assert np.max(np.abs(divergence(state, grid))) < 1.0e-12


def test_rt0_solver_smoke_runs_one_step():
    opts = SolverOptions.from_dict({"Nx": 8, "Ny": 8, "nSteps": 1, "dt": 0.01, "makeFigures": False})
    result = run_rt0_solver(opts, verbose=False)

    assert np.isfinite(result.history["faceVelRelL2"][-1])
    assert result.history["divLinf"][-1] < 1.0e-11
