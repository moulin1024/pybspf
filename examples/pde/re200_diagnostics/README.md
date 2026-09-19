# Re=200 diagnostic checkpoint

These research scripts reproduce the spatial-projection and ripple investigation
of the rational/BSPF obstacle-flow solver. Run from the repository root with the
GPU environment documented in `docs/jax_re200_ripple_investigation.md`.

Scripts keep generated output under `build/immersed_flow/re200_ripple_study`.
The compact recorded results are versioned under `docs/data/re200_ripple`.
Full field archives remain generated build artifacts. Some scripts require the
reference runs identified by their explicit `build/immersed_flow/re200_gpu*`
input paths; regenerate those with `examples/pde/immersed_channel_flow.py` and
the case settings in the investigation report before running the comparison.

Main diagnostics:

- `source_audit.py`: reconstruct the initial and evolved vorticity rate, compare
  with the local PDE, and check the GPU mass solve and field reconstruction.
- `initial_projection_audit.py --nx 145 --ny 33`: directional spatial control;
  optional `--degree`, `--laurent`, and `--samples` vary rational accuracy.
- `startup_control.py`: smooth convection startup on t=0…2, then the unchanged
  Navier–Stokes equations through t=20.
- `weak_form_audit.py`: advective/rotational equivalence and energy balance.
- `plot_source_audit.py`: produce the source diagnosis figure after its input
  audits have completed.

Boundary-fit audit scripts intentionally stop plan construction before volume
assembly and compare alternative fits; they are isolated research diagnostics,
not production preprocessing. The checkpoint does not claim a ripple remedy.
