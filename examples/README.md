# Maintained examples

Install the core and model packages as described in the root README; install
`bspf-sim[air-sea]` for the air-sea CLI. Operation notebooks use `pybspf`;
PDE notebooks use explicit `bspf_models` imports. Enable JAX x64 before
constructing plans. No source-directory `PYTHONPATH` is required.

Previous NumPy pressure demonstrations are under
`legacy/numpy_cupy/examples` and require a separate legacy environment.
Research outputs and reference data retain their original locations.

For two exact B-spline boundaries with Dirichlet data, run
`python examples/pde/spline_annulus_convergence.py --adaptive --resonance-scan`.
The [spline-annulus reference solver](../docs/spline_annulus.md) covers Poisson,
modified Helmholtz, and oscillatory Helmholtz with domain-only source sampling.
This dense NumPy/SciPy accuracy prototype does not require a fitted volume mesh.

For a deterministic high-wavenumber random-phase stress test, run
`python examples/pde/spline_annulus_turbulent_mms.py`.
The [MMS report](pde/SPLINE_ANNULUS_TURBULENT_MMS.md) records both successful
solves and rejected source resolutions.

For fixed-boundary Grad–Shafranov on a spline annulus, run
`python examples/pde/spline_annulus_grad_shafranov.py`.
The [GS notes](pde/SPLINE_ANNULUS_GS.md) describe the operator transformation,
profile iteration and independent acceptance checks.

- [Local quintic/septic B-spline source extension](pde/BSPLINE_SOURCE_EXTENSION.md):
  stabilized domain-only patch fits, FFT particular solutions and resolution audits.

- [Full-section, magnetic-axis and free-boundary GS prototype](pde/SPLINE_FREE_BOUNDARY_GS.md):
  single-axis limited equilibria with prescribed external coils and free-space boundary updates.
