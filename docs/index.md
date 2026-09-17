# pybspf Docs

This directory contains the package-level documentation for the in-repo `pybspf`
package.

## Read First

- [README.md](/Users/moulin/Workspace/pybspf/README.md): install, quick start, and current project status
- [api.md](/Users/moulin/Workspace/pybspf/docs/api.md): current public API summary
- [design.md](/Users/moulin/Workspace/pybspf/docs/design.md): package architecture and numerical design
- [compatibility_strategy.md](/Users/moulin/Workspace/pybspf/docs/compatibility_strategy.md): transition policy between the package API and the legacy module
- [refactor_backlog.md](/Users/moulin/Workspace/pybspf/docs/refactor_backlog.md): phased migration record

## Current Shape

The package code lives in [`src/pybspf`](/Users/moulin/Workspace/pybspf/src/pybspf) and is organized around:

- backend selection and device validation
- uniform grids and knot generation
- spline basis and endpoint operators
- residual correction and KKT solve helpers
- operation-family modules in `ops/`
- user-facing operator classes in `operators/`

The legacy monolithic implementation [`bspf1d.py`](/Users/moulin/Workspace/pybspf/bspf1d.py) is still present for compatibility and regression comparison.

## JAX implementation and examples

- [JAX package](../jax/README.md)
- [Notebook suite](../examples/README.md)
- [Boundary-driven Alfvén waves](jax_alfven_example.md)
- [Complete correction of the JAX Euler–Bernoulli example](jax_beam_correction.md)
- [Nonperiodic KdV test case](jax_kdv_example.md)
- [Self-consistent Landau damping study on an open finite interval](jax_landau_open_example.md)
- [Focusing NLSE bright-soliton example](jax_nlse_example.md)
- [Noise-aware JAX differentiation in 1D–3D](jax_noise_differentiation.md)
- [Open parallel kinetic dynamics (1z1v)](jax_parallel_kinetic_example.md)
- [Historical diagnosis of the initial beam and Schrödinger errors](jax_pde_accuracy_diagnosis.md)
- [JAX PDE notebook migration](jax_pde_examples.md)
- [Sine–Gordon kink–antikink collision](jax_sine_gordon_example.md)
