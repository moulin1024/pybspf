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
