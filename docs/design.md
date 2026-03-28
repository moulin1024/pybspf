# Design Notes

## Design Goal

The package is organized around the mathematical objects the user works with,
rather than around one monolithic implementation file.

The package architecture is currently:

```text
src/pybspf/
  backend.py
  grid.py
  knots.py
  basis.py
  boundary.py
  correction.py
  kkt.py
  ops/
  operators/
```

## Numerical Model

The operator follows a split formulation:

1. fit a constrained B-spline approximation to the sampled data
2. compute a residual `r = f - f_spline`
3. correct derivatives or antiderivatives using FFT-based operations on the residual

This is why the package separates:

- spline basis construction
- endpoint constraint operators
- residual correction
- constrained KKT solves
- high-level operation families

## Package Layers

### Foundational layer

- [`backend.py`](/Users/moulin/Workspace/pybspf/src/pybspf/backend.py)
- [`grid.py`](/Users/moulin/Workspace/pybspf/src/pybspf/grid.py)
- [`knots.py`](/Users/moulin/Workspace/pybspf/src/pybspf/knots.py)

This layer defines backend/device rules, uniform-grid metadata, and knot generation.

### Spline operator layer

- [`basis.py`](/Users/moulin/Workspace/pybspf/src/pybspf/basis.py)
- [`boundary.py`](/Users/moulin/Workspace/pybspf/src/pybspf/boundary.py)
- [`correction.py`](/Users/moulin/Workspace/pybspf/src/pybspf/correction.py)
- [`kkt.py`](/Users/moulin/Workspace/pybspf/src/pybspf/kkt.py)

This layer contains the reusable numerical kernels used by the main operator.

### Operation layer

- [`ops/differentiation.py`](/Users/moulin/Workspace/pybspf/src/pybspf/ops/differentiation.py)
- [`ops/integration.py`](/Users/moulin/Workspace/pybspf/src/pybspf/ops/integration.py)
- [`ops/interpolation.py`](/Users/moulin/Workspace/pybspf/src/pybspf/ops/interpolation.py)

This layer groups public operations by behavior rather than by class size.

### Public API layer

- [`operators/bspf1d.py`](/Users/moulin/Workspace/pybspf/src/pybspf/operators/bspf1d.py)
- [`operators/piecewise.py`](/Users/moulin/Workspace/pybspf/src/pybspf/operators/piecewise.py)
- [`__init__.py`](/Users/moulin/Workspace/pybspf/src/pybspf/__init__.py)

This layer exposes the package-facing API.

## Compatibility Design

The repository still keeps [`bspf1d.py`](/Users/moulin/Workspace/pybspf/bspf1d.py) as a legacy compatibility implementation and regression reference.

That allows the package to:

- migrate functionality incrementally
- verify numerical behavior during refactors
- avoid breaking downstream scripts immediately

The compatibility policy itself is documented in [compatibility_strategy.md](/Users/moulin/Workspace/pybspf/docs/compatibility_strategy.md).

## Current Tradeoff

The package owns the public module structure and most foundational implementation now, but some operation-family functions still delegate internally to legacy numerical bodies. This keeps risk low while the public API and package boundaries stabilize under test coverage.

The next meaningful cleanup after Phase 8 would be replacing those remaining internal delegations with fully package-native numerical implementations.
