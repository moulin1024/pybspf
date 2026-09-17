# Compatibility and migration

Use `from pybspf import BSPF1D, BSPF2D, PiecewiseBSPF1D` for new code.
`pybspf.bspf1d` and `pybspf.bspf2d` remain aliases. Existing package exports,
including solver and decomposition functions, remain available.

The root `bspf1d.py` is a source-checkout compatibility shim over
`legacy/bspf1d.py`. It is not part of the installed wheel. Legacy implementations
remain frozen as regression references. Package operations do not import them.

The generalized derivative API replaces legacy methods:

```python
result = op.derivatives(f, orders=(1, 2))
d1, d2, spline = result[1], result[2], result.spline
# For several signals: op.derivatives_batched(f_matrix, orders=(1, 2))
```

Intentional corrections in the core refactor:

- Invalid grid coordinates, derivative orders, regularization, and backend
  combinations now fail at the boundary of the operation.
- `correction="none"` now actually disables Fourier differentiation correction.
- Complex fitting retains imaginary data.
- Neumann fluxes require `order >= 2`; `order=1` has only value constraints.
- Piecewise construction raises for short segments instead of omitting samples.
- Repeated interpolation on different grids no longer reuses stale evaluations.
- Partial definite integrals include only the residual within the requested bounds.
- GPU antiderivatives return device arrays; explicit host conversion is the caller's job.
- The previously ignored `use_fft=True` interpolation option raises an explicit error.

Matrix and vector BLAS operations may differ at floating-point roundoff. Regression
tolerances allow small absolute errors near zero while retaining relative checks.
The Poisson solver convention mismatches remain visible test failures; this
refactor does not silently redefine those research methods.
