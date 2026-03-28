# BSPF Compatibility Strategy

## Status

The preferred user-facing API is now the package API:

```python
from pybspf import BSPF1D, PiecewiseBSPF1D, Grid1D
```

The root-level [`bspf1d.py`](/Users/moulin/Library/CloudStorage/Dropbox/Workspace/pybspf/bspf1d.py) remains in the repository as the **legacy compatibility implementation**.

## What Is Considered Stable

For new code:

- import from `pybspf`
- treat `BSPF1D` as the canonical 1D operator
- treat `PiecewiseBSPF1D` as the canonical piecewise wrapper

For existing code:

- imports from `bspf1d.py` are still expected to work
- the lowercase alias `pybspf.bspf1d` is still available

## Current Compatibility Policy

1. The package API is the main development target.
2. The root-level legacy module is kept to avoid breaking downstream scripts and notebooks.
3. Behavioral changes should be validated against the legacy implementation with regression tests before package-native replacements are accepted.
4. New internal development should happen under `src/pybspf/`, not by adding new features directly to the monolithic legacy file.

## Migration Guidance

Preferred migration path for downstream users:

1. Replace `from bspf1d import bspf1d` with `from pybspf import BSPF1D`.
2. Replace `from bspf1d import PiecewiseBSPF1D` with `from pybspf import PiecewiseBSPF1D`.
3. Keep the old imports only where legacy notebooks or scripts require zero code churn.

## Repository Policy

- [`bspf1d.py`](/Users/moulin/Library/CloudStorage/Dropbox/Workspace/pybspf/bspf1d.py) is treated as a frozen legacy reference unless a bug fix is required for compatibility.
- `src/pybspf/` is where the package architecture, tests, and new refactors should continue.
- Regression tests should keep comparing package behavior to the legacy implementation until the package is considered fully independent.
