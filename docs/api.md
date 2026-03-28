# API Summary

## Main Imports

Preferred package imports:

```python
from pybspf import BSPF1D, PiecewiseBSPF1D, Grid1D
```

Compatibility alias:

```python
from pybspf import bspf1d
```

Legacy import path:

```python
from bspf1d import bspf1d, PiecewiseBSPF1D
```

## `BSPF1D`

Primary constructor:

```python
op = BSPF1D.from_grid(
    degree=5,
    x=x,
    knots=None,
    n_basis=None,
    domain=None,
    use_clustering=False,
    clustering_factor=2.0,
    order=None,
    num_boundary_points=None,
    correction="spectral",
    use_gpu=False,
)
```

Main methods:

- `differentiate(f, k=1, lam=0.0, neumann_bc=None)`
- `differentiate_1_2(f, lam=0.0, neumann_bc=None)`
- `differentiate_1_2_3(f, lam=0.0, neumann_bc=None)`
- `differentiate_1_2_batched(f, lam=0.0, neumann_bc=None)`
- `fit_spline(f, lam=0.0, neumann_bc=None)`
- `definite_integral(f, a=None, b=None, lam=0.0)`
- `antiderivative(f, order=1, left_value=0.0, match_right=None, lam=0.0)`
- `interpolate(f, lam=0.0, use_fft=False)`
- `interpolate_split_mesh(f, refine_factor, lam=0.0, neumann_bc=None)`
- `enforced_zero_flux(f)`

Core attributes commonly used in advanced code:

- `grid`
- `degree`
- `order`
- `num_bd`
- `basis`
- `end`
- `BW`
- `Q`

## `PiecewiseBSPF1D`

Constructor:

```python
pw = PiecewiseBSPF1D(
    degree=5,
    x=x,
    breakpoints=[...],
    min_points_per_seg=16,
    **bspf_kwargs,
)
```

Current method surface:

- `differentiate_1_2(f, lam=0.0, neumann_bc_global=None)`

Useful attributes:

- `segments`
- `breakpoints`
- `x`

Each entry in `segments` is a dictionary containing:

- `i0`: starting grid index
- `i1`: ending grid index
- `op`: segment-local `BSPF1D` operator

## `Grid1D`

Constructor:

```python
grid = Grid1D(x, atol=1e-13, use_gpu=False)
```

Attributes and properties:

- `x`
- `dx`
- `omega`
- `trap`
- `a`
- `b`
- `n`

## Backend Notes

- `use_gpu=True` requires CuPy to be installed.
- The package enforces explicit backend consistency and rejects implicit NumPy/CuPy mixing.
- In the current repository environment, tests exercise the CPU path and the explicit no-CuPy failure path.
