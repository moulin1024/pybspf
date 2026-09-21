# pybspf

B-spline fitting plus Fourier residual correction for sampled functions on
uniform grids. The core operators use NumPy/SciPy on CPU or CuPy/CuPyX on CUDA.

## Install

Python 3.10 or newer is required. From this checkout:

```sh
python -m pip install .
# Development installation:
python -m pip install -e '.[dev]'
# CUDA 12 installation (choose a matching CuPy build for other CUDA versions):
python -m pip install '.[gpu]'
```

## Differentiate a signal

```python
import numpy as np
from pybspf import BSPF1D

x = np.linspace(0.0, 2.0 * np.pi, 257)
f = np.sin(x) + 0.1 * x
op = BSPF1D.from_grid(degree=5, x=x, n_basis=24)

result = op.derivatives(f, orders=(1, 2), lam=1e-8)
df, d2f = result[1], result[2]
spline = result.spline

# Reuse the operator and its cached factorization for more signals.
batch = np.column_stack([f, np.cos(x)])  # (samples, signals)
batched = op.derivatives_batched(batch, orders=(1, 2), lam=1e-8)
area = op.definite_integral(f)
x_fine, f_fine = op.interpolate(f)
```

`differentiate(f, k=1)` returns `(derivative, spline)`. The multi-order methods
return a `DerivativeResult` indexed by derivative order (1–4). Real and complex
signals are supported for fitting and differentiation; integration and
interpolation currently require real signals. Computation uses float64 or
complex128.

## Two dimensions

```python
from pybspf import BSPF2D

y = np.linspace(0.0, 1.0, 129)
xx, yy = np.meshgrid(x, y)
field = np.sin(xx) + yy**3  # (len(y), len(x))
op2 = BSPF2D.from_grids(x=x, y=y, degree_x=5, degree_y=5)
dx, spline_x = op2.partial_x(field)
laplacian = op2.laplacian(field)
```

Axis 0 is y; axis 1 is x. Batched 1D solves underpin the 2D methods.

For a no-slip pressure projection using `div(Q grad p) = div(Q raw)`, see the
[2D tensor pressure solver](docs/pressure_projection2d.md) and
[runnable example](examples/pressure_projection2d.py). It includes the BSPF line
operators, tensor inverse, and seven-mode wall-pressure completion extracted
from the 3D NS benchmark.

## GPU usage

```python
import cupy as cp
from pybspf import BSPF1D

x_gpu = cp.linspace(0.0, 2.0 * cp.pi, 257)
f_gpu = cp.sin(x_gpu)
op_gpu = BSPF1D.from_grid(degree=5, x=x_gpu, use_gpu=True)
result_gpu = op_gpu.derivatives(f_gpu, orders=(1, 2))
df_cpu = cp.asnumpy(result_gpu[1])  # explicit transfer when needed
```

Set `use_gpu=True` explicitly. Factories may upload coordinate and knot arrays
during construction; computational methods require samples on the selected
backend. Array outputs stay on that backend. `definite_integral` returns a Python
float, which synchronizes a GPU scalar. No CUDA environment variables are changed
by the core package; configure your CUDA installation outside Python.

| Feature | NumPy | CuPy |
| --- | --- | --- |
| 1D fitting and derivatives, including complex/batched samples | Yes | Implemented; CUDA parity tests provided |
| 2D derivatives and Laplacian | Yes | Implemented; CUDA parity tests provided |
| Piecewise derivatives | Yes | Implemented |
| Real definite integrals and antiderivatives | Yes | Implemented; CUDA parity tests provided |
| Interpolation and endpoint zero-flux repair | Yes | Not supported |
| Problem-specific Poisson/Schrödinger solvers | CPU implementations | Not a general GPU API |

GPU tests skip when CuPy or a CUDA device is unavailable. Implementation support
does not imply validation on every CUDA/CuPy combination.

## Contracts and limitations

- Coordinates must be finite, strictly increasing, one-dimensional, and uniformly
  spaced. The grid includes both physical endpoints.
- `lam` is finite, nonnegative regularization. Reuse one operator for a fixed grid.
- `correction="none"` disables Fourier correction for differentiation.
- Neumann constraints require `order >= 2`; this counts value and derivative
  constraints at each endpoint.
- `PiecewiseBSPF1D` splits at supplied breakpoints. Every segment must meet
  `min_points_per_seg`; short segments raise an error instead of leaving zeros.
- Interpolation uses a spline plus linearly interpolated residual. `use_fft=True`
  raises `NotImplementedError`.
- Partial definite integrals integrate the residual over the requested interval.
  Bounds must be within the grid; reversed bounds change the sign.
- The specialized Poisson solver suite has four pre-existing failures involving
  PDE conventions and an obsolete entry point. See the [assessment](docs/assessment.md)
  before relying on those workflows. This checkout is not release-ready yet.

## Development

```sh
python -m pytest                    # full suite, including known solver failures
python -m pytest -m gpu             # CUDA parity checks
python -m pytest -m performance     # timing comparisons; use a quiet machine
python -m pip wheel . --no-deps
```

Tests are collected from `tests/`, not executable research examples. If unrelated
globally installed pytest plugins interfere, run with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

Read the [API](docs/api.md), [architecture](docs/design.md),
[compatibility notes](docs/compatibility_strategy.md), and
[assessment and next steps](docs/assessment.md).
