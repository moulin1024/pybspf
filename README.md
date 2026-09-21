# pybspf

JAX B-spline plus Fourier calculus in one to three dimensions. Version 0.2
uses the former JAX implementation as the only maintained numerical core.
CPU and GPU execution use JAX; NumPy/SciPy remain available for host setup.
Importing the library does not change JAX precision or device configuration.

## Install

```sh
python -m pip install -e '.[host,test]'
python -m pip install -e './packages/models[precision,test]'
python -m pip install -e './packages/sim[air-sea,test]'
```

Install only the first package for numerical calculus. Models contain PDE
solvers; the simulation package contains CLI, checkpoint and reporting workflows.
Select the JAX device installation appropriate to your environment separately.

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pybspf import plan_1d, differentiate

x = jnp.linspace(0.0, 1.0, 65)
p = plan_1d(x)
df = differentiate(p, jnp.sin(2 * jnp.pi * x))
```

## Package boundaries

- `pybspf`: bases, plans, calculus, Galerkin spaces, generic time integration.
- `bspf_models`: elliptic, fluid, kinetic, plasma, wave and air-sea models.
- `bspf_sim`: simulation configuration, execution, checkpoints, output and CLI.

```python
from bspf_models.fluids.navier_stokes import plan_navier_stokes2d
from bspf_models.kinetic.nonlinear_itg import NonlinearITG
from bspf_sim.air_sea.platform import run
```

`bspf-air-sea` retains its `run`, `resume`, `report` and `validate` commands.
See [migration](docs/migration.md), [architecture](docs/design.md),
[API](docs/api.md), and [examples](examples/README.md).

## Validation

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 python -m pytest
python -m build --no-isolation
python -m build --no-isolation packages/models
python -m build --no-isolation packages/sim
```

Tests enable x64 explicitly. GPU tests require a JAX GPU environment.
The default suite excludes the archived NumPy/CuPy implementation.

## Archive

The previous object-oriented `pybspf` API, solvers, tests and backend adapters
are preserved in [legacy/numpy_cupy](legacy/numpy_cupy/README.md). That package
is unmaintained and must be installed only in a separate environment. Neither
`BSPF1D` nor the old `pybspf` import path is a compatibility alias in 0.2.
