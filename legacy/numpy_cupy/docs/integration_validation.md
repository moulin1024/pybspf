# JAX spatial integration validation (1D–3D)

Tested with float64/complex128 JAX and the corrected project-local OpenBLAS,
using four OpenMP threads. No numerical implementation changes were needed.

The manufactured fields have two components:

- `exp(sum(a_i*x_i))`, with `a = (0.4, -0.3, 0.2)`;
- `exp(sum(c_i*x_i))`, with `c = (0.4+2.3j, -0.3+1.7j, 0.2-2.1j)`.

The domains are `[-0.4,1.3]`, `[0.2,1.7]`, and `[-0.7,0.9]`, truncated to
one/two axes for 1D/2D. These smooth fields are nonperiodic and exercise both
growth/decay and oscillation. References use the exact exponential integral,
not numerical quadrature or the BSPF interpolant. Each box integral is a product
of `exp(c_i*a_i)*expm1(c_i*(b_i-a_i))/c_i`.

Plans use degree 9, 18 basis functions, lambda 1e-6, and the optional Chebyshev
endpoint estimator with 12 modes on 16 boundary points. The regularized estimator
uses its default alpha 1e-12.

## Regression checks

The [new integration tests](../jax/tests/test_integration.py) passed in all three
dimensions on unequal grids (33, 41, 49 points, truncated by dimension):

- Eager and JIT full-domain integration, including real input.
- JIT sub-box integration with traced bounds and complex component batches.
- Reversed bounds, zero volume, and NaN for out-of-domain bounds.
- Integration along every axis, preserving the remaining axes/components.
- First and second antiderivatives with nonzero integration constants.
- Autodifferentiation of a box bound against the exact face integral.

The four existing integral/primitive tests also passed with the default
finite-difference endpoint estimator, including polynomial exactness and the
fundamental theorem for the represented interpolant.

```sh
OMP_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python docs/diagnostics/run_with_local_blas.py -m pytest \
  -c jax/pyproject.toml jax/tests/test_integration.py -q
OMP_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python docs/diagnostics/run_with_local_blas.py -m pytest \
  -c jax/pyproject.toml jax/tests/test_bspf.py -k 'integral or primitive' -q
```

## Convergence

Each grid has N points per spatial axis. Errors below are maximum absolute
errors across the two components. Sub-boxes move each lower endpoint inward by
0.13 and each upper endpoint inward by 0.17, so bounds generally fall off-grid.

| Dimension | N | Full-domain error | Sub-box error |
|---|---:|---:|---:|
| 1D | 17 | 9.981e-10 | 6.428e-11 |
| 1D | 33 | 6.547e-14 | 2.245e-15 |
| 1D | 65 | 4.653e-16 | 8.882e-16 |
| 2D | 17 | 8.550e-10 | 4.892e-11 |
| 2D | 33 | 5.689e-14 | 1.337e-15 |
| 2D | 65 | 4.441e-16 | 4.996e-16 |
| 3D | 17 | 7.429e-10 | 4.385e-11 |
| 3D | 33 | 4.856e-14 | 3.109e-15 |
| 3D | 65 | 1.332e-15 | 8.882e-16 |

All repeated JIT volume integrals returned identical outputs in the three
repeat checks per grid. These low-wavenumber smooth fields reach float64
roundoff by N=65; this does not imply the same accuracy for unresolved high
wavenumbers or nonsmooth fields.

Raw results and the sweep driver are retained in
`build/openblas-atomic/integration-convergence.json` and
`build/openblas-atomic/integration_convergence.py`.
