# BSPF masked pressure projection in 2D

`PressurePoisson2D` extracts the pressure core of
`BSPF_3D64_T2_20260917/ns3d.py` into an independent NumPy/SciPy solver. It has no
runtime dependency on that folder and contains no NS time integrator or forcing.

The [JAX backend](jax_pressure_projection2d.md) provides the same pressure
algorithm and endpoint options with immutable plans, JIT, vmap, and field
autodiff. Its arrays use `(nx, ny)`, following the JAX package convention.

## Equation and API

Let `Q` be one on strictly interior nodes and zero on all four walls. The solver
uses the same strong BSPF first derivative for gradient and divergence:

```text
S p = b,                     S = div(Q grad)
b = div(Q raw),              projected = Q (raw - grad(p))
```

This is a discrete pressure Schur equation, not a conventional boundary-value
solver for `Delta p = f`. In particular, passing samples of the continuum
Laplacian to `solve` is generally incorrect. Boundary equations are those of
the masked operator; no normal-flux rows are substituted. A compatible scalar
RHS lies in the range of `S`, which is a stronger requirement than zero mean.
Incompatible data is rejected by checking the original, unregularized equation.

```python
import numpy as np
from pybspf import PressurePoisson2D

x = np.linspace(0, 1, 64)
y = np.linspace(0, 2, 72)
solver = PressurePoisson2D(x, y)
xx, yy = np.meshgrid(x, y)
p_exact = np.exp(xx + 0.5*yy)
raw = np.stack([p_exact, 0.5*p_exact], axis=-1)

projected, result = solver.project(raw)
p = result.pressure
print(result.schur_residual_linf, result.wall_gradient_fit_linf)

# Equivalent scalar solve with wall-gradient completion:
b = solver.divergence(solver.mask[..., None] * raw)
result = solver.solve(b, wall_gradient=raw)
```

Scalars have shape `(ny, nx)`, following the package convention. Vectors have
shape `(ny, nx, 2)` with components `(x, y)`. Grids are uniform, increasing and
include both endpoints. Data must be finite and real; this solver is CPU-only.
Only wall entries of `wall_gradient` are used, but the argument has full vector
shape. These entries specify both Cartesian gradient components, not outward
normal derivatives. The least-squares fit need not satisfy arbitrary data
exactly; its residual is reported separately.

When used in NS, `raw` is `-(u dot grad)u + nu*laplacian(u) + force`.
`projected` is then the acceleration for an RK stage, without a `1/dt` factor.
Starting with a divergence-free no-slip velocity, combinations of these stage
accelerations preserve the constraints up to numerical error.

## Tensor inverse and completion

Each line uses the source's QR-constrained spline fit and Taylor least-squares
endpoint jets. The derivative is `D = F + (B' - F B) P`: FFT differentiation
plus a spline correction. Defaults are `q=9`, `n_basis=32`, `degree=13`, and
`baseline_points=14`. Unlike the original builder, invalid parameter/grid
combinations are rejected instead of silently reducing the basis or degree.
For a small audit grid, use `q=2, n_basis=5, degree=4, baseline_points=4` with
at least six nodes per axis (the spectral/nullity checks must also pass).

The line operator is `H = D M D`, with zero endpoint entries in `M`. Eliminating
the two endpoint values gives

```text
A = H_ii - H_ie inv(H_ee) H_ei.
```

After eliminating boundary rows, the 2D interior system is a Kronecker sum of
the x and y line matrices. Separate eigendecompositions yield denominators
`lambda_y + lambda_x`. There are two null eigenmodes per line, so four tensor
denominators are shifted to `-10`. The four corner coordinates are also lifted
with `-10`. The inverse consists of endpoint elimination, two tensor transforms,
modal division, and boundary reconstruction.

The extraction calls this inverse directly and performs two refinement steps.
The source used the same inverse as a GMRES preconditioner; no outer Krylov
iteration is necessary for this exactly separable geometry. Every public solve
checks the final unlifted residual against `atol + rtol*norm(b)` (Euclidean norm).
The default tolerances are `atol=1e-9`, `rtol=1e-10`.

The masked gradient has eight null modes: four tensor modes and four corner
coordinate vectors. Three nonconstant tensor modes plus the four corners are
determined by minimizing the unweighted wall-gradient residual. The seven-column
wall matrix is factored with QR instead of the original normal-equation Cholesky,
avoiding a squared condition number. A trapezoidal volume mean fixes the constant.
No volume-by-seven basis is retained; null fields are generated as needed.

`solve(b)` without `wall_gradient` returns the zero-mean pressure representative
selected by the lifts. It does **not** impose homogeneous wall-gradient data.
Likewise, `project(raw, completion=False)` omits wall completion. These options
preserve the interior projection but can change pressure by nonconstant null
modes. `project` uses the completed pressure to compute the returned field,
so its divergence reflects the same pressure as the reported Schur residual.

## Validation and cost

Run the analytic example:

```sh
PYTHONPATH=src python examples/pressure_projection2d.py
python -m pytest tests/test_pressure_poisson2d.py -q
```

Tests compare against an independently assembled small dense Schur matrix and
full-nullspace least-squares solution, check smooth analytic gradients on square
and rectangular grids (including both FFT parity cases), projection idempotence,
null-completion invariance, and rejection of incompatible scalar right-hand sides.
The dense global matrices exist only in the tiny test reference.

Setup uses dense one-dimensional factorizations, not a global Poisson matrix.
For an `N x N` grid, tensor transforms cost `O(N^3)` per solve and retained
storage is `O(N^2)` for fixed spline parameters. The derivative application uses
line FFTs plus low-rank factors. Arbitrary geometry, variable coefficients,
prescribed Neumann/Dirichlet data, and a general NS solver are outside this API.

## Grid convergence study

Generate the semilog/log–log comparison, observed orders, and solve residuals:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python examples/pressure_projection2d_convergence.py
```

This requires matplotlib. The script saves PNG/PDF plots and all measurements
in `build/pressure_convergence2d/convergence.json`. It fixes every BSPF parameter
and tests 13 square grids from 34 to 256 endpoint-inclusive nodes per axis.
All node counts are even to keep FFT parity consistent. For three analytic
pressures, it projects the **analytic** gradient, then compares the recovered
zero-mean pressure with the exact samples in trapezoidal weighted L2. Using the
discrete gradient as input would mostly measure inversion accuracy instead of
spatial convergence, so this study deliberately avoids doing that.

The measured oscillatory pressure error drops from `1.06e-3` to `5.57e-12`.
On the common fit window `N=80..256`, a power law gives approximately `h^9.26`:
the log10 error-fit RMSE is `0.0206`, compared with `0.2258` for an exponential
in N. The localized analytic pressure also settles near order ten before
approaching roundoff. The smooth `exp(x+0.5*y)` case is already near roundoff
on the coarsest grid and cannot establish a convergence rate.

These observations support high-order **algebraic** h-convergence for these
fixed parameters over the tested range, not sustained exponential convergence.
That behavior is consistent with using a fixed finite number of endpoint jets
and a fixed spline degree. It does not establish what happens when the jet
order, spline degree, or basis size is increased with grid resolution.

## Improving endpoint estimation

The optional Chebyshev endpoint map ports the differentiation work in
`jax/src/bspf_jax/endpoints.py` to NumPy/SciPy. It changes only the linear
sample-to-jet map: the spline QR fit, FFT derivative, masked Schur equation,
tensor inversion, and wall completion retain their structure. Consequently,
the line matrices and their factorizations must be rebuilt with the new map;
it must not be used only for the RHS or only for the pressure gradient.

```python
solver = PressurePoisson2D(
    x, y,
    endpoint_method="chebyshev",
    chebyshev_modes=14,       # local polynomial degree 13
    baseline_points=18,      # samples in each local endpoint window
    endpoint_regularization=1e-12,
)
```

The default remains `endpoint_method="taylor"` to reproduce the source. Its
14 samples fit a polynomial of degree **eight**, since q=9 determines the
number of Taylor coefficients. More samples alone do not raise that degree.
The Chebyshev estimator decouples fit degree from the number of jets: fit
degree 13, for example, then evaluate only derivatives 0 through 8. It copies
endpoint values exactly and solves regularized least squares through augmented
QR with a normalized fourth-power modal penalty.

Run `examples/pressure_endpoint_comparison.py` with `PYTHONPATH=src` to reproduce
the comparison and save plots plus JSON measurements. At N=128, weighted
relative pressure errors were:

| Configuration | Oscillatory | Localized |
| --- | ---: | ---: |
| Original Taylor fit | 3.94e-9 | 4.01e-11 |
| Chebyshev M12/P16 | 3.73e-12 | 7.81e-13 |
| Chebyshev M14/P18 | 2.28e-13 | 1.09e-12 |
| Chebyshev M16/P20 | 1.61e-12 | 3.28e-12 |
| M14/P18 plus q=11, degree=15 | 4.77e-13 | 3.96e-14 |
| M14/P18 plus 48 splines | 1.29e-13 | 1.10e-13 |

This supports improving endpoint estimates first, followed by controlled tests
of matching order and spline resolution. M12/P16 gives a lower fine-grid floor
in these tests, while M14/P18 reaches high accuracy earlier on the oscillatory
case. M16/P20 is not uniformly better. These clean-data studies do not validate
robustness to noisy RHS or NS stage data, or establish a universal best setting.
Fixed-degree local fits still have algebraic truncation error; these improvements
do not establish asymptotic exponential convergence. A future adaptive choice
must balance truncation against derivative-weight amplification, while keeping
the assembled map fixed and linear during each pressure solve.
