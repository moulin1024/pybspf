# Nonperiodic KdV test case

`examples/pde/kdv_1d.ipynb` tests BSPF on a genuinely nonperiodic finite interval:

```
u_t + 6 u u_x + u_xxx = 0,    -6 <= x <= 6.
u(x,t) = (c/2) sech²(sqrt(c)/2 * (x-x0-c*t)),    c=2, x0=-2.
```

The analytic traveling soliton supplies the initial field and three boundary
data: `u(-6,t)`, `u(6,t)`, and `u_x(6,t)`. Its restriction to this interval is
an exact solution with those data. The largest endpoint-value mismatch is
1.383e-2 and the right slope reaches about -1.95e-2. These are substantial
compared with numerical error; this is not a periodic or negligible-tail test.
The originally considered periodic construction was replaced, not retained.

## Weak formulation and boundary lifting

`plan_kdv(spatial_plan, quadrature_order=8)` assembles a dense BSPF weak system.
Let Q, G1, G2 evaluate the actual trial functions and their first/second
derivatives on resolved Gauss nodes, and let W contain quadrature weights.
Split the full nodal field into interior samples q and the two prescribed
endpoint values g(t). Interior test functions vanish at both endpoints.
Integrating the dispersion term twice gives

```
M q_t = A q + F g + d_right * h - M_boundary g_t
        + 3 G1_interior.T W (Q_interior q + Q_boundary g)²,

M = Q_interior.T W Q_interior,
A = -G2_interior.T W G1_interior - d_left.T d_left,
h = u_x(right,t).
```

The symmetric part of A is
`-(d_left.T d_left + d_right.T d_right)/2`. Thus the homogeneous linear
problem is dissipative in the mass norm. Assembly explicitly retains that
identity to roundoff. Both endpoint values are imposed strongly, while the
right slope is supplied as a natural weak load: the notebook measures its
actual full-BSPF residual. JAX JVP supplies `g_t`; omitting the boundary-mass
lifting term would solve a different time-dependent problem.

A naive third-derivative collocation matrix with eliminated boundary values
had unstable positive-real eigenvalues on the tested grids. The stable weak
operator is still strongly nonnormal: a prototype modal eigenbasis had
condition number about 2.3e9. Therefore `integrate_kdv` uses ETDRK4 matrix
functions in mass-scaled coordinates, avoiding eigenvector inversion.
Augmented matrix exponentials evaluate phi-functions without dividing by the
operator, so small eigenvalues do not create cancellation singularities.
The dense setup is O(N³), intended for modest 1D problems.

The boundary callback returns `[left_value, right_value, right_slope]` and must
be differentiable JAX code. The evolution returns all closed-grid samples.
Output times must be uniformly spaced and increasing. Stiff nonautonomous
boundary forcing can reduce observed temporal order, so the notebook tests
step refinement rather than claiming automatic fourth-order convergence.

## Integral diagnostics

With nonzero boundary flux, mass and the quadratic integral need not be
constant. In particular,

```
d/dt integral(u dx) = -[3u² + u_xx]_left^right.
```

The notebook compares their entire histories with closed-form integrals of
the reference over [-6,6]. The mass changes visibly as the soliton travels;
asserting conservation on this interval would be wrong.

## Development CPU validation

Corrected local OpenBLAS, JAX x64, degree 7, 32 spline functions, nine-point
endpoint stencil, 129 samples, eight-point Gauss quadrature, T=2,
dt=0.00015625. Maximum errors over all saved times:

| Check | Result |
| --- | ---: |
| Full field error | 2.458e-10 |
| Coarse 65-point field error at the same dt | 4.694e-8 |
| Field error with doubled dt | 9.472e-10 |
| Difference after increasing Gauss order to ten | 2.856e-12 |
| Dirichlet value residual | 0 |
| Actual right-slope residual | 5.333e-9 |
| Finite-interval mass error | 2.144e-10 |
| Finite-interval quadratic-integral error | 6.295e-12 |

Unit tests check the discrete dissipativity identity and a time-dependent,
nonperiodic polynomial solution `u=x³-6t` in the linear limit. That test checks
nonzero, unequal endpoint values, a prescribed nonzero slope, and the boundary
mass term under JIT. The notebook adds nonlinear analytical-field, boundary,
space/time, quadrature, and integral checks. Existing NLSE and linear
Schrödinger notebooks are regression-tested after the shared quadrature refactor.

```sh
OMP_NUM_THREADS=4 PYTHONPATH=jax/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python docs/diagnostics/run_with_local_blas.py -m pytest \
  -c jax/pyproject.toml jax/tests/test_kdv.py
OMP_NUM_THREADS=4 PYTHONPATH=jax/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python docs/diagnostics/run_with_local_blas.py -m pytest \
  -c jax/pyproject.toml jax/tests/test_examples.py -k kdv
```
