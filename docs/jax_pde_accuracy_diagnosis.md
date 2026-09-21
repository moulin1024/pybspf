# Historical diagnosis of the initial beam and Schrödinger errors

The large errors come primarily from the new PDE discretization, not an
inherent BSPF differentiation limit. Two choices in the migration compromise
accuracy: under-resolved weak-form quadrature and second-order midpoint phase
error. Norm/energy conservation and permissive notebook assertions did not
establish adequate solution accuracy.

Both production notebooks have now been corrected. The Schrödinger correction
is recorded below; the [complete beam correction](jax_beam_correction.md) uses
resolved curvature factors, stable SVD modes, exact forced evolution, all four
boundary constraints, and a converged analytical reference. The initial error
measurements below are historical, not the current notebook results.

## 1. Nodal trapezoidal assembly loses the accurate spatial operator

The migrated weak form uses `M=E.T@W@E` and
`K=(Dk@E).T@W@(Dk@E)`, with nodal trapezoidal weights. This under-resolves
products of the oscillatory BSPF cardinal trial functions and their derivatives,
especially near endpoints. High-order pointwise differentiation does not make
this quadrature high-order. With mesh-dependent trial functions, one cannot
assume the usual fixed-integrand trapezoidal error estimate uniformly applies.

The diagnostic keeps exactly the same BSPF trial space and boundary extension
but evaluates the actual spline-plus-Fourier trial functions and derivatives
between samples. It splits quadrature at every data node and spline knot,
then uses eight-point Gauss–Legendre quadrature on each interval. Both mass
and stiffness are integrated; beam forcing is integrated consistently too.

| Beam samples | Trapezoid static displacement error | Resolved-quadrature error |
| ---: | ---: | ---: |
| 33 | 1.431e-3 | 6.062e-11 |
| 65 | 6.459e-4 | 1.300e-9 |
| 129 | 3.032e-4 | 5.773e-9 |

The exact static solution is the quartic `x²(x²-4x+6)/24`, so no modal
truncation or time error enters this comparison. At 33 points, the first beam
eigenvalue's relative error falls from -1.195e-2 to 3.622e-10. Increasing the
resolved beam grid eventually exposes conditioning/roundoff in the dense
fourth-order system; refinement is not automatically beneficial once this
floor is reached.

For Schrödinger, the relative error in the tenth Neumann eigenvalue on [0,20]
using degree five falls from -9.932e-5 to 6.878e-12 at 129 samples. Symmetry
preserves a discrete invariant, but an inaccurate spectrum still generates
wrong propagation speeds and reflection phases.

Increasing quadrature from eight to twelve points per subinterval changed
the beam mass/stiffness matrices by about 9.4e-14/7.9e-14 in relative Frobenius
norm (N=33), and the Schrödinger matrices by 1.0e-9/5.9e-11 (N=257).
The large original errors are therefore not explained by this remaining
quadrature sensitivity; near the much smaller corrected-error floor, conditioning
and quadrature must still be checked.

## 2. Midpoint adds phase error after spatial assembly is corrected

Independent generalized symmetric eigendecomposition gives the exact evolution
of each semidiscrete system, separating space and time errors. For a mode of
frequency omega, midpoint advances with modified frequency

```
omega_h = (2/dt) atan(omega*dt/2)
        = omega - omega³*dt²/12 + O(dt⁴).
```

For Schrödinger, omega is the spatial eigenvalue k²; its leading phase error
therefore scales like `T*k^6*dt²/12`. Unconditional stability and exact discrete
norm do not prevent appreciable phase error.

Measured maximum complex field errors for the original packet on [0,20],
T=2.5, N=257, degree seven:

| Assembly | Exact semidiscrete evolution | Midpoint dt=0.001 | dt=0.0005 | dt=0.0001 |
| --- | ---: | ---: | ---: | ---: |
| Nodal trapezoid | 7.143e-2 | 7.199e-2 | 7.156e-2 | 7.143e-2 |
| Resolved Gauss | 2.335e-6 | 4.873e-3 | 1.218e-3 | 4.871e-5 |

The continuum reference uses analytic cosine coefficients of the Gaussian;
extending the Gaussian integral to the real line has negligible error for
this packet centered ten Gaussian widths from either boundary.

For the beam at N=33, degree five, T=3:

| Assembly | Exact semidiscrete evolution | Midpoint dt=0.0005 | dt=0.00025 | dt=0.0001 |
| --- | ---: | ---: | ---: | ---: |
| Nodal trapezoid | 9.297e-3 | 9.303e-3 | 9.299e-3 | 9.298e-3 |
| Resolved Gauss | 2.425e-7 | 7.428e-6 | 3.548e-6 | 6.703e-7 |

The beam comparison uses ten analytic cantilever modes with stable exponential
expressions and 256-point Gauss quadrature for projection. Its finite modal
reference and unresolved high modes limit interpretation of the smallest
transient errors; the static test above is a cleaner spatial accuracy check.
The notebook's six-mode reference must also be strengthened as solver error
is reduced.

## Required correction

1. Replace nodal mass/stiffness/forcing sums by resolved integration of the
   actual BSPF trial functions, checking quadrature-order independence.
2. Use phase-accurate evolution for these autonomous linear models: exact
   generalized-eigenmode propagation is a useful reference; a higher-order
   time method or a properly converged smaller step is needed for evolution.
3. Set tolerances from independent spatial/time/reference refinement, and
   test eigenvalues and the exact static beam as well as conservation.
4. Address dense-system conditioning separately rather than increasing N
   to compensate for inaccurate quadrature.

Reproduction (SciPy is used only for independent diagnostic eigendecomposition):

```sh
OMP_NUM_THREADS=4 \
  python docs/diagnostics/run_with_local_blas.py scratch/diagnose_pde_weak.py
OMP_NUM_THREADS=4 \
  python docs/diagnostics/run_with_local_blas.py scratch/diagnose_pde_phase.py
```


## Implemented Schrödinger correction

The notebook now calls `galerkin_1d(..., quadrature_order=10)` and
`integrate_schrodinger(M, K, initial, times)`. Both are implemented in JAX;
the latter uses Cholesky mass scaling and a Hermitian eigendecomposition,
then applies exact discrete modal phases. The analytic cosine solution is
used only for validation, not numerical propagation.

A controlled endpoint/degree study found that the previous 2.33e-6 remaining
error peaks near the reflecting boundary and closely tracks trial-space
interpolation error for physically relevant cosine modes. Explicitly imposing
Neumann endpoint rows did not fix it. Local Chebyshev endpoint fitting with
8 modes over 16 samples was worse for this clean oscillatory signal. Merely
raising degree to nine with 24 spline functions gave 2.96e-6 error. Increasing
the spline basis count reduced numerical sensitivity, while spatial refinement
reduced the endpoint approximation error. These conclusions are configuration
specific, not a general rejection of Chebyshev endpoints or higher degree.

The adopted parameters are degree 7, 48 spline functions, a nine-point endpoint
stencil, 513 samples on [0,20], and T=2.5. Actual notebook execution on the
corrected CPU BLAS runtime measured:

| Check | Result |
| --- | ---: |
| Maximum complex field error, N=257 | 2.439e-6 |
| Maximum complex field error, N=513 | 5.038e-9 |
| Difference between Gauss orders 10 and 12 | 2.154e-10 |
| Difference between 96- and 128-mode continuum references | 1.318e-12 |
| Discrete norm drift | 2.220e-15 |

The notebook asserts field error below 2e-8, at least 20-fold reduction under
spatial refinement, quadrature difference below 1e-8, and norm drift below
1e-11. This replaces the former permissive 0.025 field-error tolerance.
Dense matrix conditioning and boundary approximation still limit ultimate
accuracy; this is a measured result for the specified packet, not a general
machine-precision or exponential-convergence guarantee.

`scratch/diagnose_schrodinger_accuracy.py` reproduces the parameter study using
an independent SciPy generalized eigensolve. The production notebook and
library evolution use only JAX numerical routines. Unit tests also exercise
complex Hermitian mass matrices, exact modal phases, polynomial weak-form
integrals, an exact static beam, and a continuum Neumann eigenmode.


## Implemented beam correction

The beam notebook and MP4 have been regenerated. Maximum transient displacement
error is now 1.20e-9, compared with the former 1.92e-3. Resolved quadrature,
consistent loading, SVD of the curvature factor, exact modal phases, strong
full-field clamp/free-end constraints, and a 256-mode analytical reference
replace the original assembly, midpoint evolution, and six-mode comparison.
See [the beam correction and measured limitations](jax_beam_correction.md),
including independently measured third-derivative boundary roundoff.
