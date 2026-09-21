# Degree-9 differentiation convergence diagnosis

The early power-law region in the current 2D example is primarily caused by
the endpoint derivative estimator. Finite spline smoothness is a separate
asymptotic limitation, but does not explain the dominant error in this run.
This corrects the earlier inference from the degree-5 experiment: its endpoint
comparison cannot be extrapolated to degree 9.

## Controlled experiment

The test uses the current smooth squared-radius field and seeded cosine
background from `examples/operation/differentiate_2d.ipynb`. The measured
quantity is the full-grid relative L2 error in the x derivative against JAX AD.
All cases use float64, degree 9, 18 basis functions, constraint order 8,
regularization 1e-6, and the same N-by-N grid. Only endpoint data changes.

| N | 9-point stencil | 11-point stencil | 13-point stencil | Analytic endpoint derivatives |
|---:|---:|---:|---:|---:|
| 97 | 2.81408e-8 | 2.47092e-8 | 2.46497e-8 | 2.46447e-8 |
| 129 | 1.02638e-9 | 4.59450e-11 | 2.72778e-11 | 2.72002e-11 |
| 193 | 2.29760e-11 | 3.74548e-13 | 4.43012e-13 | 1.65269e-13 |
| 257 | 1.52424e-12 | 1.34964e-13 | 3.23063e-13 | 6.43193e-14 |
| 513 | 1.32281e-13 | 2.09235e-13 | 4.67100e-13 | 1.19077e-13 |

The analytic endpoint data was passed through the existing `boundary` argument;
no differentiation kernel was changed. A 1D plan treats y slices as batches,
which is exactly the x-axis operation performed by the tensor plan.
Endpoint derivatives were calculated from closed-form tanh derivative
polynomials, with spot checks against 70-digit mpmath differentiation through
order 7. Stable sech-squared evaluation avoids cancellation in the tanh tails.

## Implementation mechanism

`src/pybspf/plans.py` builds a one-sided polynomial stencil by solving a
Taylor/Vandermonde system and dividing derivative weights by powers of dx.
For degree 9 the default constraint order is 8, imposing derivatives 0 through 7.
With nine boundary samples, these are derivatives of a degree-8 interpolant.
In general the m-th derivative stencil has truncation error O(h^(9-m)); the
seventh derivative therefore has only second-order local accuracy. Symmetry
can cancel leading terms in this particular field. These local orders are
not the observed global BSPF convergence order.

`src/pybspf/operators.py` inserts these estimated derivatives directly
into the equality-constraint right-hand side. There is no defect correction or
accuracy check on that data. Once the Fourier approximation error decreases
below the endpoint-induced error, the latter produces a power-law region.
Increasing spline degree alone does not make the endpoint estimator spectral.
The NumPy/CuPy endpoint helper uses the same Taylor-stencil construction.

## Other checks and limits

- An independent NumPy/SciPy reconstruction agrees with the JAX derivative to
  relative 3.34e-14 at N=129, much smaller than the 1.03e-9 baseline error.
- Constraint-row scaling of the KKT system leaves the early power-law region
  essentially unchanged. It is not a demonstrated fix for this error.
- The current FFT period is N*dx on a closed grid, explicitly documented in
  `jax/DESIGN.md`. Omitting the final sample gives period (N-1)*dx, but with
  analytic endpoints the errors at N=129 are 2.72e-11 and 2.48e-11 respectively.
  That convention does not explain the much larger baseline error.
- Higher endpoint matching and a single polynomial without interior spline
  knots do not substantially change the error at N=129 with analytic endpoints.
- At large N, all these float64 calculations approach a roughly 1e-13 floor.
  Wider stencils can worsen this floor; their high derivative weights amplify
  sample and arithmetic errors. These tests do not isolate every roundoff source.
- Finite-degree spline knots and finite endpoint matching still preclude a
  general claim of exponential asymptotic convergence. Analytic endpoint data
  reaches the numerical floor too quickly here to measure that asymptotic tail.

## Recommended next step

Treat endpoint estimation as a separate numerical component. An 11-point
stencil is a useful immediate comparison, but not a universal replacement.
Evaluate LDC or another stable endpoint estimator against the analytic-endpoint
baseline, measuring both endpoint accuracy and the final derivative error.
Keep the exact-endpoint option for manufactured-solution diagnostics only.
No library kernels or notebook parameters were changed during this diagnosis.

## Follow-up: updated standalone Chebyshev endpoint implementation

The standalone folder was updated after the earlier inspection. It now includes
`02_low_degree_chebyshev_noise.py`. Its source and README identify LDC as
**Low-Degree Chebyshev differentiation**, not Local Defect Correction. The
relevant component is `local_chebyshev_boundary_matrix`, a regularized local
Chebyshev least-squares fit followed by endpoint differentiation. Full-LDC is
a separate whole-function differentiation method, not an endpoint correction
iteration.

The endpoint matrix was imported directly from that file and applied to the
current field samples. Its output was passed to the JAX `boundary` argument.
The comparison below keeps degree 9, constraint order 8, 18 basis functions,
regularization 1e-6, and the Fourier operation unchanged. Chebyshev alpha is the
archive default 1e-12, and penalty power is 4. M is the number of modes, so the
polynomial degree is M-1; P is the number of samples in each endpoint window.

| N | Current FD9 | Chebyshev M8/P40, adapted to q=8 | Chebyshev M12/P16, q=8 |
|---:|---:|---:|---:|
| 65 | 1.49116e-5 | 1.46856e-1 | 1.50176e-5 |
| 129 | 1.02638e-9 | 1.30326e-3 | 6.78595e-11 |
| 193 | 2.29760e-11 | 1.32840e-5 | 9.17792e-13 |
| 257 | 1.52424e-12 | 5.07074e-7 | 1.08241e-13 |
| 513 | 1.32281e-13 | 1.05239e-8 | 2.58977e-13 |

The actual archived BSPF class was also evaluated without changing its degree-8,
20-basis, q=4, M8/P40, regularization and Fourier-cutoff defaults. Its errors
were 1.32351e-3 at N=129 and 2.08456e-6 at N=257. Those results involve several
algorithm changes and are not an isolated endpoint comparison.

M12/P20, M16/P24 and M16/P20 were also tested. Increasing mode count or window
width was not uniformly beneficial: M16/P24 gives 4.46591e-11 at N=129 but
1.09464e-12 at N=513, while its coarse N=65 error is 7.02765e-3. At N=129 the
P40 window spans approximately 1.91 physical units; fitting this field over
that broad interval with a degree-7 polynomial introduces substantial bias.

Conclusion: the local Chebyshev estimator is a useful candidate for improving
the middle of this convergence curve, but the archive defaults should not be
copied wholesale. This is a clean-data test, not a noise-robustness validation,
and M12/P16 was selected after a small parameter sweep rather than established
as a universal default. The local fit currently uses normal equations; a
future implementation should compare QR/SVD solves and quantify high-order
endpoint noise amplification. No estimator was installed as a new default.
