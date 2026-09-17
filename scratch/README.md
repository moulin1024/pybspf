# Joint noise-regularized BSPF prototype

`noise_regularized_bspf.py` is an isolated NumPy/SciPy experiment, not a change
to the `bspf_jax` API. It uses the JAX library only for comparison baselines.

From the repository root:

```sh
OMP_NUM_THREADS=4 PYTHONPATH=jax/src python docs/diagnostics/run_with_local_blas.py \
  scratch/noise_regularized_bspf.py
```

Without the corrected local BLAS, use the original environment with
`OMP_NUM_THREADS=1 PYTHONPATH=jax/src python scratch/noise_regularized_bspf.py`.
Requires NumPy, SciPy, JAX, and Matplotlib.

Optional arguments: `--samples 513 --realizations 32 --noise 1e-4
--output build/noise-regularized-bspf`. Noise is standard deviation divided by
the clean signal's RMS; the selector receives the resulting *absolute* standard
deviation. The clean derivative is used only to score results.

## Formulation

For samples y, jointly fit spline Bc and a periodic residual r:

```
minimize ||Bc + r - y||² + alpha * (gamma ||R c||² + ||D^p r||²)
subject to mean(r) = 0.
```

The prototype uses p=2, gamma=1, degree-5 splines, and 18 basis functions.
`R` evaluates the p-th spline derivative at Gauss points, with weights giving
`||R c||² = (N/L) integral |s^(p)|²`. Quadrature is exact for the squared spline
derivative. Fourier coefficients use the orthonormal FFT and physical angular
frequencies; their period is N*dx, as in the existing BSPF implementation.

Eliminate the residual analytically:

```
H_k = 1/(1 + alpha*|omega_k|^(2p)), H_0 = 0
r_hat = H * FFT(y - Bc)
S = 1-H
```

Then solve the small-column augmented least-squares problem

```
[ sqrt(S) FFT(B) ] c ~= [ sqrt(S) FFT(y) ]
[ sqrt(alpha*gamma) R ]  [       0        ]
```

using QR. There are no normal equations or dense Fourier matrices in this
algorithm. The zero-mean residual fixes the constant-mode ambiguity. There are
no hard endpoint measurements, estimated endpoint jets, or post-fit cutoffs.
`apply` differentiates both components of the jointly fitted representation.

`discrepancy_select` searches 81 alpha values between 1e-14 and 1e2, selecting
separately for each realization the fit closest to residual norm sqrt(N)*sigma.
It reports selection at search boundaries and residual/target ratios. This is
an approximate discrepancy rule, not an oracle for optimal derivative error.

## Checks and outputs

Every run first compares the FFT/QR algorithm against an independent dense real
Fourier/spline augmented least-squares solve on a small grid, including the
resulting derivative. It also checks zero residual mean and reproduction of an
unpenalized affine signal. Failure raises an assertion.

The default experiment uses the same nonperiodic signal as the noisy notebook,
513 samples, seed 20260917, and 32 paired realizations. Outputs include:

- `report.json`: method errors, selected regularization and discrepancy diagnostics.
- `results.npz`: inputs, exact derivative, fitted signal, joint derivative, alphas.
- `comparison.png`: one realization and spatial RMS errors over the ensemble.

At noise 1e-4, mean relative L2 derivative errors were:

| Method | Whole domain | First/last 40 points | Interior |
|---|---:|---:|---:|
| Existing FD9 BSPF | 2.516% | 4.662% | 1.853% |
| Existing M8/P40 endpoint-only BSPF | 1.860% | 1.889% | 1.853% |
| Joint regularization | 0.567% | 1.444% | 0.136% |

No selections hit the alpha search limits. Residual/target ratios were
0.957–1.043. These compare complete configurations: the existing baselines use
degree 9 and hard endpoint constraints, while the prototype uses degree 5 and
no endpoint constraints. They are not an isolated comparison of one parameter.

A second run with `--noise 1e-2 --output build/noise-regularized-bspf/high-noise`
also passed the independent checks. Whole-domain errors were 251.6% (FD9),
186.0% (endpoint-only M8/P40), and 6.21% (joint regularization). No alpha
selections hit the search limits; residual/target ratios were 0.970–1.026.

## Limits

Real-valued 1D input on odd uniform grids only; first derivative output;
known independent homoscedastic noise. The default demo needs more than 80
samples for its boundary/interior mask. No automatic noise estimation,
heteroscedastic weights, soft endpoint priors, or JAX port is implemented.

Boundary errors remain noticeably larger than interior errors. The example
validates the formulation and shows noise reduction, but does not establish
optimal parameters, resolution guarantees, or superiority over other smoothers.
Changing units changes physical derivative penalties and the relevant alpha
range. A production API would need a broader validation and parameter policy.
