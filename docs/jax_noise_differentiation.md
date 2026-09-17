# Noise-aware JAX differentiation in 1D–3D

The caller supplies an **a priori estimate** `noise_std` of the measurement-noise
standard deviation. The operator receives observed samples; it does not inject
noise, infer the noise level, or require a clean reference. `noise_std=0`
preserves the original BSPF operator; a positive estimate enables joint
spline/Fourier regularization. This is an explicit
switch between formulations, not a claim that the joint fit approaches the
original constrained interpolant continuously as noise tends to zero.

`noise_std` is a measurement-noise estimate, **not the smoothing coefficient**
and not a relative factor. For measurements in metres with an estimated noise
standard deviation of 0.01 metres, pass `noise_std=0.01`. For relative noise eta, convert with
`noise_std = eta * RMS(signal)`. Multiplying both the data and its noise estimate
by the same factor preserves the discrepancy choice (up to numerical rounding).
The actual smoothing coefficients are the per-axis `alpha` values in diagnostics.

```python
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import bspf_jax as bspf

x = jnp.linspace(0., 1., 65)
plan = bspf.plan_1d(x, noise_std=sigma_prior)  # your a priori noise estimate
# samples are your acquired measurements; no clean signal is passed
slope = jax.jit(bspf.differentiate)(plan, samples)
fit = jax.jit(lambda p, f: bspf.differentiate(p, f, order=0))(plan, samples)
choices = jax.jit(bspf.noise_diagnostics)(plan, samples)
```

Use `plan_2d(x,y,noise_std=sigma_prior)` or `plan_3d(x,y,z,noise_std=sigma_prior)` with the
same `differentiate`, `derivatives`, `gradient`, `mixed_partial`, `hessian`,
`laplacian`, `divergence`, and `curl` functions. Array conventions are unchanged:
spatial axes first, trailing batches/components; vector fields have a leading
component axis. Both real and complex fields and odd/even uniform grids work.
The notebook [demonstrates all three dimensions](../examples/operation/differentiate_noisy_1d_3d.ipynb).

## Model and parameter selection

For each 1D axis, solve the family of problems

```
minimize ||Bc + r - f||² + alpha (||R c||² + ||D^p r||²)
subject to mean(r) = 0.
```

`R` integrates the squared p-th spline derivative by exact Gauss quadrature,
scaled by N/domain_length to match the discrete data/Fourier norm. The periodic
residual uses period N*dx and orthonormal FFT coefficients. There are no noisy
endpoint constraints. Its spectral shrinkage is
`H[k] = 1/(1 + alpha*abs(omega[k])**(2*p))`, with H[0]=0 fixing the constant mode.
Eliminating r leaves a small-column augmented least-squares problem, solved
using JAX QR and triangular solves, without normal equations or a dense Fourier
matrix. Fourier and spline components are regularized together.

- Numeric `noise_std` is sqrt(E|epsilon|²), not variance, percentage, or relative noise.
  For complex noise it includes both real and imaginary contributions.
- The noise model assumes independent homoscedastic samples. All components of
  a batch use the same noise level and share an alpha per physical axis.
- `noise_penalty_order=2` penalizes curvature by default. Higher derivatives may
  need a higher penalty order, which must not exceed the spline degree.
- `noise_alphas` optionally supplies a finite positive strictly increasing grid
  of at least two physical regularization strengths. Default: 81 log-spaced
  values, 1e-14 through 1e2, multiplied by `(domain_length/(2*pi))**(2*p)`.
- Choose the candidate whose aggregate residual norm is closest to
  `noise_std * sqrt(number_of_scalar_samples_including_batches)`.
  This is a discrepancy heuristic, not an oracle for derivative error.
- `noise_diagnostics` returns a tuple of per-axis selections: `index`, `alpha`,
  `residual_ratio`, `at_search_edge`, and the resolved `noise_std`. A ratio near one meets the target; a grid
  edge or poor ratio requires inspecting the noise assumption or search range.

For noisy plans `lam` must be zero. It belongs to the original clean-data KKT
fit and does not control this regularization. Endpoint-estimator settings do
not participate in the noisy fit; noisy plans have no endpoint constraints.
Supplied `boundary` jets and `correction=False` raise errors rather than silently
changing the joint model. `with_regularization` applies only to clean plans.

## Multidimensional meaning

Select one alpha per axis from **the original unmodified field**, aggregating
all slices/batches. Freeze those selections, then apply the tensor product of
axis operators. For example:

```
gradient_x(f) = D_x(alpha_x) S_y(alpha_y) S_z(alpha_z) f
mixed_xy(f)   = D_x(alpha_x) D_y(alpha_y) S_z(alpha_z) f
laplacian(f)  = sum_i D_i²(alpha_i) product_{j != i} S_j(alpha_j) f
```

S is the fitted-value operator. `order=0` returns the tensor-smoothed field,
not the original noisy samples. This is a separable tensor construction, not
one isotropic multidimensional variational optimization. Selection on raw
samples preserves the original noise units and avoids incorrectly applying
sigma to an already-differentiated or smoothed field. Hessian entries use the
same choices, so mixed derivative ordering agrees at fixed selections.

Do not compute a mixed derivative by repeatedly calling `differentiate` on its
own result with the original noise level: that would adapt again on transformed
noise. Use `mixed_partial`, `hessian`, or a direct higher-order derivative.
Divergence/curl select independently for each input vector component. Adaptive
operators are nonlinear, so usual identities involving separately selected
operator compositions are not guaranteed.

Factories are checked outside JIT. Application and selection are JAX-native,
JIT-compatible, and work with vmap. Selection uses a discrete argmin: autodiff
propagates through the selected operator locally, not through selection changes.
A batch call shares selections; vmap over separate calls can select separately.
Tensor plans must have all clean axes or all noisy axes with the same a priori
noise estimate. An imperfect estimate is allowed; it need not equal a known
noise-generation amplitude. Overestimating it generally favors more smoothing,
while underestimating it retains more noise.

## Scope and verification

This addition supports differentiation and fitted samples. `decompose`,
`tensor_decompose`, `endpoint_jets`, interpolation, integration, and primitives
reject noise-aware plans; use clean plans for those APIs. Their existing
contracts remain unchanged. Noisy fitting is not an interpolating decomposition
of the raw samples, so returning the old decomposition silently would be wrong.

Tests compare the result to an independently assembled dense Fourier/spline
least-squares reference in 1D–3D, including complex batches, even grids,
selected discrepancy ratios, mixed derivatives, gradients, fitted values and
Laplacians. Other tests cover Hessian symmetry, vector operators, piecewise
AD, invalid parameters, unchanged zero-noise output, and paired noise reduction.

The seeded smooth-field gradient experiment (sigma=0.001, grids 33/35/37 per
axis, truncated by dimension) observed these relative L2 errors:

| Dimension | Original BSPF | Joint, supplied sigma |
|---|---:|---:|
| 1D | 4.273% | 2.482% |
| 2D | 8.363% | 3.404% |
| 3D | 4.947% | 2.289% |

Boundary bias remains; these are fixed test cases, not a guarantee of improvement
for every field or noise spectrum. Candidate selection scans rather than
materializing alpha-by-volume arrays. Precomputed storage scales as
O(number_of_alphas * basis_count * sum(axis_lengths)); selection still does
real work per candidate and axis. No performance claim is made here.

Synthetic noise is generated only in validation datasets, so we can measure
errors against analytic references. The inference path receives neither the
clean reference nor the actual perturbation. In real use, provide your acquired
samples and an a priori estimate from instrument specifications, separate
calibration, or domain knowledge. The `noise_std` in diagnostics echoes that
supplied estimate; only the regularization alpha is selected from the data.
