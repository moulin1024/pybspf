# Noisy 1D differentiation: effect of local Chebyshev endpoints

The runnable [notebook](../examples/operation/differentiate_1d_noisy.ipynb) calls
`pybspf` directly. It is registered in the notebook execution tests. Here LDC
means the archive's local low-degree Chebyshev endpoint estimator; this does not
compare against the separate Full-LDC algorithm or iterative defect correction.

## Experiment

- Signal: exp(0.2x) + sin(2.3x) + 0.2 cos(5.1x), on [0, 2pi], N=513.
- Exact derivative: 0.2 exp(0.2x) + 2.3 cos(2.3x) - 1.02 sin(5.1x).
- Independent Gaussian sample noise of standard deviation sigma × RMS(signal).
- 64 paired realizations, NumPy generator seed 20260917; the same noise arrays
  are reused for every method and scaled across sigma levels.
- Fixed degree 9, 18 basis functions, constraint order 8, lambda 1e-6, and
  default knots. No Fourier cutoff or whole-field smoothing.
- FD uses 9 endpoint samples. Chebyshev fits use either 12 modes/16 samples or
  8 modes/40 samples, with alpha 1e-12 and penalty power 4.
- Float64 JAX, outer JIT, corrected BLAS at four OpenMP threads.

## Results

Mean relative L2 derivative error over the 64 realizations:

| Relative noise sigma | FD9 | LDC M12/P16 | LDC M8/P40 |
|---:|---:|---:|---:|
| 0 | 3.261e-13 | 9.378e-14 | 7.272e-7 |
| 1e-10 | 2.704e-8 | 2.187e-8 | 7.276e-7 |
| 1e-8 | 2.704e-6 | 2.187e-6 | 2.024e-6 |
| 1e-6 | 2.704e-4 | 2.187e-4 | 1.883e-4 |
| 1e-4 | 2.704e-2 | 2.187e-2 | 1.883e-2 |
| 1e-2 | 2.704 | 2.187 | 1.883 |

Noise gain is the mean realization-wise ratio
`norm(D(noisy)-D(clean))/norm(noisy-clean)`, with both norms restricted to the
reported region. Units are inverse length. At sigma=1e-4:

| Method | Whole domain | First/last 40 points | Interior |
|---|---:|---:|---:|
| FD9 | 211.99 | 398.34 | 147.90 |
| LDC M12/P16 | 171.64 | 255.58 | 147.92 |
| LDC M8/P40 | 147.82 | 147.15 | 147.92 |

For this signal/resolution, M12/P16 reduces whole-domain noise gain by about
19%, while M8/P40 reduces it by about 30%. The larger improvement is localized
near the boundaries; interior noise remains essentially unchanged because the
Fourier residual is unfiltered. M8/P40 reduces boundary gain by about 63%, but
introduces a clean-signal bias of 7.27e-7 and is worse at the lowest noise level.
It is not a universally preferable parameter choice or a new default.

The clean errors from single-vector calls and zero-noise batched calls differ
slightly at roundoff. The notebook reports both. Plot bands show 10th–90th
percentiles across realizations, not confidence intervals.

Raw arrays and the plot from the measured four-thread run are retained at
`build/openblas-atomic/noisy-1d-ldc.json` and `noisy-1d-ldc.png` in that directory.
The notebook stores no outputs and regenerates the figures when run.

## Why the archived noise experiment can show a larger benefit

Source inspection of `02_low_degree_chebyshev_noise.py` distinguishes three
mechanisms that the endpoint-only notebook deliberately does not conflate:

1. `LowDegreeChebyshev` (Full-LDC, lines 155–173) forms a low-degree Chebyshev
   representation over the whole evaluation interval. It limits the
   representation's bandwidth, rather than only estimating endpoint jets.
2. `BSPFLocalChebyshev` (lines 114–135) combines M8/P40 endpoints with a Fourier
   residual cutoff `ceil(24*log(N))`. At N=513 this keeps modes through 150,
   rather than all modes through 256. It also uses constraint order 4, degree 8,
   20 basis functions, and different regularization/knots from our experiment.
3. The archived noise is `(1 + normal_noise) * interpolation_defect` (lines
   202–205). Its pointwise mean is the interpolation defect, its amplitude
   varies in space, and it changes with resolution and interpolation order r.
   It is not fixed-amplitude, zero-mean white noise. The archive's comparison
   is Full-LDC versus filtered BSPF-local-Chebyshev, not an endpoint on/off study.

A follow-up ablation uses exactly the current notebook signal, seed, 64
realizations, sigma=1e-4, and plans. It changes only the residual multiplier to
`1j*omega*(abs(mode)<=K)`, leaving the spline derivative untouched:

| Endpoint method | No cutoff (K=256) | Archive cutoff K=150 | K=32 |
|---|---:|---:|---:|
| FD9 | 2.704e-2 | 2.067e-2 | 1.221e1 |
| M12/P16 | 2.187e-2 | 1.366e-2 | 1.493e1 |
| M8/P40 | 1.883e-2 | 8.405e-3 | 1.143e-3 |

Numbers are mean total derivative relative L2 errors. M8/P40 clean-signal errors
for those cutoffs are 7.272e-7, 9.728e-7, and 1.465e-6 respectively. Thus genuine
noise suppression is possible when endpoint stabilization and interior
bandwidth control are combined, with a bias/noise tradeoff. K=32 is an
illustration for this smooth signal, not a selected general default.

Aggressive filtering is disastrous with the noisier endpoint estimators here.
The spline and residual derivatives can be individually large and cancel;
filtering the residual alone can destroy that cancellation. Endpoint jets,
constraint order and residual filtering must therefore be evaluated together.
No filter was added to the public implementation by this diagnostic experiment.
