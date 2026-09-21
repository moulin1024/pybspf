# Random-wave Poisson MMS on curved domains

This is a harder, broadband verification of the full-space BSPF strong-residual
least-squares backend. It uses the same convex and non-star-shaped C-shaped
periodic cubic B-spline domains as the earlier Poisson example. It is a scalar
manufactured solution with a prescribed spectrum, not a turbulent-flow simulation.

## Manufactured data

For 64 random waves,

```
u*(x) = sum_j A_j cos(k_j . x + phi_j)
grad u* = -sum_j A_j k_j sin(k_j . x + phi_j)
f = -Delta u* = sum_j A_j |k_j|^2 cos(k_j . x + phi_j)
g = u* on the exact spline boundary.
```

The fixed seed is 20260918. Magnitudes are sampled uniformly within eight
logarithmic shells, eight waves per shell; orientations and phases are uniform
on [0,2pi). The wavevectors are continuous, not integer box Fourier modes, so
the manufactured field need not be periodic on the surrounding rectangle.

The prescribed scalar variance spectrum has envelope E(k) proportional to
k^(-5/3). Before global normalization,

```
A_j = sqrt(2 E(|k_j|) Delta_k_shell / 8).
```

Normalize so sum(A_j^2)/2 = 1, the phase-ensemble variance. This does not assert
that the spatial variance on either finite domain is exactly one. Minimum
wave number is pi; three separate bandwidths have maxima 4pi, 8pi, and 12pi.
All amplitudes, vectors, phases, and shell assignments are saved to NPZ.

## Numerical experiment

Use all 33, 41, or 49 BSPF factors per direction. Background endpoints are
unconstrained; no low-eigenmode truncation is applied. Direct MPFR evaluation
supplies values and first/second derivatives. The column-scaled oversampled
matrix combines the domain equation and boundary values and is solved by
SVD, with relative singular cutoff 1e-14. Effective rank is recorded.

The same factorization solves all three forcing columns for a given geometry
and resolution. No exact interior values enter the solve. Interior exact
values and derivatives are used only after solving, for independent errors.

Baseline interior sampling is the subset of a (2N+1) x (2N+3) shifted grid
inside the domain; offset is 0.37. Boundary sampling is order-24 Gauss per
spline span. Independent checks use an offset of 0.61 on a 137 x 139 grid,
and order-40 boundary integration. Reported domain norms are sampled norms.
The gradient and PDE residual are evaluated with analytic BSPF derivatives,
and compared with independently specified closed-form MMS derivatives.

## Results

Independent relative solution errors at N=49, baseline sampling:

| Maximum wave number | Convex domain | Non-star-shaped C domain |
|---|---:|---:|
| 4pi | 2.61266e-10 | 1.48007e-8 |
| 8pi | 6.45124e-7 | 4.04054e-5 |
| 12pi | 3.84897e-5 | 9.46914e-4 |

All six geometry/bandwidth sequences decrease monotonically from N=33 through
41 to 49. For the hardest 12pi case, convex errors are
7.28608e-3 -> 4.16481e-4 -> 3.84897e-5; C-domain errors are
2.77639e-1 -> 2.01387e-2 -> 9.46914e-4. At N=49 its gradient errors are
3.30082e-4 / 8.30792e-3 and relative PDE residuals are
1.74316e-4 / 4.19970e-3 (convex / C).

Thus near-roundoff accuracy for the previous simple analytic solution does
not extend to broadband MMS at the same resolution. The scalar solution,
its gradient, boundary trace, and strong residual have different accuracy
requirements. The high-bandwidth C-shaped case is not fully resolved.

An additional fixed-N=49 experiment increases interior sampling to a
(3N+1) x (3N+3) grid. Results are stored separately under
`build/embedded_poisson_random_mms_sampling/`; the baseline convergence plots
are not silently replaced with denser-sampling results. A fixed SVD cutoff
can retain a different rank when sampling changes, so this is a combined
sampling/rank-sensitivity check, not a pure quadrature estimate.

For 12pi, denser sampling gives relative solution errors 4.89802e-5 (convex)
and 2.10621e-3 (C), compared with baseline 3.84897e-5 and 9.46914e-4.
Meanwhile the relative PDE residual drops to 5.86535e-5 and 9.48014e-4.
This change is material: the finest tested solution is not sampling-independent,
and reducing the least-squares PDE residual alone does not ensure that the
solution or boundary error decreases. The denser C-domain boundary error is
1.23680e-2. The benchmark therefore exposes both approximation and
sampling/conditioning sensitivity; it does not certify high-band accuracy.

## Reproduce and outputs

From the repository root, with JAX float64 enabled by the scripts:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/bspf-mpl
python examples/pde/embedded_poisson_random_mms.py
python examples/pde/embedded_poisson_random_mms.py --nodes 49 --sample-factor 3 \
  --out build/embedded_poisson_random_mms_sampling
python examples/pde/render_embedded_poisson_random_mms.py
python -m pytest -q packages/models/tests/test_random_wave_mms.py
```

`build/embedded_poisson_random_mms/summary.json` contains all 18 baseline
configurations, including effective ranks and independent residuals.
`random_wave_validation.png`/`.pdf` show the hardest-band solution, error,
and convergence. `random_wave_spectrum.png` shows wavevectors and shell
variance density. The validation images evaluate the solution on a separate
200 x 200 cell-center grid.

Three automated analytic checks pass: reproducibility and variance
normalization; gradient/Laplacian finite-difference cross-checks including
the Poisson sign and nonconstant forcing; and the exact single-wave identity.
The complete baseline run also verifies the six decreasing solution-error
sequences. No general stability or spectral convergence theorem is inferred
from these finite experiments.
