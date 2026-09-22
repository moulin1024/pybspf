# Multiscale random-wave MMS on a B-spline annulus

Run from the repository root with the model package and Matplotlib installed:

```sh
python examples/pde/spline_annulus_turbulent_mms.py
```

The default output directory is `build/spline_annulus_turbulent/`. This is a
scalar manufactured-solution stress test, not a Navier–Stokes turbulent flow.
It uses the two exact B-spline boundaries from `spline_annulus_convergence.py`,
with Dirichlet data on both components.

## Construction

For seed 20260922, construct 48 cosine waves in four radial bands centered at
Q/8, Q/4, Q/2 and Q. Directions, radii and phases are randomized reproducibly.
Wavevectors are physical angular wavenumbers (radians per coordinate unit),
not selected from the solver's Fourier dictionary:

    u(x) = sum_j a_j cos(q_j · x + phi_j)
    (-Delta + sigma) u = sum_j a_j (|q_j|^2 + sigma) cos(q_j · x + phi_j).

Amplitudes scale as |q|^(-5/6), with sum(a_j^2)/2 = 1. The discrete shell
weights have a k^(-5/3)-type scaling; this is not a claim of a continuous
Kolmogorov spectrum. The cutoffs are Q = 12, 24, 48. The separate PDE parameter
is sigma = 0, +64, -64 (Poisson, modified Helmholtz, oscillatory Helmholtz).
The exact solution is used for boundary values and independent validation;
the solver receives the analytic source, not an exact particular solution.

## Validation and observed results

Source fits use mode radii 12, 18, 24 and require independent relative maximum
error below 1e-7 for all three operators before accepting a cutoff. These
correspond to 625, 1369, 2401 Fourier columns, respectively. They are distinct
from the boundary unknown counts below.

The boundary solver uses order 10 and an adaptive indicator tolerance of 1e-7.
Independent checks use 1669 interior points, shifted boundary samples, and
48 near-boundary points at distances 1e-2, 1e-4, 1e-6. Twelve near-boundary
points also receive a higher-quadrature comparison. Final acceptance requires
relative interior L2 error < 1e-6, relative maximum interior/boundary/near-wall
errors < 1e-5, and relative quadrature change < 1e-8, as well as convergence
of the boundary indicator. These diagnostic gates do not replace or relax
the source-fit gate.

Historical baseline on 2026-09-22, before the interior-collar fix, with padding 2:

| Q | sigma | Source mode radius | Boundary unknowns | Interior relative L2 | Near-wall relative max |
|---|---|---|---|---|---|
| 12 | 0 | 12 | 220 | 3.81e-9 | 2.84e-8 |
| 12 | +64 | 12 | 360 | 9.89e-12 | 2.08e-10 |
| 12 | -64 | 12 | 360 | 3.82e-10 | 6.02e-9 |
| 24 | 0 | 18 | 240 | 5.51e-9 | 4.64e-8 |
| 24 | +64 | 18 | 360 | 3.33e-11 | 3.40e-10 |
| 24 | -64 | 18 | 360 | 3.84e-10 | 7.46e-9 |

All six solves passed. Q = 24 failed the first source level and passed the
second. Q = 48 failed every source level: at radius 24 its source validation
errors were approximately 0.1273, 0.1222, 0.1332 for sigma 0, +64, -64.
**No accepted PDE solution was produced for Q = 48.** The immediate limitation
is the current global Fourier source representation/extension at the tested
resolutions; this experiment does not establish a failure of the boundary
integral formulation itself. It also does not establish a high-frequency
convergence rate or turbulent-flow capability.

`mms.json` records all wavevectors, amplitudes and phases; `results.json`
retains rejected source levels and all solution diagnostics. The six field
NPZ files contain independent validation points, exact values, computed values
and errors. `exact_fields.png` shows analytic reference fields, including the
unresolved Q = 48 case; it must not be interpreted as three computed solutions.

The three parameterized tests in `test_spline_annulus_turbulent_mms.py` check
analytic forcing against a fourth-order finite-difference Laplacian, gradients
against finite differences, and seed reproducibility.

## High-frequency source correction

The source dictionary has component bandwidth `2*pi*m/(padding*box_width)`.
At mode radius 24, padding 2 gives bandwidths (39.46, 45.53), which leave
insufficient room for the Q = 48 source. Padding 1.5 gives (52.62, 60.71)
with the same 2401 columns. A diagnostic with padding 1.5 and the original
Cartesian-only training reduced the three source errors to 3.340e-7,
3.208e-7, 3.492e-7, still failing the unchanged 1e-7 gate. Reducing padding
further to 1.25 was worse (about 2.5e-5): bandwidth alone is not enough.

The Cartesian training grid also leaves gaps alongside both curved boundaries.
The fix adds two interior normal-offset collars: distances `1e-6*width` and
`0.25*width/samples`, with `4*samples` parameters per component. All samples
are checked to lie inside the physical domain. The validation grid, normal
offset and curve parameters differ from training; its near-wall count now
scales with resolution. No exterior source values or manufactured wavevectors
are supplied to the fit.

At Q = 48 this adds 1664 collar samples to the existing 7838 volume samples,
without increasing the Fourier column count or refining the volume grid.
The three source errors become 1.960e-9, 1.884e-9, 2.049e-9. SVD cutoff and
all acceptance tolerances remain unchanged. These are sampled error checks,
not certified uniform error bounds.

The core retains padding 2 as its default; the high-frequency benchmark now
explicitly defaults to padding 1.5 and exposes `--source-padding`. Smaller
padding is a bandwidth/continuation tradeoff, not a universal improvement:
with padding 1.5, Q = 12 needs source mode radius 18 instead of the historical
12 to pass the same gate. Q = 24 also passes at radius 18, Q = 48 at 24.

Reproduce the corrected full benchmark without overwriting the historical run:

```sh
python examples/pde/spline_annulus_turbulent_mms.py \
  --cutoffs 12 24 48 --source-padding 1.5 \
  --out build/spline_annulus_turbulent_resolved
```

The corrected Q = 48 end-to-end results (same independent PDE checks):

| sigma | Boundary unknowns | Interior relative L2 | Near-wall relative max |
|---|---|---|---|
| 0 | 930 | 7.26e-10 | 2.83e-10 |
| +64 | 540 | 6.65e-10 | 1.43e-8 |
| -64 | 1160 | 7.13e-10 | 2.72e-9 |

The boundary indicator is below 1e-7 in all three cases. Maximum independent
boundary error is below 1e-7; the largest higher-quadrature change is 1.14e-9.
The full rerun completed with nine of nine PDE runs passing and no unresolved
cutoffs. The annulus, existing panel-Poisson and MMS test suites passed together
(31 tests).
Boundary unknowns do grow for this high-frequency field: fixing the source
does not eliminate the need to resolve its boundary trace. Dense source SVD
and dense boundary solves remain scalability limits.

An additional model regression uses unrelated off-dictionary wavevectors,
all three operators, and independent near-wall probes at distances 1e-7 and
3e-3. It asserts that source callbacks are never evaluated outside the domain.
