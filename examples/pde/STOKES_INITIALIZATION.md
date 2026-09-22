# Stokes initialization: separating approximation and solve errors

Run from the repository root with the core and models packages installed:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/bspf-mpl \
python examples/pde/diagnose_stokes_initialization.py \
  --out build/immersed_flow/stokes_initialization_new
```

The output directory must not already exist. The script leaves historical data
and production solver defaults unchanged. CPU/MPFR construction uses four
workers by default; `--basis-workers 1` selects serial construction.

The physical domain is [-1,5] x [-1,1], with the original eccentric elliptical
obstacle, parabolic inlet, no-slip velocity boundaries and Laplacian zero
traction outlet. The background grid is 73 x 33. Sponge strength is zero in all
methods: a homogeneous lightning Stokes reference does not solve the original
spatially varying sponge problem. No convection or time stepping is involved.

The reference is freshly constructed by the separate boundary-only lightning
implementation at orders 96 and 120, with independent boundary checks and a
common-grid refinement comparison. The 120-order solution supplies the targets.
It shares the rational representation family with the rational correction; it
is independent of BSPF volume assembly, not an unrelated reference algorithm.

For each of `svd`, `factor`, and `rational`, the script computes:

- The production Stokes initialization.
- The best projection of the reference onto exactly the same affine velocity
  space, using the discrete H1 norm (velocity and all first derivatives), with
  the same lift and constraints as the production solve.
- Velocity and vorticity relative grid L2 errors on 401 x 161 common samples,
  H1 errors on volume quadrature, independent shifted hole-wall samples,
  the linear backward error, and the similarity of the two vorticity error maps.

Grid L2 here means an unweighted Euclidean norm on common fluid samples, as in
`compare_immersed_stokes.py`; it is not an exact domain integral. The H1 best
projection uses the method's finite quadrature rule, not a certified continuous
best approximation. Four exact outer corners and points inside/on the hole are
excluded from grid error comparisons. No finite-width boundary region is hidden.

`projection_ripple.png` displays both error maps with the same color scale.
Its section plots autoscale vertically, so residuals in the most accurate
method remain visible. No smoothing or filtering is applied. Error-map cosine
similarity is an uncentered Euclidean metric, not a rigorous error estimate;
at the reference/roundoff floor it is not useful for diagnosing ripple.

`--quadrature-factor 4` runs an integration control with the same unknown grid;
`--methods svd factor` selects a subset. Use a fresh output directory for each
control. Each completed method writes its state, raw fields, metrics, and plot.

For zero body force and zero sponge, the rational method's homogeneous Stokes
lift already supplies the solution. The production code sets its weak viscous
load to zero by Green's identity. Success in this case validates initialization
and the boundary representation; it does not validate general volume accuracy,
the buffered initialization, or subsequent nonlinear Navier-Stokes evolution.

## Observed results (2026-09-22)

Fresh CPU/MPFR run on `feat/refactor`, source HEAD `bc4bdf0`, quadrature factor
2.5, 17,881 physical quadrature points:

| Space | Solve velocity H1 error | Best-projection velocity H1 error | Solve vorticity grid L2 error |
| --- | ---: | ---: | ---: |
| SVD, rcond=1e-10 | 5.067922% | 5.067908% | 5.679404% |
| Analytic wall factor | 2.583948% | 2.583943% | 2.911328% |
| Rational boundary correction | 1.6814e-9% | 1.6813e-9% | 2.7478e-9% |

The rational discrepancy is at the reference-comparison floor: reference order 96 to 120 changes velocity by at
most 8.52e-12 and vorticity by 2.276e-9; the rational solve's maximum vorticity
difference is 2.282e-9. Both methods use the same rational representation family,
so this is numerical agreement with the refined reference, not a rigorous error
bound or a validation of all possible reference errors.

The SVD solve and best-projection vorticity error maps have cosine similarity
0.9999974332. Their difference has norm only 0.2266% of the solve's error.
For the wall-factor space these numbers are 0.9999981390 and 0.1930%.
Raw error maps and sections show the same oscillations in each solve/projection
pair. Thus replacing the Stokes solve by a reference projection within the
existing affine space does not remove the artifact. The SVD linear relative
backward error is 1.54e-17 (Frobenius matrix norm in the denominator).

At this resolution the wall factor improves accuracy but retains visible
oscillations; the rational representation resolves this specific homogeneous,
zero-sponge Stokes initialization down to the reference-comparison floor.
The production solver and example defaults were not changed. Buffered Stokes
and subsequent NS evolution require separate checks.

Artifacts: `build/immersed_flow/stokes_initialization_20260922/` contains
`report.json`, `reference.npz`, one raw NPZ per method and
`projection_ripple.png`. The existing independent reference boundary/derivative/
Stokes-equation pytest passed during this investigation.

Quadrature control: raising the factor from 2.5 to 4 increases physical
quadrature samples from 17,881 to 44,445 without changing the SVD space.
The solve vorticity changes by 2.953e-10 relative grid L2
and at most 1.728e-08 pointwise. The best-projection vorticity changes
by 2.341e-09 relative grid L2. The percent-level errors and
oscillatory patterns persist. This rules out volume quadrature as the dominant
source in this SVD initialization test. Control artifacts are in
`build/immersed_flow/stokes_initialization_q4_20260922/`; the field-to-field
comparison is `quadrature_control.json` in the main output directory.
