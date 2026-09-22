# Grad–Shafranov on an exact B-spline annulus

Implementation: `packages/models/src/bspf_models/plasma/spline_annulus_gs.py`.
Run `python examples/pde/spline_annulus_grad_shafranov.py` after installing the
core/model packages and Matplotlib. Results go to `build/spline_annulus_gs/`.

The default example uses degree-five periodic B-splines built from the same
control polygons as the annulus Poisson experiment, translated to physical
major radius R0 = 3. This CHANGES the actual boundaries and gives C4 rather
than C2 knot joins. Use `--degree 3` for the original cubic geometry. The physical
annulus must stay at R > 0. Geometry uses local coordinates (R-R0, Z), but
source/boundary callbacks and solution evaluation use physical (R, Z).

## Equation and implementation

The sign convention matches the existing GS models:

    -Delta* psi = mu0 R^2 p'(psi) + F F'(psi),
    Delta* = d_RR - (1/R) d_R + d_ZZ.

The substitution psi = sqrt(R) u yields

    -Delta u + 3 u/(4 R^2) = S(R,Z,sqrt(R) u)/sqrt(R).

Picard iteration treats S/sqrt(R) - 3u/(4R^2) as the effective Poisson source.
Both Dirichlet data become g/sqrt(R). This reuses the domain-only Fourier
source extension (including interior collars), analytic Fourier particular
solutions and exact-geometry boundary layer potentials. Source SVD, boundary
LU are reused. Layer evaluation applies bounded blocks directly to the density;
no dense source-grid or validation-grid layer matrices are cached. Near/self
quadrature is reevaluated on each application using cached panel geometry.
There is no fitted volume mesh. Plotting triangulations do not enter the solve.

Intermediate effective sources can fail the final extension tolerance: they
contain the boundary correction from an unconverged iterate. Intermediate fits
are therefore permitted without acceptance, but a returned solution MUST pass
all three separate checks:

1. The effective source fit on independent source validation points.
2. The full transformed GS residual, evaluated against the source at the NEW
   flux, including its complex numerical error.
3. Independent boundary probes with higher quadrature order.

The first two are relative maximum errors (with a 1e-14 scale floor for the PDE
residual); the boundary error is normalized by max(1, max|g|). Iteration
exhaustion, nonfinite residuals and failed boundary checks raise errors. The
residual uses the analytic Laplacian of the particular solution plus the fact
that the boundary correction is harmonic; the example separately differentiates
the computed flux with a fourth-order stencil in the ORIGINAL GS operator.
No exact interior flux or its derivatives enters the solver.

## Interface

```python
from bspf_models.plasma.spline_annulus_gs import SplineAnnulusGSPlan

plan = SplineAnnulusGSPlan(
    domain, major_radius=3.0, modes=32, samples=136, padding=1.5,
    order=18, subdivisions=2,
)
solution = plan.solve_profiles(
    p_prime=lambda psi: 0.01 + 0.002*psi**2,
    ff_prime=lambda psi: 0.02 + 0.01*psi**2,
    mu0=1.0,
    boundary_flux=(0.0, 0.02),  # outer, inner
    tolerance=1e-8, source_tolerance=1e-8,
)
psi = solution.flux(physical_points)
```

`solve(source, boundary_flux)` also supports a prescribed scalar/spatial source;
`nonlinear=True` accepts `source(physical_points, psi_values)`. Relaxation and
iteration limit are configurable. This is a Picard accuracy prototype: strongly
nonlinear profiles or regions approaching R=0 can require a different iteration
or preconditioner. There is no free-boundary update, axis treatment, X-point
handling or automatic source/panel adaptation in this new wrapper.

The annulus is a meridional computational region with an excluded inner
component; it is not the full simply connected cross-section of a standard
solid tokamak plasma. Its two prescribed flux boundaries define this experiment.

## Validation cases

- Nonpolynomial Solov'ev analytic flux with a logarithmic homogeneous term.
  Both pressure and toroidal sources are nonzero; analytic traces are prescribed
  on the arbitrary spline boundaries, for numerical regression.
- Oscillatory MMS psi = 0.1 sin(5(R-3)) cos(4Z), with an independently written
  original-GS source containing the radial first-derivative term.
- Flux-dependent profiles as in the example above, with constant outer/inner
  fluxes 0 and 0.02. An independent fourth-order finite-difference GS residual
  is checked at interior probes. This case has no manufactured exact solution.

The automated tests also cover an analytic polynomial, profile iteration,
R>0 and interior-evaluation requirements, iteration exhaustion and rejection
of unresolved source data even with a deliberately loose PDE tolerance.

## Original cubic geometry limitation

On the original C2 cubic boundaries, Solov'ev and oscillatory MMS passed.
The nonmanufactured constant-boundary quadratic-profile case, however,
stagnated at a relative source/PDE residual of about 5.20e-8 with mode radius
32 (4225 columns), so the requested 1e-8 gates rejected it. More Picard
iterations did not improve this. Raising boundary order from 14 to 18 did
not remove the earlier intermediate-source error. The successful analytic
MMS cases do not establish that arbitrary equilibria have equally smooth
Fourier extensions.

Reproduce that rejected case with:

```sh
python examples/pde/spline_annulus_grad_shafranov.py --degree 3 --modes 32 \
  --out build/spline_annulus_gs_cubic
```

The degree-five comparison is a different geometry, not a repair or a claim
of 1e-8 convergence on the original cubic domain.


## Completed degree-five run (2026-09-22)

Run: `python examples/pde/spline_annulus_grad_shafranov.py --degree 5 --modes 32 --out build/spline_annulus_gs_quintic32`.
The Fourier dictionary has 4225 columns; each boundary panel has order 18 and
each original knot span is split in two. mu0 = 1, R0 = 3, final source and
nonlinear PDE tolerances are both 1e-8, and the boundary tolerance is 1e-7.

- Solov'ev: relative flux L2 error 4.80e-10; boundary error 3.43e-14.
- Oscillatory MMS: relative flux L2 error 4.45e-8 (iteration tolerance 1e-7).
- Nonlinear case: p'(psi)=0.01+0.002 psi^2, FF'(psi)=0.02+0.01 psi^2;
  four Picard iterations; final relative GS residual 5.74e-9, effective source
  fit error 5.47e-9, boundary error 7.14e-13.
- Independent original-operator FD residual: 2.36e-8, at 24 interior probes,
  fourth-order differences with h=3e-4. Its acceptance threshold is 1e-5;
  it is a separate finite-difference check, not a 1e-8 derivative certificate.
- Four automated GS tests pass, including a genuinely quadratic flux source
  and an unresolved-source rejection with a deliberately loose PDE tolerance.

The degree-five case at mode radius 24 was also rejected (about 4.12e-8 source
error). Thus increasing boundary smoothness alone did not resolve the problem.
These are resolution/geometry comparisons, not a proof isolating the sole
cause of the Fourier error. Rejected run histories are preserved in
`build/spline_annulus_gs_comparison/failed_runs.json`.

`plot_spline_annulus_gs.py build/spline_annulus_gs_quintic32` draws accepted
interior flux samples together with the prescribed boundary traces. Its
triangulation/interpolation is only for display, not part of the PDE method.

## Optional B-spline extension

Use `--source-backend bspline` to replace
the global source fit while retaining FFT particular solutions and the boundary
integral solver. See [construction, resolution and validation](BSPLINE_SOURCE_EXTENSION.md).
Source/PDE gates are unchanged; insufficient resolution is still rejected.

A resolved B-spline run passes all three cases at the existing tolerances:
`--spline-degree 7 --spline-spans 11 --patch-radius .13 --patch-samples 46
--sample-grid 257 --fft-grid 2049`. Its nonlinear source/PDE errors are
1.11e-9 / 3.36e-9. The default coarse B-spline settings are insufficient for the
nonmanufactured equilibrium; see the linked resolution study and accepted run.

## Bounded-memory layer application

`AnnulusPanelPlan.apply_layer(points, density, ..., block_size=512)` computes
far-field quadrature charges before target evaluation, then immediately reduces
each target block into the result. Near/self quadrature and double-layer jump
averaging retain the reference formulas. `AnnulusSolution.interior` and
`boundary` use this path, as do GS iteration and flux evaluation.

GS accepts `layer_block_size=512`; this also controls its boundary checks.
The dense boundary system/LU remains, and `potential_matrix` is retained for
boundary assembly and explicit reference audits. There is no cached rectangular
target-by-boundary operator. For the resolved example this removes the 810 MiB
training and 441 MiB validation operator caches (1.22 GiB combined).

This implementation uses NumPy/SciPy on the host, not GPU kernels. It trades
repeated layer evaluation for lower memory; it does not claim faster complete
GS iteration. Near/self panel quadrature geometry caches remain. The resolved
B-spline timings above were measured BEFORE this change and must not be used
as timings for the streamed path.

## Full sections and free plasma boundaries

`SplineSection` now supports a full poloidal section without an inner hole.
See [the full-section/free-boundary extension](SPLINE_FREE_BOUNDARY_GS.md) for
magnetic-axis diagnostics and the separate coil/plasma Green-boundary iteration.
The fixed-boundary plan alone remains a Dirichlet solver.
