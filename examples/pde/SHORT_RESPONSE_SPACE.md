# Geometric short-time response enrichment

This experiment augments the existing rationally corrected 73 x 33 BSPF space.
The original space, rational Stokes lift, physical viscosity, nonlinear form,
open-outlet condition and IMEX midpoint integrator are retained. There is no
solution smoothing, spectral filter, ramp, artificial viscosity, or force-specific
snapshot basis. This is a host-only research adapter, not a new solver default.

## Added functions

`short_response_space.py` implements streamfunction modes near the ellipse,
horizontal channel walls and inlet. Their curl is exactly divergence-free.
Analytic second-order jets give velocities and velocity gradients.

For the ellipse, let `r = sqrt(((x-cx)/a)^2 + ((y-cy)/b)^2)` and
`d = min(a,b) * (r-1)`. This is a smooth radial coordinate proportional to the
normal distance at the wall, not the exact Euclidean distance. Modes have form

```
psi = d^2 * exp(-d/ell) * {1, cos(k theta), sin(k theta)} * R(x,y)
```

where `R` has double zeros on the rectangle. For the straight walls and inlet,
use exponential decay in the actual wall distance, tangential Chebyshev
polynomials, the same rectangle factor, and `(r^2-1)^2` to enforce the hole wall.
All added functions and their gradients vanish on all boundaries, including the
outlet. Existing functions retain the original open-outlet freedom and obstacle
circulation degree of freedom. No new boundary condition is imposed on the
combined space.

Two geometry-based scale sets are tested relative to
`ell0 = sqrt(nu * 0.02 / 2) = 0.01238278375`:

- Two scales: `ell0 * [0.7, 2]`.
- Four scales: `ell0 * [0.5, 1, 2, 4]`.

Tangential orders are 16 around the ellipse and 24 along each straight boundary.
These choices are independent of the FEM target and frozen force. Scales remain
fixed during each NS run. The added functions are normalized in the velocity H1
inner product, projected off the entire original space, and orthonormalized.
Near-dependent added directions are discarded at relative energy threshold
`1e-10`; no original-space direction is discarded.

## Verification protocol

1. `validate_response_modes.py` checks wall traces and compares analytic
   gradients/Hessians with independent finite differences at curved and straight
   near-wall probes.
2. `validate_response_quadrature.py` refines quadrature factors 4, 6, 8 for the
   complete added-mode energy Gram matrix. Report diagonal relative errors and
   Frobenius errors after diagonal normalization; these do not alone certify
   every nearly dependent linear combination.
3. `enrich_ns_response.py` reuses the exact frozen force and fine curved P4/P3 FEM
   response from `FROZEN_NS_FORCE.md`. Actual Galerkin, best H1, and best energy
   responses use the same enriched space. At changed quadrature points, the
   existing reference mesh/state is read and analytically evaluated without
   remeshing or solving a new PDE; its DOF order is checked against frozen samples.
4. Start NS from the unchanged homogeneous rational Stokes lift at Re=20,
   dt=0.02, no sponge, and evolve to t=1. Save raw fields and compare with the
   preceding original-space run on exactly the same 401 x 161 display grid.
5. Refine the integration rule and compare both frozen response and NS evolution.
   Hole, wall, inlet and outlet-flux checks supplement the visual comparison.

The upstream diagnostic is the unscaled fourth y-difference of raw vorticity,
RMS on x in (-0.85,-0.45), y in (-0.8,0.8). It is a fixed-sampling roughness
indicator, not an exact NS error norm. No independent nonlinear NS reference is
used here. The independent reference establishes short-time *linear* response
accuracy, while NS tests establish changes in the observed ripple.

The user's observation that ripple becomes more apparent as Reynolds number
increases is consistent with `ell0 ~ Re^(-1/2)` at fixed dt. This scaling is a
mechanism argument; it does not establish nonlinear accuracy or stability at all
Reynolds numbers.

## Reproduction

Requires the already generated fine frozen FEM data and its optional research
dependencies; see `FROZEN_NS_FORCE.md`. Use fresh output directories.

```sh
python examples/pde/validate_response_modes.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/validate_response_quadrature.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/enrich_ns_response.py --levels 2 4 \
  --out build/immersed_flow/enriched_response_new
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/enrich_ns_response.py --levels 4 --quadrature-factor 6 \
  --out build/immersed_flow/enriched_response_q6_new
MPLCONFIGDIR=/tmp/bspf-mpl python examples/pde/render_enriched_response.py \
  --out build/immersed_flow/enriched_response_new
```

## First results (2026-09-22)

At quadrature factor 4, the original 2059-dimensional space is retained. The
new two-scale space adds 214 independent directions; the four-scale space adds
406. Errors below are weighted discrepancies against the fine FEM reference,
not certified continuum errors.

| Space | Actual frozen-response H1 | Best H1 | Actual velocity L2 |
| --- | ---: | ---: | ---: |
| Original | 17.6981% | 16.6337% | 2.8324% |
| 2 scales | 1.290032% | 1.201034% | 0.206977% |
| 4 scales | 0.090071% | 0.084266% | 0.014108% |

The four-scale discrepancy is now comparable to the reference's last H1 mesh
refinement difference (0.083950%). Its last digits are not a resolved accuracy
claim. The actual/best-energy H1 gap is 0.00334% of the reference norm.

The fixed-grid upstream fourth-difference RMS at t=1 decreases from
`5.79708e-4` to `5.18036e-5`, a factor of 11.19. Both runs start at exactly
`4.93018e-6`. Raw maps and sections show strong ripple suppression, with residual
small oscillations; this is not evidence of fully converged nonlinear accuracy.
The final peak speed is 1.392178 (original 1.392519), obstacle speed is below
1.05e-11, wall/inlet defects are below 4.9e-10, and the flux defect is 1.56e-12.

Added-mode Gram quadrature checks find a worst diagonal discrepancy of 0.525%
for factor 4 versus factor 8, and a diagonally scaled Frobenius discrepancy of
0.0401%. Factor 6 versus factor 8 gives 2.49e-9 and 3.07e-10 respectively.
Therefore a full factor-6 response and NS rerun is included rather than relying
only on the initial factor-4 result.

Initial outputs: `build/immersed_flow/enriched_response_20260922/`.

## Raising Reynolds number

Use a staged Re=20, 50, 100, 200 study with the same acceptance protocol. At each
Re, recompute the characteristic short-time scale from that viscosity and time
step, span several layer lengths, and check quadrature convergence for those
new functions. Increasing Re by ten at fixed dt reduces the nominal length by
sqrt(10). The observed low-Re/high-Re contrast is consistent with this mechanism,
but the nonlinear force also changes as NS evolves.

Normal enrichment alone does not resolve tangential gradients or detached wake
shear layers. Independently vary angular/straight-wall orders and bulk nx, ny.
Check the time step separately by halving it in a *fixed spatial space*: changing
layer lengths with dt during that comparison would confound spatial and temporal
errors. The midpoint viscous treatment does not by itself establish temporal
accuracy or the stability of explicit convection.

For attached high-Re flow this follows the physical idea of supplying rapid
wall-normal variation in advance. A thin Prandtl layer is not present in every
flow at every Re; low-Re viscous influence can occupy the whole domain, and
separation requires resolving interior shear rather than only wall-attached
functions. The enrichment is a prior on representable scales, not a prescribed
boundary-layer solution or a turbulence closure.

## Completed quadrature refinement

The full factor-6 rerun retains the same 406 added directions. Its frozen-response
H1 discrepancy is 0.089839% and best-H1 discrepancy 0.084131%, consistent with
factor 4 and the current reference-resolution limit. At t=1 its upstream
roughness is 5.18019e-5, still 11.19 times below the original-space result.
Changing quadrature from 4 to 6 changes this scalar diagnostic by 0.00325%.
Final identical-grid relative differences are 1.06e-8 in velocity and 7.52e-7
in vorticity. These checks do not identify integration error as the cause of
remaining NS oscillations. Max-speed diagnostics sampled on different volume
quadratures should not be compared as if they used identical probe locations.

The full rerun and comparison are saved under
`build/immersed_flow/enriched_response_q6_20260922/`, including `refinement.json`.
See `CLASSICAL_BOUNDARY_LAYER_ENRICHMENT.md` for the subsequent theory-guided
space design proposals and their distinction from the implemented experiment.
