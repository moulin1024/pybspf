# Entry of ripple during NS evolution

This audit continues the no-sponge, Re=20 rational-space short run. It keeps the
production spatial and temporal operators unchanged and separates the initial
spatial RHS, linear algebra, and temporal integration.

Run from the repository root with the core and model packages installed:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python examples/pde/audit_ns_ripple_source.py \
  --baseline build/immersed_flow/rational_ns_nosponge_20260922 \
  --out build/immersed_flow/ns_source_audit_new
MPLCONFIGDIR=/tmp/bspf-mpl python examples/pde/render_ns_ripple_source.py \
  --out build/immersed_flow/ns_source_audit_new
```

The output directory must be new. The baseline is required to have grid 73 x
33, Re=20, dt=0.02, final time 1, rational walls, quadrature factor 4 and zero
sponge strength. The computational rectangle is [-1,5] x [-1,1]. All construction
uses CPU/MPFR; x64 is explicitly enabled by the diagnostic application.

## Measurements

The initial unforced rational Stokes lift has zero reduced state. With zero
sponge, its weak viscous load is zero by the existing Green-identity treatment.
The initial NS acceleration is the mass projection of the convective load
(including the production outlet treatment). Thus it can be audited before a
time step is taken.

On a 31 x 161 fluid patch x in [-0.85,-0.35], y in [-0.8,0.8], compare the curl
of the discrete acceleration with the local pressure-free equation:

```
omega_t = -u * omega_x - v * omega_y + nu * (omega_xx + omega_yy)
```

Local derivatives use independent fourth-order centered finite differences at
h=0.002 and 0.001. They are diagnostic only. Acceleration is evaluated directly
from the original BSPF coefficients and rational correction, without a second
projection. Reconstruction is checked against resident quadrature operators.

Further checks include:

- The mass solve's relative residual and initial linear-load balance.
- Equivalence of advective and rotational weak convective loads, including the
  outlet kinetic-head term, measured in the inverse-mass dual norm.
- Projection of a smooth pressure gradient, with pressure zero at the outlet,
  so its exact weak load should vanish. This is a specific consistency check,
  not a proof for every pressure field.
- Independent hole-boundary samples for the acceleration.
- One-step increments divided by dt for dt=0.02, 0.01, 0.005, 0.001, showing
  whether they approach the oscillatory semidiscrete rate as dt decreases.
  These are initial-rate tests at different end times, not equal-time solutions.
- A full dt=0.01 run to t=1, compared with the saved dt=0.02 baseline on the
  same physical output grid. Reduced coordinates are never directly compared.

Unscaled fourth differences in y emphasize high-frequency variation. Their
magnitude depends on sampling and is not an exact error norm. Error-map cosine
similarity describes the pattern, not accuracy. The initial-rate patch uses
dy=0.01; the final-time roughness comparison uses the original dy=0.0125 and
x in (-0.85,-0.45), y in (-0.8,0.8). These two diagnostic magnitudes should not
be compared directly.

This audit can locate where ripple first enters the discrete evolution. It does
not by itself prove late-time persistence, exclude every startup compatibility
effect, identify an optimal replacement space, or constitute a validated fix.

## Results, 2026-09-22

The initial discrete vorticity rate already oscillates before any time step.
The local continuous-equation rate is smooth on the same upstream patch.

| Initial rate quantity | Measured value |
| --- | ---: |
| Continuous-equation rate RMS | 2.16922 |
| Discrete rate RMS | 2.30165 |
| Difference RMS | 0.671945 |
| Continuous rate fourth-difference RMS | 3.59299e-5 |
| Discrete rate fourth-difference RMS | 1.09893e-2 |
| Change in continuous rate when diagnostic h is halved | 1.59e-9 RMS |

The discrete-rate roughness is 305.85 times the continuous-rate roughness.
The initial residual's fourth-difference spectrum, averaged across x after a
Hann window in y, peaks at 7.006 cycles per unit y. This is near the nominal
33-node Fourier cutoff of 8 cycles per unit y; that observation supports the
spatial interpretation but is not a directional convergence test.

The BSPF and rational contributions to the rate's fourth-difference RMS are
0.0110406 and 0.00105156 respectively. This decomposition depends on the chosen
representation, so it should not be interpreted as two independent physical
errors. It shows that the current rational correction does not cancel the
background-space oscillation in this projected acceleration.

Checks found a mass-solve relative residual 1.28e-15, zero initial linear load,
advective/rotational relative dual defect 5.57e-10, and a relative projected
velocity norm 4.73e-10 for the tested smooth pressure gradient. Direct rate
reconstruction differs from resident quadrature values by 1.26e-11 relative.
Initial acceleration on independent hole samples has maximum speed 6.97e-11.
These checks do not support an algebraic solve error, this weak-form identity,
or the tested pressure-gradient inconsistency as the dominant explanation.

| First-step dt | Cosine with initial high-frequency spatial residual |
| --- | ---: |
| 0.02 | 0.977594 |
| 0.01 | 0.992290 |
| 0.005 | 0.997636 |
| 0.001 | 0.999885 |

The first-step incremental rate approaches the already-oscillatory spatial RHS
as dt decreases. At equal final time t=1, halving dt from 0.02 to 0.01 changes
velocity by 0.006937% and vorticity by 0.021459% in relative grid L2. The upstream
roughness is 5.797082e-4 versus 5.795415e-4, a 0.028756% change; the two roughness
patterns have cosine similarity 0.9999992483.

The supported entry mechanism is spatial ringing when the nonlinear force is
mass-projected into the finite, globally supported BSPF+rational divergence-free
velocity space. Accurate boundary satisfaction and energy/weak-form identities
do not ensure that the curl of this projected acceleration reproduces the local
vorticity equation. The homogeneous Stokes lift represents the initial velocity
very accurately, but this does not establish accurate representation of the
subsequent inhomogeneous acceleration.

This is consistent with the earlier Re=200 GPU investigation, but all numbers
above are fresh Re=20 CPU measurements with sponge OFF. It locates the entry
stage and rules out time-step error as the dominant cause of this t<=1 pattern.
It does not yet separate all boundary-startup compatibility effects from a
persistent spatial approximation defect for this exact no-sponge case. No
production remedy, extra filtering, viscosity or altered force was introduced.

The next useful spatial experiment is to approximate an independently computed
inhomogeneous Stokes/Helmholtz response to the frozen nonlinear force, comparing
actual and best-approximation velocity-gradient errors, followed by directional
resolution controls. A compatible vorticity/velocity formulation is a candidate
to investigate, not a validated replacement inferred from these checks alone.

Artifacts are in `build/immersed_flow/ns_source_audit_20260922/`: `report.json`,
`derived_metrics.json`, `fields.npz`, `half_dt_final.npz`, and
`source_diagnosis.png`. The baseline is preserved unchanged.
