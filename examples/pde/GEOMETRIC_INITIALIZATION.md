# Boundary-compatible initialization without LARS

The channel example accepts `--initial-state compatible --wall-method factor`.
No rational Stokes extension is constructed. `ImmersedFlowPlan.stokes_state` is
now lazily computed, so this path also avoids an unused steady Stokes solve.
Existing callers that request `stokes_state` retain the same solve and result.

With the existing analytic geometry factor q, use

```
psi_0 = q psi_channel + (1-q) C,
C = psi_channel(x_center, y_center),
u_0 = d_y psi_0,  v_0 = -d_x psi_0.
```

On the obstacle, q and its gradient vanish. On the rectangle, q=1 and its
gradient vanishes. Thus the initial velocity satisfies the prescribed inlet
and no-slip data and is analytically divergence-free. The outlet remains a
natural weak boundary condition; pointwise initial traction is not imposed.
The scalar q is an explicit function of geometry, not a LARS approximation.

The existing circulation column represents C(1-q). A mass solve expresses its
known velocity correction in the normalized basis. This is not a steady
momentum or pressure solve. Tests compare independent interior evaluations to
the analytic formula, check wall/inlet traces and flux, and forbid construction
of the rational extension. The state is explicitly verified to be non-Stokes.

## Re=200 diagnostic

```sh
python examples/pde/immersed_channel_flow.py \
  --wall-method factor --initial-state compatible \
  --re 200 --dt .01 --time 1 --save-every .1 \
  --buffer-strength 0 --quadrature-factor 3 \
  --out build/immersed_flow/geometric_initial_re200_dt001_new
python examples/pde/inspect_immersed_ripple.py \
  --out build/immersed_flow/geometric_initial_re200_dt001_new
```

BLAS uses library defaults (10 threads observed on the test machine). No
single-thread environment override, smoothing, artificial viscosity, or sponge
was used. This first isolation experiment uses the original geometric factor
space, without the previous response enrichment. Time integration remains
second-order IMEX midpoint.

The saved run is `build/immersed_flow/geometric_initial_re200_dt001_20260922`.
It completed t=1 with 2060 retained degrees of freedom and 25,434 volume points.
Setup took 34.3 s; advancement plus diagnostics/output took 36.8 s. Maximum
sampled speed decreased from 2.089 initially to 1.672 at t=1. Final obstacle
wall speed was 2.96e-15; prescribed outer velocity error was 2.05e-13; the
maximum relative section-flux error was 1.79e-13.

The initial field is smooth, but significant oscillations appear during
evolution. The same fixed-grid upstream fourth-difference RMS diagnostic gives:

| Time | RMS |
| ---: | ---: |
| 0 | 8.3384e-6 |
| 0.1 | 3.7390e-3 |
| 1 | 1.0884e-2 |

This diagnostic is not a solution error norm. The experiment shows that LARS
is not necessary for ripple to occur; it does not isolate spatial resolution,
nonlinear projection, or time-discretization errors. It is not a comparison at
identical trial space/initial data with the previous enriched rational runs.

## Weighted response enrichment

`geometric_enriched_channel.py` embeds the same compatible initial state in a
larger space, with zero initial enrichment coefficients. Its default adds four
exponential thin scales at each of the obstacle, top, bottom and inlet. The
optional `--broad-lengths .1 .2` also includes the earlier broad modes.
All added streamfunctions have double zeros on the prescribed boundaries.
The geometric lift is not Stokes: its viscous load is retained in the enlarged
system. No LARS construction or steady Stokes solve is used.

```sh
python examples/pde/geometric_enriched_channel.py \
  --re 200 --dt .01 --time 1 \
  --out build/immersed_flow/geometric_thin_re200_dt001_new
```

An independent positive-quadrature audit gates time advancement. Its reference
points are used only for verification, not at each time step. Acceptance uses
relative tolerances 1e-6 for mass/stiffness and 1e-5 for nonlinear loads.

The first geometric run with four thin plus two broad scales
(`geometric_enriched_re200_dt001_20260922`) failed this audit before advancement:
mass 2.42e-7, stiffness 1.84e-6, perturbed nonlinear load 6.00e-4. Raw component
comparisons were about 1e-13, while the final combinations were much less
accurate, indicating amplification by near dependence and cancellation.
This is a failed validation, not an NS result or evidence of ripple reduction.

Removing the broad modes retained 427 thin directions. At local order 12,
mass (1.10e-7) and stiffness (8.48e-7) passed, but the perturbed nonlinear load
(3.78e-4) still failed. Thus broad-mode dependence alone does not explain the
failure. This run also stopped before time advancement. Local weighted-rule
convergence must be checked independently of the background mesh.

At local order 16 the bilinear discrepancies decreased to 3.23e-9 (mass) and
3.12e-8 (stiffness), but the nonlinear discrepancy remained 5.58e-4 against the
original reference. Separating BB, BE, EB and EE reproduces the implementation
to 2.1e-14; their relative discrepancies against that reference were 1.15e-4,
2.88e-3, 4.19e-2 and 2.40e-4, respectively. This does not yet establish which
quadrature is inaccurate: cubic nonlinear products require an independent
reference-convergence check too. No NS trajectory was accepted from these runs.

For a controlled alternative, `--wall-method rational` selects the existing
LARS Stokes lift with zero perturbation coefficients. It still undergoes the
same nonlinear audit. This changes the initial field, so comparisons with the
geometric compatible run must not be described as identical-initial-state tests.
LARS handles the boundary/Stokes background, not the nonlinear projection.
In the existing rational-wall implementation it also supplies homogeneous
Stokes boundary corrections to the tensor basis, used throughout evolution.
Switching wall methods therefore changes both the trial space and the initial
field; it is not an experiment that isolates only initialization.

## Accepted thin-space integration and evolution

Increasing the independent reference to 602,368 points (144 normal layer
points, minimum 40 tangent points per reference panel) left the nonlinear
discrepancy essentially unchanged: 5.5813e-4. Increasing only the local normal
order was therefore insufficient. The local straight-wall tangent budget was
still 1.5 times the background line size, appropriate for lower products but
inadequate for the nonlinear triple products in this test.

Using a tangent factor of 3 with normal order 16 passed the independent audit:
mass 2.41e-9, stiffness 1.12e-8, lift convection 7.82e-10, perturbed convection
2.23e-7. No angular refinement was needed. The background still uses 25,434
points. The four local rules use 11,440 / 25,680 / 25,680 / 16,080 points;
their counts depend on the fixed approximation orders, not global refinement
with Reynolds number. The runner now defaults to these rules, with no automatic
quadrature refinement during its audit.

`geometric_thin_cubic_re200_20260922` completed Re=200, dt=.01, t=1. It has 2060
background plus 427 thin degrees of freedom. Setup took 73.6 s; evolution and
diagnostics took 50.5 s. The separate independent audit took 471 s. BLAS retained
its default multithreading. Final wall speed was 4.45e-15, outer velocity error
1.87e-13, and relative section-flux error 8.08e-12.

The initial sampled fields match the un-enriched run. At t=1 the fixed-grid
upstream fourth-difference RMS decreased from 1.0884e-2 to 7.6648e-3 (29.6%).
The raw vorticity still shows substantial ripple: this is a validated partial
improvement, not ripple elimination. The current thin scales end at 0.0157;
sqrt(nu*t) at t=1 is 0.0392, motivating a separate trial with response lengths
0.04 and 0.08.

```sh
python examples/pde/compare_geometric_enrichment.py \
  --baseline build/immersed_flow/geometric_initial_re200_dt001_20260922 \
  --enriched build/immersed_flow/geometric_thin_cubic_re200_20260922
```

The follow-up with broad lengths 0.04 and 0.08 retained 578 added directions
but failed validation before advancement: mass 3.48e-7, stiffness 5.20e-6,
total perturbed convection 1.05e-5. Raw block errors were much smaller than
the final normalized combination, so near dependence remains a limitation of
that larger space. Its separate BB/BE/EB/EE diagnostic initially used inconsistent
definitions of B (whether broad modes belonged to it); that diagnostic was
invalid and is annotated in the saved summary. The total audit and its failure
are unaffected. The reference decomposition has since been aligned with the
production partition.

## LARS alternative

The four-scale LARS run with the corrected tangent rule passed nonlinear
integration (9.31e-7) but failed stiffness integration (2.49e-6), predominantly
in the background block. It stopped before advancement.

`--stiffness-quadrature-factor 4` enables a separate, one-off assembly of the
rational background stiffness, restricted to the zero-sponge case. It preserves
the existing background coordinates, mass matrix, nonlinear volume points and
time-step rule. The scaled raw stiffness Gram matrix is transformed once,
avoiding an additional volume-by-background rotation. The finer linear-only
rule has 44,445 points and took 26.8 s in the recorded test. It is independent
of Reynolds number for this fixed geometry and background space. This option
still requires the full independent audit to pass before advancement.

```sh
python examples/pde/geometric_enriched_channel.py \
  --wall-method rational --stiffness-quadrature-factor 4 \
  --re 200 --dt .01 --time 1 \
  --out build/immersed_flow/lars_thin_linear4_re200_new
```

Without broad responses, the adapter now shares the already-assembled bulk
operators rather than copying five 25,434-by-background arrays. This avoids
about 2.1 GB of duplicate arrays at this resolution; it does not alter their
values or the integration rule.
