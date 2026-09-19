# BSPF curl-residual formulation experiment

Status: research prototype, **not a validated ripple correction**. The
production BSPF solver remains unchanged. This experiment follows the
[force-localization audit](jax_re200_ripple_investigation.md), which identified
nonlocal curl errors in the finite velocity-space projection.

## What was tested

Keep the original rationally corrected BSPF velocity space and add

```text
ell² integral beta(x) curl(v_h) * curl(R_momentum) dx
```

to its momentum equation. The momentum residual contains time, convection,
physical viscosity and the outlet sponge. The pressure gradient disappears
under curl. The channel is unforced; an externally forced implementation would
also need the curl of the force. That API and its manufactured tests were not
implemented because these preliminary channel controls failed to establish a
useful correction.

Writing `Omega`, `Omega_x`, `Omega_y` and `LapOmega` for vorticity evaluation
matrices and `W` for quadrature, the added terms are

```text
M_new = M + ell² Omega.T W beta Omega
L_new = L + ell² Omega.T W beta (-nu LapOmega + sigma Omega + sigma_x V)
C_new = C + ell² Omega.T W beta (u omega_x + v omega_y).
```

The matching static lift terms are included. In particular, the sponge curl
contains `sigma_x*v`; it is not simply `sigma*omega`. Both IMEX stage solves
use the augmented mass and linear matrices. The matrix is generally
nonsymmetric, so the prototype uses GPU LU instead of assuming Cholesky
applies. The diagnostic mass solve remains SPD. No output is filtered and
physical viscosity is not increased.

This form is consistent for an exact smooth solution because its added
residual vanishes. **That alone does not prove stability or accuracy of the
finite-dimensional evolution.** This is a curl-test/Petrov experiment, not a
least-squares residual penalty with a manifestly nonnegative quadratic form.

The two length conventions tested are

```text
k_x = pi*(nx-1)/L_x, k_y = pi*(ny-1)/L_y
ell_wave² = 1/(k_x²+k_y²) = .0002533029591
ell_cell² = pi²*ell_wave² = .0025.
```

These correspond to a Fourier derivative length and a combined cell spacing;
they are exploratory normalization choices, not calibrated optimal parameters.
The unweighted test has beta=1. The boundary-compatible variant uses

```text
beta = product_over_boundaries d_b²/(d_b²+ell²),
```

where outer distances are Cartesian and the ellipse uses
`d_hole=(q-1)/|grad(q)|`. Both beta and its gradient vanish on every boundary.
Consequently the extra vector test `curl(beta*curl(v_h))` vanishes there. This
addresses test-function boundary compatibility; it is not a stability proof.

## Analytic GPU derivatives and controls

BSPF derivatives through fourth order use a finite Fourier sum and analytic
spline recurrences. This avoids differentiating the singular cardinal quotient
at a node. The rational correction is a homogeneous Stokes field, so its
vorticity is harmonic: its Laplacian is exactly zero. First derivatives of its
vorticity use the existing second derivatives of the analytic rational basis.

Independent checks found:

- New line derivatives of orders 0–2 match production evaluation within
  1.88e-10 relative for x and 6.24e-14 for y.
- Centered finite-difference checks of third derivatives agree within 9.3e-8
  relative. Fourth-derivative checks reach 9.9e-6 relative for x and 2.2e-8 for
  y at h=.0002. Halving h worsens the x fourth-derivative comparison, consistent
  with cancellation in that reference. This is a diagnostic check, not a
  high-precision certification of every high-order matrix row.
- Rational vorticity gradients match separate finite-difference evaluations
  within 5.8e-12 relative.
- Setting augmentation to zero reproduces an original GPU step within
  6.78e-16 maximum coefficient difference.
- Candidate mass backward residuals are about 1e-15; time loops pass a device
  transfer guard. Geometry/basis metadata and explicit output diagnostics
  retain the original plan's host API.

The initial physical state is held fixed to the original Stokes state. The
augmented steady linear state differs by only 5.68e-8 velocity L2 in the
unweighted test and 2.05e-9 in the weak boundary-compatible test. This avoids
mistaking a different initialization for a changed projection.

## Initial improvement is insufficient

On x=[-.85,-.35], y=[-.8,.8], the original initial local vorticity-equation
residual has RMS .66967. The unweighted curl test lowers it to .20476 and
reduces its y fourth-difference RMS from .010946 to .003344. However, that run
becomes unusable around t=.6 at dt=.01. Reducing dt fourfold still leads to a
non-finite solution by t=.625.

A homogeneous linear control, without convection or lift, decays for the
sampled perturbation in both formulations over t=.25, .5 and 1. This rules out
that particular linear test as an explanation of the rapid failure; it does
not prove stability of the entire nonnormal linear operator or isolate every
coupling in the nonlinear failure.

The weak boundary-compatible test remains finite through t=2 at dt=.01, but
the unfiltered upstream fourth-difference diagnostic is **.003328**, compared
with **.003012** for the matched original run: about 10.5% worse. It is not a
ripple fix despite the modest improvement of the initial residual.

Independent outer-wall errors at t=2 are 2.89e-9 for this candidate and 2.76e-9
for the matched baseline. Neither meets the existing 1e-9 wall criterion at
this checkpoint. The candidate's hole-wall error is 3.27e-10 and its maximum
relative flux error is 5.61e-11. Boundary accuracy must remain an explicit gate
for any eventual correction.

The stronger cell-scale test reduces the initial residual RMS to .39941 but
shows runaway growth by t=2.4 at dt=.01. A smaller step changes the onset and
severity, but does not establish a useful correction. At dt=.0025, t=3,
upstream vorticity RMS is 22.14 and D4y .401. Raising quadrature from 2.5 to 4
still gives RMS 22.72 and D4y .343, with maximum volume speed 9.79. Thus neither
control produces a usable ripple-free flow. These runs remain finite through
their stopping time; they must not be described as proven blow-ups at dt=.0025.
The weak boundary-compatible test costs about 11.55 ms/step including the
probe/diagnostic synchronization, versus 11.29 ms/step in the matched
zero-strength harness. Curl setup added 10.89 seconds, including compilation.
These are short-run measurements, not optimized production timings.

## Consequence for the persistent goal

The direct curl-test augmentation is not being promoted. The useful result is
that improving the frozen force projection is insufficient: a correction also
needs controlled dynamics. Analytic derivative and diagnostic code is retained
for a subsequent formulation with an energy/stability argument and explicit
forcing consistency.

An established direction is least-squares vorticity-residual stabilization,
which tests with the curl of the spatial differential operator and adds a
nonnegative residual quadratic form in the stationary Oseen problem. Its
published analysis does not automatically validate an unsteady, global BSPF
implementation; that extension still needs independent checks.
[Ahmed et al., WIAS preprint 2740](https://www.wias-berlin.de/preprint/2740/wias_preprints_2740.pdf).

## Reproduction

Use the CUDA/GPU Python environment from the main BSPF investigation. The
control reads `build/immersed_flow/re200_ripple_study/source_audit_fields.npz`
for the independent initial local-PDE reference. Run sequentially:

```sh
python examples/pde/re200_diagnostics/curl_derivative_audit.py
python examples/pde/re200_diagnostics/curl_linear_control.py
python examples/pde/re200_diagnostics/run_curl_controls.py
python examples/pde/re200_diagnostics/summarize_curl_residual.py
```

All changes are isolated under `examples/pde/re200_diagnostics`. The original
production implementation and time integrator are unchanged. These are research
controls; no production regression or manufactured-accuracy result is claimed
for the new form.

- [Comparison and checks](data/re200_ripple/curl_residual/comparison.json)
- [Derivative checks](data/re200_ripple/curl_residual/derivative_checks.json)
- [Linear control](data/re200_ripple/curl_residual/linear_control.json)
- [Raw-field comparison at t=2](data/re200_ripple/curl_residual/t2_comparison.png)
