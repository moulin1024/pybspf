# Boundary-stable 2D KH: compatible BSPF and physical layer enrichment

The boundary stripe instability seen in the previous `96 x 80` weak KH result
is removed in the tested runs without a sponge, filter, reduced-order boundary
stencil, increased viscosity, or a change to the prescribed velocity boundaries.
The implementation is `jax/src/bspf_jax/stream_navier_stokes.py`.

## What the controlled experiments established

1. The original strong collocation method has a separate semidiscrete inflow
   instability, including in constant advection. Time refinement does not cure
   that operator. The earlier one-dimensional Galerkin test addressed this.
2. Changing momentum and pressure to paired weak forms preserved smooth-PDE
   accuracy, but left grid-scale boundary stripes in nonlinear KH. Small weak
   divergence was not sufficient evidence of a correct resolved flow.
3. A compatible streamfunction Galerkin method made the *continuous* velocity
   reconstruction pointwise divergence-free. On a uniform `64 x 64` grid, it
   **still produced stripes at t=6**. Thus unresolved weak divergence was not
   the only source of the observed KH boundary oscillations.
4. Keeping the same compatible method, same viscosity, same forcing, and same
   four fixed boundaries, adding six one-dimensional exponential trial
   functions removed the stripes. This isolates approximation of the thin
   boundary layer as the decisive change in this comparison.

At the outflow, a vortex-induced transverse velocity must return to the
prescribed value zero. The convection-diffusion scale is approximately
`delta=nu/|U|`, about `0.002` here. Uniform x spacings are about `0.095` at 64
nodes and `0.063` at 96 nodes. The smooth/global trial representation cannot
adequately resolve this thin transition at those sizes. The resulting
oscillations contaminate the interior, including the inflow side.

An independent continuous scalar PDE checks this mechanism:

\[
u'-0.002u''=1,\quad u(0)=u(1)=0,
\qquad u(x)=x-\frac{e^{(x-1)/0.002}-e^{-1/0.002}}{1-e^{-1/0.002}}.
\]

At 48 nodes, ordinary BSPF weak Galerkin has max error `0.2832`; adding the
analytic layer function gives `3.38e-14`. The forcing is specified analytically,
not manufactured using the discrete matrix. See
`scratch/check_boundary_layer_enrichment.py` and
`build/kh_stream/layer_accuracy/`.

The KH solution has no analytic reference. Its layer interpretation is supported
by the controlled enrichment experiment, physical scale, resolved profiles,
and grid comparisons; it is not an assertion that every possible KH boundary
artifact has this cause.

## Discretization and direct solve

Represent the velocity by

\[
(u,v)=(U(y),0)+(\partial_y\psi,-\partial_x\psi).
\]

The perturbation streamfunction is tensor-product BSPF. Both its value and
normal derivative vanish on each wall, so the original fixed velocity is
preserved. Mixed derivatives commute in the continuous reconstruction, hence
`div u=0` at arbitrary physical points, not just in a finite test space.
This is the compatibility principle used in divergence-conforming methods;
see [Evans and Hughes, divergence-conforming B-splines](https://doi.org/10.1142/S0218202513500139).
The particular BSPF-plus-Fourier/enrichment implementation here is our own
construction, not a claim to inherit all results of that paper automatically.

Use curl test functions. Pressure gradients then have zero contribution by
integration by parts. This **eliminates pressure from the 2D velocity evolution**;
it does not secretly replace the pressure Schur complement by a scalar
Laplacian. A physical pressure field is not currently returned.

For one-dimensional streamfunction bases define

\[
M=\int B^TB,\quad K=\int B'^TB',\quad J=\int B''^TB''.
\]

The kinetic mass and viscous bilinear form are

\[
\mathcal M=K_x\otimes M_y+M_x\otimes K_y,
\]
\[
\mathcal H=J_x\otimes M_y+2K_x\otimes K_y+M_x\otimes J_y.
\]

After one-dimensional mass orthonormalization and stiffness diagonalization,
`M=I`, `K=diag(lambda)`. Each RHS evaluation inverts the kinetic mass through

\[
\dot a_{ij}=F_{ij}/(\lambda_i^x+\lambda_j^y).
\]

Only one-dimensional matrices and tensor arrays are used. There is no global
2D matrix, iterative pressure solve, or refinement. The time integrator is
explicit RK4, including viscosity. Implicit Stokes integration is not claimed.

Convection uses the rotational force `(v*omega,-u*omega)` with
`omega=-Delta psi-U'`. Its contribution to unforced homogeneous kinetic energy
is zero algebraically. The viscous form is positive semidefinite, yielding

\[
\frac{d}{dt}\frac12 a^T\mathcal Ma=-\nu a^T\mathcal Ha\le0.
\]

The actual KH experiment has a nonzero fixed lift and a maintaining force;
physical energy exchange and KH growth are allowed. Monotone total-energy
decay is not asserted for that driven problem.

## Enrichment, without damping or order reduction

In x, add

\[
e^{-(x-x_L)/\delta_k},\quad e^{-(x_R-x)/\delta_k},
\qquad \delta_k\in\{0.002,0.008,0.032\}.
\]

These are six **one-dimensional basis functions**, tensored with the y basis.
They are unknown solution components, not a damping coefficient or a buffer
region. The differential equation, physical viscosity, and boundary conditions
are unchanged. No original streamfunction trial function is discarded. The
BSPF component retains degree 13, 32 splines, q=9, Cheb12/window16. The output
nodes remain uniform; Chebyshev-distributed physical nodes are not required.

Quadrature is subdivided near the walls to integrate the new functions
accurately. These extra integration points are not time-evolution grid nodes.
Analytic derivatives of the exponentials and spline/Fourier basis are used.

The final setup uses weighted QR of the full enriched space before imposing
the clamped boundary constraints, rather than forming mass normal equations. This matters because some exponentials become nearly represented
by BSPF as n grows. QR retains all columns; no singular-vector truncation or
filter is applied. BSPF/exponential cancellation is combined in MPFR113 before conversion to
float64, followed by well-conditioned reorthogonalization. Runtime is JAX float64.
The pre-QR pilot results differed only at small roundoff levels; final setup
and regression checks are recorded separately. The final 96x80 rerun differs
from the pilot by `1.69e-10` in nodal velocity; its mass-orthogonality defect is
`2.50e-14` and it stores approximately 4.23 MiB of one-dimensional factors
(excluding JAX runtime and field work arrays).

The original two-component nodal velocity space is replaced by derivatives of
a compatible scalar BSPF space. Thus the claim is retained high-order PDE
accuracy and exact incompressibility, not identical velocity coefficients to
the previous collocation solver.

## Validation results

All KH cases use domain `[-3,3] x [-1,1]`, nu=.002, thickness=.12, amplitude=.03,
wavelength=1.5, fixed `(tanh(y/.12),0)` on all four walls, and a constant load
maintaining the base. No run in this report uses a sponge.

- At `t=6`, continuous physical reconstruction comparisons (integrated over a
  common quadrature grid that resolves the thin layers):

  | comparison | full relative velocity L2 | central relative velocity L2 |
  |---|---:|---:|
  | 64x64 vs 96x80 | 0.5298% | 0.1798% |
  | 96x80 vs 128x96 | 0.3173% | 0.03560% |

  The norm is relative to the perturbation, excluding the fixed base. Central
  means `|x|<2, |y|<.5`. These are grid differences, not exact PDE errors. The
  full-domain result still has finite spatial error, especially near boundaries.
- In the final 96x80 rerun, maximum sampled pointwise divergence was
  `1.15e-14`, fixed-boundary velocity error `1.49e-13`, and nodal maximum speed
  `1.3862`. The 13 targeted stream, weak, and original NS regression tests pass.
- On 64x64, dt=.004 vs .001 at t=6 gives relative kinetic-norm difference
  `8.03e-11` and nodal velocity difference `1.33e-10`. The removal of stripes is
  not caused by increasing temporal damping.
- The enriched 64x64 case continued to `t=12` without the former stripe pattern.
  Over the saved 0..12 samples, nodal max speed stayed below 1.93 and ended near
  1.062. This is a finite-time experiment, not an all-time/all-flow guarantee.
- Smooth analytic manufactured NS, including nonzero pressure gradients,
  retains high-order accuracy. The enriched 64x64 t=.04 error was `6.03e-12`.
  The final MPFR/QR setup gives `8.997e-13` at 96x96 for the same analytic test.
  A pressure-gradient-only force projects to roundoff. This test is independent
  of the discrete momentum operator.
- `jax/tests/test_stream_navier_stokes.py` checks continuous pressure-gradient
  removal, mixed derivative cancellation at off-grid points, clamped wall
  values, viscous energy balance, constant-advection neutrality, polynomial
  reproduction, and MPFR second derivatives (including odd/even node counts).

Large wall vorticity is expected for the resolved layer: a transverse velocity
change of order .8 over thickness .002 gives a gradient of order 400. Figures
explicitly mark vorticity color clipping. The velocity profiles distinguish a
resolved sharp layer from oscillations extending into the domain.

## Usage and artifacts

```sh
python -m pip install -e 'jax[weak-ns]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src:scratch \
 MPLCONFIGDIR=/tmp/pybspf-mpl python scratch/run_kh_stream.py \
 --nx 96 --ny 80 --T 6 --dt .002 --layers .002 .008 .032 \
 --out build/kh_stream/final96 --render
```

`--layers` without values disables enrichment for a controlled comparison.
Use `--resume <checkpoint.npz>` only with matching basis parameters. Checkpoints
are written every .2 time units so interruption does not discard all progress.
Currently the KH helper uses the specified `[-3,3] x [-1,1]` benchmark domain;
the core unforced/forced solver accepts other rectangular uniform grids.

Useful outputs under `build/kh_stream/`:

- `boundary_fix.png`: original weak vs enriched compatible KH, same 96x80 grid.
- `boundary_cuts.png`: actual reconstructed velocity and resolved thin layer.
- `final96/kh_stream.mp4`: movie from saved computed states.
- `long_time.png`: 64x64 fields at t=6 and t=12.
- `grid_comparison.json`, `grid96_128.json`, `time_comparison.json`.

The formulation and experiments here are two-dimensional. A 3D compatible
velocity-space formulation has not been implemented by this change.
