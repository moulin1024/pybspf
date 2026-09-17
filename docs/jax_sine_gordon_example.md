# Sine–Gordon kink–antikink collision

The [notebook](../examples/pde/sine_gordon_1d.ipynb) solves
`u_tt = u_xx - sin(u)` on [-4,4] until t=10. With c=0.6,
γ=1/sqrt(1-c²), collision time tc=5 and center xc=0.4, the reference is

    u(x,t) = 4 atan(sinh(γ c (t-tc)) / (c cosh(γ (x-xc)))).

This is an exact nonlinear two-soliton scattering solution, not a sum of two
single kinks or a breather. It is continuous through collision: at t=tc the
field is zero, but the velocity is nonzero. The regression test differentiates
the expression twice to verify the PDE, including at the collision point.
See [Durham's two-soliton illustrations](https://maths.dur.ac.uk/users/P.E.Dorey/SOLITONS_2025_26/SGpictures/SG_Kinkantikink.html).

Initial displacement/velocity and both moving Dirichlet traces come from this
reference. The spatial shift yields unequal endpoints (maximum mismatch 1.539).
The finite interval has significant boundary energy transfer; it is not a
periodic or negligible-tail approximation. Numerical fields are evolved from
initial data by the reusable solver, not replaced by the analytic solution.

## Infrastructure

`integrate_sine_gordon` accepts an unconstrained first-derivative `Galerkin1D`
system and returns displacement and velocity histories. Full BSPF cardinal
trial functions are evaluated at resolved Gauss points. After partitioning
interior and boundary samples, the weak equation is

    M_ii q_tt = -K_ii q - K_ib g - Q_i.T W sin(Q_i q + Q_b g) - M_ib g_tt.

JAX JVPs supply g_t and g_tt. The mass factorization is performed before the
RK4 time loop. Endpoint displacement/velocity are prescribed at every stage
and output; initial data must be compatible. No exact kink information enters
the solver except through user-specified initial and boundary callables.

This is dense small-1D infrastructure. It assumes a first-derivative weak form,
real compatible data and twice differentiable boundary callables. RK4 is
explicit, neither unconditionally stable nor exactly energy conserving;
callers must resolve the fastest spatial wave modes. x64 is enabled by the
notebook, not by package import.

## Measured validation

CPU/JAX with the corrected local BLAS build; degree 7, 24 spline functions,
9 endpoint samples, 129 spatial samples, Gauss order 8, dt=0.002, 201 outputs.
Maximum errors include all stored times, including collision, and grid samples.

| Quantity | Result |
| --- | ---: |
| Displacement error, N=65 | 3.046e-7 |
| Displacement error, N=129 | 1.307e-9 |
| Velocity error, N=129 | 4.090e-8 |
| Dirichlet residual | 0 |
| Displacement change on halving dt | 4.355e-11 |
| Displacement change, Gauss 8 to 10 | 6.515e-11 |
| Reference energy quadrature, 256 vs 512 points | 7.816e-14 |
| Energy error against independent analytic-field quadrature | 4.498e-9 |
| Integrated numerical boundary-power balance residual | 9.594e-8 |

The energy density is u_t²/2 + u_x²/2 + 1-cos(u), so E'=[u_t u_x].
Numerical energy rises from 15.401856 to 19.997198 near collision and returns
to 15.401856 as the solitons separate. Independent Gauss–Legendre quadrature
integrates the analytic energy density over the same finite interval. The
balance check integrates measured boundary power with BSPF on the 201-point
time grid; its residual also includes time-quadrature and boundary-derivative
error. Two spatial grids demonstrate refinement, not an exponential-rate claim.

The notebook plots field snapshots, an energy-density space–time diagram,
field error and energy transfer. Regression tests retain single-kink coverage
and add the analytic two-soliton PDE residual and numerical evolution through
collision. The notebook remains registered in the executable example suite.

## Animation

`scratch/render_sine_gordon_mp4.py` reads the notebook's setup and solver call,
computes 401 states with dt=0.001, and writes
`examples/pde/results/sine_gordon_collision.mp4` (25 fps). It shows numerical
and exact fields, energy density and finite-interval energy. The animation's
maximum field error is 1.334e-9; no interpolated or reference-only frames are
used. The script also runs the notebook's energy-reference and balance checks.
