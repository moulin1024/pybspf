# Self-consistent Landau damping study on an open finite interval

The [notebook](../examples/pde/landau_open_1d.ipynb) evolves nonlinear electron
Vlasov–Poisson, f_t+v f_z-E f_v=0, phi_zz=int(f)dv-1, E=-phi_z.
Ions are fixed at unit density. The normalized electron thermal speed, plasma
frequency and Debye length are one. The phase-space rectangle is [-40,40] x
[-6,6], and the simulated time is 0..12.

## Nonperiodic model

Both endpoint potentials are grounded. Incoming electrons are supplied by a
Maxwellian reservoir: v>0 on the left, v<0 on the right. Outgoing traces evolve
freely. On velocity faces, the sign of -E at each spatial quadrature point
selects Maxwellian inflow or computed outflow. There is no periodic wrapping.
Grounded potential conditions are physical boundaries, not transparent ones.
The Maxwellian is normalized using the BSPF finite-velocity integral so the
unperturbed state is an exact numerical equilibrium.

The initial perturbation is a compact cos^8 window of half-width 24, centered
at z=3, modulating a carrier k=0.5 with amplitude 1e-3. A window-weighted mean
correction removes its net charge using BSPF integration. It vanishes near both
reservoirs. The solver retains the nonlinear electric force, although this
amplitude is small. A localized packet contains a range of wavenumbers, so its
field energy need not follow one exponential envelope.

## BSPF Poisson and kinetic evolution

`poisson_dirichlet` solves phi''=source using first and second BSPF primitives.
Writing G=int(source), H=int int(source) with zero left constants, the affine
correction enforces both endpoint potentials. For zero potentials,

    phi = H - (z-z_left) H(z_right)/L,
    E = -G + H(z_right)/L.

No Poisson differentiation matrix is inverted. The immutable Vlasov–Poisson
plan precomputes these primitive maps and the BSPF velocity moment weights.
`integrate_vlasov_poisson` updates density and field at every RK4 stage.
Tensor weak transport and upwind fluxes extend the open parallel-kinetic
infrastructure. Multiplication by the spatially varying electric force is
projected at resolved spatial quadrature points. Evolving f-fM preserves the
homogeneous equilibrium exactly. No filtering or positivity clipping is used.

## Diagnostics and interpretation

Electric energy W=int(E²)/2 is compared with integrated electron work
-int(E int(vf)dv)dz. Perturbation free energy combines W with the relative
Maxwellian entropy integral int int[f log(f/fM)-f+fM]. For small perturbations
its particle part is int int[(f-fM)²/(2fM)]. This tracks field energy moving
into particle perturbations without claiming irreversible entropy production.

Boundary relative-free-energy fluxes are measured at all four phase-space
faces. Every spatial/velocity integral and time antiderivative in these
diagnostics uses BSPF. Velocity truncation can introduce charge-loss effects;
balances are checked rather than presumed exact. Grounded endpoints imply
zero imposed voltage work. Small-ratio series avoid cancellation in the
relative entropy near equilibrium.

## Measured validation

Default: 129x129 points, degree 7, 24 spline functions per axis, nine endpoint
samples, Gauss order 8, dt=0.02, 121 output states. Corrected CPU BLAS/JAX:

| Quantity | Result |
| --- | ---: |
| Minimum f, no clipping | 6.026e-9 |
| Grounded potential residual | 1.237e-18 |
| Direct primitive vs precomputed field map | 1.104e-15 |
| Differentiated Gauss-law residual | 2.786e-8 |
| Final / initial electric energy | 0.131694 |
| Initial perturbation free energy | 1.446578e-5 |
| Free energy lost through boundaries by t=12 | 1.456% |
| Relative free-energy balance residual | 2.251e-9 |
| Field–particle exchange balance residual / initial W | 4.190e-9 |
| 65 vs 129 spatial field difference / peak E | 3.090e-6 |
| 129 vs 257 velocity field difference / peak E | 1.790e-9 |
| Halved-step field difference / peak E | 7.464e-9 |
| Reservoirs at ±60 vs ±40: field difference / peak E | 2.015e-3 |

The larger-domain run uses 193 spatial samples to preserve spacing and the
same localized perturbation. It compares fields on the shared [-40,40]
interval. The differentiated Gauss-law check includes the approximation from
refitting primitive samples for numerical differentiation; field construction
itself uses the first primitive directly.

The electric field loses about 86.8% of its initial energy while only about
1.46% of total perturbation free energy exits the boundaries. Most of the
field energy becomes particle free energy, and velocity-space phase mixing is
visible. These checks support Landau-type collisionless damping rather than
field decay dominated by escape. They do not establish a single homogeneous
Fourier-mode damping rate. Weakly damped long wavelengths contribute to the
late-time packet response. Boundary effects remain measurable, and longer-time
studies require additional velocity refinement and recurrence checks.

Regression tests verify nonzero/batched Dirichlet Poisson integration,
electric-field signs, equilibrium preservation under JIT, nonlinear field
updates and time-step refinement. The notebook executes the physical balances
and space/velocity/time/domain comparisons as assertions.

## Animation

`scratch/render_landau_open_mp4.py` executes the notebook's main solve and
Poisson/free-energy checks at 301 output times, preserving dt=0.02. It exports
`examples/pde/results/landau_open_phase_space.mp4` (1280x800, 25 fps, 12.04 s).
Panels show the computed distribution perturbation (fixed color scale in units
of 1e-4), electric field and free-energy exchange with boundary transfer. Frames
are computed states, not temporal interpolations. The denser output-time check
has relative free-energy and exchange residuals of 2.25e-9 and 5.41e-9.
