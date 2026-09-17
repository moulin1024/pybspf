# Boundary-driven Alfvén waves

The [notebook](../examples/pde/alfven_1d.ipynb) solves linear ideal shear-Alfvén
waves on a straight guide field with static density and uniform equilibrium
pressure:

    rho(z) xi_tt = (B0²/mu0) xi_zz,
    u_perp = xi_t, b_perp = B0 xi_z.

Normalized units are L=B0=mu0=rho_L=1, z in [0,1], t in [0,4]. Density is
`1 + 1.5*(1+tanh((z-0.5)/0.08))`. Initial displacement and velocity vanish.
The left displacement is A sin^8(pi t/Td) for 0<t<Td, and zero otherwise;
A=1e-3, Td=0.5. The right displacement is zero throughout. The compact pulse
uses sin^8 rather than the initially proposed sin^4 for C7 switch-on/off
regularity. This improves spatial resolution of the propagating fronts.

Endpoint values differ by up to A while driving. After driving, both endpoints
are fixed. Their spatial derivatives need not agree, and waves reflect from
physical walls. No periodic wrapping or independent magnetic boundary data is
imposed. The smooth density gradient causes internal scattering. This is a
linear wave model, not full compressible or nonlinear MHD.

## Reusable JAX infrastructure

`plan_alfven` evaluates density at resolved Gauss nodes and forms

    M = Q.T W rho Q,
    K = (B0²/mu0) G.T W G.

`integrate_alfven` evolves interior displacements and velocities using RK4.
With prescribed endpoints g(t), the interior equation includes both -K_ib g
and -M_ib g_tt. JAX differentiation supplies boundary velocity/acceleration.
Callers must provide compatible initial data, increasing times, and a stable
step for the chosen grid and Alfvén speed. The dense operators target modest
1D problems; this is not an adaptive or unconditionally stable solver.

`alfven_energy` integrates the physical kinetic/magnetic energies at assembly
quadrature points. `alfven_boundary_power` independently evaluates
`[B0²/mu0 xi_t xi_z]_left^right` using full BSPF endpoint derivatives.

The shared quadrature geometry was extracted into `_quadrature_rule`, retaining
the existing Galerkin integration rule. `interpolate(..., derivative=k)` now
evaluates derivatives of the original spline/Fourier interpolant off-grid;
it does not fit a second interpolant to derivative samples.

## BSPF integration diagnostics

The notebook uses `integrate` for spatial energy and `antiderivative` for
accumulated boundary work. Squaring the fields generates finer scales. The
129-point state is therefore evaluated on 513 points before constructing and
integrating the energy density. Differentiating the original interpolant at
those points is essential: refitting nodal magnetic derivatives introduces an
additional approximation. No new PDE solve is performed on the diagnostic grid.

The resolved weak energy is retained as a separate diagnostic. Its drift tests
time stepping; its difference from the BSPF spatial integral tests the energy
integration. Boundary power is computed from the fields, not from energy
increments. After the drive stops both endpoints do no work, so energy should
remain constant even while the waves reflect inside the cavity.

## Independent reference and measured checks

For uniform density, the same driven problem has the method-of-images solution

    xi(z,t) = sum_m [g(t-2m-z) - g(t-2m-2+z)].

The pulse is causal; four image pairs suffice over t<=4. This verifies boundary
driving and repeated wall reflections independently of the BSPF discretization.
The nonuniform case is assessed with mesh, time and quadrature refinement,
not assigned an unverified analytic reference.

Default: 129 points, degree 7, 24 spline functions, 9 endpoint samples, Gauss
order 8, dt=0.0005 and 401 output times. CPU/JAX with corrected local BLAS:

| Check | Result |
| --- | ---: |
| Dirichlet displacement residual | 0 |
| Uniform-cavity max error / A | 2.792e-8 |
| 65 vs 129 max displacement difference / A | 5.814e-4 |
| 129 vs 257 difference / A | 2.432e-6 |
| Halved-step displacement change / A | 2.622e-8 |
| Gauss 8 vs 10 change / A | 2.950e-11 |
| Injected energy | 1.653904465e-5 |
| Max energy-minus-work residual / max energy | 1.817e-7 |
| BSPF vs weak-quadrature energy / max energy | 3.044e-10 |
| Relative weak-energy drift after driving | 6.009e-10 |

The boundary-work residual includes temporal integration and endpoint-gradient
errors. Spatial comparisons do not establish a universal exponential rate;
the compact pulse has finite C7 regularity. The notebook contains assertions
for all these checks and is registered in the executable example suite.

## Animation

`scratch/render_alfven_mp4.py` executes the notebook's setup, main solve and
BSPF energy checks, then renders all 401 computed states to
`examples/pde/results/alfven_boundary_driven.mp4` (1280x800, 25 fps, 16.04 s).
The panels show displacement with independently prescribed endpoints,
transverse magnetic perturbations and energy versus integrated boundary work.
The highlighted band marks the central density transition; magnetic plot limits
cover the full computed history, including peaks during wall reflections.
