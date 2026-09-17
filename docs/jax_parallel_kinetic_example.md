# Open parallel kinetic dynamics (1z1v)

The [notebook](../examples/pde/parallel_kinetic_1d.ipynb) solves

    f_t + v f_z + a f_v = 0,  a=q E_parallel/m=0.3,

on z in [0,1], v in [-2.5,2.5], t in [0,1.2]. The field is prescribed and
constant; this is collisionless parallel transport, not a self-consistent
Vlasov–Poisson or full gyrokinetic model. No Poisson solve, collisions, magnetic
mirror force or perpendicular dynamics are included.

## Physical boundaries and reference

Two unequal warm Gaussian phase-space pulses are initially centered outside
opposite ends of the spatial interval, at (-0.2,1) and (1.2,-0.8). Their
restriction supplies initial interior values; incoming traces supply the
reservoir pulses. Their exact accelerated continuation is

    f(z,v,t) = F(z-vt+a t²/2, v-a t).

The solver only receives initial values and inflow traces. It evolves interior
and outgoing values numerically. The two streams interpenetrate without
collisions, shift in velocity under the applied field, and escape.

At z=0 only v>0 is incoming. At z=1 only v<0 is incoming. Since a>0, the lower
velocity face is incoming and the upper face outgoing. Velocity-boundary
contributions are retained in the balances even when small. Endpoints are
never identified or wrapped; opposite spatial boundary distributions differ
by as much as 0.9992 in this test. The reservoirs can be changed independently
through the `inflow(t,z,v)` callable.

## Numerical infrastructure

`plan_parallel_kinetic(z_plan, v_plan, acceleration=..., quadrature_order=8)`
forms tensor-product resolved BSPF weak transport operators. Each axis has
mass M=Q.T W Q and weak derivative S=G.T W Q. Its symmetric part is set to
one half of the endpoint surface matrix, preserving integration by parts to
roundoff. Velocity multiplication is projected with the same quadrature and
split into positive and negative parts for the spatial boundary fluxes.

`integrate_parallel_kinetic` advances the (nz,nv) coefficients with JAX RK4.
Incoming fluxes use prescribed data; outgoing fluxes use the computed field.
Velocity inflow changes sides with the sign of acceleration, and vanishes for
zero acceleration. Both signs and zero are regression tested under outer JIT.
The homogeneous-inflow operator is dissipative in the tensor mass norm; a
regression test checks norm decay and nonzero freely evolving outgoing traces.

Inflow is weakly imposed, not assigned exactly to nodal endpoint values.
Actual incoming trace errors are therefore measured. The method has no
positivity limiter, filter or clipping. Explicit time steps must resolve the
transport stability limit. Dense axis operators are intended for modest 1z1v
examples, not high-dimensional gyrokinetic simulations.

## BSPF integral diagnostics

Velocity moments and spatial integrals use `bspf_jax.integrate`; accumulated
boundary transfer and electric work use `antiderivative` on output times.

    N' = int v(f_left-f_right) dv + a int(f_vmin-f_vmax) dz
    K' = a int int v f dz dv + 1/2 int v³(f_left-f_right) dv
         + a/2 int(vmin² f_vmin-vmax² f_vmax) dz

Mass is normalized to one in K=int int v²f/2. All fluxes use actual computed
traces. Residuals consequently include weak boundary-trace and BSPF integration
errors as well as time stepping. Particle number and kinetic energy are not
assumed constant in an open, electrically driven system.

## Measured checks

Degree 7; 16/24 spline functions in z/v; 9 endpoint samples; 65x129 nodes;
Gauss order 8; dt=0.0005; 121 output states. CPU/JAX with corrected local BLAS:

| Quantity | Result |
| --- | ---: |
| Maximum distribution error against characteristics | 2.702e-8 |
| Maximum incoming trace error | 2.702e-8 |
| Minimum distribution value, without clipping | -1.297e-8 |
| Coarse 33x65 reference error | 9.134e-5 |
| Change on halving the time step | 1.495e-8 |
| Change from Gauss order 8 to 10 | 7.179e-12 |
| Relative particle-number reference error | 2.983e-10 |
| Relative particle-balance residual | 4.112e-9 |
| Relative kinetic-energy balance residual | 5.678e-9 |

Time error is visible at this resolution; halving dt reduces the reference
error to 1.442e-8. No claim of universal positivity or exponential convergence
is made. The notebook executes these assertions and plots phase-space slices,
number density, boundary particle transfer and reference error. Five focused
regression tests cover acceleration signs, affine characteristic solutions,
constant preservation, validation and homogeneous-inflow dissipation.

## Phase-space animation

`scratch/render_parallel_kinetic_mp4.py` reads the notebook setup, solver and
balance checks. It computes 301 states at the same internal dt=0.0005 and
exports `examples/pde/results/parallel_kinetic_phase_space.mp4` (1280x800,
25 fps, 12.04 seconds). Panels show f(z,v), number density and particle balance;
arrows identify incoming/outgoing spatial characteristics. The fixed color
scale is 0..1, while the numerical minimum (including negative undershoots) is
reported in each frame. No numerical clipping or temporal interpolation is
used. Maximum distribution error is 2.702e-8; particle and energy balance
residuals remain below 6e-9 on the denser output-time grid.
