# BSPF KH with open vertical faces

The streamfunction JAX backend supports `x_boundary="open"`. The old fixed
boundary condition remains the default. Horizontal faces keep their prescribed
velocity `(tanh(y/.12), 0)`. Both vertical faces are open, because the upper and
lower shear streams travel in opposite directions. This changes the physical
boundary-value problem; it is not another stabilization of the old fixed walls.

Let `n` be the outward normal, `s=u·n`, and `u_ref=(tanh(y/.12),0)` (zero when no
lift is supplied). On the vertical faces the physical Laplacian traction is

\[
 \nu\partial_n u-pn=\min(s,0)(u-u_{ref}).
\]

For local outflow this is zero traction: the transverse velocity is free and a
vortex need not collapse to zero transverse velocity at the edge. Local inflow
couples weakly to an external shear-flow reservoir. This is a Robin condition,
not an exact Dirichlet inflow and not a mathematically reflection-free boundary.
The traction convention uses `nu*partial_n(u)`, consistent with the volume
`nu*Laplacian(u)` weak form, rather than symmetric-stress traction.

The sign-dependent boundary term is motivated by the modified-traction
backflow conditions discussed by
[Dong and Shen (2015)](https://www.math.purdue.edu/~shen7/pub/Don.S15.pdf).
Our incoming reference-flow variant and streamfunction implementation are
specified here; we do not implement that paper's pressure-correction algorithm.
For zero reference velocity the boundary contribution to kinetic energy is

\[
 -\tfrac12s|u|^2+\min(s,0)|u|^2=-\tfrac12|s||u|^2\le0.
\]

For incoming flow with nonzero reference it is
`-|s| |u-u_ref|²/2 + |s| |u_ref|²/2`: the reservoir can supply energy.
This is a semidiscrete boundary energy balance, not unconditional stability of
explicit RK4 or a claim that the driven KH flow has decreasing energy.

## Compatible direct formulation

The original BSPF scalar space is retained. The x direction has no essential
streamfunction constraints. The y direction still has `psi=psi_y=0` at its two
ends. Thus the perturbation velocity `(psi_y,-psi_x)` is divergence-free
pointwise, vanishes on horizontal faces, and is unconstrained on vertical faces.
The net perturbation flux through each vertical face is zero because psi has
equal values at the two horizontal endpoints. This is appropriate for the
symmetric zero-net-flux shear benchmark; arbitrary prescribed net throughflow
would require a different lift.

The kinetic mass remains `Kx⊗My + Mx⊗Ky`. One-dimensional orthonormalization and
stiffness diagonalization still yield division by `lambda_x+lambda_y` at every
stage. The constant x mode is permitted; clamped y makes the total denominator
positive. No global 2D matrix, iterations, or refinement are introduced.

Rotational convection absorbs kinetic energy into total pressure
`P=p+|u|²/2`. Therefore its boundary load must be

\[
 \min(s,0)(u-u_{ref})-\tfrac12|u|^2 n.
\]

Omitting the second term would impose the wrong physical traction. Boundary
loads are integrated against the same curl test functions as the volume load.
Pressure is eliminated from the velocity evolution; no physical pressure field
is reconstructed. `stream_ns_boundary_load` accepts an **additional** physical
traction load, including when validating nonzero prescribed pressures.

The open KH runs use **no exponential enrichment, sponge, filter, or artificial
viscosity**. They keep degree-13 BSPF and physical viscosity `.002`. The shear is
maintained with the same constant base-balancing load as the fixed-wall runner.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=scratch \
 MPLCONFIGDIR=/tmp/pybspf-mpl python scratch/run_kh_stream.py \
 --nx 96 --ny 80 --T 12 --dt .002 --x-boundary open --layers \
 --out build/kh_stream/open96 --render
```

`--layers` with no arguments disables exponential enrichment. Output nodes are
uniform; quadrature is internal. The checkpoint of a fixed-boundary run is not
compatible with the open basis. `boundary_linf` now measures only the boundaries
where velocity is prescribed; `vertical_boundary_v_linf` records the free
transverse velocity separately.

Tests in `packages/models/tests/test_stream_open_boundary.py` independently check the
boundary kinetic-energy flux, pointwise divergence, retained horizontal
conditions, and a continuous manufactured stationary Navier–Stokes solution
with nonzero open-face velocities and pressure. The manufactured traction,
advection, diffusion, and forcing are analytic, not formed by calling the
numerical RHS. Fixed-boundary regression tests are also retained.

## External absorbing extension

A later option keeps the region of interest `[-3,3]×[-1,1]` intact and extends
the computational domain to `[-5,5]×[-1,1]`. The term

\[
 f_{abs}=-\sigma(x)(u-U),\qquad U=(\tanh(y/.12),0)
\]

is supported only in `3<|x|<5`. It models relaxation to an external background
flow. The equations inside the region of interest contain no absorption, but
incompressible pressure coupling can still transmit effects from the exterior;
zero local coefficient does not imply an identical interior solution.

For `s=(|x|-3)/2`, use `sigma=sigma_max*S(s)`, where

\[
 S(s)=\frac{e^{-1/s}}{e^{-1/s}+e^{-1/(1-s)}}\quad(0<s<1),
 \qquad S=0\ (s\le0),\quad S=1\ (s\ge1).
\]

This C-infinity ramp introduces no derivative jump at the absorption interface.
The initial perturbation keeps the exact original analytic formula for `|x|<=3`.
Outside it, the analytic continuation is multiplied by `1-S(|x|-3)`, so it is
smooth and vanishes for `|x|>=4`. The projected finite-dimensional initial
velocity need not be identical between two different computational domains.

Absorption is integrated against the curl test space, preserving exact
incompressibility. With mass-normalized y basis its modal load is

\[
 -M_\sigma a\Lambda_y-K_\sigma a,\qquad
 M_\sigma=\int\sigma B_x^TB_x,\quad K_\sigma=\int\sigma B_x'^TB_x'.
\]

Its perturbation-energy work is exactly
`-integral sigma*|u-U|² <= 0` at quadrature level. Only two additional 1D
matrices are stored; time integration still uses the same direct diagonal
inertia solve. The explicit RK4 time step must also resolve the damping rate.
No artificial viscosity, modal filter, or lower-order closure is used.
Smooth compact support is not global analyticity, so retaining BSPF order does
not by itself prove exponential convergence of the whole extended problem.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=scratch \
 MPLCONFIGDIR=/tmp/pybspf-mpl python scratch/run_kh_stream.py \
 --nx 160 --ny 80 --T 12 --dt .002 --x-boundary open --layers \
 --extension 2 --sponge-strength 4 \
 --out build/kh_stream/extended160_s4 --render
```

The x spacing is `10/159=0.0628931`, slightly finer than the original
`6/95=0.0631579`. The external boundaries remain open; the horizontal fixed
velocity boundaries are unchanged. The movie marks `x=±3` with dashed lines
and shades the extensions. Absorption is now an explicit modeling option, not
part of the default solver.

The companion `scratch/check_sponge_sensitivity.py` reuses saved basis factors
and the identical initial state to repeat the simulation at another damping
strength. This isolates damping strength from resolution, basis setup, and
initial data. It reports physical quadrature-weighted velocity differences
inside the interest region and central region.


### Open-only control results

The 96×80 open-only run reached t=12 with maximum sampled speed 2.1289,
pointwise divergence below 7.11e-15, and horizontal boundary error below
3.79e-14. Seven open/fixed regression tests passed. At t=12 the 64×64 vs
96×80 quadrature-weighted perturbation velocity differences were 2.21% over
the full domain and 0.215% in |x|<2, |y|<.5. The full-domain L-infinity
velocity difference was 1.18, concentrated near boundaries: the open-only
boundary region is not spatially converged. These results do not establish
reflection-free vortex exit. At t=6 the 64×64 dt=.004 vs .002 nodal transverse
velocity difference was 2.71e-6 and vorticity difference 1.93e-4.

### Extended-domain validation

The 160×80, dt=.002, sigma_max=4 run reached t=12. Its maximum sampled speed
was 2.4077 over the run; final speed was 1.3419. Pointwise divergence stayed
below 7.994e-15 and the prescribed horizontal velocity error below 4.119e-14.
At t=12 the full-domain nodal vorticity maximum was 12.8002 and the region-of-
interest maximum was 10.2305. The largest vorticity sampled at any time was
42.2379, so final values should not be mistaken for run-wide maxima.

Two additional tests passed for exact zero interior absorption, equality of
the factored damping load to direct vector-force quadrature, its negative
perturbation-energy work, unchanged zero perturbation, and taper derivatives.
Together with the seven preceding open/fixed tests this is nine passing checks.

Saved artifacts under `build/kh_stream/`:

- `extended160_s4/kh_stream.mp4`: 61 computed frames, H.264, 1210×660, 12.2 s.
- `extension_full.png`: full domain at t=6 and 12, with extension boundaries.
- `extension_comparison.png`: open-only versus extended absorption, cropped
  to the same interest region, with comparable spatial spacing.

Horizontal-wall shear layers remain because those physical boundary conditions
were retained. Color clipping is labeled explicitly in figures and video.

With sigma_max reduced from 4 to 2 at the identical resolution, initial modal
state, dt, and T=12, the quadrature-weighted **perturbation** velocity L2
difference is 2.9174% in |x|<3 (full y extent), and 0.9582% in |x|<2 (also full
y extent). Absolute velocity L-infinity differences are .07606 and .03939,
respectively. Thus the extension visibly improves the boundary behavior but
this particular two-unit extension is **not demonstrated independent of its
absorption parameters**. Enlarging the extension and repeating a domain-length
study would be needed before treating it as a quantitatively validated
unbounded-domain approximation. No such additional long computation was run.
Both completed simulations and the movie renderer exited normally.
