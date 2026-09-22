# Full-section and free-boundary GS prototype

The magnetic axis is an interior O point of the poloidal flux at **R > 0**.
It is not the coordinate singularity R=0. The new `SplineSection` geometry
removes the inner hole; the existing `SplineAnnulus` API remains available.
The computational section can include the magnetic axis without imposing any
artificial inner Dirichlet trace. R=0 remains unsupported by the sqrt(R)
transformation.

## Fixed full section and magnetic axis

```python
from bspf_models.elliptic.spline_annulus import SplineSection
from bspf_models.plasma.spline_annulus_gs import SplineAnnulusGSPlan

domain = SplineSection(outer_scipy_bspline)
plan = SplineAnnulusGSPlan(domain, major_radius=3., source_plan=source_plan)
solution = plan.solve(source, (outer_flux,))
axis = solution.magnetic_axis(guess=[3., 0.])
```

`source_plan` must belong to this same domain; it can be the B-spline/FFT backend.
`boundary_flux=None` supplies zero data on every domain boundary, preserving
annular defaults. `magnetic_axis` locates a nondegenerate local maximum (or
minimum with `sign=-1`), verifies the gradient and Hessian signature, and rejects
unresolved points. Its derivatives are diagnostic finite differences: fourth
order in coordinate directions, second order for the mixed Hessian. The step
is explicit and every stencil must stay inside the computational domain. It
is not a general search for all O and X points.

The nonpolynomial Solov'ev regression includes the axis inside the domain and
uses independent analytic source and boundary values. With Fourier radius 16,
72 source-grid points per direction and order-12 panels subdivided twice, a
probe run gives maximum sampled flux error 1.24e-9 and axis
(2.99999999990, 3.9e-13), compared with the exact (3,0).

## Actual free plasma boundary

`FreeBoundaryGSPlan` wraps a full-section plan. The spline computational boundary
lies in vacuum and is NOT a prescribed plasma interface. On every outer iterate:

1. Set edge flux to the total flux at a prescribed limiter point.
2. Build a flux-dependent plasma current and normalize its volume integral to
   the prescribed total toroidal current.
3. Update computational-boundary flux from prescribed external filament coils
   PLUS the current plasma volume integral of the free-space ring Green function.
4. Solve GS on the full section, including vacuum.
5. Recompute profiles, the full nonlinear PDE residual and boundary closure
   against the new solution; mix the next iterate with damped Anderson iteration.

This boundary update allows a changing plasma shape inside a fixed vacuum box.
It does not confuse moving a prescribed Dirichlet curve with free-boundary GS.
The ring Green function is the standard complete-elliptic-integral solution,
with a stable small-parameter formula and no singularity softening. It obeys
`-Delta* psi = mu0 R j_phi`; its volume weights represent current in amperes.
The mathematical convention can also be checked against
[FreeGS](https://github.com/freegs-plasma/freegs) and its
[equilibrium documentation](https://freegs.readthedocs.io/en/stable/creating_equilibria.html).
No FreeGS solver code is used or installed.

The implemented profile family is

    j_phi = A [beta R/R0 + (1-beta) R0/R] max(psi-psi_edge,0)^power.

This corresponds to

    p' = A beta/R0 max(psi-psi_edge,0)^power,
    FF' = mu0 A (1-beta) R0 max(psi-psi_edge,0)^power.

A is determined by the specified positive total plasma current. Pressure is
zero at the edge and is the analytic integral of p'. `power` controls interface
regularity; it is part of the physical profile choice, not an invisible smoother.
This is a restricted profile family, not an arbitrary experimental profile
reconstruction or coil-current optimization.

## Acceptance and limits

- Final inner source, GS PDE and boundary checks retain their requested gates.
- Outer current consistency, full nonlinear PDE and updated Green-boundary
  consistency must all pass the requested outer tolerance.
- A higher-order independent volume rule checks total current and boundary
  Green flux. If it fails, increase `volume_order`; no tolerance is relaxed.
- Only one connected limited plasma, separated from the computational boundary,
  is supported. Connectivity and wall separation are sampled diagnostics, not
  certified continuous topology tests.
- `plasma_boundary` traces the edge flux about a chosen axis and requires a
  closed star-shaped contour. Missing crossings or sampled recrossings raise.
- `exterior_flux` evaluates the free-space coil and plasma field outside the
  computational section. Interior field evaluation uses the high-order GS solve.
- No X-point/separatrix, diverted equilibrium, multiple plasma islands, conducting
  wall response, coil optimization or R=0 treatment is implemented here.
- The outer iteration can fail for incompatible coils, limiter/current choices,
  unresolved thin profiles or an unsuitable initial state. Failures are explicit.
- All large Green interactions and layer potentials are applied in blocks.
  The small dense boundary LU, local source factorizations and at most eight
  Anderson history directions remain. This is CPU code, not a GPU implementation.

## Synthetic-coil experiment

```sh
python examples/pde/spline_free_boundary_gs.py \
  --out build/spline_free_boundary_limited
```

Coordinates and currents are nondimensional, with mu0=1. The example uses two
fixed coils at (5,+2), (5,-2), each current -0.72; total plasma current 1;
a limiter point at (3.6,0); and a circular-control-polygon spline vacuum boundary
centered at R0=3. The initial Gaussian is a seed only, not a prescribed final
plasma boundary. The two coils are outside the computational section.

The initial example tolerances are explicitly exploratory: outer 1e-3 and inner
1e-4. A run passing these gates must not be described as a 1e-8 free-boundary
solution. `results.json` records parameters, each residual, failures and diagnostic
checks; `fields.npz` and `equilibrium.png` are written only after successful
solution and boundary extraction. The separate full-section Solov'ev accuracy
result does not establish the same accuracy for a moving plasma interface.

Increase `--sample-grid`, `--fft-grid` and `--volume-order` together with tighter
inner/outer tolerances for a resolution study. Fixed-degree high-order source
approximation cannot overcome insufficient physical sampling or limited profile
regularity at the plasma edge.
