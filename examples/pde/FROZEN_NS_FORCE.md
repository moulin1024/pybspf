# Frozen nonlinear-force response and best approximation

The force is fixed once from the smooth, unforced rational Stokes initial field:

```
f0 = -(u0 · grad) u0
```

Every BSPF and finite-element solve uses this same continuous function. The
initial rational coefficients and their SHA256, source parameters, force
samples on production quadrature, and response arrays are retained. The reference
uses Gmsh curved meshes and scikit-fem Taylor-Hood elements. It shares the physical
force and geometry, but no BSPF basis, mass matrix, pressure projection, or
stiffness assembly. The optional research dependencies `gmsh`, `meshio`, and
`scikit-fem` are not added to the core or model installation requirements.

## Problems and projections

Solve the homogeneous-boundary response on the original exterior-ellipse domain:

```
alpha * v - mu * Delta(v) + grad(p) = f0
                                      div(v) = 0
```

Velocity is zero at the inlet, horizontal walls and ellipse. The outlet at x=5
has zero Laplacian traction `(mu*v_x - p, mu*w_x)`. There is no sponge. These are
response fields, not total velocities including the initial Stokes lift.

Two operators are compared:

- Steady forced Stokes: alpha=0, mu=nu=0.0153333333333.
- Helmholtz-Stokes: alpha=1, mu=(dt/2)*nu=0.000153333333333, with dt=0.02.
  This is the implicit block of the production midpoint method, applied to the
  frozen force. It does not include nonlinear updates of the force.

The production space is fixed at 73 x 33, rational boundary correction,
CPU/MPFR, quadrature factor 4 (44445 physical samples). Four responses are retained:

1. Actual Galerkin solve with the frozen-force load.
2. Best H1 projection of the independent response onto exactly the same
   homogeneous velocity space, minimizing velocity and all first derivatives.
3. Best energy projection, minimizing alpha times velocity L2 plus mu times
   the gradient seminorm; this is the metric associated with the solved operator.
4. Best L2 velocity projection, for comparison of derivative amplification.

All best projections use the same production quadrature and basis as the actual
solve. FEM gradients are evaluated analytically inside each curved element;
their divergence is measured, not silently set to zero. Projection loads include
both independent ux and vy components. The resulting best H1 is a discrete
quadrature optimum, not a certified continuous best approximation.

## Reference verification

Quadratic curved meshes describe the ellipse; increasing spatial resolution and
polynomial degree checks both approximation and geometry effects. Boundary and
interior derivative behavior are not inferred merely from the linear residual.

The reference is first calibrated by an independent divergence-free manufactured
flow with a smooth streamfunction vanishing with its gradient on every velocity
boundary and with zero outlet traction. Its force uses automatic differentiation
of that continuous expression, not assembled FEM matrices. P3/P2 mesh refinement
(scale 2 to 1) reduces the gradient discrepancy from 1.0668% to 0.1695%; P4/P3
reduces it from 0.13095% to 0.01107%. Physical-coordinate polynomial reproduction
is within 8e-15. A constant pressure-gradient test gives velocity coefficients
below 8e-12, including the small-viscosity Helmholtz operator.

The P3/P2 references use scales 1, 0.7 and 0.5. Their last Helmholtz H1 difference
is still 2.49%, so these coarse references are retained but not used alone to
justify the final result. P4/P3 controls at scales 0.7 and 0.5 follow. Refinement
differences are empirical uncertainty indicators, not rigorous continuum error
bounds. All reported field norms use unchanged production volume quadrature.

For visual maps, strictly interior points on the common 401 x 161 display grid
are used, excluding hole points and outer boundary nodes. Volume norms include
the near-wall region. Derivative comparisons on that display grid can show small
FEM element-scale residuals and must be read with the refinement controls.

## Reproduction

Use new output directories; historical results are not overwritten:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python examples/pde/validate_frozen_force_fem.py --degree 4 \
  --out build/immersed_flow/fem_validation_new
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python examples/pde/compare_frozen_ns_force.py --degree 4 --scales 0.7 0.5 \
  --out build/immersed_flow/frozen_force_new
MPLCONFIGDIR=/tmp/bspf-mpl python examples/pde/render_frozen_ns_force.py \
  --out build/immersed_flow/frozen_force_new
```

The archived NPZ files store FEM states and independently evaluated reference
fields; MSH files preserve each reference mesh. `report.json` contains convergence
and approximation metrics. Figures compare actual, H1, energy, and L2 projection
errors. No solver defaults, filter, artificial viscosity or physical forcing in
the production NS example are changed by this investigation.

## Results, 2026-09-22

The two runs have identical frozen-coefficient SHA256
`b08fce2db50d7ee16e5dc35b8349c75e3863737657fa0d2c45cd90f690ef4e87`.
The independently evaluated force load differs from the original initial NS
explicit RHS by only 1.2451e-15 relative. The input has therefore not been
replaced by a manufactured forcing chosen to fit the tested BSPF space.

The final reference uses curved P4/P3 Taylor-Hood elements, 19,367 triangles,
315,540 scalar-component velocity unknowns in total and 89,277 pressure unknowns.
The preceding P4/P3 mesh has 10,890 triangles. Their weighted relative H1
response differences are 0.006054% for steady Stokes and 0.083950% for Helmholtz;
vorticity differences are 0.004660% and 0.052976%, respectively. The fine
reference linear residuals are 4.67e-13 and 9.04e-13. The Helmholtz reference
uncertainty is far smaller than the tested-space discrepancy below. The small
Stokes discrepancies should not be interpreted as certified errors to all shown
digits, given the empirical reference-refinement difference.

| Response | Actual H1 discrepancy | Best H1 discrepancy | Best energy H1 discrepancy | Actual vorticity discrepancy |
| --- | ---: | ---: | ---: | ---: |
| Steady Stokes | 0.026851% | 0.026850% | 0.026850% | 0.027222% |
| Helmholtz-Stokes | 17.698087% | 16.633730% | 17.698038% | 17.743593% |

These are relative, physically weighted volume-quadrature differences against
the fine FEM reference, not exact continuum error bounds. H1 includes velocity
and all four first derivatives; no smooth initial background is added to dilute
the relative response error.

For Helmholtz, the best H1 error squared is 88.33% of the actual H1 error squared.
Within this discrete diagnostic and reference accuracy, most of the gradient
error cannot be removed merely by choosing different coefficients in the current
space. The actual and best-energy responses differ by only 5.70e-6 of the reference
vorticity grid norm, while both retain the same clear upstream oscillations.
This does not support additional assembly/projection-construction error as the
main cause. The remaining difference from best H1 reflects the different
optimization metric: the equation minimizes its energy error, not H1 error.

Best H1 is not uniformly better at every location. On the upstream patch
x in (-0.85,-0.35), y in (-0.8,0.8), actual vorticity error RMS is 0.29627,
whereas best H1 has 0.45977. Both oscillate. Best H1 lowers the global H1 error
by redistributing error; changing projection weights alone has not removed ripple.
The actual and best H1 Helmholtz vorticity error maps have cosine similarity 0.9451.

A supported interpretation is that the same frozen force is easily resolved
after steady viscous smoothing, but not in the short-step response. The natural
Helmholtz viscous length is sqrt(mu/alpha)=0.01238, compared with background
spacings dx=0.08333 and dy=0.0625. The independent reference exhibits sharp
near-wall response. Current rational corrections are homogeneous Stokes fields;
they adapt the geometry but do not explicitly supply Helmholtz boundary-layer
response modes. This explains why excellent homogeneous Stokes initialization
can coexist with poor time-dependent acceleration representation. The length
comparison is a mechanism interpretation, not a general convergence theorem.

The evidence favors enriching or changing the response space (for example,
operator-aware near-wall functions) before changing projection algebra. Such a
replacement still needs independent approximation and NS evolution tests. This
frozen linear-response experiment does not certify a nonlinear fix or resolve
all possible startup-compatibility and late-time behavior questions.

Full P3/P2 evidence is preserved in
`build/immersed_flow/frozen_force_20260922/`; the final P4/P3 comparison is in
`build/immersed_flow/frozen_force_p4_20260922/`. See `projection_comparison.png`,
`response_errors.png`, `report.json`, and `interpretation_metrics.json` in the
latter directory. Both manufactured-reference calibration runs are retained in
`frozen_force_fem_validation_20260922/` and
`frozen_force_fem_p4_validation_20260922/` under the same build parent.
