# Complete correction of the JAX Euler–Bernoulli example

The notebook and MP4 now use the corrected solver. The model is unchanged:
a unit cantilever, initially at rest, subject to a constant unit distributed
load from t=0 to t=3. The grid still contains 129 samples.

Maximum displacement error decreased from **1.92e-3 to 1.20e-9**.

## Corrections

1. **Resolved quadrature.** Mass and bending stiffness integrate the actual
   BSPF trial functions and curvature at Gauss nodes split by every data node
   and spline knot. Eight and ten Gauss points independently check assembly.
2. **Consistent load.** The force is `load * Q.T @ weights`, using the same
   trial functions and quadrature, rather than nodal trapezoidal weights.
3. **Conditioning.** Retain the curvature matrix `G=weak.derivative_values`.
   If `L L.T = rhoA*M`, compute the SVD of
   `sqrt(EI*W) G L^{-T}`. Its singular values are angular frequencies.
   This avoids squaring the derivative factor's condition number by solving
   an eigenproblem for the dense fourth-order stiffness matrix. All discrete
   modes are retained; no low-mode filtering is applied.
4. **Time evolution.** `integrate_elastic` applies exact sine/cosine formulas
   for the autonomous constant-load system. Stable sinc expressions handle
   small/zero frequencies. There is no midpoint time-discretization error.
5. **Boundary conditions.** The notebook now imposes all four full-field
   conditions: left value/slope and right curvature/third derivative. This
   reduces the trial space by four nodal degrees of freedom and is checked
   under spatial refinement. It is a nonperiodic cantilever, not a periodic
   Fourier beam. Independent BSPF differentiation checks every residual.
6. **Reference.** Use 256 continuum cantilever modes, checked against 512.
   Roots solve `cos(b)+sech(b)=0`; scaled decaying exponentials avoid overflow
   and hyperbolic cancellation. The exact dimensionless load integral is
   `2*sigma/b`, and each mode has unit integral of its square. No projection
   quadrature or BSPF modes enter the reference. The zero initial displacement
   is exact, using `2*sin²(omega*t/2)`.
7. **Energy diagnostic.** Compute kinetic and bending energy about static
   equilibrium directly from quadrature-point velocities and curvatures.
   Avoid cancellation in quadratic forms using the squared stiffness matrix.

A variant retaining natural weak free-end conditions attained 5.79e-10 field
error, but its independently evaluated right shear residual reached 1.95e-3.
The final notebook uses all four constraints: its field error is slightly larger
(1.20e-9), while its actual endpoint residuals are much smaller. This tradeoff
is explicit rather than reporting only the favorable field norm.

## Measured validation

JAX x64 on the development CPU, corrected local OpenBLAS. Degree 5, 16 spline
functions, seven-point endpoint stencil, Gauss order 8. Errors cover 301 output
times on [0,3] and every sampled spatial point.

| Check | Result |
| --- | ---: |
| Maximum displacement error, 129 samples | 1.198e-9 |
| Maximum displacement error, 65 samples | 1.533e-8 |
| Difference between Gauss orders 8 and 10 | 1.283e-12 |
| Difference between 256- and 512-mode references | 1.530e-13 |
| Static quartic-solution error | 5.708e-12 |
| Fundamental frequency relative error | 2.230e-11 |
| Maximum relative energy drift | 2.704e-12 |
| Left displacement residual | 0 |
| Left slope residual | 5.168e-12 |
| Right moment residual (EI*w_xx) | 3.776e-10 |
| Right shear residual (EI*w_xxx) | 4.536e-7 |

The third-derivative residual amplifies floating-point cancellation when the
full BSPF derivative is independently re-evaluated from reconstructed samples.
In a separate arithmetic check, applying the constraint row to the extension
first gave 8.62e-9, multiplying a preassembled derivative by the reconstructed
field gave 1.63e-8, and a fresh BSPF derivative evaluation gave 4.54e-7.
These are mathematically equivalent evaluations with different cancellation.
It is not 1e-9 merely because the displacement is that accurate. All four
conditions are imposed through full differentiation rows, and the notebook
asserts the independently measured residuals at tolerances appropriate to
these derivative orders. The sudden load excites a high-frequency modal tail;
these measurements do not imply exponential convergence uniformly in time.

The notebook asserts displacement error below 5e-9, at least fivefold spatial
refinement improvement, quadrature difference below 1e-9, reference difference
below 1e-12, static error below 1e-10, and energy drift below 1e-9. The old
3e-3 displacement tolerance is gone.

Unit tests independently check mode norms/load integrals with Gauss quadrature,
analytical boundary conditions, static deflection, the first frequency,
transient refinement, nonuniform output times, positive density/rigidity
scaling, and zero-frequency constant acceleration. They cover both the natural
weak free-end space and the final fully constrained space.

## Reproduction and artifacts

- Notebook: `examples/pde/euler_bernoulli_1d.ipynb`
- Updated animation: `examples/pde/results/euler_bernoulli_beam_corrected.mp4`
- The original `euler_bernoulli_beam.mp4` path is also refreshed.
- Renderer: `scratch/render_beam_mp4.py`; it executes the notebook's numerical
  cells and assertions rather than maintaining another beam solver.

```sh
OMP_NUM_THREADS=4 PYTHONPATH=jax/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python docs/diagnostics/run_with_local_blas.py -m pytest \
  -c jax/pyproject.toml jax/tests/test_elasticity.py jax/tests/test_galerkin.py
OMP_NUM_THREADS=4 PYTHONPATH=jax/src MPLCONFIGDIR=/tmp/pde-mpl \
  python docs/diagnostics/run_with_local_blas.py scratch/render_beam_mp4.py
```
