# Triangle boundary–interior Poisson prototype

`simplex_poisson.py` is a standalone NumPy/SciPy research baseline, outside the
maintained JAX core. It tests the boundary–interior decomposition idea; it does
not implement Fourier modes, triangular B-splines, or a new approximation space.
No changes to existing NS experiments are required.

## Discretization

Solve `-Δu=f` with Dirichlet data on a polygon triangulation. On each straight,
nondegenerate triangle, use the complete degree-p Bernstein basis in barycentric
coordinates. Coefficients supported at vertices and edges form the trace;
indices with all three barycentric exponents positive form zero-trace bubbles.
Shared edge coefficients are oriented by global vertex IDs. The resulting space
is C0 conforming, independent of triangle vertex orientation.

Partition the local stiffness matrix into trace T and interior I blocks:

```
H = -K_II^{-1} K_IT
z =  K_II^{-1} f_I
u_I = H u_T + z
S = K_TT - K_TI K_II^{-1} K_IT
b = f_T  - K_TI K_II^{-1} f_I
```

The columns of `[Id; H]` are discrete harmonic lifts: they are energy-orthogonal
to every interior bubble, not necessarily pointwise harmonic. Assemble S on the
mesh skeleton, impose Dirichlet trace coefficients, solve, and reconstruct each
interior independently. This is standard static condensation of conforming
high-order FEM and is algebraically equivalent to the full solve. It does not
by itself improve approximation or suppress underresolved convection artifacts.

Positive Duffy/Gauss quadrature of order p+4 assembles the operator/load;
independent order p+9 evaluates errors. Smooth nonpolynomial loads are integrated
numerically. Dirichlet data are interpolated on each edge in degree p (not imposed
exactly for arbitrary nonpolynomial traces). The benchmark has linear boundary
data, represented exactly. p=1,2 have no interior bubbles.

## Reproduce

From the repository root, with NumPy, SciPy, Matplotlib and pytest installed:

```sh
MPLCONFIGDIR=/tmp/bspf-simplex-mpl python examples/pde/simplex_poisson.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q examples/pde/test_simplex_poisson.py
```

No thread-count environment variable is set. Outputs go to `build/simplex_poisson`.
The square mesh uses Delaunay triangles on vertices whose interior locations are
perturbed with seed 42. Refinement levels are not nested.

Manufactured solution: `u=x+y+sin(pi*x)sin(pi*y)` on the unit square.
Forcing: `f=2*pi^2*sin(pi*x)sin(pi*y)`.

## Initial results (2026-09-22, CPU)

| p | 32 triangles L2 | 128 triangles L2 | 512 triangles L2 |
|---|---:|---:|---:|
| 1 | 7.369e-2 | 1.842e-2 | 4.733e-3 |
| 2 | 4.278e-3 | 5.764e-4 | 6.955e-5 |
| 3 | 3.216e-4 | 2.079e-5 | 1.334e-6 |
| 4 | 2.230e-5 | 7.988e-7 | 2.295e-8 |
| 5 | 1.275e-6 | 2.231e-8 | 3.701e-10 |

At p=5 on 512 triangles: 6561 total coefficients, 3489 skeleton coefficients,
3169 free global unknowns after prescribing the boundary. The remaining 3072
interior coefficients are recovered locally. Relative free-system residual is
2.53e-15. This is a reduction in global system size, not a measured speedup.
On 32 triangles, full-versus-condensed coefficient differences across p=1..5
are at most 1.20e-14. Six tests pass, covering partition of unity, zero bubble
traces, quadratic reproduction with nonzero boundary data and mixed triangle
orientations, continuity, condensation equivalence and cubic convergence.

## Limits and next comparisons

Only scalar constant-coefficient Poisson, straight triangles, and all-Dirichlet
boundaries are implemented. Meshes must be valid manifold triangulations without
degenerate cells. Bernstein bases at substantially higher degree need conditioning
study; small residuals are not condition-number estimates. There is no high-Re,
three-dimensional, cut-cell, or nonoscillatory claim.

Before promoting this to a library API, compare mass/stiffness conditioning and
operator cost against orthogonal triangular bases at equal approximation order;
then investigate whether an independently chosen boundary lift or interior space
improves accuracy per degree of freedom. Test nonconvex domains and corner
singularities separately from the present smooth manufactured solution.

## Follow-up

See [boundary/interior comparison](SIMPLEX_BSPF_EXPLORATION.md) for polynomial
versus Fourier interior tests and residual-selected edge modes, including matched
DOF accounting and the absence of a demonstrated end-to-end speedup.
