# BSPF Poisson on closed B-spline domains

This benchmark solves `-Delta u = f` with Dirichlet data on two smooth, simply
connected domains inside `[-1,1]^2`. The first is a convex ellipse-like periodic
cubic B-spline. The second is a rounded C-shaped periodic cubic B-spline with
an empty visibility kernel. Neither boundary is replaced by a polygon for
operator assembly. Both curves are C2, not globally C-infinity.

## Discrete method

The field space consists of tensor products of the existing unrestricted BSPF
factors, restricted to the physical domain. The background square has no
physical boundary condition. The one-dimensional parent space has 33 nodes;
the benchmark retains 6, 10, 14, 18, or 20 eigenmodes per direction. This is a
modal truncation study in one fixed parent space, not an h-refinement study.

`_stream_line(..., clamped=False, dirichlet=False)`, `stream_evaluate_line`, and
the shared `tensor_product` kernel are reused. Arbitrary-point values and
derivatives are evaluated directly through the existing MPFR BSPF evaluator,
then stored in float64. There is no intermediate Hermite interpolation.

Symmetric Nitsche imposes the curved Dirichlet boundary:

```
a(u,v) = integral_Omega grad(u).grad(v)
       - integral_Gamma (dn u) v - integral_Gamma (dn v) u
       + tau integral_Gamma u v
l(v)   = integral_Omega f v - integral_Gamma (dn v) g
       + tau integral_Gamma g v.
```

The penalty is four times the discrete trace constant bounding
`||dn v||_Gamma^2 / ||grad v||_Omega^2`, computed on the positive stiffness
eigenspace after restricted-mass whitening. Constant modes have zero normal
derivative. This is a numerically estimated bound for the retained space and
quadrature, not a mesh-independent analytical estimate. Matrix positivity and
quadrature sensitivity are checked explicitly.

The restricted mass SVD drops squared singular values below `1e-12` times the
largest. The retained rank is reported, and `1e-10`/`1e-14` sensitivity runs are
included. This truncation changes the approximation space; it is not hidden
inside the linear solver. The benchmark uses dense Cholesky after assembly.
It does not claim the original rectangular tensor inverse solves the curved
problem exactly, nor does it claim a scalable matrix-free implementation.

## Geometry and quadrature

Boundary positions, tangents, normals, and arc-length weights come directly
from the periodic cubic spline. Gauss integration is split at its knot spans.

Volume integration uses vertical slices and **all** boundary intersections.
Spline knots and vertical tangencies split the x intervals. Each root is
bracketed on a monotone spline segment; this avoids spurious roots of nearly
degenerate cubic polynomials. The C-shaped case has two interior intervals
on some slices and therefore exercises the non-star-shaped path.

A beta(6,6) endpoint substitution gives sixth-power flattening in x. It handles
square-root/cube-root inverse branches near ordinary tangencies and joins onto
straight vertical pieces. Extra subdivisions with width at most 0.25 resolve
the BSPF factors. The geometry is not approximated by an indicator mask at
background quadrature nodes. Root finding and numerical integration still
have finite tolerances.

The provided curves are regular and simple by construction. The API assumes
regular, simple, counterclockwise input; it is not a general self-intersection
repair or topology-validation library. Non-star-shapedness is checked by
infeasibility of a finite set of necessary inward tangent half-plane
constraints: even this subset admits no possible star center. This is a
floating-point numerical certificate, not an interval-arithmetic proof.

## Reproduce

From the repository root:

```sh
PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/bspf-mpl \
  python examples/pde/embedded_poisson.py
PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest -q jax/tests/test_embedded_poisson.py
```

The script enables JAX float64 before basis construction. Its optional
`--reuse-cache` flag reads trusted local sample pickles under the output
directory. Regenerate these files if geometry, quadrature, or basis code changes.

The manufactured solution is

```
u_exact = exp(x) cos(y) + x^2 + y^2
f = -4
g = u_exact on the spline boundary.
```

Assembly uses Gauss orders 12, 16, and 20. Errors use independent order-24
volume and boundary samples. Reassembly at order 24 separately measures the
remaining quadrature sensitivity. All errors are against the manufactured
solution, not against another numerical solution. Boundary RMS is absolute;
L2 and gradient errors are relative. Sampled maxima are not certified suprema.

Outputs under `build/embedded_poisson/`:

- `summary.json`: geometry checks, error tables, ranks, penalties, matrix
  condition numbers, quadrature changes, and cutoff sensitivity.
- `convex_solution.npz`, `nonstar_solution.npz`: coefficients and sampled fields.
- `validation.png`: geometry, solution, error, and convergence comparison.

The tests also compare volume moments with independent boundary integrals,
check Green's identity and its Poisson sign, check the affine harmonic problem,
and compare solutions assembled on independent quadrature rules.

## Recorded validation

With 20 modes per direction, order-20 assembly and independent order-24 errors:

| Domain | Relative L2 | Relative gradient | Boundary RMS | Retained DOFs |
|---|---:|---:|---:|---:|
| Convex | 6.42765e-5 | 2.58607e-3 | 3.09110e-5 | 396 / 400 |
| Non-star-shaped C | 3.40369e-4 | 1.28536e-2 | 2.31256e-4 | 400 / 400 |

Increasing from 6 to 20 modes reduces relative L2 from 1.47158e-2 to 6.42765e-5
on the convex domain, and from 2.15922e-2 to 3.40369e-4 on the C domain. The
order-20 to order-24 reassembly changes the solution by 1.37426e-6 and
5.12861e-7 respectively in relative L2. Quadrature error is therefore still
measurable, particularly on the convex domain, and is not described as zero.

Independent volume/boundary areas agree to relative 5.1e-15 and 2.6e-13.
The C domain has an infeasible tangent-half-plane LP and two interior intervals
on some vertical slices. The mass-cutoff sweep retains 392/396/400 convex DOFs
and gives L2 errors 6.41369e-5 / 6.42765e-5 / 6.43578e-5; all 400 C-domain DOFs
are retained throughout that sweep. Six distinct automated checks pass.

This establishes a curved-domain Poisson benchmark. It does not by itself
establish asymptotic spectral convergence, arbitrary-geometry conditioning,
or a moving plasma-vacuum interface algorithm. The finite C2 geometry, basis
truncation, quadrature, and mass cutoff must each remain visible in further
accuracy studies.

## Full-space accuracy diagnosis and a strong-residual comparison

The preceding 20-mode errors are not an accuracy limit of BSPF on these curves.
`examples/pde/embedded_poisson_approximation.py` fits the exact function on one
interior grid and checks a different grid plus the spline boundary. It is only
an approximation diagnostic, not a PDE solver. Using all 33 parent factors
instead of 20 lowers independently sampled relative function errors from
8.19e-5 / 2.43e-4 to 1.51e-13 / 4.41e-14 (convex / C-shaped). SVD with a
relative singular cutoff of 1e-14 is used, retaining 1069/1089 directions in
the full tensor candidate spaces. Truncating the low stiffness eigenmodes
does not preserve the approximation properties of the complete BSPF space.

`examples/pde/embedded_poisson_full_space.py` then solves the actual Poisson
problem using **only f and boundary g**, with the complete 33 x 33 candidate
space and direct BSPF second derivatives. It minimizes a weighted, oversampled
strong PDE residual together with the boundary-value residual. SVD acts on
the rectangular, column-scaled matrix directly; no normal equations, domain
mass whitening, or Nitsche penalty are used. This is a distinct experimental
backend, not a demonstration that full-space Nitsche already has this accuracy.

The finer solve uses the interior points of a 67 x 69 background grid with
an offset of 0.37, plus order-24 boundary points. Independent checks use an
offset of 0.61 on a 101 x 103 background grid, and order-32 boundary points.
The following relative function/gradient norms are **sampled norms**, not the
Gauss-integrated norms of the earlier table. The SVD cutoff here is 1e-14.

| Domain | Relative function | Relative gradient | Boundary RMS | PDE residual RMS |
|---|---:|---:|---:|---:|
| Convex | 3.10700e-15 | 1.02744e-13 | 1.70720e-14 | 5.88543e-12 |
| Non-star-shaped C | 9.72615e-15 | 3.48467e-13 | 6.28277e-14 | 2.66646e-12 |

The finer systems retain 1077/1089 and 1089/1089 directions respectively.
Both a coarser sampling grid and a 1e-12 SVD cutoff are also recorded. The
coarse C-domain PDE residual is 1.80e-7 despite a training residual near 1e-13,
which demonstrates why independent residual checks are essential. Refining
the sampling lowers that independent residual to 2.67e-12.

Reproduce with the same environment variables as above:

```sh
python examples/pde/embedded_poisson_approximation.py
python examples/pde/embedded_poisson_full_space.py
```

Data are in `build/embedded_poisson_diagnosis/approximation.json` and
`full_space_poisson.json`. These results demonstrate near-roundoff function
accuracy for the current smooth manufactured solution on both geometries.
They do not establish general stability of strong-residual sampling, equality
of Nitsche and collocation backends, or accuracy for nonsmooth solutions. The
main actionable finding is to preserve the full BSPF approximation space and
use stable algebra before attributing the former error floor to curved geometry.

Render the full-space solution, absolute error, and strong PDE residual with
`python examples/pde/render_embedded_poisson_full_space.py`. This evaluates
both backends on the same independent 180 x 180 cell-center grid. The output
`full_space_fields.png` (also PDF) shows the two domains with shared color
scales; `accuracy_comparison.png` compares sampled solution and gradient
errors. White pixels are outside the physical domain. Logarithmic display
values are floored at 1e-16, while the saved arrays and reported metrics use
the unfloored values. These plots and `plot_metrics.json` are saved under
`build/embedded_poisson_diagnosis/`.
