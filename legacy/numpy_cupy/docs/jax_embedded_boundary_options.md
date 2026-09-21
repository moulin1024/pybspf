# Embedded Poisson: endpoint controls and boundary treatment

## Completed endpoint comparison

The previous embedded BSPF runs already used Chebyshev endpoint jets: q=9,
12 Chebyshev modes over 16 uniform samples, regularization 1e-12, spline degree
13 and 32 splines. This is a correction of a possible interpretation that
Chebyshev treatment had not yet been used.

The shared `_stream_line` and embedded `background_line` now expose endpoint
method, sample count, Chebyshev mode count and regularization. Defaults and
the `StreamLine` storage layout are unchanged. Arbitrary-point evaluation
continues to use the same stored projector and MPFR kernel.

At N=49, seed 20260918, kmax=12pi, baseline 2N+1 interior sampling, identical
boundary sampling, SVD cutoff, and independent validation points:

| Chebyshev modes / window points | Convex relative solution error | C-domain relative solution error |
|---|---:|---:|
| 12 / 16 (previous baseline) | 3.84897e-5 | 9.46914e-4 |
| 12 / 12 | 4.60415e-6 | 1.09757e-4 |
| 14 / 18 | 3.49857e-5 | 1.47232e-3 |
| 16 / 20 | 2.92523e-5 | 7.70012e-4 |

The 12/12 configuration improves the two solution errors by approximately
8.36x and 8.63x. Its independent relative PDE residuals are 3.72935e-5 and
5.80817e-4, and relative boundary errors are 3.31859e-5 and 7.59348e-4.
It is an improvement for these tests, not a new global default. In particular,
the default is left unchanged for existing NS and MHD callers. Increasing
fit order and window width is not monotonically beneficial at fixed N.

An independent 1D screen samples sin/cos at 49 frequencies from pi to 12pi,
interpolates from N=49 nodal data, and checks 801 points. This is a direct
interpolation/derivative test, not the curved Poisson solve. The screen and
all endpoint variants are saved under `build/embedded_poisson_endpoint/`.
Twenty-five existing endpoint and stream-NS regression tests pass.

Reproduce (with `PYTHONPATH=jax/src`, float64 enabled by the entry points):

```sh
python examples/pde/embedded_poisson_endpoint_screen.py
python examples/pde/embedded_poisson_random_mms.py --nodes 49 \
  --chebyshev-modes 12 --endpoint-points 12 \
  --out build/embedded_poisson_endpoint/m12p12
```

Other compared configurations use 14/18 and 16/20 respectively. All retain
the same q, spline degree/count, regularization and PDE least-squares backend.

## Proposed high-accuracy boundary architecture (not implemented)

For constant-coefficient Poisson on a smooth simply connected domain Omega,
split u=p+h. Extend f smoothly into a surrounding rectangle and compute a
particular solution -Delta p=f_ext using the rectangular solver. On Omega,
solve Delta h=0 with h=g-p on its actual B-spline boundary. A second-kind
Laplace boundary integral equation can determine h from boundary unknowns.

This avoids restricting and whitening the entire two-dimensional background
space and avoids balancing an approximate PDE residual against a boundary
penalty. It does not make the original cut-basis Nitsche or least-squares
method identical to a rectangular solve. The architecture is explicitly a
BSPF volume solver plus a boundary-integral correction. Boundary density may
itself use a one-dimensional BSPF representation, but its operator is new.

Source extension must be smooth; zero extension across the physical boundary
does not preserve high-order accuracy. Artificial outer boundary conditions
must also permit a smooth particular solution. Blindly imposing zero
Dirichlet data at rectangle corners can introduce compatibility problems.
A periodic smooth extension with a separate polynomial mean-source lift is
one alternative; this would be an auxiliary periodic problem, not a physical
periodic boundary condition on Omega.

Nyström or Galerkin quadrature should split at the B-spline knot spans.
The present cubic boundaries are C2, not analytic periodic curves, so global
exponential quadrature convergence must not be presumed. Evaluation near
the curve needs special nearly-singular quadrature, such as corrected panel
quadrature or QBX. Gradients/normal derivatives require their own accuracy
checks; accurate Dirichlet values alone are insufficient for MHD coupling.

Relevant primary literature:

- Fryklund, Lehto, Tornberg, [Partition of Unity Extension](https://arxiv.org/abs/1712.08461):
  smooth forcing extension with a spectral volume solve and boundary integral
  correction; the paper reports tenth-order convergence down to 1e-14 in its
  examples. This is evidence for the architecture, not a benchmark of our code.
- Helsing and Ojala, [Close evaluation of layer potentials](https://portal.research.lu.se/en/publications/on-the-evaluation-of-layer-potentials-close-to-their-sources/):
  why native quadrature fails near the boundary and how corrected evaluation
  can recover near-machine accuracy.
- Stein, Guy, Thomases, [IBSE](https://arxiv.org/abs/1506.07561):
  an alternative that smoothly extends the unknown field for Cartesian
  high-order discretization. This is more directly aligned with a future
  volume NS/MHD framework, but finite regularity extension does not by itself
  imply machine precision or a fully spectral convergence rate.

Before implementation, compare the SAME random-wave MMS on a rectangle,
then independently verify the particular solution and harmonic correction.
For a stronger geometry test, add a harmonic logarithmic source with its
singularity in the exterior C-shaped opening. It is smooth inside Omega but
cannot be smoothly continued through the entire box, which specifically
tests the advantage of a separate boundary correction over a global trial
space that must represent both effects.
