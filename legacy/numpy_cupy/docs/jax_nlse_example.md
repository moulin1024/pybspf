# Focusing NLSE bright-soliton example

`examples/pde/nlse_1d.ipynb` is a separate JAX notebook for

```
i psi_t = -psi_xx - 2 |psi|² psi.
```

Its infinite-line reference is
`sech(x-x0-2*k*t) exp(i*(k*x+(1-k²)*t))`, with x0=-3 and k=0.5.
On [-28,28] over 0<=t<=4, the pulse moves from -3 to 1. The finite problem
uses natural Neumann weak boundaries; the reference tail there is at most
2.78e-11. It is not an exact finite-box solution, but the boundary mismatch
is below the measured field error. No wall collision is included.

The notebook keeps model setup, a library call, analytical validation, and
plotting. It also propagates the same initial sech pulse with the linear
Schrödinger infrastructure to show how dispersion changes the shape when
nonlinearity is absent.

## Numerical infrastructure

`galerkin_1d` now exposes `values` and `quadrature_weights` in its immutable
PyTree. These describe the same quadrature used to assemble the mass matrix.
With Q denoting trial-function values at these points, the cubic term is
projected consistently:

```
i M q_t = K q - g Q* W (|Qq|² Qq).
```

A pointwise nonlinear phase rotation at data nodes would not generally
preserve the dense-mass weak-form norm and is not used.

`integrate_nlse(weak, initial, times, coupling=2., substeps=40)` diagonalizes
the constant linear part and advances the modal state with fourth-order
interaction-picture (Lawson) RK4. Linear phases are exact; the nonlinear
projection is evaluated at four RK stages. A JAX scan returns histories of
free nodal coefficients. Norm and Hamiltonian are monitored rather than
claimed to be exactly conserved by time integration. This is a dense 1D
building block with O(N*Nquad) nonlinear-stage cost. Time refinement remains
necessary; highly oscillatory content can require smaller steps.

## Measured validation

Development CPU, corrected local OpenBLAS, JAX x64. Default: N=513,
degree=7, 48 spline functions, nine-point endpoint stencil, eight-point Gauss
quadrature per node/knot subinterval, dt=0.0025, T=4.

| Check | Measured value |
| --- | ---: |
| Maximum complex field error, fine run | 7.781e-10 |
| Field error at N=257, same time step | 1.518e-8 |
| Field error at doubled time step | 1.237e-8 |
| Difference between the two time steps | 1.159e-8 |
| Maximum relative norm drift | 4.619e-13 |
| Maximum relative Hamiltonian drift | 3.834e-12 |
| Maximum width error | 2.152e-10 |

Raising Gauss order from eight to ten gave 7.783e-10 maximum field error,
consistent with the reported accuracy. At N=257, dt=0.02/0.01/0.005 gave
errors 3.10e-6/2.01e-7/1.87e-8; further time refinement reached a spatial floor
around 1.5e-8. These controls motivate separate space and time refinement in
the notebook. The width reference is pi/sqrt(12).

The notebook's assertions check the full complex solution without phase
alignment, filtering, or renormalization. Unit tests cover the zero-coupling
linear limit, both focusing/defocusing cubic signs, fourth-order nonlinear
phase convergence, outer JIT, and basic argument validation. The unchanged
linear Schrödinger notebook is rerun as a regression check.
