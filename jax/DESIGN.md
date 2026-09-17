# Mathematical contract

The constrained interpolating formulation below is the default (`noise_std=0`).
Positive a priori noise estimates use the [joint regularization and tensor-product contract](../docs/jax_noise_differentiation.md),
with discrete discrepancy-based selection and no endpoint constraints.

## Constrained spline fit

On a closed uniform grid `x_i = a + i*dx`, let `B[i,j] = B_j(x_i)` be a
clamped B-spline basis of degree p, and let W contain trapezoid weights.
Given samples f, solve

```math
\min_c \; \|Bc-f\|_W^2 + \lambda\|c\|_2^2,
\qquad Cc = d.
```

C evaluates derivatives 0 through q−1 at the left endpoint, followed by the
same derivatives at the right endpoint. The default d is estimated from local
polynomial stencils, `d = E f`. Users may supply complete endpoint jets instead.
The stencil width is independent of p and q; the default q=p−1 and width=p
match the existing BSPF formulation. Wider stencils can improve smooth-function
accuracy but may worsen conditioning.

The matrix convention, matching the existing primal solution, is

```math
\begin{bmatrix}
2(B^T W B + \lambda I) & -C^T \\
C & 0
\end{bmatrix}
\begin{bmatrix}c\\\mu\end{bmatrix}
=
\begin{bmatrix}2B^T Wf\\d\end{bmatrix}.
```

A plan stores the LU factorization explicitly. Applying it uses a matrix RHS
for all trailing batches. A new lambda produces a new factorization and a new
plan; there is no hidden cache or mutation. Positive lambda changes the fit;
zero lambda requires sufficient independent samples. The checked constructor
rejects obvious underdetermination and exactly singular factorizations. It does
not promise good conditioning for every knot/grid/stencil choice.

`decompose` returns `(coefficients, spline, residual)` satisfying `f=s+r`.
With prescribed jets the operation is affine in f. With inferred jets it is linear.
The spline constraints do **not** imply the final Fourier-corrected derivative
satisfies arbitrary prescribed boundary data exactly.

### Endpoint estimator selection

`endpoint_method="finite_difference"` preserves the Taylor-stencil default.
`endpoint_method="chebyshev"` replaces only the precomputed endpoint blocks.
On each endpoint window, map coordinates to [-1,1] and solve
`min_c ||V c - f_window||^2 + alpha ||P c||^2`, where V contains Chebyshev
polynomials and `P[j,j]=(j/(modes-1))**penalty_power` with its first two entries
zero. The implementation factors the augmented matrix `[V; sqrt(alpha)*P]`
using QR, then evaluates analytic polynomial derivatives with the physical
`(2/window_width)**k` scaling. It overwrites the value rows with exact endpoint
sample selection. Reflection supplies right-end weights on the uniform grid.

`boundary_points`, `chebyshev_modes`, `chebyshev_alpha` and
`chebyshev_penalty_power` configure this local fit. These are constructor
parameters; `Plan1D.boundary_blocks` is an ordinary PyTree array of shape
`(2, constraint_order, boundary_points)`, ordered left then right. Each block
acts only on its corresponding boundary slice; even overlapping windows are
handled independently. No zero-padded global endpoint matrix is constructed
or stored. This replaces the former dense `boundary_map` plan field; use the
public `endpoint_jets` function to apply either estimator. Storage and work per
slice are O(constraint_order * boundary_points), independent of axis length.
JIT, batching,
autodiff, tensor composition and explicit boundary overrides require no new
kernel branches. No Fourier filtering or defect-correction iteration is added.

## Basis implementation

`basis_matrix(t, x, degree=p, derivative=k)` uses the Cox–de Boor recurrence.
Repeated-knot denominators contribute zero, with safe division to avoid NaNs in
automatic differentiation. Physical derivatives follow

```math
B'_{i,p}=\frac{p}{t_{i+p}-t_i}B_{i,p-1}
       -\frac{p}{t_{i+p+1}-t_{i+1}}B_{i+1,p-1}.
```

The right endpoint is evaluated by its left-hand limit. Derivatives above p are
zero for the spline part; the Fourier residual can still have nonzero higher
derivatives. Interior knots in checked plans must be simple. Derivatives of a
piecewise basis at interior knots follow the right-hand branch, so derivatives
beyond the basis smoothness are not globally continuous.

## Fourier residual and differentiation

For r=f−Bc, define its discrete Fourier coefficients R. The residual interpolant
has period **T=n*dx**, not `(n−1)*dx`, preserving the original sampled-grid BSPF
FFT convention. Do not discard the final physical grid sample as if it were a
periodic duplicate: the input is a closed nonperiodic grid whose spline fit
removes endpoint jets before Fourier correction.

```math
D^k f = B^{(k)}c + \mathcal F^{-1}[(i\omega)^k R].
```

For real data, use the real part of the full complex Fourier interpolant.
At even n, this corresponds to a cosine Nyquist mode: odd Nyquist derivatives
vanish on grid nodes. Complex data uses the full signed-frequency convention,
including the negative Nyquist mode. Consequently composing two sampled first
derivatives is not identical to the direct second derivative in all cases.
`laplacian` and Hessian diagonals use direct second derivatives.

`max_derivative` controls the precomputed derivative matrices (default 4).
`derivatives(..., orders=(1,2,4))` shares one fit and FFT. `correction=False`
returns only derivatives of the fitted spline.

## Tensor composition in 1D–3D

A tensor plan holds independent axis plans; no Kronecker matrix is materialized.
Moving one axis to the leading position converts all other dimensions to batch
columns. Mixed partials compose these linear directional operators. With
inferred boundary jets the operators on distinct axes commute up to roundoff.
Mixed partials with arbitrary separately prescribed face jets are not exposed:
compatibility of intersecting boundary data needs a separate PDE-level contract.

Write S_i for the spline fit along axis i and F_i=I−S_i. The complete decomposition
is

```math
f=\prod_{i=1}^{d}(S_i+F_i)f.
```

`tensor_decompose` returns all 2^d sampled components keyed by strings such as
`SSF`. F denotes the residual to be represented by Fourier modes, not an array
of Fourier coefficients. The components reconstruct f exactly up to roundoff;
the API does not claim orthogonality or compressed storage. Requesting all eight
3D components has correspondingly greater memory cost.

Vector components are leading, then spatial axes, then batches. Gradient adds
one component axis; Hessian adds two. Curl returns a scalar in 2D and a vector in
3D. Volume integration removes spatial axes and retains batch dimensions.

## Interpolation and integration

`interpolate` evaluates **the same spline plus trigonometric residual** between
grid nodes; it intentionally differs from the old piecewise-linear residual
interpolator. Real Nyquist modes are evaluated as cosines off-grid. Tensor-grid
interpolation applies these one-dimensional interpolants successively.

The exact spline primitive is formed by extending the knot vector at both
ends and taking cumulative weighted coefficients:

```math
\widetilde c_0=0,\qquad
\widetilde c_{i+1}=\widetilde c_i+
 c_i\frac{t_{i+p+1}-t_i}{p+1}.
```

The Fourier residual integral uses

```math
\int_a^b e^{i\omega(x-x_0)}dx
=(b-a)e^{i\omega((a+b)/2-x_0)}
\operatorname{sinc}\left(\frac{\omega(b-a)}{2\pi}\right).
```

Here sinc is normalized as sin(pi*x)/(pi*x). This handles the zero mode without
singular division and admits differentiation with respect to either bound.
`integrate` and `antiderivative` therefore integrate the same interpolant.
The first primitive sets its left value. The second sets both left value and
left slope; the zero mode contributes its linear or quadratic primitive.

Evaluation points and integration bounds must stay inside the physical grid
interval. Application functions return NaN for out-of-domain requests even
under JIT; they do not implicitly extrapolate. Reversed bounds are supported.

## Boundaries between setup and computation

`plan_1d`, `plan_2d`, and `plan_3d` are checked setup functions for concrete grids.
They perform host-side validation before JAX assembly. Plans are immutable
PyTrees with degree/order metadata static and numerical arrays dynamic.
Computational functions have no file I/O, host callbacks, device transfers, or
configuration updates. JIT staging and device placement belong to the caller.

Changing grid lengths, basis size, derivative order, or tensor dimension may
require recompilation. Reuse shapes and plans across simulation steps. The
factory requires x64 because stencil/KKT conditioning can overwhelm float32;
importing the package does not globally enable it.

## Fixed-step time integration

`rk4_step(rhs, y, t, dt)` implements the classical four-stage Runge–Kutta rule.
`integrate_rk4` applies it with `lax.scan` over output intervals and a static
`lax.fori_loop` over internal substeps. Both accept floating/complex array or
PyTree states. The returned history includes the initial state; each leaf gains
a leading time dimension. Parameters captured by the RHS remain differentiable.

Time points must be finite and strictly increasing, and the caller chooses a
stable internal step size. Different output intervals may have different widths;
each gets the same number of substeps. This is not an adaptive solver and does
not estimate a CFL limit. State projection or boundary conditions belong to the
RHS/model contract, not to hidden mutations inside the integrator.

`endpoint_jets` exposes the default endpoint estimates used by the spline fit,
so a model can replace selected derivatives with prescribed values through JAX
functional updates. As above, prescribed spline jets are not a strong boundary
condition on the total corrected derivative. The migrated diffusion notebook
therefore checks the actual unconstrained derivative of its evolved field at
the boundary, alongside mass and solution error.
