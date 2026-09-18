# Nonperiodic KH with same-space weak BSPF

This implementation replaces strong momentum collocation with the original
BSPF trial space evaluated in a Galerkin weak form. It does **not** replace BSPF
with finite differences. Only one-dimensional matrices are assembled; all 2D
operations are tensor contractions. Pressure projection is direct, with zero
iterations and zero refinement. The NS time integrator is unsplit RK4, including
explicit viscosity; the separate direct Helmholtz primitive is not an implicit
Stokes solver.

## Running

```sh
python -m pip install -e 'jax[weak-ns]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src \
 MPLCONFIGDIR=/tmp/pybspf-mpl python scratch/run_kh_weak.py \
 --nx 160 --ny 112 --T 6 --dt .002 --validate
```

`--no-video` skips rendering; `--render-only --out <existing-directory>` creates
an MP4 from saved computed frames. Rendering requires matplotlib and ffmpeg.
The default output is `build/kh_weak/160x112/`.

Use `bspf_jax.plan_weak_navier_stokes2d(x, y)` and
`weak_ns_rk4_step(plan, velocity, dt, force_load)` for the new backend.
Enable `jax_enable_x64` before setup. `force_load` is a **weak integrated load**,
not a nodal acceleration; `weak_ns_load` integrates a physical force at the
plan's tensor quadrature points. The previous strong and rejected SBP variants
remain separate for comparisons.

## Approximation and quadrature

Uniform nodes, QR-constrained degree-13 splines with 32 basis functions, nine
endpoint jets, and 12-mode Chebyshev fits on 16 endpoint nodes are retained.
MPFR with 113-bit precision is used only to evaluate the spline-plus-Fourier
cardinal basis during setup. All matrices and runtime JAX computations are
float64. This avoids the cancellation previously found in naive float64 basis
assembly; it does not change the approximation space.

Let `B` and `G` evaluate basis values and first derivatives at positive-weight
Gauss quadrature points. Velocity increments use the interior cardinal
functions `B_I`, hence vanish exactly on every wall. Pressure tests use the
**full** cardinal basis, including endpoint functions. Nonlinear products use
overintegration on spline knot intervals; the quadrature order grows with
resolution. No sponge, filter, or artificial viscosity is present.

For zero boundary velocity, the convective load is assembled as

\[
N_c(u)=-\tfrac12 B_I^TW(u\cdot\nabla u_c)
       +\tfrac12\sum_dG_{d,I}^TW(u_du_c).
\]

Forward evaluations and transpose contractions use identical quadrature. Thus
`u.T N(u)=0` algebraically. Viscosity uses `K=G_I.T W G_I`, giving the negative
physical dissipation. The energy law for the homogeneous unforced problem is

\[
\frac{d}{dt}\frac12 u^TM_vu=-\nu u^TK_vu.
\]

Nonzero fixed KH boundary velocities and the maintained-base forcing can
exchange energy. KH perturbations can physically grow; total monotone energy
decay is not asserted for this driven experiment.

## Direct pressure projection with the correct Schur complement

A mass-orthogonal projection using the old **strong nodal divergence** was
screened and rejected: it was energy stable but failed to remove a smooth
physical pressure gradient accurately. The final implementation uses paired
weak divergence and gradient. It does not substitute a scalar Laplacian for
the actual Schur complement.

In each direction, with `M=B_I.T W B_I`, define

\[
S=B^TWB_I,\qquad C=B^TWG_I,
\qquad A=CM^{-1}C^T,\quad H=SM^{-1}S^T.
\]

The weak divergence of zero-wall increments is

\[
D=[C_x\otimes S_y,\ S_x\otimes C_y],
\]

and its exact mass-adjoint pressure Schur matrix is

\[
DM_v^{-1}D^T=A_x\otimes H_y+H_x\otimes A_y.
\]

Neither 2D matrix is formed. Since `H` is singular, directly solving an
`(A,H)` eigenproblem would be invalid. Instead, the code solves

\[
AV=(A+H)V\Lambda,\qquad V^T(A+H)V=I.
\]

Consequently `V.T A V=Lambda` and `V.T H V=I-Lambda`. The 2D modal denominator is

\[
\lambda_i^x(1-\lambda_j^y)+(1-\lambda_i^x)\lambda_j^y.
\]

Each line has two exact zero and two exact unit eigenvalues. Eight tensor
pressure null modes are pseudoinverted to zero; other modes are retained.
These include discrete pressure gauge modes beyond the physical constant.
Pressure itself is therefore not uniquely recoverable from the projection.
The projected velocity is unique in the mass norm. The projection API returns
velocity increments, not a claimed uniquely normalized physical pressure.

Only one-dimensional factors and a scalar 2D reciprocal-denominator array are
stored. Tensor transforms use dense 1D matrices in this version; compressed
transforms are not enabled. In the strict assembly taxonomy this is partial
assembly / matrix-free with respect to the global operator, not absence of all
stored matrices. Nonlinear quadrature work arrays are also 2D and can dominate
memory; no claim of optimal 3D quadrature memory usage is made.

`weak_ns_helmholtz` separately solves `(alpha M + beta K)u=load` with homogeneous
Dirichlet conditions by symmetric generalized eigendecomposition. Simply
composing it with projection would introduce a time-splitting error because
diffusion and the no-slip projector do not generally commute. The current NS
integrator deliberately advances the unsplit semidiscrete equation with RK4;
ordinary advective and viscous time-step restrictions still apply.

## Boundary conditions and diagnostics

Domain `[-3,3] x [-1,1]`, viscosity `.002`, base `(tanh(y/.12),0)`, perturbation
amplitude `.03` and wavelength `1.5`, as in the original nonperiodic KH test.
All four boundaries hold the base velocity fixed. There is no periodic wrapping
and no absorbing layer. A fixed load equal to the negative weak momentum load
of the base maintains the discrete shear. This is a driven, bounded KH problem,
not an unbounded or periodic reference benchmark.

`weak_ns_divergence` returns the L2 projection of physical divergence into the
full pressure space, represented at the nodes. This is the constraint enforced
by the solver. `weak_ns_pointwise_divergence` separately evaluates the original
BSPF nodal derivative. They are different diagnostics; a tiny weak residual
does not establish pointwise incompressibility or spatial convergence.

## Validation

`jax/tests/test_weak_navier_stokes.py` independently checks:

- the tensor projection against an explicitly assembled small dense constraint
  null-space projection, including an odd grid size;
- nonlinear projected energy dissipation and fixed walls;
- the direct Helmholtz solve against a dense solve;
- polynomial diffusion and constant-flow consistency;
- removal of an analytically specified pressure gradient.

`scratch/validate_weak_ns_accuracy.py` uses the analytic solution
`u=exp(-t) curl(psi)`, with a smooth nonperiodic streamfunction vanishing together
with its normal derivative on the walls, and pressure `exp(-t) cos(x+2y)`.
The force is analytically differentiated and independently integrated, not
manufactured from the discrete NS matrix. At `T=.04`, `dt=.0005`, initial results:

| n per axis | max continuous-PDE velocity error |
|---:|---:|
| 40 | 1.4090e-9 |
| 64 | 1.5346e-11 |
| 96 | 6.4238e-13 |
| 128 | 1.0356e-12 |

The highest resolutions reach a float64 projection/roundoff floor; these four
points alone do not prove asymptotic exponential convergence. KH tests and
plots are recorded in `build/kh_weak/`, with separate time/grid comparisons.

## Follow-up: boundary stripes and their resolution

The original weak-velocity implementation above is retained as a diagnostic
comparison, not the recommended KH formulation: its t=6 boundary stripes were
not removed by weak energy stability alone. The follow-up uses a compatible
BSPF streamfunction and exponential enrichment of the physical outflow layers.
See [the controlled diagnosis, direct formulation, and validation](kh_stream_boundary_enrichment.md).
