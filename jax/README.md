# BSPF in JAX

An independent functional implementation of **B-spline + Fourier (BSPF)**
calculus in one, two, and three spatial dimensions. The import name is
`bspf_jax`. This directory intentionally has no `__init__.py`: it must not shadow
third-party `jax`.

The implementation does not call the existing NumPy/CuPy operators. B-spline
recurrences, endpoint stencils, constrained linear algebra, FFTs, interpolation,
and integration are all expressed with JAX. NumPy is used only for validation
of static geometry in the checked constructors, and SciPy in reference tests and optional DCT/HODLR pressure compression setup.

## Install and run

The experimental [mapped spline Poisson solver](../docs/jax_mapped_spline_poisson.md)
uses four exact ellipse-to-rectangle maps with conforming scalar B-splines,
matrix-free GPU assembly and a resident GPU conjugate-gradient solve.
Its [mapped Navier–Stokes extension](../docs/jax_mapped_navier_stokes.md)
evolves the compatible divergence-free curl space with exact no-slip boundaries,
viscous patch coupling and GPU IMEX time stepping.

For a rectangle with a fixed eccentric analytic hole, see the
[smooth-field immersed Poisson prototype](../docs/jax_immersed_poisson.md).
`ImmersedPoissonPlan` reuses BSPF rectangular derivatives and the tensor
Poisson inverse, jointly fits the physical field and its hole extension, and
caches a dense SVD for repeated right-hand sides. This accuracy reference
uses NumPy/SciPy on the host; it is not a GPU/JIT implementation.

The [immersed channel-flow prototype](../docs/jax_immersed_channel_flow.md)
adds divergence-free Navier–Stokes flow around the hole, prescribed parabolic
inflow, no-slip walls, and a smooth outlet buffer with implicit damping.
Its optional [rationally corrected BSPF space](../docs/jax_hybrid_rational_flow.md)
includes the rational field in mass, diffusion and nonlinear terms, with
continuous unsteady NS MMS validation on a fixed background grid.

From the repository root:

```sh
python -m pip install -e './jax[test,notebook]'
python -m pytest -c jax/pyproject.toml jax/tests
```

Choose a JAX accelerator installation for your hardware using the
[official installation instructions](https://docs.jax.dev/en/latest/installation.html).
This project does not select devices or modify global JAX configuration.

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import bspf_jax as bspf

x = jnp.linspace(0.0, 1.0, 129)
plan = bspf.plan_1d(x, degree=5, n_basis=20, boundary_points=7)
f = jnp.sin(x) + x**2

df = jax.jit(bspf.differentiate)(plan, f)
split = bspf.decompose(plan, f)  # coefficients, spline, residual
integral = jax.jit(bspf.integrate)(plan, f)
loss_grad = jax.grad(lambda u: jnp.sum(bspf.differentiate(plan, u)**2))(f)
```

For 3D, the same primitives compose without a new solver class:

```python
x = jnp.linspace(0.0, 1.0, 33)
y = jnp.linspace(-1.0, 1.0, 35)
z = jnp.linspace(0.0, 2.0, 37)
plan = bspf.plan_3d(x, y, z, degree=3, n_basis=10, boundary_points=5)
X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
f = X**2 + Y**2 + Z**2
lap = jax.jit(bspf.laplacian)(plan, f)  # approximately 6 everywhere
```

See [the mathematics and API contract](DESIGN.md). Runnable examples are
notebooks only:

- [1D fitting, calculus, JIT and autodiff](examples/01_bspf_1d.ipynb)
- [2D/3D tensor and vector calculus](examples/02_bspf_2d_3d.ipynb)
- [Nonperiodic 3D Taylor–Green differentiation](../examples/operation/differentiate_3d.ipynb)

For CPU execution with the development OpenMP OpenBLAS build, launch numerical
processes with one BLAS thread while retaining JAX JIT:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m jupyterlab
# Or run tests from the repository root:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -c jax/pyproject.toml jax/tests
```

Set these before importing numerical libraries; restart existing notebook
kernels after changing the launch environment. The installed OpenMP OpenBLAS
0.3.32 build produced incorrect concurrent solves in both JAX and SciPy;
`OPENBLAS_NUM_THREADS` alone did not control that OpenMP build. The package does
not mutate process-wide threading settings at import. See the
[diagnosis and regression evidence](../docs/3d_jit_diagnosis.md).

A [corrected project-local OpenBLAS build](../docs/corrected_blas_build.md)
supports concurrent solves with multiple OpenMP threads on the development Mac.
After building it, launch from the repository root:

```sh
OMP_NUM_THREADS=4 python docs/diagnostics/run_with_local_blas.py -m jupyterlab
```

This selects the local library for the new process without replacing conda's
libraries. The single-thread workaround remains necessary for the original build.

## Noise-aware differentiation

For noisy samples, use the joint regularization path instead of relying on
endpoint estimation alone:

```python
plan = bspf.plan_1d(x, noise_std=sigma_prior)  # a priori absolute noise std estimate
df = jax.jit(bspf.differentiate)(plan, samples)  # samples are your noisy measurements
diagnostics = bspf.noise_diagnostics(plan, samples)
# The same noise_std option works with plan_2d and plan_3d.
```

The default `noise_std=0` preserves clean-data behavior. A positive value is the
caller's a priori estimate of sample-noise standard deviation, in data units.
The operator receives noisy observations; it does not add noise, estimate the
noise level, or need a clean reference. Multidimensional derivatives also smooth
transverse axes. See the [API, mathematics and limitations](../docs/jax_noise_differentiation.md)
and [minimal 1D–3D notebook](../examples/operation/differentiate_noisy_1d_3d.ipynb).

## Optional Chebyshev endpoint estimator

Finite differences remain the default. To estimate endpoint derivatives by a
regularized local Chebyshev fit instead:

```python
import bspf_jax as bspf

plan = bspf.plan_2d(  # or plan_1d / plan_3d
    x, y, degree=9, n_basis=18, lam=1e-6,
    endpoint_method="chebyshev",
    boundary_points=16, chebyshev_modes=12, chebyshev_alpha=1e-12,
)
```

`boundary_points` is the number of
samples per endpoint window; `chebyshev_modes` is the number of terms (polynomial
degree plus one). Modes default to `degree+1`; the Chebyshev window defaults to
`min(grid_size, 2*chebyshev_modes)`. Require at least as many modes as constrained
derivatives, and at least as many samples as modes. An augmented QR solve avoids
normal equations. `chebyshev_alpha` controls this local fit, independently of
the spline regularization `lam`; `chebyshev_penalty_power` defaults to 4.
Endpoint values are copied exactly from the data. Explicit `boundary` jets
still override the estimator at application time.

Both estimators store only two compact endpoint blocks and multiply them by
the first/last `boundary_points` samples. Their storage is independent of the
global grid length; `endpoint_jets(plan, f)` remains the public application API.

The 12-mode/16-sample choice helped the current smooth 2D example, but is not a
universal optimum. Wider windows can introduce bias and high derivative orders
amplify roundoff/noise. This option is local Chebyshev fitting, not Local Defect
Correction. It does not add the standalone benchmark's Fourier cutoff.

## 2D pressure projection

An experimental `transform_backend="dct_hodlr"` option applies DCT plus
hierarchical low-rank factors directly, with zero refinement by default. Install
`pip install -e 'jax[compression]'` for its optional host setup dependency.
See the [compressed direct-transform benchmark](../docs/jax_layered_pressure.md).

`plan_pressure_poisson2d` builds the masked pressure solver with either the
original Taylor jets or the optional Chebyshev estimator:

```python
plan = bspf.plan_pressure_poisson2d(
    x, y, endpoint_method="chebyshev",
    chebyshev_modes=14, baseline_points=18,
)
projected, result = jax.jit(bspf.project_pressure2d)(plan, raw)
assert bool(result.converged)
```

Use uniform endpoint-inclusive grids and `(nx, ny, 2)` vector data. This solves
`div(Q grad p) = div(Q raw)` with exact zero wall projection and optional
wall-gradient completion. It supports JIT, vmap, and field autodiff. See the
[pressure API and validation notes](../docs/jax_pressure_projection2d.md) for
the scalar solve, diagnostics, shape differences from NumPy, and limitations.

## 3D pressure direct core

`plan_pressure_poisson3d(x, y, z, transform_backend="dense")` and
`transform_backend="dct_hodlr"` provide the same masked tensor direct core.
Use `jax.jit(bspf.solve_pressure_poisson3d)(plan, rhs)` for compatible
`div(Q grad p) = rhs`. The application always uses zero refinement. Identical
axis grids share their factors; transforms process 2048 grid lines per batch.

This includes face elimination/recovery and null-mode lifts, but **does not yet
include physical wall-gradient completion or a 3D NS integrator**. For unique
pressure recovery tests, form `rhs = bspf.pressure_action3d(plan, p)` and use
`jax.jit(lambda a, f: bspf.solve_pressure_poisson3d(a, f, lifted=True))(plan, rhs)`.
Check `result.converged`. See the [64³–256³ comparison](../docs/jax_pressure3d_benchmark.md)
for zero-refinement accuracy, timings, memory, and the benchmark command.
A separate [FD + PyAMG comparison](../docs/jax_pressure3d_pyamg.md) reports
sparse assembly, AMG setup, V-cycle/CG iterations, and distinguishes discrete
recovery error from continuous-PDE discretization error.

## Functional design

For a computed nonperiodic Kelvin–Helmholtz example using this pressure solver,
see [the NS setup, validation, and MP4 workflow](../docs/jax_kh_nonperiodic.md).
The reusable kernels are `plan_navier_stokes2d`, `ns_rhs`, and `ns_rk4_step`;
the example uses fixed shear boundary velocities with explicit base forcing
and boundary damping.

- `Plan1D` and `TensorPlan` are frozen registered PyTrees holding explicit arrays.
  They do not contain mutable caches, methods that mutate state, or device flags.
- `plan_1d` validates geometry and precomputes a KKT factorization **outside JIT**.
  Reuse that plan for changing fields. `tensor_plan(px, py, pz)` allows different
  degree, stencil width, knots, and regularization on each axis.
- Numerical application functions are pure. They accept a plan and arrays and
  return arrays/PyTrees; they compose with `jit`, `vmap`, `jvp`, and `grad`.
- `with_regularization(plan, lam)` returns a new plan and is differentiable in
  `lam`. Its low-level contract requires a nonnegative finite scalar and an
  invertible KKT system; traced values cannot be checked by Python exceptions.
- Enable x64 before construction. The factory rejects disabled x64 rather than
  silently using float32 for a potentially ill-conditioned KKT system.

The distinction between dynamic array leaves and static metadata follows the
[JAX PyTree model](https://docs.jax.dev/en/latest/custom_pytrees.html).
Shape-changing options must be static for
[JIT compilation](https://docs.jax.dev/en/latest/jit-compilation.html).

## Shape and transformation contracts

Spatial axes are **leading axes in physical order**:

| Dimension | Scalar/batched field | Vector field |
| --- | --- | --- |
| 1D | `(nx, ...)` | `(1, nx, ...)` |
| 2D | `(nx, ny, ...)` | `(2, nx, ny, ...)` |
| 3D | `(nx, ny, nz, ...)` | `(3, nx, ny, nz, ...)` |

Use `meshgrid(..., indexing="ij")`. This explicitly differs from the older
NumPy facade's `(ny, nx)` convention. Trailing dimensions represent batches or
channels. Complex fields retain complex values throughout the calculus.

```python
from functools import partial

dxy = jax.jit(partial(bspf.mixed_partial, orders=(1, 1)))
dxx = jax.jit(partial(bspf.differentiate, axis=0, order=2))
# Equivalent: jax.jit(bspf.differentiate, static_argnames=("axis", "order"))
```

`axis`, `order`, `orders`, and `correction` are static configuration when passed
to JIT. Bounds, field values, endpoint jets, and integration constants can be
traced arrays. Constructors intentionally do not support tracing grid validation;
this release tests differentiation with respect to fields, evaluation points,
integration bounds, and regularization, not end-to-end geometry optimization.

## Scope

Implemented: constrained fits; directional and mixed derivatives; multi-order
and batched derivatives; gradient/divergence/curl/Hessian/Laplacian; complete
2/4/8-component sampled tensor decompositions; arbitrary-coordinate and tensor-grid
interpolation; axis and volume integrals; first/second antiderivatives.

This includes spatial calculus, fixed-step RK4, and small dense 1D weak-form systems
with linear implicit midpoint and exact linear propagation. General PDE solvers, adaptive time integrators,
piecewise domain segmentation, shocks, wet/dry fronts, and positivity-preserving
schemes are not part of this subproject yet. Endpoint jets constrain the **spline
fit**, not a complete PDE boundary treatment. There is no Poisson solver here.

Tests cover CPU float64/complex128. GPU/TPU performance and accuracy must be
verified on the respective hardware; JAX device portability alone is not a
hardware validation result.

## Validation record

On the development CPU with JAX 0.10.0 and x64 enabled:

- 41 tests passed (37 numerical/unit checks and 4 migrated notebook smoke tests),
  including independent SciPy basis/KKT references, real and
  complex inputs, batched and mixed derivatives, tensor reconstruction, vector
  calculus, interpolation, primitives, boundary jets, JIT, vmap, and autodiff.
- Both notebooks executed with their numerical assertions passing.
- The standalone wheel built and passed an isolated installed-package smoke test.
  Importing it leaves the JAX precision configuration unchanged and does not
  import the existing `pybspf` package.
- For exp(x) on [0,1], degree 5, 16 basis functions, and a seven-point endpoint
  stencil, max first/second derivative errors decreased under refinement:

  | Samples | First derivative | Second derivative |
  | --- | --- | --- |
  | 33 | 3.01e-10 | 4.37e-8 |
  | 65 | 3.37e-11 | 2.35e-9 |
  | 129 | 3.69e-12 | 4.62e-10 |

These are correctness checks for specified smooth problems, not a guarantee of
accuracy for arbitrary signals or a GPU performance benchmark.

## Time-dependent examples

`rk4_step(rhs, state, t, dt)` and
`integrate_rk4(rhs, initial, times, substeps=1)` provide pure fixed-step RK4 for
real/complex array or PyTree states. Histories include the initial state and
have a leading time axis. `rhs` and `substeps` must be static under JIT. Output
times must be finite and strictly increasing; each interval is subdivided into
exactly `substeps` internal steps. Choose a stable time step for your spatial
operator: no adaptive/CFL controller is implied.

`endpoint_jets(plan, field, axis=0)` exposes the estimated spline endpoint
constraints as `(2, q, *other_axes_and_batches)`. Use functional `.at` updates to
replace individual derivatives before passing `boundary=` to the spatial
operator. These constrain the spline part; inspect the final field's boundary
residual independently.

The first batch of migrated repository examples is indexed in
[examples/README.md](../examples/README.md). Notebook smoke tests execute their
actual code cells in isolated, headless Python processes (Matplotlib required).


`galerkin_1d(plan, derivative_order=1, constraints=())` assembles dense mass,
stiffness, and extension matrices from clean BSPF derivative rows and
trapezoidal quadrature. Homogeneous essential constraints are `(side, order)`
pairs (side 0 = left, 1 = right); remaining boundary conditions follow the weak
form. For example `derivative_order=2, constraints=((0, 0), (0, 1))` describes
a clamped/free beam. Reconstruct samples with `weak.extension @ coefficients`.
This is a quadrature weak form, not an exact spline Galerkin discretization;
its matrices use O(N²) storage and are intended for small 1D problems.

`integrate_linear_midpoint(operator, initial, times, substeps=1)` advances
`y' = operator @ y`, factoring once per output interval and reusing its dense
propagation matrix for internal steps. It accepts vectors or columns of
batched states, including complex data, under JIT. It is second order in time
and preserves quadratic invariants of the discrete system to roundoff; phase
accuracy still requires time refinement. See the PDE notebooks for complete
model setup, exact references, and executable accuracy/conservation checks.


For accurate conservative PDE spectra, use
`galerkin_1d(plan, quadrature_order=10)`. This integrates the actual BSPF trial
functions and their derivatives with composite Gauss quadrature, splitting at
knots and data nodes. Use at least `degree+1` points per subinterval and verify
quadrature convergence. Omitting the option retains the initial nodal
trapezoidal assembly, which can substantially under-resolve these integrals.
For consistent constant forcing, `integrate(plan, weak.extension)` returns the
integrals of the trial functions.

`integrate_schrodinger(mass, hamiltonian, initial, times)` solves the autonomous
linear equation `i M y_t = H y` using a JAX generalized Hermitian eigensystem
and exact modal phases. It eliminates time-discretization error and supports
JIT, including complex Hermitian matrices. The caller supplies positive-definite
mass and Hermitian Hamiltonian matrices. Dense setup costs O(N³); spatial and
floating-point errors remain. This is not a nonlinear or time-dependent PDE
integrator. The Schrödinger notebook uses it with resolved BSPF quadrature and
checks the complex field against an independent continuum solution.


`integrate_nlse(weak, initial, times, coupling=2., substeps=1)` evolves the
cubic equation `i M q_t = K q - g Q* W (|Qq|² Qq)` in JAX. Positive coupling
is focusing. `weak.values` and `weak.quadrature_weights` expose the trial-function
values and integration weights used by the cubic projection; construct the
weak form with resolved Gauss quadrature. Initial data and histories are free
nodal coefficients, reconstructed with `weak.extension` if constrained.

The fourth-order interaction-picture (Lawson RK4) method uses exact linear
modal phases, four projected nonlinear stages per step, and a compiled scan.
It removes the linear explicit-RK4 stability restriction, but step refinement
remains necessary for nonlinear and oscillatory accuracy. Norm and energy are
not preserved exactly. Dense setup costs O(N³), and each nonlinear stage costs
O(N*Nquad). This is a small 1D nonlinear PDE building block, not a scalable
multidimensional or adaptive solver. See
[the bright-soliton notebook](../examples/pde/nlse_1d.ipynb) for a minimal model,
independent analytic reference, and space/time/conservation checks.


`plan_kdv(spatial_plan, quadrature_order=8)` and
`integrate_kdv(plan, initial, times, boundary=boundary, substeps=...)` provide
finite-interval KdV evolution. The JAX callback `boundary(t)` returns
`[u(left,t), u(right,t), u_x(right,t)]`. Endpoint values are lifted strongly,
including their time derivatives via JVP; the right slope is a natural weak
load whose residual must be checked. Histories include all closed-grid samples.
There is no periodic endpoint identification. With nonzero boundary flux, mass
and quadratic norm generally change.

The homogeneous linear weak operator is dissipative in its mass norm. Dense
matrix-function ETDRK4 handles its strong nonnormality without relying on an
ill-conditioned eigenbasis. Output times must be uniformly spaced and increasing;
the caller chooses a stable, resolved time step. Stiff boundary forcing can
reduce the observed temporal order. See the
[nonperiodic KdV notebook](../examples/pde/kdv_1d.ipynb) and
[validation note](../docs/jax_kdv_example.md).


`elastic_modes(weak, density=1., rigidity=1.)` returns ascending angular
frequencies and mass-normalized modes using an SVD of the mass-scaled
`weak.derivative_values`. This avoids squaring the bending operator's condition
number. `integrate_elastic(weak, displacement, velocity, times, force=force, ...)`
returns displacement and velocity histories for constant forcing. Stable sinc
formulas cover rigid zero-frequency modes. The method has no time-step error;
space, quadrature, conditioning, and boundary checks are still required.

`bspf_jax.references.cantilever_step_response` supplies an independent convergent
modal benchmark using stable root/mode formulas and analytical load coefficients.
The [corrected beam notebook](../examples/pde/euler_bernoulli_1d.ipynb) enforces
all four full-field boundary conditions and validates every component separately.
See [the beam validation record](../docs/jax_beam_correction.md).

### Sine–Gordon with moving boundaries

`integrate_sine_gordon(weak, initial, velocity, times, boundary=..., substeps=...)`
solves `u_tt = u_xx - sin(u)` on a finite interval. Pass an unconstrained
`galerkin_1d(plan, derivative_order=1, quadrature_order=8)` system and a twice
JAX-differentiable callable returning the two Dirichlet values. Both returned
histories contain all closed-grid samples. Boundary velocity and acceleration
are differentiated automatically; the latter enters the mass lifting term.
The nonlinear force uses resolved quadrature, and explicit RK4 requires a
stable wave step. This dense infrastructure is for small 1D problems. See
[the notebook](../examples/pde/sine_gordon_1d.ipynb) and
[validation](../docs/jax_sine_gordon_example.md).

### Boundary-driven Alfvén waves

`plan_alfven(spatial, density=..., magnetic_field=1., permeability=1.,
quadrature_order=8)` assembles the density-weighted mass and magnetic-tension
stiffness for a static, straight-field linear Alfvén model. Density can be a
positive scalar or a vectorized function. `integrate_alfven(model, xi0, v0,
times, boundary=..., substeps=...)` imposes two moving endpoint displacements
and includes their acceleration in the mass lifting term. It returns full-grid
displacement and velocity histories. `alfven_energy` and
`alfven_boundary_power` provide quadrature energy and physical endpoint power.

The [notebook](../examples/pde/alfven_1d.ipynb) also uses BSPF `integrate` for
spatial energy and `antiderivative` for boundary work. For nonlinear products,
evaluate fields on a finer grid first: `interpolate(..., derivative=1)`
evaluates derivatives of the original fitted spline/Fourier interpolant
directly, avoiding a new fit to derivative samples. See
[validation](../docs/jax_alfven_example.md).

### Open parallel kinetic transport (1z1v)

`plan_parallel_kinetic(z_plan, v_plan, acceleration=..., quadrature_order=8)`
assembles a tensor weak BSPF transport operator for `f_t + v f_z + a f_v = 0`.
`integrate_parallel_kinetic(model, f0, times, inflow=..., substeps=...)` returns
`(nt,nz,nv)` samples. The vectorized callable `inflow(t,z,v)` supplies incoming
traces only: positive velocities at the left spatial end, negative velocities
at the right, and the appropriate velocity face selected by the acceleration.
Outgoing traces evolve freely. Inflow is weakly imposed with upwind fluxes.
The explicit RK4 step must satisfy transport stability requirements; the scheme
has no positivity limiter. The field is prescribed, with no Poisson solve or
collisions. See the [notebook](../examples/pde/parallel_kinetic_1d.ipynb) and
[validation](../docs/jax_parallel_kinetic_example.md).

### Self-consistent open Vlasov–Poisson

`poisson_dirichlet(plan, source, left=0., right=0.)` solves `phi''=source`
by first/second BSPF antiderivatives and an affine boundary correction, returning
`(phi, E=-phi')`. It accepts trailing batch axes and uses no Poisson matrix solve.

`plan_vlasov_poisson(z_plan, v_plan, temperature=1., quadrature_order=8)` sets
up electrons, fixed unit-density ions, Maxwellian inflow reservoirs and grounded
endpoint potentials. `integrate_vlasov_poisson(model, initial, times,
substeps=...)` evolves the nonlinear kinetic equation and recomputes Poisson
at every RK stage, returning distribution, potential and field histories.
`vlasov_poisson_fields(model, distribution)` evaluates the same field mapping.
The equilibrium Maxwellian is normalized with BSPF velocity integration.
Explicit time steps and velocity resolution must be checked; no positivity
limiter or energy-conservation guarantee is imposed. See the
[open-domain notebook](../examples/pde/landau_open_1d.ipynb) and
[validation](../docs/jax_landau_open_example.md).

### Experimental SBP comparison (not the BSPF accuracy-preserving fix)

`plan_navier_stokes2d(p, closure="sbp84")` selects uniform-grid SBP derivatives
(interior order 8, boundary order 4), compatible viscosity, split advection,
and an orthogonal tensor-direct pressure projection with zero refinement.
Use `ns_divergence` and `ns_project_velocity` with this option. It changes the
spatial discretization and does not retain BSPF spectral accuracy; the default
`closure="bspf"` is unchanged. See [the energy estimate and KH validation](../docs/kh_energy_stable_closure.md).

The subsequent [1D weak BSPF study](../docs/bspf_weak_advection_diffusion_1d.md)
retains the original BSPF trial space and demonstrates energy stability with
high-order PDE accuracy. It is a research prototype with JAX x64 time-stepping
validation, not yet integrated into the NS solver.

### Nonperiodic KH without a sponge

The compatible 2D streamfunction backend keeps high-order BSPF approximation,
constructs pointwise divergence-free velocity, and directly inverts its tensor
Poisson kinetic mass. Optional exponential trial functions resolve the thin
physical outflow layers without adding a damping term or lowering boundary
order. The KH example retains the original four fixed velocity boundaries.

See [diagnosis and validation](../docs/kh_stream_boundary_enrichment.md),
`bspf_jax.plan_stream_navier_stokes2d`, and `scratch/run_kh_stream.py`. Install
`bspf-jax[weak-ns]` for its SciPy/MPFR host setup; runtime remains JAX float64.


For open vertical boundaries, pass `x_boundary="open"` to
`plan_stream_navier_stokes2d`. Local outflow has zero Laplacian traction;
local inflow couples to the reference shear through a boundary-only Robin term.
Horizontal velocity remains fixed. The pointwise-divergence-free direct tensor
solve is retained. See [open KH boundary conditions](../docs/kh_open_boundary.md)
for equations, energy balance, limitations, and the movie runner.

The KH runner `scratch/run_kh_stream.py` now defaults to an external buffer of
width **1 on each side**: the region of interest is `[-3,3] x [-1,1]`, and the
computational domain is `[-4,4] x [-1,1]` on a `128 x 80` grid. It uses the
ordinary open boundary, peak sponge strength 4, and no exponential enrichment.
Absorption is exactly zero inside the region of interest. `--accuracy` retains
the fixed-boundary, no-buffer validation configuration. Explicit options still
override these defaults; `--extension 0` disables the default sponge as well.

To configure absorption through the library, build an extended-domain plan
and call `plan_stream_sponge(plan, interior=(-3, 3), strength=4)`. Pass the
returned object as `sponge=` to `stream_ns_rhs` or `stream_ns_rk4_step`.
It relaxes only the perturbation to the prescribed lift, with an exactly zero
coefficient in the interior and a smooth exterior ramp. Example runner options:
`--nx 128 --ny 80 --x-boundary open --layers --extension 1 --sponge-strength 4`.


Dynamic open boundaries are available with `x_boundary="dynamic", boundary_D0=1`.
`with_stream_dynamic_boundary(open_plan, D0=1)` reuses existing factors and
adds a direct generalized-eigenvalue inertia inverse. Volume operators and
initial projection remain unchanged. Dynamic outflow remains optional.
See [dynamic outflow experiment](../docs/kh_dynamic_outflow.md) for the exact
reference-flow boundary law, tests, KH comparison, and limitations.


Dynamic outflow can be combined with `plan_stream_sponge` on an extended domain.
The [matched narrow-layer study](../docs/kh_hybrid_outflow.md) compares widths
0.5 and 1 against width 2. At D0=1 and peak damping 4 the narrower layers stayed
stable, but the dynamic condition did not preserve the wider-domain interior
solution; see the reported quantitative differences before reducing the layer.
