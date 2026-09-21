# JAX PDE notebook migration

**Accuracy update:** Schrödinger and the beam use resolved Gauss quadrature and
exact linear evolution, with maximum field errors about 5e-9 and 1.2e-9.
See the [historical diagnosis](jax_pde_accuracy_diagnosis.md) and the
[complete beam correction](jax_beam_correction.md). Initial midpoint results
at the end of this document are retained as a historical record.

All notebooks in `examples/pde/` now use `pybspf` for spatial operators and
JAX for numerical assembly and time evolution. Matplotlib handles visualization;
there are no source-path modifications, SciPy evolution routines, external
`py-pde` dependencies, or duplicated solvers in these notebooks. Stored outputs
are cleared. Install the JAX notebook extra and execute the cells to reproduce
results.

## Model and formulation choices

- **Burgers:** same traveling-front problem and moving Dirichlet data. Only
  interior samples are evolved; exact endpoint values are reconstructed at
  each RK4 stage. A single `derivatives` call shares the fit/FFT for first and
  second derivatives. The tutorial default is N=129, viscosity 0.05, T=2,
  degree 7, dt=0.001. The previous N=1024, viscosity 0.01 research configuration
  had a much sharper front. Refining space and respecting explicit diffusion
  stability are necessary when changing viscosity.
- **Schrödinger:** same complex Gaussian packet, domain [0,20], Neumann walls,
  and T=2.5. N=513, degree 7, 48 spline functions, Gauss order 10, and exact
  discrete linear evolution (no time-step error). The external finite-difference comparison
  is replaced by an independent continuum cosine expansion using 96 modes and
  2049 quadrature points; 128 modes check reference truncation.
- **Euler–Bernoulli:** same unit cantilever under a suddenly applied unit
  constant load, initially at rest, T=3. N=129, degree 5, resolved Gauss order 8,
  and all four clamp/free-end conditions constrained on the full BSPF field.
  Exact forced modal evolution uses an SVD of the mass-scaled curvature factor.
  A stable 256-mode analytical reference is checked against 512 modes.
- **2D diffusion:** already migrated; unchanged by this migration. Its
  spline-jet boundary treatment measures the corrected field's actual flux
  rather than claiming exact strong enforcement.

The Schrödinger and beam discretizations intentionally change from ad-hoc
endpoint repair and strong collocation to symmetric quadrature weak forms.
A probe of direct full-field endpoint elimination found complex eigenvalues
in the nominally self-adjoint spatial operators. Merely porting those matrices
to JAX would retain spurious growth modes in conservative time evolution.

## Reusable infrastructure

`galerkin_1d(plan, derivative_order=k, constraints=..., quadrature_order=...)`
assembles `M=Q.T@W@Q` and `K=G.T@W@G`, where Q and G evaluate the physical
trial functions and their derivatives on resolved quadrature nodes. Homogeneous
full-field constraints are `(side, derivative_order)` pairs. The beam uses
`((0,0),(0,1),(1,2),(1,3))`; Schrödinger retains natural weak Neumann conditions.

`integrate_schrodinger` and `integrate_elastic` provide exact discrete linear
evolution. The elastic solver retains `weak.derivative_values` and applies SVD
to the mass-scaled derivative factor instead of diagonalizing its squared
stiffness matrix. Constant forcing is integrated consistently. See the beam
note for stable energy diagnostics and its analytical reference.

These are small dense 1D building blocks, not scalable tensor PDE solvers.
Spatial convergence, boundary residuals, conditioning, and reference truncation
remain independently tested. `integrate_linear_midpoint` is still available,
but is no longer used by either corrected conservative example.

## Initial midpoint validation on the development CPU (historical)

JAX x64, corrected local OpenBLAS, four OpenMP threads; maximum errors are over
all saved times and spatial samples:

| Problem | Field error | Difference with half dt | Conservation / boundary check |
| --- | ---: | ---: | --- |
| Burgers | 3.98e-9 | 9.70e-13 | Dirichlet residual 1.67e-16 |
| Schrödinger | 1.59e-2 | 3.75e-3 | Absolute norm drift 4.11e-14 |
| Cantilever | 1.92e-3 | 6.57e-6 | Relative energy drift 2.96e-7; clamp slope 5.18e-12 |

Schrödinger refinement from 257 to 513 samples decreased the field error from
7.20e-2 to 1.59e-2 at the same step. Cosine-reference tail differences were
1.32e-12. The beam's four-versus-six-mode difference was 9.57e-6, substantially
below the spatial error. Beam energy is measured about its static equilibrium;
the unshifted kinetic plus bending energy alone is not conserved under a load.
Conservation alone is not an accuracy test. The beam's ill-conditioned dense
fourth-order system has measurably greater roundoff drift than Schrödinger.

Unit tests check polynomial bending energy, full clamp enforcement, positive
mass/stiffness, the Neumann constant mode, complex norm conservation, midpoint
second-order convergence, JIT/PyTree use, and rejection of noisy plans.
The updated Schrödinger result and implementation are recorded in the linked
accuracy diagnosis above; the midpoint numbers in this table are historical.

The four PDE notebooks are registered in `packages/models/tests/test_examples.py`; their
numerical assertions run verbatim in fresh headless processes.

```sh
OMP_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python docs/diagnostics/run_with_local_blas.py -m pytest \
  -c jax/pyproject.toml tests/test_galerkin.py tests/test_time_integration.py
OMP_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python docs/diagnostics/run_with_local_blas.py -m pytest \
  -c jax/pyproject.toml packages/models/tests/test_examples.py -k pde
```

The notebook runner sets one OpenMP thread before starting each fresh process;
its numbers can differ slightly from the four-thread record above. On an
unpatched affected OpenBLAS installation, use `OMP_NUM_THREADS=1` before Python
starts. CPU validation does not establish GPU/TPU behavior or performance.

## Sine–Gordon: nonperiodic kink–antikink collision

The [Sine–Gordon notebook](../examples/pde/sine_gordon_1d.ipynb) solves
`u_tt = u_xx - sin(u)` with exact moving Dirichlet traces. Its shifted kink–antikink collision has unequal, time-dependent endpoint
values on a short interval, so the boundary test is substantial. The reusable JAX
solver uses quadrature projection, boundary acceleration lifting and RK4.
Measured displacement error is 1.3e-9; energy is checked against independent
quadrature of the analytic two-soliton field and boundary transfer, not assumed constant. See
[validation and limitations](jax_sine_gordon_example.md).

## Boundary-driven Alfvén waves

The [Alfvén notebook](../examples/pde/alfven_1d.ipynb) adds a linear
variable-density cavity with independently driven/fixed endpoints. It checks
a uniform cavity against a method-of-images solution, refines the
inhomogeneous simulation, and uses BSPF spatial integration and time
antiderivatives to check energy against boundary work. See the
[formulation and validation](jax_alfven_example.md).

## Parallel kinetic dynamics with open reservoirs

The [parallel kinetic notebook](../examples/pde/parallel_kinetic_1d.ipynb)
evolves two unequal warm beam pulses on a finite 1z1v rectangle under a
prescribed constant parallel acceleration. It supplies only incoming boundary
data; outgoing particles leave without wrapping. Resolved tensor weak forms
and upwind boundary fluxes provide stable transport. BSPF moments and time
antiderivatives check particle and kinetic-energy balances, including finite
velocity-domain fluxes. See [validation](jax_parallel_kinetic_example.md).

## Self-consistent Landau damping study on a finite interval

The [open-domain Landau notebook](../examples/pde/landau_open_1d.ipynb) adds
nonlinear electron Vlasov–Poisson with grounded potentials and Maxwellian
incoming reservoirs. Poisson is solved with BSPF antiderivatives at every RK
stage. A localized small perturbation produces damped field oscillations and
velocity-space phase mixing; free-energy flux and field–particle work separate
these from boundary escape. Mesh, velocity, time-step and domain-size
comparisons are included. See [validation](jax_landau_open_example.md).
