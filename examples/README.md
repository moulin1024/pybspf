# Examples: staged migration to JAX

Examples stay in notebooks and call the numerical infrastructure directly.
Install from the repository root, then select that environment as the kernel:

```sh
python -m pip install -e './jax[test,notebook]'
python -m pytest -c jax/pyproject.toml jax/tests/test_examples.py
```

The migrated notebooks enable JAX x64 explicitly, use `bspf_jax`, and contain no
source-path modifications or embedded spline solvers. All plotting happens after
JAX computations. Their stored outputs are cleared to avoid presenting stale
NumPy results as JAX results. First calls include compilation; times are not
performance claims.

## Ready: JAX notebooks

| Notebook | What it demonstrates | Checks |
| --- | --- | --- |
| [1D differentiation](operation/differentiate_1d.ipynb) | Nonlinear-phase signal; fit and Fourier derivative | Analytical derivative and mesh refinement |
| [Noisy 1D differentiation](operation/differentiate_1d_noisy.ipynb) | Paired Gaussian-noise study of FD versus local Chebyshev (LDC) endpoints | Clean bias, 64 realizations, total error and boundary/interior noise amplification |
| [Noise-aware differentiation in 1D–3D](operation/differentiate_noisy_1d_3d.ipynb) | Unified `noise_std` API; joint spline/Fourier regularization | Analytic gradients, paired noise, per-axis discrepancy diagnostics |
| [1D integration](operation/integrate_1d.ipynb) | Recover the signal from sampled derivatives | Primitive error, definite integral, refinement |
| [2D differentiation](operation/differentiate_2d.ipynb) | Seeded smooth modes plus a circular tanh transition | Analytical x derivative and refinement |
| [3D differentiation](operation/differentiate_3d.ipynb) | Shifted Taylor–Green vortex restricted to a nonperiodic box; Chebyshev endpoint estimation | Full analytic Jacobian, curl, divergence, Laplacian, face mismatches and refinement |
| [Burgers](pde/burgers_1d.ipynb) | Traveling viscous front; stage-wise Dirichlet data and JAX RK4 | Exact field, boundary residual, halved step |
| [Schrödinger](pde/schroedinger_1d.ipynb) | Reflecting Gaussian packet; resolved BSPF weak form and exact linear phases | Cosine-series reference, spatial/quadrature refinement, norm conservation |
| [Focusing NLSE](pde/nlse_1d.ipynb) | Traveling bright soliton; resolved cubic projection and fourth-order interaction-picture evolution | Full complex reference, space/time refinement, norm, energy, and width |
| [Open-domain Landau damping](pde/landau_open_1d.ipynb) | Nonlinear self-consistent Vlasov–Poisson; Maxwellian inflow and grounded finite endpoints | BSPF Poisson primitives, phase mixing, field–particle exchange, free-energy boundary flux, spatial/velocity/time/domain comparisons |
| [Open parallel kinetics](pde/parallel_kinetic_1d.ipynb) | Two unequal warm beams in a prescribed parallel field; 1z1v with incoming-only reservoir data | Characteristic reference, refinement, incoming trace error, negativity, BSPF particle and energy balances |
| [Boundary-driven Alfvén waves](pde/alfven_1d.ipynb) | Variable-density, line-tied cavity; a driven left footpoint and fixed right endpoint | Independent uniform-cavity reference, space/time/quadrature refinement, BSPF energy and boundary-work integrals |
| [Sine–Gordon](pde/sine_gordon_1d.ipynb) | Kink–antikink collision on a finite interval; moving Dirichlet data and resolved sine projection | Exact displacement/velocity, space/time/quadrature refinement, boundary energy transfer |
| [Nonperiodic KdV](pde/kdv_1d.ipynb) | Finite-interval soliton with moving Dirichlet values and prescribed right slope | Field, boundary residuals, space/time/quadrature refinement, finite-interval mass and quadratic integral |
| [Euler–Bernoulli beam](pde/euler_bernoulli_1d.ipynb) | Loaded cantilever; four full-field constraints and exact modal evolution | 256/512-mode reference, static solution, frequency, space/quadrature refinement, all boundary residuals, energy |
| [2D diffusion](pde/diffusion_2d.ipynb) | Cosine-mode heat equation; JAX spatial operators and RK4 | Field error, trapezoid mass drift, actual boundary flux, halved time step |

Defaults are smaller than the previous research sweeps. In particular, the 1D
signal uses beta=1.2 (beta=1.05 restores the sharper original signal), and the
2D tanh transition is wider. The 2D comparison now uses the **same constrained
spline fit**, with and without Fourier correction, instead of an independently
implemented unconstrained tensor fit.

JAX fields use `(nx, ny, nz, ...)` with `meshgrid(indexing="ij")`; the old
NumPy facade uses `(ny, nx)` in 2D. This change is made explicitly in the notebooks.
The diffusion example constrains first-derivative spline jets; it measures the
full corrected field's flux, which is not guaranteed to vanish exactly.

The former duplicate `pde/diffusion_2d.py` is replaced by its notebook, so the
same example no longer has two independently maintained implementations.

## PDE discretizations and remaining research workflows

**Accuracy status:** Schrödinger and the beam now use resolved quadrature and
exact linear propagation. Maximum field errors are about 5e-9 and 1.2e-9,
respectively, for their stated examples. See the
[historical diagnosis](../docs/jax_pde_accuracy_diagnosis.md) and
[complete beam correction](../docs/jax_beam_correction.md).

All ten notebooks in `pde/` now use JAX. Burgers evolves interior samples with
stage-wise boundary values. Schrödinger and the cantilever use resolved
quadrature weak forms with exact linear evolution. Schrödinger's zero flux is
a natural weak condition. The beam now constrains all four clamp/free-end
conditions using full BSPF derivative rows and checks their residuals.
The dense weak-form infrastructure is for modest 1D systems, not large tensor grids.
See [PDE validation](../docs/jax_pde_examples.md) for the changed formulations,
defaults, measured errors, and limitations.

`navier_stokes/`, `bvp/`, and `turbulence/` remain research workflows outside
this migrated suite. The old Poisson examples remain removed pending a new
implementation.

The 3D vortex uses a box of length `3*pi/2` in each direction. Translation alone
would preserve periodicity on a full-period box; the shortened box and measured
opposite-face mismatches establish nonperiodic boundary data. The initial grid
is rectangular `(33, 41, 49)` to exercise axis layout before cubic refinement.
This is a differentiation benchmark, not a vortex time-evolution simulation.
The 3D vector operators use outer JIT. On the development OpenMP OpenBLAS build,
launch Jupyter with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m jupyterlab`
and start a fresh kernel. These settings prevent the observed concurrent BLAS
solve failure; the notebook test runner sets them before starting Python.
See [the runtime diagnosis](../docs/3d_jit_diagnosis.md).

## BSPF incompressible cavity

[`pde/cavity_2d.py`](pde/cavity_2d.py) runs a unit-square, regularized lid-driven
cavity with the existing divergence-free BSPF streamfunction basis. Defaults:
Re=100, 41×41 nodes, a smooth startup from rest, and second-order semi-implicit
time stepping. It saves flow plots, numerical states, and diagnostics. See the
[formulation, commands, and verification](../docs/jax_cavity_2d.md).

The [MHD cavity extension](pde/mhd_cavity_2d.py) releases two localized magnetic
islands inside the same conducting cavity and advances a matching NS control.
It evolves both velocity and magnetic flux, checks energy exchange, and saves
computed states for the [animation renderer](pde/render_mhd_cavity.py). See
[the MHD equations, boundaries, and verification](../docs/jax_mhd_cavity.md).

The [axisymmetric tokamak example](pde/tokamak_axisymmetric.py) solves a
self-consistent BSPF Grad–Shafranov equilibrium with fixed filament coils and
advances a deformable, full-vector incompressible linear vertical mode. Its
exterior uses a low-density resistive approximation to vacuum. See the
[model, validation, and limits](../docs/jax_tokamak_axisymmetric.md) and
[animation renderer](pde/render_tokamak_axisymmetric.py).

The [plasma–vacuum free-boundary example](pde/tokamak_plasma_vacuum.py) replaces
that exterior fluid with a current-free magnetic vacuum and includes its
interface stress through a variational coupling. Its default now uses BSPF for
equilibrium, plasma displacement, and mapped vacuum; the old FEM vacuum is an
explicit comparator (`--vacuum-method fem`). See the [all-BSPF implementation](../docs/jax_tokamak_all_bspf.md)
and [equations and
verification](../docs/jax_tokamak_vacuum.md). The [passive wall stabilization
benchmark](pde/tokamak_confined.py) studies confinement of the elongated plasma
using a shaped ideal conducting shell; see its [scope and commands](../docs/jax_tokamak_confined.md).

The [velocity–induction MHD runner](pde/tokamak_velocity_mhd.py) now uses the
shared stream-NS differential, inertia and RK4 kernels, with tensor-Poisson
preconditioned vacuum solves. It retains the linear displacement solver as an
independent reference; see [actual reuse and validation](../docs/jax_tokamak_velocity.md).

## Initial migration validation (historical)

All four notebooks executed headlessly with their assertions enabled. The full
JAX suite passed 41 tests on the development CPU. The default diffusion run had
maximum field error about 9.9e-9, mass drift about 1e-15, and final boundary flux
about 2.1e-7. These results describe the stated cosine-mode problem, not a
general guarantee of stability or exact conservation. Accelerator execution has
not been checked for this migration.


The separate focusing-NLSE notebook keeps the linear Schrödinger example intact.
It uses a sech soliton on a wide domain, so finite-boundary effects are negligible
over the validation interval. `integrate_nlse` projects the cubic term using the
same resolved quadrature as the BSPF weak form; it does not apply a nodal phase
rotation to a dense mass matrix. The method is fourth order, not exactly
conservative, and the notebook measures field, shape, norm, and energy errors.


The KdV notebook deliberately uses a short nonperiodic interval, with endpoint
values differing by about 0.014 and a nonzero prescribed right slope. Its
reference is an exact finite-interval restriction with matching boundary data,
not a negligible-tail periodic approximation. Mass changes through the boundaries;
whole integral histories are checked against the reference. See
[the KdV validation note](../docs/jax_kdv_example.md).

### BSPF Poisson on curved B-spline domains

`pde/embedded_poisson.py` validates symmetric-Nitsche BSPF Poisson on a convex
periodic spline domain and a C-shaped domain with an empty visibility kernel.
It uses direct spline geometry, multiple-interval slice quadrature, independent
error quadrature, and modal/quadrature/cutoff studies. See
[the curved-domain validation note](../docs/jax_embedded_poisson.md).

`pde/embedded_poisson_random_mms.py` adds a 64-wave analytic random-spectrum
Poisson MMS, three bandwidths, full-space resolution studies, and a separate
sampling-sensitivity run. `pde/render_embedded_poisson_random_mms.py` plots
the fields, errors, and spectra. See [the random-wave MMS note](../docs/jax_random_wave_poisson_mms.md).

`pde/embedded_poisson_smooth_extension.py` couples two unknown BSPF fields
through exact spline-normal traces up to second order. The control and renderer
scripts distinguish interface matching from physical solution accuracy. This
experimental extension currently performs worse than direct residual solving;
see [the implementation and validation note](../docs/jax_bspf_smooth_extension.md).

`pde/normal_continuation_validation.py` separately checks value, Cartesian
gradient and Laplacian errors for continuation along analytic spline normals,
including degree/width sensitivity and seam checks. See
[the normal-collar validation](../docs/jax_normal_continuation.md).

`pde/regularized_normal_poisson.py` adds modal cutoff regularization, a shared
BSPF representation for seam continuity, and unknown-field collar constraints
in a Poisson least-squares solve. See [the coupled experiment](../docs/jax_regularized_normal.md).

`pde/convex_poisson.py` implements the accuracy benchmark for
`bspf_jax.ConvexPoissonPlan`: exact convex spline geometry, an arclength
H^(3/2) boundary norm, analytic BSPF derivatives, H2 coefficient scaling,
and reusable SVD solves. See [the solver guide](../docs/jax_convex_poisson_solver.md).

## 固定网格的凸区域 Poisson 输出

`pde/convex_poisson_fixed_grid.py` 使用 `ConvexPoissonGridPlan` 返回指定网格上的解。
`pde/benchmark_convex_poisson_grid.py` 复用已求出的 MMS 系数，比较缓存张量求值与
逐点求值的成本。它只衡量输出阶段；参考求解后端仍使用稠密 SVD。
详见 [固定网格接口](../docs/jax_convex_poisson_fixed_grid.md)。

`pde/convex_poisson_tensor.py` 对照保持原曲边积分的无矩阵算子、单层迭代、
两层粗空间修正和参考 SVD。实验后端同时提供投影粗空间与代数谱粗空间，
默认拒绝未收敛结果。详见 [两层求解器说明](../docs/jax_convex_poisson_tensor.md)。

## 内孤立波斜坡算例

[`pde/isw_slope`](pde/isw_slope/README.md) 用 `pybspf.ClosedBSPFLine` 和
`bspf_jax.mapped_boussinesq.MappedBoussinesq` 构造二维无滑移 Boussinesq 算例。
保留冻结初场、变水深映射、float64 RK4/CG、检查点和 NetCDF 导出。
安装与单线程运行命令见算例说明。
