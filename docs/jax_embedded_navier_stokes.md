# Fixed-obstacle BSPF Navier–Stokes

The public module is `bspf_models.fluids.embedded_navier_stokes`. It supports a
fixed axis-aligned elliptical obstacle inside a 2D Cartesian rectangle, constant
positive viscosity, fixed velocity boundary data, and optional natural right
outflow. It imports neither `scratch` nor `examples`, and uses no complex-analysis
or Goursat representation. General moving geometry and 3D are not implemented.

**Structural incompressibility is now the default.** The former pressure-stabilized
formulation requires explicit `incompressibility="stabilized"`. The previous p=9 Re=1000
results were produced with that former formulation and remain unchanged. The new
p=7 run starts from its own structurally constrained Stokes field.

## Quick start

Install `pybspf` and `packages/models`, then run:

```sh
python examples/pde/embedded_navier_stokes.py --reynolds 100 --steps 10
```

```python
import jax
import numpy as np
from bspf_models.fluids.embedded_navier_stokes import plan_embedded_navier_stokes2d

jax.config.update("jax_enable_x64", True)

def boundary(points, tag):
    velocity = np.zeros_like(points)
    if tag == "outer":
        inlet = np.isclose(points[:, 0], -2)
        velocity[inlet, 0] = 1.5 * (1 - (points[inlet, 1] / 1.5)**2)
    return velocity  # tag == "hole": no slip

plan = plan_embedded_navier_stokes2d(
    boundary, dt=1e-4, viscosity=0.38 / 100,
    cells=3, degree=3, order=40, convection_order=40,
    edges=[np.linspace(-2, 4, 4), np.linspace(-1.5, 1.5, 4)],
    outflow=True, linear_backend="dense",
)
state = plan.stokes_initial_state()
state = plan.advance(state, 10)
fields = plan.evaluate(np.array([[1., 0.], [2., 0.]]), state)
# Columns: u, v, recovered p, ux, uy, vx, vy
```

This is an execution example, not a high-Re resolution study. The initializer
solves Stokes at the plan's viscosity. Boundary tangential velocity is imposed
weakly by SIPDG/Nitsche; its normal component is explicitly constrained.

## What structural mode enforces

The spatial setup builds three sets of linear constraints on the velocity:

1. Zero volume divergence in the represented derivative space.
2. Zero normal-velocity jump on every inter-aggregate face.
3. The prescribed normal velocity on Dirichlet boundaries.

Two constraint backends are available. `dense_reference` compresses quadrature
rows and performs a bounded global SVD. `implicit_qr` first constructs a local
zero-divergence basis on each aggregate (small dense SVDs), then applies
SuiteSparseQR to the sparse global normal-trace operator. Its Q and global null
basis remain implicit. It never converts the global constraint matrix to dense.
Neither construction is an analytic commuting-complex proof for arbitrary
enriched functions. Numerical rank and independent physical audits both matter.

The mixed time-step operator is

```
[alpha M + nu K   D.T] [u]      [momentum load]
[D                0 ] [lambda] = [d            ]
```

The multiplier block is exactly zero. There is **no pressure jump penalty in the
continuity equation**, and no viscosity-dependent relaxation of incompressibility.
The generalized multiplier includes volume and trace reactions. Scalar pressure
is recovered separately from that reaction using the original weak pressure
gradient and jump regularization. This diagnostic recovery cannot change the
velocity or its constraints. Its accuracy must still be assessed independently.
Closed domains use zero-mean recovered pressure; the natural outlet fixes the
pressure level in open domains.

After construction, a different quadrature rule (`max(order+7, 2*degree+9)`) builds
independent audit operators. Initialization and every accepted time step must
satisfy:

- `||div u|| <= tol * max(||grad u||, 1)`;
- `||jump(u.n)|| <= tol * max(||u||_L2, 1)`;
- `||u.n-g.n||_boundary <= tol * max(||g.n||_boundary, 1)`.

The default `incompressibility_tolerance` is `1e-9`. These are nondimensional,
quadrature-based norms. They check the reconstructed field, not just the reduced
constraint rows. A small mixed-system residual alone never makes a structural
step valid. Failed audits are rejected; the implementation does not fall back to
the old stabilized formulation or silently weaken tolerances.

## Current setup and runtime limits

The default `constraint_backend="dense_reference"` is capped at **2048 velocity
DOFs**, even with `linear_backend="sparse"`. It retains pure JAX runtime after
NumPy/SciPy setup. JAX `"dense"` triangular solves are capped at 4096 total
unknowns. JAX `"sparse"` uses sequential substitution with O(nnz) storage;
reverse-mode differentiation through its dynamic loops is not supported.

For larger cases explicitly select:

```python
plan = plan_embedded_navier_stokes2d(
    boundary, dt=1e-4, viscosity=0.00038,
    cells=11, degree=7, order=96, convection_order=48,
    edges=channel_edges, outflow=True,
    constraint_backend="implicit_qr", linear_backend="host_sparse",
    constraint_rank_tolerance=1e-9,
    **geometry_prior_options(7),
)
# ... advance/evaluate ...
plan.close()  # release the native QR worker
```

Here `channel_edges` is a pair of arrays with 12 entries each, and
`geometry_prior_options` is imported from the solver module. The runner
`scratch/obstacle_stokes/structural_p7_run.py` contains the exact graded benchmark
configuration. The implicit Q stays sparse/implicit, but numerical rank is determined by a
dense SVD of the QR range core. Core storage is capped at 512 MB; SVD workspace
and retained factors require additional memory. This is not an asymptotically
scalable sparse-only rank algorithm. It needs installed
SuiteSparse development headers/libraries and a C++ compiler. Set
`BSPF_SUITESPARSE_PREFIX` if they are outside the standard search prefixes.
A small native worker is compiled locally and isolated from Python to avoid
conflicting OpenMP runtimes. No package download is performed.

The large-case backend is **CPU NumPy/SciPy/SuiteSparse, not pure JAX**, and is
neither JIT-compatible nor differentiable. It shares the state/diagnostic API.
It solves `Z.T H Z` with projected CG, applying Z through local sparse blocks
and implicit QR actions. A scalar BDF2 momentum LU is reused as preconditioner;
these are linear iterations, not nonlinear steady-state iterations. Pressure
recovery uses a separate sparse factorization and cannot feed back into velocity.
Arbitrary-point evaluation remains a host operation for all backends.

Rank filtering can restrict the useful enriched velocity space. Inspect
`constraint_rank`, `unconstrained_velocity_dofs` and `constraint_rank_history`.
The corrected implicit backend first protects the complete local divergence-free
polynomial subspace, then filters only the complementary enriched modes. Boundary
lifting uses this polynomial subspace whenever the normal data are representable.
These curl-polynomial dictionaries construct velocity coordinates; there is no
streamfunction PDE or scalar Poisson solve replacing the velocity formulation.

The trace rank uses singular values of a core exported from the **same** implicit
QR factorization. The effective relative threshold is `max(requested, 1e-9)` to
limit amplification of representation roundoff; it is reported as
`effective_svd_tolerance`. A smaller requested value does not override this floor.
Independent physical audits remain mandatory, since SVD rank alone does not
certify physical normal-flux accuracy. Tangential no-slip remains a weak Nitsche
condition and must be measured separately.

## Time scheme, forcing and validity

The integrator uses backward Euler startup followed by IMEX BDF2/AB2: implicit
viscosity/constraints, explicit upwind convection, and a fixed step size. A single
BDF2 factor is reused, including as the BE startup's linear preconditioner. There
are no nonlinear iterations. Convection includes volume advection, interior
upwind corrections and prescribed incoming traces. High-frequency stability
still constrains dt; enforcing incompressibility does not remove that restriction.

Use sampled force and vector-Laplacian outlet traction:

```python
def load_at(t):
    return plan.load(force=my_force(t, plan.force_points),
                     traction=my_traction(t, plan.traction_points,
                                          plan.traction_normals))
state = plan.advance(state, 20, load=load_at)
```

Sample arrays have shape `(number_of_points, 2)`. Traction means
`nu * partial_n velocity - p * normal`. The load callback is evaluated at the new
time. Structural constraint rows are fixed boundary data and must not be replaced
with time-dependent pressure equations.

With the JAX backends, `plan.step(state, load)` is JIT/scan-compatible. All
backends return `(state, diagnostics)`.
Diagnostics contain `linear_residual`, physical `kinetic_energy`, `valid`,
`divergence_l2`, `normal_jump_l2`, and `boundary_normal_l2`. `advance` raises on
invalid output. When using `step` in `jax.lax.scan`, inspect **every** validity
flag. Validity is not a spatial/temporal convergence certificate or a tangential
boundary-error estimate. No arbitrary physical energy cutoff is imposed.

## Initial data and restart

`stokes_initial_state` supplies compatible initial data. `initialize` checks
shape/finiteness and, in structural mode, the independent incompressibility
audits. It rejects incompatible velocity rather than silently projecting it.

`EmbeddedNSState` is a JAX pytree with `coefficients`, `previous_coefficients`,
`previous_convection`, `step`, and `time`. In structural mode, the coefficient tail
contains **generalized multipliers** for the dense reference, or a velocity-sized
**momentum reaction vector** for implicit QR; neither contains the old pressure
basis coefficients.
Use `evaluate` to obtain scalar pressure.

```python
np.savez("checkpoint.npz", **state._asdict())
# Rebuild the identical plan, then:
import jax.numpy as jnp
from bspf_models.fluids.embedded_navier_stokes import EmbeddedNSState
with np.load("checkpoint.npz") as data:
    state = EmbeddedNSState(*(jnp.asarray(data[k]) for k in EmbeddedNSState._fields))
```

Save the entire plan configuration and boundary definition alongside the state.
Changing dt, basis, geometry, viscosity, constraint rank settings, or formulation
requires reinitialization. Old stabilized checkpoints are not structurally
compatible checkpoints. Adaptive stepping and automatic checkpoint conversion
are not implemented.

## Legacy geometry-prior sequence

`geometry_prior_options(p)` retains the p=5,7,9,11 geometric enrichment settings.
Structural mode applies the selected backend's size limits, rank checks and
independent audits to any such enriched space. The old large benchmark can still be explicitly constructed
with `incompressibility="stabilized"`; that formulation solves `B u - C p = g`,
so it does not certify divergence-free velocity. Its structural diagnostic fields
are NaN. This compatibility option must not be described as satisfying the new
incompressibility requirement.

## Validation

`test_structural_navier_stokes.py` covers manufactured velocity and recovered
pressure at two viscosities and with enrichment; independent divergence, normal
jump and boundary audits; gradient-force velocity invariance; a closed-domain
pressure gauge; sparse runtime; incompatible data rejection; the setup size cap;
and second-order evolution of a nonstationary constrained field.

`test_embedded_navier_stokes.py` retains explicit stabilized-mode regression tests
for the previous prototype. Existing tensor-grid and stream NS APIs are unchanged.
The previous Re=1000 t=1 data remain a stabilized-method result with appreciable
divergence and must not be relabeled as a structural-method result.

A small Re=1000 check (3x3 background cells, degree 3, dt=1e-4, 20 steps to
t=0.002, initialized from its own Stokes field) produced:

| Quantity at final step | Value |
| --- | ---: |
| Independent-quadrature divergence L2 | 4.46e-13 |
| Interior normal-jump L2 | 4.75e-14 |
| Boundary-normal mismatch L2 | 3.18e-10 |
| Relative net boundary flux | 3.49e-14 |
| Mixed linear residual | 1.71e-15 |

The local run record is `build/obstacle_stokes/ns/structural_re1000_smoke.json`.
This is a constraint verification case, not a replacement for the p=9 solution
through t=1 and not a resolution or accuracy certificate.

The sparse-backend tests additionally exercise redundant QR constraints, affine
lifts, manufactured velocity/pressure and unsteady forcing in both closed and
outflow domains. They forbid CSR/CSC `.toarray()` calls throughout setup and
execution, guarding against materializing the input constraint matrix. The
explicitly bounded QR range core is dense and reported in rank metadata.

The reproducible rank probe (`scratch/obstacle_stokes/structural_rank_probe.py`)
found maximum velocity errors at three fixed manufactured-solution samples of
2.74e-10 at rank tolerance 1e-5 and 7.53e-4 at 1e-9; the respective recovered
pressure errors were 2.95e-10 and 9.28e-3. Both cases had tiny divergence and
normal-trace norms. This is an observed approximation defect of strict numerical
rank selection, not evidence that the p=7 flow has that same error. Results are
preserved in `build/obstacle_stokes/ns/structural_rank_sensitivity_before_fix.json`.

## Corrected-rank audits

Run `scratch/obstacle_stokes/audit_polynomial_reproduction.py` for independent
volume/boundary integration of degree-3 and degree-5 manufactured polynomial
solutions, including the geometry-enriched degree-5 case. The enriched case has
velocity L2 error 8.24e-10, pressure 1.22e-8, gradient 1.98e-8 and tangential
boundary error 4.41e-11 with the corrected implementation.

`audit_rank_fix_channel.py` checks the existing degree-7 channel at its **Stokes
initial field**, not at t=0.1. Independent divergence, normal jump and normal
boundary norms are 3.16e-13, 1.86e-13 and 1.14e-13. Hole tangential L2/max errors
are 2.12e-5 / 6.63e-5. The QR core is 2784 x 4512 (100.5 MB); cached-base setup
plus this solve/audit takes about 151 s on the development machine. These data
establish reproduction and boundary enforcement checks, not spatial convergence
or a performance win against another method.

## Historical Re=1000, p=7 run to t=0.1 (before rank correction)

The sparse structural run completed 1000 steps of dt=1e-4 from its own Stokes
initial field. It used 23,488 velocity coefficients, 11,112 local
divergence-free coordinates and 8,481 free coordinates after trace constraints.
All time steps passed the independent order-103 audits. Final norms were
1.938e-13 (divergence), 1.591e-12 (interior normal jump), and 1.221e-10 (boundary
normal mismatch); relative net flux was 1.044e-14. The full Dirichlet velocity
mismatch, including weakly imposed tangential data, was 4.806e-2.

Median regular step time (excluding the first ten steps) was 0.871 s. Total
run wall time including cached-base loading, structural setup, initialization,
and output was 1024.3 s; original base assembly was cached from a separate
50.3 s setup. The last step required seven projected linear CG iterations.
This is the optional CPU backend, not a JAX performance measurement.

Artifacts are under `build/obstacle_stokes/ns/structural_re1000_p7_t01`: JSON
contains every step and audit; NPZ contains five field snapshots and restart
history; `_final.png` and `_history.png` show fields and constraint histories.
The accuracy caveat above applies to this run.


## Thin-ellipse migration after rank correction

`scratch/obstacle_stokes/thin_airfoil_structural.py` runs the structural velocity
solver on a 7 x 7 graded Cartesian cut grid, ellipse axes (1, 0.1), chord 2,
viscosity 0.002 and unit freestream at 5 degrees in body coordinates. The outer
boundary is a rectangle [-60,60] x [-40,40] with uniform velocity on every side;
it is **not** the elliptical outer boundary of the earlier body-fitted SEM case.
No streamfunction PDE is solved. Pressure retains the separate weak recovery.

The source placement now caps inward-normal depth using the exact opposite
intersection of that normal with the ellipse. A curvature-radius rule alone can
put sources back into the fluid for a thin ellipse. This cap is geometry-only.

Both p=5 and p=7 completed 20 steps at dt=1e-4, ending at t=0.002 from their own
compatible Stokes initial fields. All independent incompressibility checks
passed. The independently integrated **hole** wall errors at the final time are:

| Degree | Velocity coefficients | Tangential L2 | Tangential maximum | Normal L2 |
| --- | ---: | ---: | ---: | ---: |
| 5 | 8764 | 3.528e-04 | 6.920e-04 | 1.033e-11 |
| 7 | 12612 | 2.562e-04 | 6.636e-04 | 2.680e-10 |

The p=7 final divergence and interior normal jump are 3.57e-13 and 8.10e-11.
Tangential L2 decreases only about 27%, and its maximum barely changes. Thus the
migration passes its structural and short-time checks, but this two-case study
**does not demonstrate high-order no-slip convergence or a resolved Re=1000 wake**.
The geometry enrichment budget also changes with degree; this is not a pure
single-factor p-convergence experiment. Full JSON records and restart states are
in `build/obstacle_stokes/ns/thin_airfoil_structural_p{5,7}.{json,npz}`.
Timings were collected alongside validation jobs, not as controlled performance
benchmarks. The main regression suite passed 32 tests, and the updated sparse
suite (including the new thin-source check) passed all 9 tests.


## Obstacle no-slip constraints

Set `wall_enforcement="constraint"` in `plan_embedded_navier_stokes2d` to add
obstacle tangential velocity to the existing structural constraint system.
It uses the same rank-revealed projection, polynomial-preserving local spaces,
and affine lifting. The divergence, interface-normal and wall-normal constraints
remain active. Outer tangential data still use Nitsche. The library default stays
`"nitsche"` for compatibility; the thin-airfoil runner now defaults to
`"constraint"` and writes `_noslip` files. Select `--wall-enforcement nitsche` to
reproduce the earlier baseline.

`wall_slip_error(state.coefficients)` returns an independently integrated hole
wall tangential L2 mismatch. In constraint mode, initialization and every step's
`valid` flag additionally require this error to be within the structural
tolerance scaled by the prescribed tangential data (unit lower bound).
The existing three normal/divergence diagnostics retain their original meaning.

At the same thin-airfoil grid, p, dt=1e-4 and final t=0.002:

| Degree | Treatment | Hole tangent L2 | Hole tangent max | Free velocity coordinates |
| --- | --- | ---: | ---: | ---: |
| 5 | Nitsche | 3.528e-04 | 6.920e-04 | 2067 |
| 5 | Constraint | 9.809e-12 | 7.082e-11 | 1972 |
| 7 | Nitsche | 2.562e-04 | 6.636e-04 | 3518 |
| 7 | Constraint | 9.338e-12 | 3.264e-11 | 3399 |

All 20 steps pass the independent checks. The p=7 final divergence is 3.58e-13,
interior normal jump 8.01e-11, and hole normal mismatch 1.12e-11. Median CG
iterations after the first five steps fall from 10 to 8 (p=5) and 13 to 11 (p=7).
Runs overlapped other verification, so their wall times are not controlled speed
comparisons. This remains a short evolution from Stokes initial data, not a
resolved Re=1000 wake benchmark.

An independent smooth zero-wall-velocity Stokes MMS checks approximation loss
(`scratch/obstacle_stokes/audit_no_slip_accuracy.py`). Its analytic curl defines
manufactured data only; the numerical formulation still solves velocity.

| Degree | Treatment | Relative velocity L2 | Relative gradient L2 |
| --- | --- | ---: | ---: |
| 5 | nitsche | 2.519e-03 | 6.656e-03 |
| 5 | constraint | 7.068e-03 | 1.269e-02 |
| 7 | nitsche | 1.188e-05 | 3.657e-05 |
| 7 | constraint | 3.175e-05 | 7.872e-05 |

The constrained p=5 to p=7 sequence reduces velocity error about 223-fold, but
has about 2.7 times the error of weak enforcement at the same p. Exact wall data
remove admissible trace modes; vanishing wall slip must not be interpreted as a
guarantee of better full-field accuracy. Polynomial nonzero-boundary MMS tests
also cover the new constraints in open and closed domains and recovered pressure.

The alternative `wall_penalty_factor` scales obstacle Nitsche terms and their
matching load, leaving interior penalties unchanged. Factor 16 reduces p=5 wall
L2 slip to 5.42e-5. Factor 64 fails Stokes initialization at the current 1200 total
CG-iteration budget for both p=5 and p=7; it is not the selected thin-airfoil
configuration. No failed result was accepted or used as a checkpoint.

Validation: 45 main regression tests and the additional dense/JIT wall-constraint
test passed. Result files are `thin_airfoil_structural_p{5,7}_noslip.json` and
`no_slip_accuracy.json` under `build/obstacle_stokes/ns/`.


## Bounded array projection and GPU preparation

The implicit-QR runtime profile attributed about 56% of p=5 step time to QR
projection calls (including native worker computation and communication), versus
about 8% to convection. `projector_backend="array"` now materializes the null
basis in **local divergence-free coordinates** once, in batches of 64 columns.
Runtime restrict/lift use sparse local maps and dense matrix-vector products,
with no QR worker calls. The rank selection, approximation space, affine lift,
physical tolerances and step audits are unchanged. BE/BDF2 Helmholtz operators
are also cached instead of rebuilt at every step.

This trades additional O(local_dofs * free_dofs) storage for lower runtime cost.
The individual array basis is capped at 256 MiB; exceeding this raises an explicit
error rather than allocating without bound. Use `projector_backend="implicit_qr"`
for the memory-conservative alternative. The library retains that default;
the fixed-size thin-airfoil runner defaults to `array`, adding `_array` to its
result filenames. The cap is **not** a total solver or peak-memory cap.

`plan.projector.jax_actions()` exports pure JIT-compiled JAX restrict/lift actions.
Their arrays stay on the selected JAX device, with no NumPy conversion, host
callback or native QR call inside these actions. FP64 is required explicitly.
They match the implicit and NumPy-array projections in regression tests. The
current development host exposes only a CPU device, so no GPU speed measurement
has been performed.

The complete integrator is still CPU NumPy/SciPy: CG control, sparse LU
preconditioning, convection and acceptance audits have not been moved to the
GPU. Calling the JAX projection from this CPU loop would add transfers and is
not the intended GPU architecture. The next port should keep velocity, Krylov
vectors, convection tables, operator application and audits on-device for the
whole step, use a device-suitable preconditioner, and transfer only checkpoints
and compact diagnostics. Rank revelation may remain host-side setup. At larger
problem sizes a blocked implicit device projector is needed instead of an
unbounded dense null-basis cache.

`scratch/obstacle_stokes/benchmark_projector_backends.py` compares both runtime
backends using the same plan, LU factors and checkpoint, alternating execution
order for three trials of 20 steps per backend. It requires the prior `_noslip`
checkpoints, reproducible using the thin-airfoil runner with
`--projector-backend implicit_qr`. It checks all physical acceptance flags and
the final mass-weighted velocity difference. Setup and array export are reported
separately from warmed step time. No p, dt, tolerance or quadrature is relaxed.

Paired CPU measurements (single BLAS thread; medians across the three trials):

| Degree | Implicit QR s/step | Array s/step | Speedup | Additional basis MB | Export seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.274 | 0.148 | 1.85x | 49.3 | 0.52 |
| 7 | 0.593 | 0.381 | 1.56x | 130.8 | 1.41 |

Final velocity relative L2 differences between backends are 3.50e-16 (p=5) and
4.38e-16 (p=7). Both keep their CG iteration counts; the speedup comes from
cheaper projection, not relaxed accuracy. Raw records are in
`build/obstacle_stokes/ns/projector_backend_comparison.json`. These measurements
compare projector implementations, not BSPF against SEM/FEM, and do not establish
GPU performance or the largest stable timestep.

The updated sparse-projector regression suite passes 22 tests, including batched
native-Q export, NumPy/JAX/implicit projection equivalence, the storage cap,
polynomial velocity/pressure reproduction, and array-backend manufactured time
stepping with QR-worker actions explicitly forbidden after setup.
