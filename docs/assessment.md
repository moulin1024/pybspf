# Overall examination and refactor status

The library already has a useful foundation: a `src/` package layout, package-owned
numerical operations, composition of grids/bases/constraints, factorization caches,
and numerical comparisons against a frozen legacy implementation. The main gaps
were inconsistent device handling, duplicated solve paths, stale documentation,
and correctness failures hidden behind those structural issues.

## Implemented core refactor

| Finding | Change |
| --- | --- |
| Fitting and differentiation duplicate KKT assembly and device logic | Shared validated solve in `ops/_common.py`; integration uses it too |
| Batched differentiation loops over columns | One matrix RHS solve and an FFT along the sample axis |
| 2D construction and evaluation force NumPy conversion | Backend-preserving geometry and field handling |
| Complex GPU differentiation uses half-length frequencies | Full FFT frequencies for complex input on both backends |
| Complex fitting discards imaginary parts | Common float64/complex128 normalization |
| Invalid grids reach spline/FFT operations | Explicit dimensionality, finite, increasing, and uniform checks |
| Basis cache keys only the first evaluation coordinate | Cache fixed-grid derivatives only; evaluate arbitrary grids afresh |
| Integration duplicates fitting and crosses device boundaries | Shared solve and backend-native integration arrays |
| Partial integrals add the full-domain residual | Piecewise-linear residual integration over the actual bounds |
| Piecewise code silently omits short segments | Fail clearly; preserve every sample in successful constructions |
| `correction="none"` is ignored by differentiation | Honor the setting and reject unknown strategies |
| Neumann indexing accepts value-only constraints | Require `order >= 2` for a specified flux |
| CUDA configuration contains a hardcoded workstation path | Remove environment mutation from the core package |
| Top-level imports eagerly load specialized solvers | Lazy exports retaining existing import names |
| Public methods are attached after class definition | Declare bindings in the class body |
| Array annotations describe only real NumPy arrays | Include real/complex NumPy and optional CuPy arrays |
| Legacy shim breaks regression test collection | Restore its `_Knot` compatibility export |
| README is effectively empty and API docs name removed methods | Rewrite installation, examples, shape/device contracts, architecture, and migration notes |
| Test discovery includes executable examples | Restrict default discovery to `tests/`; register GPU/performance markers |

New regression tests cover invalid grids/shapes/orders, complex fitting and
batched differentiation, empty batches, repeated refinement, evaluation-cache
collisions, partial integration, correction settings, regularization, piecewise
complex data, and optional CUDA parity. Existing numerical regressions remain.
Absolute tolerances were added only to roundoff-sensitive comparisons near zero
where matrix and vector BLAS paths differ.

## Verification on this checkout

- Full suite: **80 passed, 3 GPU tests skipped, 4 pre-existing solver failures**.
  Command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --tb=short`.
- New contract suite: **26 passed, 3 skipped**.
- The existing performance regression passed in the full run.
- Wheel build succeeded with `python -m pip wheel . --no-deps --no-build-isolation`.
- Installed-wheel imports, lazy solver exports, and the CPU README examples passed
  in an isolated Python process outside repository import paths.
- CuPy execution is not verified on this machine; the GPU tests must run on CUDA
  hardware before claiming release support.
- Existing uncommitted solver, decomposition, and example work was preserved.

CI runs the functional suite on Python 3.10 and 3.13. It deliberately leaves
existing solver failures visible; there are no blanket skips or expected-failure
markers to manufacture a green build.

## Release blockers: solver contracts

These failures were reproduced before the core edits:

1. `test_poisson2d_hybrid_dst_solver_reproduces_discrete_homogeneous_solution`
2. `test_poisson2d_hybrid_dst_solver_handles_general_dirichlet_traces`
3. `test_poisson2d_boundary_corrector_02_matches_dirichlet_edges`
4. `test_poisson2d_fft_corrected_solver_recovers_periodic_mode`

The first two tests specify a five-point **negative discrete Laplacian**, whereas
`solve_hybrid_dst` documents a **positive continuous Laplacian** and its DST helper
uses continuous spectral eigenvalues. Fixing only the sign cannot reconcile the
operators. The direct Galerkin solve also has inconsistent documentation about
its sign convention. Define the PDE contract per public method, then add
manufactured-solution and boundary-residual tests for that exact operator.

The corrector test requests a zero-mean gauge, while the implementation describes
hard Dirichlet values. Subtracting the mean would change those values. Decide
whether this object is a Dirichlet lift or a periodic correction before modifying
its gauge.

The final test calls `solve_fft_corrected_02`, which is absent. The existing
`solve_dst_corrected_02` uses a different transform and returns five values rather
than six. It should not be silently aliased to the obsolete method.

## Next architectural work

1. **Resolve the solver contracts above.** Keep core array operators and PDE
   methods separately documented; add PDE sign conventions to every solver.
2. **Split `solvers/poisson2d.py`.** At over 1,300 lines it combines Galerkin
   assembly, harmonic lifts, DST/FFT methods, boundary corrections, and POD
   experiments. Extract those by mathematical responsibility after their tests
   agree on the equations.
3. **Separate research APIs from the stable API.** Keep current re-exports for
   compatibility, but designate directional splits, Neumann precomputations, and
   experimental corrections explicitly before a versioned release.
4. **Run CUDA CI.** Exercise real/complex inputs, explicit knots, 2D axes,
   integration, and piecewise boundaries. Record supported CuPy/CUDA versions;
   avoid treating an import check as proof of device compatibility.
5. **Finish validation and typing.** Public constructor annotations can accept
   array-like CPU inputs more precisely; knot multiplicity/domain coverage and
   lower-level helper shape contracts deserve focused tests. Add type-checking
   once the remaining research interfaces stabilize.
6. **Review cache lifetime and mutation.** KKT caches are unbounded by `lam`, and
   exposed grid/basis arrays remain mutable. Define explicit cache clearing and
   thread-safety rules if operators are to be shared in long-running services.
7. **Release hygiene.** Confirm the proprietary license is intentional, choose
   publication metadata, and test wheels independently of repository imports.
   Optional NGSolve examples need their own dependency/test environment.

The old [phase backlog](refactor_backlog.md) is retained as history, not as an
accurate description of what remains to be migrated.
