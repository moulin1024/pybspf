# BSPF Refactor Backlog

This backlog turns the package reorganization plan into an execution sequence with clear deliverables and exit criteria.

## Phase 0: Stabilize the Current Baseline

Goal: protect current behavior before moving code.

Tasks:

1. Add a package scaffold under `src/pybspf` that re-exports the legacy implementation.
2. Add import-level smoke tests for the package API.
3. Document the current public entry points and known limitations.
4. Record obvious defects in the legacy code that should be fixed during migration.

Deliverables:

- installable `pyproject.toml`
- `src/pybspf` package scaffold
- minimal test scaffold
- migration backlog document

Exit criteria:

- `from pybspf import BSPF1D, Grid1D, PiecewiseBSPF1D` works
- current users can still import the legacy module directly

## Phase 1: Extract Data and Configuration Primitives

Goal: move low-risk foundational types without changing numerical behavior.

Tasks:

1. Move array aliases and shared typing helpers into `pybspf/types.py`.
2. Move `Grid1D` into `pybspf/grid.py`.
3. Move knot generation helpers into `pybspf/knots.py`.
4. Add focused tests for uniform-grid validation, spacing, and knot generation.

Deliverables:

- `types.py`
- `grid.py`
- `knots.py`
- unit tests for grid and knot logic

Exit criteria:

- the new modules are imported by the package instead of the monolith
- numerical results for grid metadata and knot vectors match the legacy behavior

## Phase 2: Extract Basis and Boundary Operators

Goal: isolate reusable spline building blocks.

Tasks:

1. Move `BSplineBasis1D` into `pybspf/basis.py`.
2. Move `EndpointOps1D` into `pybspf/boundary.py`.
3. Add tests for basis matrix shapes, derivative matrix caching, and boundary operator construction.
4. Keep all public behavior unchanged by adapting imports in the legacy operator.

Deliverables:

- `basis.py`
- `boundary.py`
- unit tests for basis and endpoint operators

Exit criteria:

- `B0`, `BT0`, and `BkT(k)` values match the legacy implementation
- endpoint constraint matrices are reproducible across the refactor

## Phase 3: Extract Residual Correction and Solver Infrastructure

Goal: centralize algorithmic kernels that are reused by multiple operations.

Tasks:

1. Move `ResidualCorrection` into `pybspf/correction.py`.
2. Create `pybspf/kkt.py` for KKT assembly, LU caching, and coefficient solve helpers.
3. Replace direct KKT assembly in the operator with calls into the solver module.
4. Add tests for solve consistency across different `lam` values.

Deliverables:

- `correction.py`
- `kkt.py`
- solver tests

Exit criteria:

- spline coefficients computed through the new solver match the legacy path
- LU caching behavior is preserved

## Phase 4: Rebuild the Main 1D Operator Around Composition

Goal: make `BSPF1D` a thin orchestrator instead of a monolith.

Tasks:

1. Create a proper `BSPF1D` implementation in `pybspf/operators/bspf1d.py`.
2. Inject or construct grid, basis, boundary, and solver helpers explicitly.
3. Preserve the user-facing constructors and method names.
4. Keep `bspf1d` as a compatibility alias.

Deliverables:

- real `BSPF1D` class in the package
- compatibility alias for `bspf1d`

Exit criteria:

- common examples run through `BSPF1D` without importing the root-level monolith
- compatibility imports still function

## Phase 5: Split Operation Families

Goal: isolate the main numerical workflows so they are easier to test and maintain.

Tasks:

1. Move differentiation logic into `pybspf/ops/differentiation.py`.
2. Move integration and antiderivative logic into `pybspf/ops/integration.py`.
3. Move interpolation logic into `pybspf/ops/interpolation.py`.
4. Keep `BSPF1D` methods as thin wrappers over these operation modules.

Deliverables:

- `ops/differentiation.py`
- `ops/integration.py`
- `ops/interpolation.py`
- operation-level tests

Exit criteria:

- each operation family can be tested independently of unrelated methods
- duplicated solve/setup code is reduced

## Phase 6: Clean Up CPU/GPU Backend Boundaries

Goal: reduce repeated backend branching and make device behavior explicit.

Tasks:

1. Consolidate CuPy detection and device validation in `backend.py`.
2. Replace repeated NumPy/CuPy branching with backend helpers where safe.
3. Add CPU/GPU parity tests for supported methods.
4. Fix or remove broken GPU-only paths discovered during test migration.

Deliverables:

- cleaned backend abstraction
- parity tests
- resolved GPU edge cases

Exit criteria:

- supported CPU/GPU methods produce numerically consistent results
- unsupported device combinations fail with clear errors

## Phase 7: Migrate Piecewise and Legacy Compatibility Layer

Goal: finish the package transition without breaking downstream users.

Tasks:

1. Move `PiecewiseBSPF1D` into `pybspf/operators/piecewise.py`.
2. Update it to depend on the package operator rather than the monolith.
3. Add piecewise tests around breakpoint handling and derivative stitching.
4. Leave the top-level `bspf1d.py` as a temporary compatibility shim or freeze it as legacy.

Deliverables:

- `piecewise.py`
- piecewise tests
- documented compatibility strategy

Exit criteria:

- package users do not need the root-level monolith for normal usage
- legacy import paths are still available during transition

## Phase 8: Documentation, Packaging, and Release Hygiene

Goal: make the package usable by others with minimal repo context.

Tasks:

1. Expand `README.md` with install, examples, supported features, and limitations.
2. Add API docs and a short design document.
3. Add CI steps for imports and tests.
4. Prepare versioning and deprecation notes for the old API.

Deliverables:

- polished README
- package docs
- CI configuration

Exit criteria:

- a new user can install the package and run the basic examples without reading the monolithic file

## Immediate Next Actions

1. Keep the scaffold package importing from the legacy file until tests exist.
2. Extract `Grid1D` and knot logic first because they are low-risk and foundational.
3. Add numerical regression tests before touching the derivative code paths.
4. Fix obvious defects only when covered by tests or when blocking migration.
