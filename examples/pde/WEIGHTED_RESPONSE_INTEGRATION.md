# Low-node exponential-weight integration

**Status: experimental full adapter, not accepted for production NS.** The
Re=200 full-space audit failed (mass 2.03e-4, stiffness 1.48e-3, perturbed
nonlinear action 4.78e-3); its time evolution was correctly blocked. Raising
the fixed background factor from 2.5 to 3 still failed the background audit.
A subsequent patchwise audit placed the observed background discrepancy in the
interior rectangular patches (9.64e-6), not the corners (6.12e-13). The corner
hypothesis was therefore not supported. These prototype changes and failed
results are retained for investigation. Work then switched, at the user's
request, to the LARS-free geometric initialization documented in
`GEOMETRIC_INITIALIZATION.md`. That experiment uses the existing factor-space
Galerkin solver, not this unvalidated weighted adapter.

The fixed-budget ray rule established Re-independent point counts but still
evaluated every response mode at 91,136 physical points. This successor pulls
the known exponential out of the integrand and uses a Gaussian rule for its
measure. It is implemented and verified for the full 432-mode thin energy Gram,
including interactions between different boundary families.

The reusable implementation is `weighted_response_assembly.py`; the three
`validate_*gram.py` / coupling scripts call it and compare saved references.

For a same-boundary product, 1/ell_eff = 1/ell_i + 1/ell_j, and

```
integral_0^H exp(-n/ell_eff)*g(n) dn
  = ell_eff * integral_0^(H/ell_eff) exp(-eta)*g(ell_eff*eta) d_eta.
```

`exponential_weight_quadrature.py` constructs positive Gaussian nodes for the
finite exponential measure using a cheap scalar seed rule and reorthogonalized
Lanczos. Expensive basis functions are evaluated only at the returned nodes.
No physical tail is deliberately truncated. Moment checks through degree 31
give relative errors below 2.3e-14 over scaled interval lengths 0.01 to 10,000.

`ResponseModes.operators(..., strip_exponential=True, family=...)` evaluates
factored values, velocities and gradients analytically. It avoids underflow
followed by multiplication by a large inverse exponential. These outputs are
weighted-integration amplitudes, not physical fields. Defaults are unchanged.

## Geometry and all thin-block interactions

- Obstacle blocks use 64 periodic angular nodes and weighted radial nodes.
- Wall blocks use 48 tangent nodes and weighted normal nodes, including fluid
  intervals on both sides of a hole intersection.
- Inlet/wall corner products use two exponential-weighted coordinates and
  retain the hole subtraction.
- Obstacle/wall products combine their affine radial exponents and reverse the
  coordinate when the combined exponent changes sign.
- Opposite-wall terms remain included on a small exact-geometry rule. They are
  extremely small in the tested high-Re cases; their relative accuracy is not
  inferred from the complete matrix norm.

The complete matrix is symmetrized at roundoff level (raw relative asymmetry
about 1e-16). Independent checks use the previously generated higher-order ray
Gram, not a fitted force or solution.

## Results and the weak-direction check

For Re=200, the complete 432-mode block with 12 normal nodes takes about 2.6 s.
Its scaled Frobenius discrepancy from the higher-order ray reference is 1.03e-9.
The corresponding Re=2000 discrepancy is about 7e-10. Several wall-block
comparisons have a reference discrepancy floor: their weighted 8- and 12-node
results agree more closely with each other than with the ray reference.

The normalized thin Gram has eigenvalues down to approximately 6e-11. Therefore
the overall matrix norm is insufficient to assess integration accuracy:

| Re | Normal nodes vs 16 | Scaled Frobenius change | Maximum relative energy-form change |
| ---: | ---: | ---: | ---: |
| 200 | 8 | 2.62e-10 | 1.69e-3 |
| 200 | 12 | 3.94e-14 | 6.17e-6 |
| 2000 | 8 | 1.23e-13 | 6.24e-6 |
| 2000 | 12 | 9.42e-14 | 7.13e-6 |

The energy-form check uses Cholesky whitening and includes float64 sensitivity
of nearly dependent directions; it is not an exact-arithmetic error bound.
Use 12, not 8, normal nodes as the current candidate for subsequent coupling
work. This is a fixed order, not an order increased with Re.

For the complete Re=200 thin block, basis-value visits decrease from 39,370,752
to 3,172,656 (12.41 times fewer). There are 61,896 block-node visits, each
evaluating only the needed boundary/scale functions; these are not 61,896
global nodes evaluating every mode. `cost_comparison.json` records the direct
same-process timing against the 91,136-point rule. Neither ratio is an overall
NS speedup.

In the direct same-process comparison, the 91,136-point rule took 8.970 s and
the weighted 12-node assembler took 2.583 s, a measured 3.47-fold speedup for
this matrix block. Scalar measure construction and block scheduling are included;
background coupling and NS evolution are excluded from both timings.

## Full background and nonlinear integration

`weighted_response_space.py` now provides the experimental `WeightedResponsePlan`.
It assembles the original six-length, 648-column enrichment jointly with the
same energy scaling and `rcond=1e-10`. It does not truncate broad and thin spaces
separately. No filter, artificial viscosity, sponge, or time-integrator change is
introduced. The original `EnrichedFlowPlan` remains available for comparisons.

The original background and the two broad lengths use a fixed bulk rule. The
432 thin modes use the pairwise exponential rules for both mass and stiffness.
Background/thin couplings use local normal coordinates and a composite Gaussian
rule for `exp(-n/ell_max)`. Smaller scales are evaluated as decaying exponential
ratios. Tangential corner panels also scale with layer thickness. The normal
panel boundaries are fixed in dimensionless coordinates; they do not multiply
with Reynolds number. Each last panel extends to the physical domain boundary.

Let `B` denote the background plus broad modes (including the lift), and let
`E = sum_g E_g` be the four thin boundary families. The background nonlinear
load is evaluated using the exact algebraic decomposition

```
N(B+E) = N(B) + sum_g [N(B,E_g) + N(E_g,B) + N(E_g,E)].
```

The first term uses the bulk rule; each correction uses its family's exponential
measure. Thin test rows integrate the complete `N(B+E)` on their own weighted
rules. Thus mixed products and thin/thin products remain present. The outlet
term is unchanged because every enrichment has zero outlet velocity.

Local original basis matrices are retained before the dense background rotation.
Rotations are applied after cross-block integration, or to a coefficient vector
before nonlinear evaluation. No full background Gram or dense Q-by-N basis
rotation is formed on local layer nodes. MPFR preprocessing retains its existing
precision and can use the existing spawn-based chunk evaluator.

`validate_weighted_response.py` checks bilinear forms and nonlinear actions
against an independent, higher-order positive ray quadrature, using deterministic
random probes and both the Stokes lift and a finite velocity perturbation. Its
acceptance thresholds are 1e-6 for projected mass/stiffness and 1e-5 for nonlinear
loads. This is a numerical quadrature audit, not a proof of NS stability or ripple
removal. Full-size validation and measured timings must be read from the run
artifacts; the thin-block speedup above is not a whole-solver speedup.

```sh
python examples/pde/validate_weighted_response.py \
  --out build/immersed_flow/weighted_full_new/small_audit.json
python examples/pde/enrich_ns_response.py \
  --out build/immersed_flow/weighted_response_re200_new \
  --weighted-quadrature --audit-weighted --quadrature-factor 2.5 \
  --family broad --levels 4 --skip-reference --reynolds 200 \
  --dt 0.01 --time 1 --frames 100
```

The earlier 91,136-point dt=0.02 run overflowed before t=1; its data do not
establish ripple removal or RK4 accuracy. The solver uses IMEX midpoint (order 2).
The subsequent old-path dt=0.01 attempt was cancelled during assembly when the
user requested full low-cost integration, and contains no trajectory.

BLAS/OpenMP thread counts are left to library defaults. Current run commands do
not force `OPENBLAS_NUM_THREADS=1` or `OMP_NUM_THREADS=1`; the observed default
OpenBLAS pool on this machine has 10 threads. Historical single-thread timings
above are retained as historical measurements, not comparable-thread benchmarks
for the new full adapter.

## Thin-block checks

```sh
python examples/pde/validate_exponential_weight_moments.py
python examples/pde/validate_exponential_weight_gram.py \
  --out build/immersed_flow/weighted_quadrature_new
python examples/pde/validate_exponential_wall_gram.py \
  --out build/immersed_flow/weighted_quadrature_new
python examples/pde/validate_weighted_thin_couplings.py \
  --out build/immersed_flow/weighted_quadrature_new
```

Results, matrices and metadata are in
`build/immersed_flow/weighted_quadrature_20260922/`. These research checks require
the reference Grams in `build/immersed_flow/scaled_quadrature_20260922/`.
