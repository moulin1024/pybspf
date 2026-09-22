# Fixed-budget scaled integration for response enrichment

The Re=200 runs with globally increased quadrature were cancelled during base
assembly at the user's request. They produced no NS trajectories. The replacement
uses thickness-scaled coordinates with a point budget independent of Reynolds
number. It changes integration nodes, not the PDE, viscosity, trial functions,
time scheme, or boundary conditions.

## Ray coordinates and positive weights

For a counterclockwise rectangle boundary segment b(s), let v=b(s)-c, where c
is the ellipse center, and R=norm(v/axes). A fluid point is

```
r = 1 + (R-1)*t,       0 <= t <= 1
x(s,t) = c + (r/R)*v
J(s,t) = cross(v,b'(s))*r*(R-1)/R**2 > 0.
```

The four ray sectors cover the exact rectangle minus ellipse. The ellipse
distance used by the response modes is d=min(axes)*(r-1), so the inner scaling
aligns with those functions exactly. It is generally not Euclidean distance.
The outer wall distance is linear along each ray as well. Geometry-only tangent
panels remain unchanged across Reynolds numbers.

Each ray has an inner layer, a bulk interval and an outer layer. Layer widths
are 32 times the largest thin response scale, converted to the local coordinate
and capped at one quarter of the ray interval. A fixed 48-node budget on each
side is split over dimensionless intervals [0,1/16], [1/16,1/4], [1/4,1].
The bulk has 32 nodes. The layer split resolves the smallest exponentials and
their quadratic/cubic products without increasing the node count. Broad scales
0.1 and 0.2 are left to the bulk and existing layer intervals.

The first single-panel 48-node prototype failed the analytic thin-layer moment
test (relative errors about 2e-4 at the largest tested Re). Redistributing the
same nodes into the three scaled panels fixed that failure; tolerances were not
relaxed. No part of the physical domain or exponential tail is discarded.

This initial implementation applies one common positive rule to all weak-form
terms. It is not yet the proposed background/BL block integrator with unchanged
background nodes. Keeping one consistent rule preserves the Gram construction
as weighted products and avoids introducing mismatched block integrals.

## Validation before NS

`validate_scaled_response_quadrature.py` checks positive weights, exterior nodes,
all polynomial moments through total degree four, and analytic radial exponential
moments including products with decay rates 2 and 3.

| Re | Points | Maximum relative layer-moment discrepancy |
| ---: | ---: | ---: |
| 20 | 91,136 | 1.71e-14 |
| 200 | 91,136 | 5.55e-9 |
| 2,000 | 91,136 | 3.75e-8 |
| 20,000 | 91,136 | 3.82e-8 |

Polynomial moment discrepancies are below 4e-14. Layer-moment references use
analytic radial antiderivatives on the same tangent rule; they independently
test radial scaling, not tangent convergence.

`validate_scaled_response_gram.py` additionally compares all 648 raw response
modes' energy Gram matrices with higher radial orders (72 layer / 64 bulk).
The production budget remains 91,136 points for both Reynolds numbers; the
148,096-point rule is only an integration-order reference.

| Re | Maximum diagonal change | Scaled relative Frobenius change |
| ---: | ---: | ---: |
| 200 | 3.30e-8 | 3.62e-9 |
| 2,000 | 1.95e-7 | 2.86e-8 |

These tests do not certify the smallest retained coupled-space singular values,
tangential convergence, nonlinear accuracy, or spatial sufficiency. A scaled
coordinate does not create missing separated shear-layer or wake dynamics.

The model gains an optional `quadrature_rule` callback; its default quadrature
is unchanged. Four selected model checks pass, including a continuous forced-NS
residual with the custom rule hook and the existing second-order time test.

## Short NS diagnostic

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python examples/pde/enrich_ns_response.py \
  --levels 4 --family broad --reynolds 200 --skip-reference \
  --scaled-quadrature --quadrature-factor 2.5 \
  --time 1 --dt .02 --check-dt-time 1 --frames 50 \
  --out build/immersed_flow/scaled_response_re200_t1_new
```

The extra geometry-only split at the old buffer start can slightly change the
point count from the standalone table; it remains independent of Re. The domain
has no sponge. Both dt=0.02 and dt=0.01 run to t=1 in exactly the same space.
The coarse trajectory is reused for output. The old Re=20 frozen FEM solution
is not treated as a Re=200 reference.

Output and run metadata are in `build/immersed_flow/scaled_response_re200_t1_20260922/`.
Integration checks are in `build/immersed_flow/scaled_quadrature_20260922/`.

The full scaled-ray NS attempt completed assembly (2059 base directions and
525 retained enrichments), but its dt=0.02 trial overflowed before t=1. No valid
terminal field was produced. The user then prioritized reducing the remaining
91,136-point integration cost, so no additional large NS assembly was started.

The lower-cost successor is documented in `WEIGHTED_RESPONSE_INTEGRATION.md`.
It extracts exponential measures and integrates scale-pair blocks with a few
weighted normal nodes, rather than evaluating all functions on the ray grid.
