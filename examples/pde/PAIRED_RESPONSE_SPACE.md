# Thin-layer / outer-compensation comparison

This experiment implements the first proposed construction in
`CLASSICAL_BOUNDARY_LAYER_ENRICHMENT.md` and compares it with the existing
four-scale response space. It keeps Re=20, dt=0.02, t=1, the same rational Stokes
initial field, full NS weak form, and zero sponge. All plotted data are raw.

## Construction

For a distance coordinate n and positive scale ell define

```
R_ell(n) = 1 - (1+n/ell)*exp(-n/ell)
psi_pair(s,n) = A(s) * [R_ell(n) - R_L(n)],  L > ell.
```

The streamfunction and its first derivative vanish at n=0 and infinity. In the
flat-wall prototype the thin part's tangential velocity integrates to A(s),
while the broad part integrates to -A(s). Total correction flux is zero; the
compensation occupies a different scale from the near-wall correction. The curl
of the streamfunction supplies both velocity components consistently.

Near zero, the primitive is evaluated with its Taylor series to avoid cancellation.
Its derivatives are analytic, including all chain and product terms. At the
ellipse, use the same radial distance coordinate as the preceding experiment;
it is not exact Euclidean distance. Boundary envelopes enforce the rectangle.
For each straight wall, remove its own squared-distance factor from the envelope
because R already supplies that double zero. A squared ellipse level enforces
the obstacle constraint. Consequently the flat-wall moment formula is a design
motivation, not an assertion that the enveloped curved mode has identical moments.

## Controlled candidate dictionaries

All groups retain the original 2059-dimensional BSPF/rational space. The same
energy normalization and rank threshold are applied to each candidate dictionary.
The four-scale candidate functions remain present in both additions; joint
rank truncation can choose a different set of normalized enrichment directions.

| Family | Local exponentials | Paired modes | Raw additional candidates |
| --- | --- | --- | ---: |
| local control | ell0*[0.5,1,2,4] | none | 432 |
| hybrid | same | ell0 and 2*ell0, each paired with L=0.2 | 648 |
| broad control | same plus 0.1 and 0.2 | none | 648 |

Here ell0=sqrt(nu*0.02/2)=0.0123827837473; angular order is 16 and straight-wall
order is 24. The two new dictionaries have equal raw candidate counts, not a
promise of equal retained rank or runtime. Their retained counts are reported.
The broad control tests whether gains merely come from adding larger scales.
The chosen outer scale is fixed from geometry before inspecting results; no
force or FEM snapshots are used to fit basis functions.

A `paired` family can also replace the local dictionary with four pair families,
for later equal-size replacement studies. It is not included in the initial
hybrid-versus-broad comparison unless explicitly listed in the results.

## Validation

`validate_response_modes.py` checks derivatives by finite differences, all
essential boundary traces, the paired primitive's exact integral identity, and
thin/broad cancellation on the half-line. The Gram-matrix quadrature check
covers the full hybrid dictionary. As before, the actual and best-H1 frozen
response are compared with independently generated curved P4/P3 FEM data.
Then NS is advanced and inspected using identical-grid raw sections and the
previous upstream fourth-difference diagnostic. A winning candidate must also
survive quadrature refinement; reducing roughness alone is not a certificate
of nonlinear accuracy.

```sh
OPENBLAS_NUM_THREADS=1 python examples/pde/validate_response_modes.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/validate_response_quadrature.py --family hybrid
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/enrich_ns_response.py --levels 4 --family hybrid \
  --out build/immersed_flow/paired_response_new
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/enrich_ns_response.py --levels 4 --family broad \
  --out build/immersed_flow/broad_response_new
```

The reference dataset and optional research dependencies are documented in
`FROZEN_NS_FORCE.md`. Production solver defaults are unchanged.

## Initial paired result (factor 4)

The hybrid dictionary retains 564 enrichment directions, versus 406 previously.
The frozen actual H1 discrepancy is 0.024266%, best H1 is 0.023616%, and velocity
L2 discrepancy is 0.001435%. These small discrepancies are below the preceding
FEM mesh-to-mesh difference, so they are not certified absolute accuracies.

At t=1 the upstream roughness falls from 5.18019e-5 to 1.94299e-5, a further
factor of 2.666 (62.49% decrease). Relative to the original space's 5.79708e-4,
the total decrease is about 29.84-fold. The final velocity grid differs from
the four-scale control by 3.62e-5 relative; the vorticity difference is 9.40e-4.
Thus a large change in the small-scale diagnostic accompanies a small change
in the overall field. None of these quantities is an independent NS error norm.

Hole speed is below 6.1e-12, wall speed below 4.7e-10, inlet error below 1.4e-10,
and outlet flux differs from prescribed inlet flux by about 1.0e-12.

The full hybrid Gram matrix has factor-6 versus factor-8 diagonal relative
change below 2.49e-9 and diagonally normalized Frobenius change 1.93e-10. The
complete factor-6 NS and same-candidate-count broad-exponential controls are
reported below.

## Broad-exponential control (factor 4)

With the same 648 candidate functions, the broad control retains 484 directions,
80 fewer than the paired hybrid. Frozen actual H1 discrepancy is 0.049217%, best
H1 is 0.046804%, and velocity L2 discrepancy is 0.006627%.

Its final NS roughness is 1.42382e-5: 3.638 times below the preceding four-scale
space and 27% below the paired hybrid. Therefore the frozen initial-response
ranking and the final NS-roughness ranking differ. The paired family performs
better on the frozen response, while the broad exponential family performs
better on this specific late-time local roughness measure with fewer retained
modes. This experiment does not establish either family as uniformly more
accurate, nor demonstrate a unique advantage of pairing over adequate scale
coverage. Their initial fields, equations and integration scheme are identical.

An interpretation to test next is that the nonlinear forcing and its relevant
response scales evolve; optimizing the initial frozen force need not optimize
a later NS trajectory. The compared norms also differ: global H1 versus local
fourth differences. An independent frozen-force audit at a later time would be
needed before attributing the remaining oscillations to a particular mechanism.

The useful outcome is a validated candidate family and evidence to span both
thin and broader response scales. The numerical comparison, rather than the
classical interpretation alone, determines which dictionary to retain.

## Paired quadrature refinement completed

At factor 6 the same 564 directions are retained. Actual frozen H1 discrepancy
is 0.024289%, best H1 is 0.023906%, and actual velocity L2 is 0.001434%.
Final NS roughness is 1.942964e-5. Factor 4 to 6 changes the roughness RMS by
0.00143%, final grid velocity by 7.50e-9 relative, and vorticity by 3.88e-7.
The improvement therefore survives the denser integration rule. Hole speed
remains below 6.1e-12, wall speed below 4.7e-10, and flux mismatch about 1e-12.

Outputs are in `build/immersed_flow/paired_response_q6_20260922/`;
`refinement.json` compares identical display grids. The flat-wall primitive
illustration is saved as `paired_mode_structure.png` in the factor-4 run.

## Final comparison with factor-6 quadrature

Both additions have completed the full t=1 trajectory with denser quadrature.
The original 2059 directions remain in every space; the counts below are the
additional retained directions.

| Space | Additional directions | Frozen actual H1 discrepancy | Final roughness RMS | Reduction from four-scale |
| --- | ---: | ---: | ---: | ---: |
| Four-scale | 406 | 0.089839% | 5.180190e-5 | 1.00 |
| Four-scale + thin/outer pairs | 564 | 0.024289% | 1.942964e-5 | 2.666 |
| Four-scale + broad exponentials | 484 | 0.048740% | 1.423293e-5 | 3.640 |

These H1 numbers measure discrepancies from the finite-resolution FEM reference,
whose preceding mesh-to-mesh H1 change was 0.083950%; they are not certified
absolute errors. Roughness is the same-grid upstream fourth-y-difference RMS
of vorticity, not an NS solution error norm. The pair decreases it by 62.49%
and the broad control by 72.52% relative to the four-scale result.

For the broad control, factor 4 to 6 changes final velocity by 6.76e-9 relative,
vorticity by 3.51e-7, and roughness RMS by 0.03735%. Its hole speed is below
9.9e-12, wall speed below 1.7e-9, inlet error below 4.6e-10, and flux mismatch
below 1.5e-12. Thus the ranking survives quadrature refinement for both spaces.

The paired family has lower roughness at t=0.1 and 0.2, the two additions are
nearly equal at t=0.3, and the broad control has lower roughness from t=0.4.
Residual stripes remain. These experiments validate Re=20 and t<=1 only;
they do not establish high-Reynolds-number robustness or time-step convergence
for the new dictionaries.

The final figure and numerical comparison are
`build/immersed_flow/paired_response_q6_20260922/paired_comparison.png` and
`paired_comparison.json`. Broad-control fields and `refinement.json` are in
`build/immersed_flow/broad_response_q6_20260922/`.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/enrich_ns_response.py --levels 4 --family hybrid \
  --quadrature-factor 6 --out build/immersed_flow/paired_response_q6_new
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/enrich_ns_response.py --levels 4 --family broad \
  --quadrature-factor 6 --out build/immersed_flow/broad_response_q6_new
python examples/pde/render_paired_comparison.py \
  --paired build/immersed_flow/paired_response_q6_new \
  --broad build/immersed_flow/broad_response_q6_new
```
