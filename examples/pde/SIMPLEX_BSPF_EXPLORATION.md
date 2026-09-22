# Boundary/interior resolution: accuracy and cost exploration

This follows `SIMPLEX_POISSON.md`. These standalone NumPy/SciPy research scripts
remain outside the maintained JAX numerical core. They test specific hypotheses,
not a claim of a new finite element method or an FFT solver on triangles.

## Experiment A: cheap traces with rich local interiors

`simplex_bspf_comparison.py` uses the common conforming space

`V(T) = trace lifts of P_p + zero-trace interior functions`.

Three variants share exactly the same QR normalization, reference contractions,
static condensation, sparse assembly and factorization code:

1. Full P3, P4, P5, P6 Bernstein FEM baselines.
2. Degree-three traces with all polynomial bubbles of degree q=5,6,7.
3. Degree-three traces with the original P3 bubble plus Fourier modes multiplied
   by `6*lambda0*lambda1*lambda2`. Interior dimensions match variant 2 exactly:
   6, 10, 15 per triangle. Modes use `2*pi*k` on the reference bounding square,
   ordered by frequency length; both sine and cosine are included until the
   budget is reached. A budget may end partway through a pair. This particular
   fixed ordering is not rotation invariant; cell vertex order is canonicalized.

All variants retain the complete P3 space. The Fourier modes vanish on every
edge, are independent of the forcing and exact solution, and are not tuned per
case. They are a simple candidate, not the only possible triangular Fourier
construction and not the original one-dimensional spline/Fourier algorithm.

Interior coordinates are whitened using QR of weighted reference gradients,
without dropping modes. Physical affine-element stiffness matrices are assembled
from three cached reference contractions and the geometry metric. This both
avoids repeated volume stiffness integration and improves local coordinates;
it does not change the approximation space. The reference mass matrix is NOT
being claimed to be an identity. Sparse skeleton LU is reused across loads.

### Benchmarks and accounting

The unit square has 32 or 128 perturbed Delaunay triangles, seed 42. Three smooth
manufactured solutions, all with zero physical boundary data:

- smooth: `sin(pi*x)*sin(pi*y)`;
- oscillatory: `sin(5*pi*x)*sin(4*pi*y)`;
- localized: `x*(1-x)*y*(1-y)*exp(-120*((x-.43)^2+(y-.57)^2))`.

Forcing is the analytic negative Laplacian. Assembly uses 24x24 positive Duffy
quadrature; errors use an independent 36x36 rule. The audit changes assembly to
36x36 and compares gradients at 40x40 on the coarse mesh, for all three loads and
P6 / degree-three trace with 15 polynomial / 15 Fourier bubbles. The maximum
relative H1 change was 1.61e-14, below the fixed 1e-8 gate.

Both total and globally solved DOFs are recorded. Reference/geometry/factorization
setup and per-load/recovery timings are separate; per-load timings are medians
of three repetitions. Error evaluation and plotting are excluded. All candidates
use the same conservative quadrature order; timings are prototype comparisons,
not optimal production implementations of each space. Thread limits are unset.

### Result: an interior-only accuracy plateau

On 128 triangles, relative H1-seminorm errors:

| Space | Free global DOFs | Total DOFs | Smooth | Oscillatory | Localized |
|---|---:|---:|---:|---:|---:|
| Full P3 | 401 | 625 | 6.968e-4 | 6.011e-2 | 1.280e-1 |
| P3 trace + 15 polynomial bubbles | 401 | 2417 | 6.025e-4 | 5.030e-2 | 1.021e-1 |
| P3 trace + 15 Fourier-type bubbles | 401 | 2417 | 6.086e-4 | 5.095e-2 | 1.036e-1 |
| Full P4 | 577 | 1089 | 3.139e-5 | 1.088e-2 | 3.705e-2 |
| Full P6 | 929 | 2401 | 3.728e-8 | 2.572e-4 | 3.358e-3 |

Keeping the trace small makes the global system small, but does not preserve
high accuracy. Rich interior spaces do not repair missing edge information.
The edgewise best-polynomial-trace diagnostic is the same for all fixed-P3-trace
candidates; it is not a rigorous volume error lower bound. The observed plateau,
together with full-space refinement, supports the trace-bottleneck interpretation.
The tested Fourier construction offers no accuracy advantage over polynomial
bubbles at equal total DOFs. This is not a negative theorem about all Fourier
extensions or BSPF designs.

For 15 polynomial bubbles, the maximum raw local interior condition number is
353.4; after reference-energy whitening it is 8.29. For Fourier bubbles the
corresponding values are 308.5 and 7.51. This is a useful coordinate improvement
available to ordinary FEM too, not a new approximation advantage. No global
condition-number or scalable iterative-solver claim is made.

## Experiment B: residual-selected skeleton modes

`simplex_skeleton_adaptive.py` fixes the interior to the 10 bubbles of P6.
The available trace dictionary is P6, represented hierarchically by vertices and
edge functions `t*(1-t)*Legendre_(k-2)(2*t-1)`, k=2,...,6. Start with all P3 trace
modes, then select additional edge modes in batches of 16. Only the discrete
operator and load are used for selection; exact solutions enter error reporting
only. This experiment supports homogeneous Dirichlet conditions only.

For selected modes S, residual r, and candidate j, the precise single-mode
squared energy improvement relative to the enriched finite space is

`gain_j = r_j^2 / (A_jj - A_jS A_SS^{-1} A_Sj)`.

This requires global solves for candidate responses. Batched selections are a
heuristic; individual gains are not additive. A cheaper alternative uses
`r_j^2/A_jj`, omitting re-equilibration of existing modes. It is a lower estimate
of the individual gain for this SPD system. Neither score estimates error outside
the chosen P6 dictionary. No tolerance or case-specific selection rule is used;
comparison budgets are the same as uniform P4/P5 traces.

### Result: useful boundary compression, not yet faster end-to-end

With 128 triangles, seed 42, 577 global unknowns, identical 1280 local interior
unknowns, and the cheaper diagonal score:

| Case | Uniform P4 trace + P6 interior | Selected trace + same interior | Error reduction |
|---|---:|---:|---:|
| Smooth | 2.764e-5 | 2.334e-5 | 1.18x |
| Oscillatory | 9.012e-3 | 3.643e-3 | 2.47x |
| Localized | 2.712e-2 | 3.358e-3 | 8.08x |

The localized case reaches essentially the full-P6 error with 577 rather than
929 free skeleton DOFs: 37.9% fewer global unknowns. Counting retained interior
DOFs too, the free-space reduction is from 2209 to 1857, only 15.9%. Full rich
operators are constructed before compression in this prototype, so memory/setup
costs of the rich reference space are NOT avoided.

With seeds 7 and 123, the same diagonal score and batch size give respective
localized improvements of 5.71x and 8.65x over uniform traces at equal DOFs.
Oscillatory improvements are 2.48x and 2.50x; smooth improvements 1.13x and 1.04x.
Additional-seed runs were concurrent: their timings should not be compared.

For seed 42, localized selection plus repeated reduced solves took about 0.019 s
with the diagonal score, versus 0.160 s with precise scores and about 0.004 s for
one full-P6 skeleton solve. These exclude shared setup/load/error evaluation.
Thus there is currently NO demonstrated end-to-end wall-clock speedup. The
precise score improves smooth/oscillatory accuracy more, but costs substantially
more. Reusing a selected trace space for many similar right-hand sides could
amortize selection; that has not been validated and may fail for changing loads.

This benefit is residual-driven hp-style trace compression, not an established
advantage unique to BSPF over mature adaptive FEM. A fair next competitor is
ordinary residual-adaptive hp FEM, not only uniform Pp spaces.

## Reproduce

From the repository root (no forced single-thread setting):

```sh
MPLCONFIGDIR=/tmp/bspf-simplex-mpl python examples/pde/simplex_bspf_comparison.py
MPLCONFIGDIR=/tmp/bspf-simplex-mpl python examples/pde/simplex_skeleton_adaptive.py
MPLCONFIGDIR=/tmp/bspf-simplex-mpl python examples/pde/simplex_skeleton_adaptive.py --score diagonal --out build/simplex_bspf_adaptive_diagonal
MPLCONFIGDIR=/tmp/bspf-simplex-mpl python examples/pde/simplex_skeleton_adaptive.py --score diagonal --seed 7 --out build/simplex_bspf_adaptive_seed7
MPLCONFIGDIR=/tmp/bspf-simplex-mpl python examples/pde/simplex_skeleton_adaptive.py --score diagonal --seed 123 --out build/simplex_bspf_adaptive_seed123
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q examples/pde/test_simplex_poisson.py examples/pde/test_simplex_bspf_comparison.py
```

Outputs contain JSON metrics and plots. The comparison also records environment
and source hashes. Fourteen targeted tests pass: original baseline tests plus
analytic gradients, zero-trace Fourier modes, cubic completeness, cell-permutation
invariance, old full-FEM equivalence, quadrature refinement, direct metric assembly,
exact single-mode energy gain, full hierarchy recovery and diagonal-score bound.

## Direction justified by these results

Pursue independently controlled trace and interior spaces, stable reusable local
operators, and inexpensive residual-driven trace allocation. Do not continue
adding Fourier interior modes while freezing the skeleton. A true BSPF trace
(spline endpoint correction plus a resolved smooth remainder) can be a subsequent
candidate, but it must beat the hierarchical polynomial trace at equal total
cost, preserve endpoint compatibility, and survive orientation/conditioning
checks. No NS/ripple, nonsmooth-domain, 3-D or high-Re claim follows from these
smooth scalar Poisson experiments.

Related established ideas include residual-free bubbles and generalized finite
elements; boundary lifting/static condensation and adaptive modal selection are
not claimed as novel:

- [Residual-free bubbles and local enrichment](https://www.sciencedirect.com/science/article/abs/pii/S0045782504004268)
- [Partition of unity FEM](https://www.sciencedirect.com/science/article/pii/S0045782596010870)
