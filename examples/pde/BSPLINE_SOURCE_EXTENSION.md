# Local high-order B-spline source extension

`bspf_models.elliptic.bspline_source.BSplineSourcePlan` replaces the global
Fourier least-squares source fit with reusable local tensor B-spline fits.
It supports degree 5 and 7. FFT/FINUFFT particular solutions and the exact spline
boundary integral correction are retained. There is no body-fitted volume mesh,
no exterior source callback and no use of exact MMS values or derivatives in
construction. This is a host-side experimental model backend, not a JAX kernel.
Install the model package's `fast` extra for FINUFFT.

## Construction

Overlapping boundary patches use Cartesian coordinates scaled by patch radius.
Each coordinate has a clamped uniform knot vector with `spans` intervals. Within
a patch the tensor product has `(degree + spans)^2` coefficients. Only physical
points from a clipped Chebyshev stencil are sampled. Each local factorization
solves

    min_c ||A c - f||_2^2 / number_of_samples + stability^2 ||J c||_2^2.

`J` contains normalized jumps of the degree-th derivative across simple knots,
for both coordinate directions. Global degree-p tensor polynomials lie in its
nullspace. This selects a smoother continuation of poorly observed exterior
coefficients without penalizing those polynomial components. The augmented
matrix is factored directly by truncated SVD, avoiding normal equations. These
are small, cached LOCAL dense factorizations; no global dense source fit is
formed. `stability` is a regularization weight, not an error tolerance or a proof
of conditioning for arbitrary geometries.

Compact C7 PU weights combine patches with exterior zero patches. Deep inside,
the existing independent Cartesian source grid is resampled with 12-point
polynomial interpolation; this bulk interpolation is not B-spline fitting.
A smooth blend joins it to boundary patches. The extension vanishes near the
periodic box edge. FFT coefficients are then inverted with the existing Poisson
or Helmholtz symbol, including the existing zero/resonant-mode treatment.

Source validation uses independent shifted physical points and close-to-wall
collars, AFTER FFT resampling. A failed source gate raises an error. Thus spline
fit bias, insufficient knots and insufficient FFT resolution remain visible.
GS retains its separate final source, full PDE and boundary gates. Intermediate
Picard fits may be unresolved, but cannot be returned as accepted solutions.

## Usage

```python
from bspf_models.elliptic.bspline_source import BSplineSourcePlan
from bspf_models.elliptic.spline_annulus import AnnulusPanelPlan

source_plan = BSplineSourcePlan(
    domain, degree=7, spans=5, stability=1e-6,
    grid_size=1025, sample_grid=129,
)
plan = AnnulusPanelPlan(domain, sigma=0.0, order=16, subdivisions=2)
solution = plan.solve(
    (outer_dirichlet, inner_dirichlet), source=physical_source,
    source_plan=source_plan, source_tolerance=1e-7,
)
```

For GS, pass this source plan to `SplineAnnulusGSPlan(source_plan=...)`.
The following resolution passed all three GS cases at their existing gates:

```sh
python examples/pde/spline_annulus_grad_shafranov.py \
  --source-backend bspline --spline-degree 7 --spline-spans 11 \
  --patch-radius .13 --patch-samples 46 --sample-grid 257 --fft-grid 2049 \
  --out build/spline_annulus_gs_bspline_resolved
```

`--degree` still controls GEOMETRY degree; `--spline-degree` controls SOURCE
approximation degree. The global Fourier reference remains the example default.

For a reproducible source-only random-wave resolution audit:

```sh
python examples/pde/benchmark_bspline_extension.py \
  --degrees 7 --spans 3 5 --cutoffs 12 24 48
```

Its JSON includes the random seed, actual wavevectors/phases, geometry degree,
source settings, setup/fit times and rejected resolutions. Diagnostic fitting
uses infinite tolerance to report errors, then explicitly marks acceptance
against the requested finite tolerance. It does not claim a PDE solve passed.

## Resolution and limitations

- Increasing `spans` refines knots within each patch; increasing `degree` changes
  approximation order. Neither adds missing physical information automatically.
- Smaller patches localize oscillations but require adequate bulk sampling and
  FFT resolution. The plan rejects bulk interpolation stencils leaving the
  physical domain. `grid_size` and `sample_grid` have different roles.
- Reducing `stability` can amplify exterior values and worsen Fourier error;
  it is not a substitute for spatial resolution.
- At fixed degree this is a high-order spline method, not exponential spectral
  convergence. Near-contact boundaries, narrow channels and rough sources can
  require new patch layouts and resolution. There is no automatic adaptivity.
- It does not eliminate global FFT storage or the physical need to resolve high
  frequencies. No speedup over an equally accurate Fourier reference is claimed
  without matched-resolution timing.
- BSPF boundary-jet/interior decomposition is a subsequent step and is not part
  of this implementation.

## Verification

The automated tests cover both degrees, domain-only callbacks, complex
polynomial reproduction, Poisson and both signs of Helmholtz, knot-refinement
improvement, invalid parameters and rejection of unresolved forcing. Eighteen
new tests pass; the existing annulus, GS and local-extension suites contribute
31 further passing tests (49 total). The models CI job installs the `fast`
extra so these tests are exercised rather than skipped for missing FINUFFT.

On the test annulus, degree-7 wave forcing `sin(12*x) cos(9*y)` with 513 FFT
points per axis improves from 1.12e-4 to 8.01e-6 to 2.16e-6 relative source error
as spans increase 3, 5, 7. All three remain rejected at 1e-7. A 513 FFT grid also
fails some smooth-source checks; the accepted elliptic tests use 1025 without
relaxing their tolerances.

The initial three-span GS run passes Solov'ev (flux L2 error 4.80e-10), but the
oscillatory case fails the final source gate at approximately 1.80e-7. This is a
resolution failure, not an accepted run. Full seven-span results are recorded
separately in the GS example output.

Seven spans pass Solov'ev and oscillatory MMS (flux relative L2 errors 4.80e-10
and 4.45e-8), but the nonmanufactured nonlinear GS case stagnates at 1.84e-6
source/PDE error and is rejected at 1e-8. Its full history is retained in
`build/spline_annulus_gs_bspline7_spans7/results.json`. An accepted analytic MMS
is therefore not evidence that the same resolution suffices for that equilibrium.

A further run with degree 7, nine spans, radius 0.16, 42 sampling nodes per
patch axis, sample grid 257 and FFT grid 1537 reduces the nonlinear plateau
to 4.78e-8 (about 38 times smaller). It is still rejected at the unchanged 1e-8
gate. Results are in `build/spline_annulus_gs_bspline_refined/results.json`.

## Accepted full GS run

The command above uses 105 patches, each with 324 coefficients, 73,744 physical
training points and a 2049-by-2049 FFT grid. The original quintic geometry,
boundary quadrature and acceptance tolerances remain unchanged.

| Check | Measured value |
| --- | ---: |
| Solov'ev flux relative L2 error | 4.80e-10 |
| Oscillatory MMS flux relative L2 error | 4.45e-8 |
| Nonlinear GS relative PDE residual | 3.36e-9 |
| Nonlinear effective-source relative maximum error | 1.11e-9 |
| Nonlinear boundary error | 5.52e-12 |
| Independent original-GS fourth-order FD residual | 1.93e-6 |

The nonlinear solution is accepted after four Picard iterations. Source and
analytic PDE gates are both 1e-8; the independent FD gate remains 1e-5, with
24 points and h=3e-4. The FD result is a separate check and does NOT certify
1e-8 derivative accuracy.

Before the streamed-layer change, measured plan setup was 34.2 seconds and
the three solves plus checks took
216.4 seconds on this machine, including the initial cached layer evaluation
matrices. These are a single run, not a controlled speed benchmark. This
resolution establishes feasibility, not a performance advantage or a general
resolution prescription. The global Fourier backend remains the default.

Full output: `build/spline_annulus_gs_bspline_resolved/results.json` (`passed=true`).
The three-resolution comparison is in `build/bspline_extension/gs_comparison.json`.
The coarse high-wave random-source audit remains rejected; success here does
not establish that its wave-48 case is resolved by the coarse settings.

The current GS path applies the boundary layer in bounded blocks and no longer
retains those dense target matrices. See `SPLINE_ANNULUS_GS.md` for the memory
contract and the distinction between historical timings and current behavior.
