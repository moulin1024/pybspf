# Narrow external absorption combined with dynamic outflow

This experiment tests whether Dong-type boundary inertia can reduce the exterior
absorption width required for this BSPF KH benchmark. The region of interest
remains `[-3,3]×[-1,1]`; absorption is exactly zero there. No lower-order closure,
filter, or artificial viscosity is introduced. The direct inertia solve and
pointwise divergence-free curl representation are unchanged.

## Controlled comparison

The existing ordinary-outflow exterior calculation uses width L=2 on each side,
160×80 nodes, maximum damping 4, and physical viscosity .002. The new runs use:

| Exterior width per side | Full domain | Grid | dx | Maximum damping |
|---|---|---|---|---|
| 2 (existing reference) | [-5,5]×[-1,1] | 160×80 | .0628931 | 4 |
| 1 | [-4,4]×[-1,1] | 128×80 | .0629921 | 4 |
| .5 | [-3.5,3.5]×[-1,1] | 112×80 | .0630631 | 4 |

Both narrow widths are run with ordinary outflow and with dynamic Dong outflow
D0=1. All four runs use T=12, dt=.002. Maximum damping is intentionally unchanged:
this tests whether a better terminal condition can handle the greater residual
perturbation, rather than compensating by simply increasing absorption.

All runs use the same continuous initial streamfunction, including its exterior
continuation and C-infinity cutoff between |x|=3 and 4. This agrees with the
existing width-2 reference. For width .5, this restricts that continuous initial
field to the shorter domain; a nonzero initial trace on an open face is valid.
Kinetic projection into different-domain bases can introduce small differences;
these are explicitly measured in the common region. The paired boundary laws at
each width start from exactly the same modal coefficients.

Comparison is against a numerical larger-domain model, not an exact infinite-
domain solution. Recall that its damping-strength sensitivity was approximately
2.9% in the full interest region at T=12. Final velocity comparisons use common
physical quadrature points and weights in the original region of interest.
Relative errors use perturbation velocity (base shear removed), and are called
**differences**, not exact PDE errors. A separate center metric uses
`|x|<2, |y|<.5`. The edge-vorticity diagnostic uses
`2.8<|x|<=3, |y|<.9`, avoiding the immediate horizontal-wall shear.

## Why a shorter layer may work, and what it changes

With the same C-infinity ramp S as in the previous experiment,
`sigma(x)=4*S((|x|-3)/L)`. A purely advected perturbation with speed one would
be attenuated by roughly `exp(-integral sigma dx)=exp(-2*L)`: approximately
.018 for L=2, .135 for L=1, and .368 for L=.5. This is only a scalar transport
estimate, not a prediction for incompressible KH. It explains why the terminal
boundary condition becomes more important as L shrinks.

Sharper spatial variation can itself affect vorticity: the curl of the
relaxation force includes `-sigma'(x)*v`, in addition to damping perturbation
vorticity. Incompressible pressure also couples the exterior to the interior.
There is therefore no guarantee that adding boundary inertia permits arbitrary
layer reduction without changing the interior solution.

For zero reference velocity and homogeneous driven-boundary data, the combined
energy law adds the negative term `-integral sigma*|u|²` to the dynamic-boundary
energy identity. A new test verifies this identity using direct physical
quadrature for the absorption work. Five dynamic-boundary tests, including this
combined case, passed. The driven shear benchmark can receive energy from its
reference flow and maintained base, so its kinetic energy need not decrease.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src:scratch \
 python scratch/run_kh_hybrid_sweep.py --extension .5 --nx 112
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src:scratch \
 python scratch/run_kh_hybrid_sweep.py --extension 1 --nx 128
```

Each call sets up one basis, then advances the ordinary/dynamic pair. The two
runs in a pair have identical initial data, viscosity, damping, and resolution.
The normal runner also supports this combination, for example:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src:scratch \
 python scratch/run_kh_stream.py --nx 112 --ny 80 --T 12 --dt .002 \
 --x-boundary dynamic --boundary-d0 1 --layers --extension .5 \
 --sponge-strength 4 --seed-cutoff-width 1 \
 --out build/kh_stream/hybrid_L0.5_s4_dynamic --render
```

`--seed-cutoff-width 1` keeps the same continuous seed as the width-2 reference;
the legacy default cutoff otherwise changes with very short extensions.

Compare completed checkpoints with `scratch/compare_hybrid_outflow.py`.


## Results

All four T=12 runs completed without a blow-up. Peak pointwise divergence was
below 4.89e-15, and horizontal velocity errors below 4.12e-14. The combined
energy test passed. The initial velocity differences from the wider reference
inside the common interest region were 1.83e-7 for L=.5 and 5.56e-7 for L=1.
The paired ordinary/dynamic runs have zero initial difference.

Final quadrature-weighted perturbation-velocity differences from the width-2
reference are:

| Width | Terminal boundary | Whole interest region | Center | Whole-region absolute L-infinity |
|---|---|---:|---:|---:|
| 1 | open | 11.268% | 5.645% | 0.2561 |
| 1 | dynamic | 11.447% | 5.617% | 0.2555 |
| 0.5 | open | 19.046% | 8.551% | 0.4373 |
| 0.5 | dynamic | 20.816% | 9.778% | 0.4637 |

The local edge-vorticity peaks over the entire run remained about 8.242 for all
four cases, close to the width-2 reference. Thus a smooth-looking boundary
diagnostic did not imply unchanged interior dynamics. At matched widths,
adding the dynamic terminal condition did not materially improve agreement
with the wide reference. In particular, the narrowest dynamic case had slightly
larger differences than its ordinary-boundary counterpart.

This test does not establish that the combination can reduce the buffer at
fixed quantitative accuracy. It also does not rule out other D0 values, damping
profiles, or stronger integrated absorption. Only D0=1, maximum damping=4 and
two shorter widths were tested; no additional parameter search was launched.
The comparison reference is itself a finite-width absorbing model, so these
percentages must not be read as exact infinite-domain PDE errors.

The setup is available as an option. Halving L halves integrated absorption at
fixed peak strength; a separate study preserving integrated damping (e.g.
increasing peak strength inversely with L) could isolate that effect, but it
also steepens force gradients and cannot be assumed to preserve the solution.

Artifacts under `build/kh_stream/`: `hybrid_comparison.json`,
`hybrid_comparison.png`, `hybrid_differences.png`, `hybrid_comparison.mp4`, and
`hybrid_L1_s4_dynamic/kh_stream.mp4`. The comparison movie displays only the
identical interest region; the latter movie also shows the actual absorption
layer and its interface.
