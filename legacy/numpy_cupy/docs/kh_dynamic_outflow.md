# Dynamic energy-stable outflow in the BSPF streamfunction backend

This optional experiment adds boundary inertia without a buffer, artificial
viscosity, filtering, or lower-order closure. It retains the original uniform
96×80 output grid and domain `[-3,3]×[-1,1]` for KH. Both vertical faces are
open; horizontal velocities remain prescribed. The underlying reference is
[Dong (2015), convective-like energy-stable open boundary conditions](https://arxiv.org/html/1506.01320).
This implementation uses that boundary law in a compatible curl Galerkin
formulation, not the paper's pressure/velocity splitting scheme.

Let `s=u·n`, `beta=nu*D0`, and

\[
 E(u)=\tfrac12(|u|^2n+s u)\,1_{s<0}.
\]

The implemented physical boundary condition is

\[
 \beta\partial_tu+\nu\partial_nu-pn=E(u)-E(U),
 \qquad U=(\tanh(y/.12),0).
\]

The prescribed `-E(U)` term maintains the background inflow reservoir, since
our two shear streams travel in opposite directions. For no lift, U=0.
This is a sharp-switch version of Dong's condition; the paper also gives a
smoothed switch. It is not an exact incoming Dirichlet condition, and its
reference-flow compensation should not be confused with the zero-boundary-load
case in the paper. The original incoming-relaxation traction remains available
with `dong_backflow=False` on `with_stream_dynamic_boundary`.

With zero external forcing, homogeneous horizontal velocities and U=0, the
semidiscrete energy relation is

\[
 \frac{d}{dt}\left(\tfrac12\int_\Omega|u|^2+
 \tfrac\beta2\int_{\Gamma_o}|u|^2\right)
 =-\nu\int_\Omega|\nabla u|^2
 -\tfrac12\int_{\Gamma_o}|u\cdot n|\,|u|^2.
\]

The driven shear benchmark has external energy input; its energy need not
monotonically decrease. This spatial energy law does not make explicit RK4
unconditionally stable. The sharp switch can also affect temporal smoothness
when flow reverses.

## Direct inverse with unchanged volume basis

The original scalar basis, physical mass and stiffness, viscous operator,
initial kinetic projection, and output reconstruction are retained. In the
existing mass-normalized basis the extra inertia adds only endpoint traces:

\[
 \mathcal M_*=(K_x+\beta G_e^TG_e)\otimes I_y
             +(I_x+\beta B_e^TB_e)\otimes K_y.
\]

The two rows of `B_e` and `G_e` are basis values and derivatives at the vertical
endpoints. Define `M*=I+beta*B_e.T B_e` and `K*=Kx+beta*G_e.T G_e`.
A one-dimensional generalized eigensolve produces
`R.T M* R=I`, `R.T K* R=diag(lambda*)`. For a modal load F, every stage solves

\[
 \dot a=R\left[\frac{R^TF}{\lambda_x^*+\lambda_y}\right].
\]

No global 2D matrix, iterative linear solve, or refinement is needed. Crucially,
the **physical** Kx is still used in viscosity; it is not replaced with K*.
The two extra stored arrays are the x transform and tensor denominator.
`stream_ns_inertia_apply` is used when constructing a base-balancing force;
multiplying an acceleration by the old diagonal mass would be incorrect.

Rotational convection still requires the kinetic-pressure boundary conversion
`-|u|²*n/2`, in addition to E(u)-E(U). Pointwise incompressibility is preserved
by the same curl representation. Physical pressure is not reconstructed.

## API and reproduction

```python
p = plan_stream_navier_stokes2d(x, y, viscosity=.002,
                              x_boundary="dynamic", boundary_D0=1.)
# Or reuse an existing open plan:
p = with_stream_dynamic_boundary(open_plan, D0=1.)
```

`D0=0` is the static Dong-traction control, not the original incoming-relaxation
boundary. The KH runner uses the same initial modal state as the open-only run
when loading those saved factors:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src:scratch \
 MPLCONFIGDIR=/tmp/pybspf-mpl python scratch/run_kh_stream.py \
 --nx 96 --ny 80 --T 12 --dt .002 --x-boundary dynamic --boundary-d0 1 \
 --basis-from build/kh_stream/open96 --layers \
 --out build/kh_stream/dynamic96_d1 --render
```

Omit `--basis-from` to rebuild the factors. No extension or sponge is used.

## Validation and limitations

Tests independently verify the augmented inertia inverse against its physical
boundary-quadrature definition, the zero-reference energy identity for D0=0,1,2,
and a transient continuous manufactured NS solution with nonzero boundary
velocities, pressure, and analytic traction. The latter includes the boundary
acceleration explicitly. Six dynamic/open tests passed.

A localized weak vortex is initialized at `(1.8,.45)` in the existing shear,
with streamfunction `.015*exp(-((x-1.8)^2+(y-.45)^2)/.18^2)*(1-y^2)^2`.
It crosses the right boundary during a short T=2.4 run. The comparison reference
uses `[-5,5]`, comparable spacing, the same seed, and **no absorption**. This
reference is another finite-domain calculation, not an exact infinite-domain
solution. Initial projection differences are recorded. All comparisons use
physical velocity and common quadrature points, not modal coefficients.

Time-integrated L2 velocity differences from that longer-domain reference:

| Boundary | Integral from t=0 to 2.4 |
|---|---:|
| Original incoming relaxation, D0=0 | 8.46972e-4 |
| Static Dong, D0=0 | 9.01610e-4 |
| Dynamic Dong, D0=1 | 8.95084e-4 |
| Dynamic Dong, D0=2 | 9.01392e-4 |

Thus D0=1 reduces this measure only about 0.72% relative to static Dong and is
about 5.68% worse than the original boundary. Some instantaneous boundary peaks
improve, but this test does **not** show a general reduction in reflection or
justify claiming equivalence to a large exterior domain. D0=1 is a first KH
trial consistent with characteristic advection speed near one, not a proven
optimal value. Probe data are in `build/kh_stream/dynamic_vortex`.

## KH outcome at 96×80, D0=1

The no-buffer run completed T=12 at dt=.002. Maximum sampled speed was 2.1863,
pointwise divergence stayed below 7.106e-15, and prescribed horizontal velocity
error stayed below 3.742e-14. It used exactly the original open96 initial modal
state (same saved basis, quadrature and seed projection).

To avoid conflating horizontal-wall shear with the vertical outflow artifact,
define the vertical-edge diagnostic region as
`2.8<|x|<=3, |y|<.9` and evaluate nodal vorticity there:

| Case | Peak over 0<=t<=12 | Value at t=12 |
|---|---:|---:|
| Original open boundary | 50.7825 | 50.7825 |
| Dynamic D0=1, no buffer | 12.4257 | 7.7961 |
| Extended absorbing exterior | 8.2418 | 2.8461 |

The dynamic condition visibly suppresses the sharp vertical-edge bands. Its
full-domain maximum at t=12 is still 43.7392, arising outside this diagnostic
region; horizontal fixed-wall shear remains. These are observed vorticity
magnitudes, not errors against an exact solution, and reducing them is not by
itself proof of better physical accuracy. The localized-vortex comparison above
still does not show better overall agreement than the previous boundary.
No KH grid/time-step convergence or broad D0 optimization was carried out in
this trial; it should not be called an exact transparent boundary or a proven
replacement for the larger domain.

All 13 checks across dynamic/open, fixed-boundary, and absorption tests passed.
The run, short probes, tests, and renderer all completed; no jobs remain.
Artifacts: `build/kh_stream/dynamic_comparison.png`,
`build/kh_stream/dynamic_vortex/comparison.png`, and
`build/kh_stream/dynamic96_d1/kh_stream.mp4`.
