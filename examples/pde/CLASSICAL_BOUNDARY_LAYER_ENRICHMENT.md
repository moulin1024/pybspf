# Classical boundary-layer theory as a trial-space design guide

Status: design conclusions from the 2026-09-22 response-space experiment and
source review. The four-scale exponential enrichment is implemented and tested;
the thin/broad pairing now has a follow-up implementation and NS comparison in
`PAIRED_RESPONSE_SPACE.md`. Matched similarity profiles, integral-moment selection
and shifted correctors remain proposals, not claimed NS results.

## Method positioning

Classical high-Re attached-flow reasoning couples an Euler/potential outer field
to a Prandtl viscous layer and its displacement feedback. The current experiment
instead starts from an exact homogeneous Stokes lift and retains a full-domain
NS Galerkin solve. Stokes is the lift/corrector, not a high-Re approximation of
the outer solution. This distinction matters when describing the method's
relationship to classical viscous–inviscid interaction. MIT's [Drela/Merchant
course](https://ocw.mit.edu/courses/16-13-aerodynamics-of-viscous-fluids-fall-2003/pages/lecture-notes/)
explicitly distinguishes thin-layer equations, integral methods, displacement
interaction, and separation.

Our design proposal is to encode these structures in admissible trial functions
and determine their coefficients through the existing full NS weak form. This
is a physical approximation-space prior. It does not impose a boundary-layer
closure, prescribe the true profile, or replace the pressure projection.

## 1. Separate the scales that arise from different balances

Use wall arc length s and true normal distance n in a valid tubular neighborhood.
The current elliptical radial coordinate is only proportional to normal distance
at the wall. A future implementation should retain exact Cartesian derivatives,
coordinate Jacobians and curvature, while using thin-layer asymptotics to select
basis shapes and scales.

| Mechanism | Characteristic scale | Candidate use |
| --- | --- | --- |
| Attached convective layer | sqrt(nu*s/Ue), up to profile conventions | Slowly varying thickness along each upstream-to-downstream wall branch |
| Front stagnation region Ue ≈ a*s | sqrt(nu/a), a>0 | Hiemenz family; avoid dividing by Ue=0 |
| Transient diffusion | sqrt(nu*tau) | Several physical response times, not only one time step |
| Current midpoint resolvent | sqrt(nu*dt/2) | The tested short-response exponentials |

Falkner–Skan similarity supplies a family for Ue=K*s^m, including Blasius and
plane stagnation flow, rather than one universal profile. The exact scaling
convention is eta=n*sqrt((m+1)*Ue/(2*nu*s)). See [MIT Falkner–Skan
notes](https://ocw.mit.edu/courses/16-100-aerodynamics-fall-2005/6ff8ad40bf9283306606a6a91494da33_16100lectre25.pdf).
The stagnation construction is also derived in [MIT's Hiemenz
notes](https://web.mit.edu/fluids-modules/www/highspeed_flows/ver2/bl_Chap2/node13.html).

The unsteady diffusion profile contains erfc(n/(2*sqrt(nu*tau))); its shape is
not the same as a steady similarity profile or one exponential resolvent.
[NTNU's Stokes first-problem derivation](https://leifh.folk.ntnu.no/teaching/tkt4140/._main035.html)
provides the exact half-space solution. Our inference is to include a small
multiscale dictionary rather than treating dt as a physical boundary-layer age.

## 2. Enrich profile shape and pressure-gradient response

A useful prototype family is the Pohlhausen quartic

```
F(eta; Lambda) = 2*eta - 2*eta^3 + eta^4
                + (Lambda/6)*eta*(1-eta)^3,  0 <= eta <= 1
Lambda = delta^2 * (dUe/ds) / nu
```

It satisfies F(0)=0, F''(0)=-Lambda, F(1)=1, F'(1)=F''(1)=0.
This gives a cheap pressure-gradient shape direction dF/dLambda and a thickness
direction obtained by differentiating the scaled profile. These boundary
conditions and the polynomial are given in [MIT's Pohlhausen
notes](https://web.mit.edu/fluids-modules/www/highspeed_flows/ver2/bl_Chap2/node12.html).

For better accuracy, solve the Falkner–Skan ODE for selected favorable and adverse
pressure gradients, tabulate smooth streamfunctions and derivatives, and include
profile differences plus sensitivities to thickness and pressure-gradient
parameter. Convert every candidate to a homogeneous streamfunction correction;
do not add the unit outer velocity of each profile to an already lifted flow.
Smoothly match any finite-support profile at its edge to the differentiability
required by our velocity-gradient assembly.

This is a dictionary of possible shapes. Do not force the actual evolving flow
to stay on the one-parameter similarity family. A local fitted beta near
stagnation, reversal, or separation is not a reliable sole selector. Nor should
adding more polynomial constraints automatically be treated as an improvement;
[Majdalani and Xuan's analysis](https://arxiv.org/abs/2009.04097) examines the
limitations of the classical Pohlhausen profile choice.

## 3. Give displacement feedback its own efficient directions

The following observation is derived directly from our implemented basis. For
an idealized flat-wall local streamfunction

```
psi_local(s,n) = A(s) * n^2 * exp(-n/ell),
u_local = partial_n psi_local,
```

its tangential correction integrates to zero over n in [0,infinity). Its positive
and negative velocity lobes both occur within the thin layer. In the actual
bounded-domain basis, envelopes and geometry modify the profile; the complete
BSPF space can already supply outer compensation. This observation does NOT
prove that the total space lacks displacement effects or that they cause the
remaining NS ripple. It motivates a more economical way to represent them.

A candidate pairing, derived here, is

```
R_ell(n) = 1 - (1+n/ell)*exp(-n/ell)
psi_pair(s,n) = A(s) * [R_ell(n) - R_L(n)],  L >> ell.
```

Both psi and its normal derivative vanish at the wall; psi tends to zero far
away. The first part carries a thin tangential-velocity correction with nonzero
integral, while the second compensates on the broad scale L. The combined mode
retains zero total flux. Dependence on s automatically supplies the normal
velocity through the streamfunction; it is not an independently prescribed
physical wall transpiration.

For finite curved geometry, apply suitable boundary envelopes or the existing
admissible correction machinery, and verify all traces after this operation.
Choose L from available outer geometry and curvature rather than always setting
it to another thin scale. Corners, overlapping layers and cut loci require
patches, not a global normal coordinate.

The pairing is related to the present dictionary by the exact identity

```
R_ell(n) - R_L(n) = integral_{ell}^{L} n^2 * exp(-n/q) / q^3 dq.
```

Thus it reorganizes and extends a continuum of present exponential modes into
an efficient matched direction; it is not a wholly unrelated mechanism.

For a flat incompressible attached layer, psi ≈ Ue*n - Ue*delta_star in the
matching region. Differentiation gives the familiar external normal-velocity
correction d(Ue*delta_star)/ds. This is our continuity-based derivation of why
near-wall velocity deficit and outer displacement must be represented together.
NASA describes the outer flow's response to the effective displaced surface in
its [boundary-layer overview](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/boundary-layer/).

## 4. Use integral physics to assess and select modes

For steady, two-dimensional, incompressible, attached flow define

```
delta_star = integral (1-u/Ue) dn
theta      = integral (u/Ue)*(1-u/Ue) dn
H          = delta_star/theta
Cf         = 2*tau_wall/(rho*Ue^2)
```

The von Karman integral relation is

```
dtheta/ds + (2+H)*(theta/Ue)*dUe/ds = Cf/2.
```

See [the integral-equation derivation](https://farside.ph.utexas.edu/teaching/336L/Fluidhtml/node115.html).
Use this relation to diagnose attached quasi-steady regions and guide an initial
thickness estimate, with explicit unsteady/curvature terms when those are
material. Do not enforce the steady formula as a closure on the transient NS
simulation. Ue and the layer edge need operational definitions; near separation
or interacting layers these integrals require care.

Proposed selection: form candidate streamfunction sensitivities, then inspect
the rank and conditioning of their responses in delta_star, theta, and wall
shear, alongside H1 approximation quality. At fixed Ue, the linearized theta
functional has weight (1-2*u/Ue)/Ue; theta is not itself a linear functional of
velocity. Finite-basis moment constraints can distinguish wall-shear, thickness
and shape directions. Energy orthogonalization should retain physically useful
moment directions rather than selecting only by visual similarity.

## 5. A more direct bridge to our actual time-step operator

A local constant-coefficient Oseen resolvent gives an operator-based refinement
of the classical picture. Fourier-transform along s with wavenumber k and use
Laplace parameter z. Curl elimination of pressure gives

```
[z + i*k*Ue - nu*(D_n^2-k^2)] * (D_n^2-k^2) psi = 0.
```

The decaying roots are |k| and

```
q = sqrt(k^2 + (z+i*k*Ue)/nu),  Re(q)>0.
```

This derivation is for locally uniform coefficients, not a theorem for our
curved/sheared NS field. It predicts paired outer/potential and viscous
responses, and can guide real/imaginary mode pairs for selected frequencies.
Curvature and mean shear would enter a more accurate local operator.

A possible global implementation uses boundary-extension differences:
`E_z(g) - E_0(g)`, where E_z is a shifted Stokes extension and E_0 the existing
Stokes extension with the same admissible boundary trace g. The difference has
homogeneous velocity traces and can be added to the current space. Boundary
traces come from a geometry/tangential basis, not a fitted nonlinear force.
This is a proposed extension operator; it has not been implemented or tested.

## 6. A concrete comparison sequence

1. Keep the tested four-scale space as control. Add paired thin/broad modes and
   measure frozen-response H1 error, displacement-related moments, wall shear,
   condition numbers, and cost per retained direction.
2. Compare matched profile sensitivities (thickness and pressure gradient) with
   an equal-cost increase in plain exponentials. Include a stagnation family.
3. Test unseen smooth forcing patterns and several viscosities/shifts against
   independent FEM, in addition to the current frozen nonlinear force. This
   guards against optimizing only one response.
4. Repeat NS with angular, bulk and temporal refinements separately. A timestep
   test must hold the spatial basis fixed. If bases later vary in time, account
   for the basis time derivative and state-transfer error.
5. For separation and detached shear layers, add wake-aligned interior modes or
   bulk refinement. Wall-attached similarity functions alone are insufficient.

The first acceptance target is better error-versus-DOF across those controls,
not an attractive profile plot. The current experimental result supports the
value of physically selected near-wall scales; it does not yet establish the
benefit of the additional classical-profile constructions above.
