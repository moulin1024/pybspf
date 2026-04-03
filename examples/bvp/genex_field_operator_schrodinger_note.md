# GENE-X Field Operators as Schrödinger-Type Systems

## Scope

This note summarizes a theoretical reformulation of the three elliptic field
operators used in the `GENE-X` field solve path:

- quasi-neutrality for the electrostatic potential `phi`
- parallel Ampere's law for the parallel vector potential `A_parallel`
- Ohm's law for the parallel electric field `E_parallel`

The goal is to place all three equations into a common
Schrödinger-type framework and to record the main consequences of that point
of view.

The code-level references behind the formulas below are:

- `upstream/genex/src/operators/field_solve_operators/op_solve_qn_eq_s.f90`
- `upstream/genex/src/operators/field_solve_operators/op_solve_amps_law_s.f90`
- `upstream/genex/src/operators/field_solve_operators/op_solve_ohms_law_s.f90`
- `upstream/genex/src/cxx/operators/op_mom_maxwells_eq.hxx`
- `upstream/genex/src/cxx/operators/op_mom_ohms_law.hxx`
- `upstream/genex/src/mesh/mesh_5d_m.f90`
- `upstream/genex/src/mesh/mesh_5d_s.f90`

## Unified elliptic form

The underlying field solver is written in the generic form

$$
\lambda u - \xi \nabla \cdot \left(c \nabla u\right) = b.
$$

At the operator level, the three field equations correspond to:

1. Quasi-neutrality:
$$
- \xi_{\rm qn} \nabla \cdot \left(c_{\rm qn} \nabla \phi\right) = b_{\rm qn}.
$$

2. Parallel Ampere's law:
$$
- \xi_A \nabla \cdot \left(c_A \nabla A_\parallel\right) = b_A.
$$

3. Ohm's law:
$$
\lambda_\Omega E_\parallel - \xi_\Omega \nabla \cdot
\left(c_\Omega \nabla E_\parallel\right) = b_\Omega.
$$

## Meaning of the Jacobian `J`

In this code, `J` is the Jacobian of the coordinate transform from laboratory
coordinates to the mesh coordinate system. It is a geometric weight, not a
current density. It enters the volume element through

$$
dV = J \, dq^1 dq^2 dq^3.
$$

The implementation uses:

- `J = 1` for slab and circular equilibria
- `J = R` for cylindrical-type cases

so `J` is the factor that makes integration and divergence operators correct in
curvilinear geometry.

## Code-specific coefficients

Let

$$
h_{\rm eff}^2 := \left(\frac{\rho_{\rm ref}}{L_{\rm ref}}\right)^2.
$$

Then the three equations use the following coefficient structure inside the
compute region.

### Ampere

$$
\xi_A = \frac{h_{\rm eff}^2}{J}, \qquad c_A = J, \qquad \lambda_A = 0.
$$

Hence

$$
- \frac{h_{\rm eff}^2}{J} \nabla \cdot \left(J \nabla A_\parallel\right) = b_A.
$$

The source term is a parallel-current moment,

$$
b_A \sim \sum_s \beta_{\rm ref} q_s \sqrt{\frac{T_s}{2m_s}}
\int v_\parallel f_s \, dW.
$$

### Ohm

$$
\xi_\Omega = \frac{h_{\rm eff}^2}{J}, \qquad c_\Omega = J.
$$

Hence

$$
- \frac{h_{\rm eff}^2}{J} \nabla \cdot \left(J \nabla E_\parallel\right)
+ \lambda_\Omega E_\parallel = b_\Omega.
$$

The reaction coefficient and source are moment-based:

$$
\lambda_\Omega \sim
- \sum_s \frac{\beta_{\rm ref} q_s^2}{2m_s}
\int v_\parallel \partial_{v_\parallel} f_s \, dW,
$$

$$
b_\Omega \sim
\sum_s \beta_{\rm ref} q_s \sqrt{\frac{T_s}{2m_s}}
\int v_\parallel \partial_t f_s \, dW.
$$

### Quasi-neutrality

$$
\xi_{\rm qn} = \frac{1}{J}.
$$

In the linearized polarization setting used by the code,

$$
c_{\rm qn} = h_{\rm eff}^2 \, \tilde c_{\rm qn},
\qquad
\tilde c_{\rm qn} \sim \frac{J}{B^2}.
$$

Hence

$$
- \frac{h_{\rm eff}^2}{J} \nabla \cdot
\left(\tilde c_{\rm qn} \nabla \phi\right) = b_{\rm qn}.
$$

The source is a charge-density moment,

$$
b_{\rm qn} \sim \sum_s q_s \int f_s \, dW.
$$

## Weighted Poisson viewpoint

The Ampere equation is already a weighted Poisson equation:

$$
- \frac{h_{\rm eff}^2}{J} \nabla \cdot \left(J \nabla A_\parallel\right) = b_A.
$$

Equivalently,

$$
- \nabla \cdot \left(J \nabla A_\parallel\right)
= \frac{J}{h_{\rm eff}^2} b_A.
$$

So in standard weighted-Poisson notation,

$$
- \nabla \cdot \left(w \nabla u\right) = f,
\qquad
u = A_\parallel, \quad w = J.
$$

The Ohm equation is the same weighted Poisson operator plus a reaction term.
The quasi-neutrality equation is the same structure with a different spatially
varying weight.

## Schrödinger-type reformulation

Consider the generic scalar equation

$$
- \frac{1}{W(x)} \nabla \cdot \left(P(x) \nabla u\right) + Q(x) u = F(x).
$$

After division by \(P/W\), this becomes

$$
- \Delta u - \nabla \ln P \cdot \nabla u + \frac{W}{P} Q \, u
= \frac{W}{P} F.
$$

Now apply the Liouville transform

$$
u = P^{-1/2} \psi
\qquad \Longleftrightarrow \qquad
\psi = \sqrt{P} \, u.
$$

Then the first-order term disappears and one obtains the Schrödinger-type form

$$
\left[- \Delta + U_P(x) + \frac{W}{P} Q(x)\right]\psi
= \frac{W}{\sqrt{P}} F,
$$

with

$$
U_P(x) :=
\frac{\Delta \sqrt{P}}{\sqrt{P}}
= \frac{1}{2} \Delta \ln P + \frac{1}{4} |\nabla \ln P|^2.
$$

## Fully unified form with the same effective semiclassical parameter

Using the same

$$
h_{\rm eff}^2 = \left(\frac{\rho_{\rm ref}}{L_{\rm ref}}\right)^2
$$

for all three operators, the transformed equations can all be written as

$$
\boxed{
\left[- h_{\rm eff}^2 \Delta + V_{\rm eff}(x)\right]\psi = S(x)
}.
$$

### Ampere

Use

$$
\psi_A = \sqrt{J} \, A_\parallel.
$$

Then

$$
\left[- h_{\rm eff}^2 \Delta + h_{\rm eff}^2 U_J\right]\psi_A
= \sqrt{J} \, b_A,
$$

where

$$
U_J := \frac{\Delta \sqrt{J}}{\sqrt{J}}.·
$$

So

$$
V_A = h_{\rm eff}^2 U_J,
\qquad
S_A = \sqrt{J} \, b_A.
$$

### Ohm

Use

$$
\psi_\Omega = \sqrt{J} \, E_\parallel.
$$

Then

$$
\left[- h_{\rm eff}^2 \Delta + \lambda_\Omega + h_{\rm eff}^2 U_J\right]
\psi_\Omega
= \sqrt{J} \, b_\Omega.
$$

So

$$
V_\Omega = \lambda_\Omega + h_{\rm eff}^2 U_J,
\qquad
S_\Omega = \sqrt{J} \, b_\Omega.
$$

### Quasi-neutrality

Use

$$
\psi_{\rm qn} = \sqrt{\tilde c_{\rm qn}} \, \phi.
$$

Then

$$
\left[- h_{\rm eff}^2 \Delta
+ h_{\rm eff}^2 U_{\tilde c_{\rm qn}}\right]\psi_{\rm qn}
= \frac{J}{\sqrt{\tilde c_{\rm qn}}} \, b_{\rm qn},
$$

where

$$
U_{\tilde c_{\rm qn}} :=
\frac{\Delta \sqrt{\tilde c_{\rm qn}}}{\sqrt{\tilde c_{\rm qn}}}.
$$

So

$$
V_{\rm qn} = h_{\rm eff}^2 U_{\tilde c_{\rm qn}},
\qquad
S_{\rm qn} = \frac{J}{\sqrt{\tilde c_{\rm qn}}} \, b_{\rm qn}.
$$

## Hamiltonian notation

All three equations can be written as

$$
\hat H_X |\psi_X\rangle = |S_X\rangle,
\qquad
X \in \{A, \Omega, {\rm qn}\},
$$

with

$$
\hat H_X = - h_{\rm eff}^2 \nabla^2 + V_X(\mathbf x).
$$

The three effective Hamiltonians are therefore

$$
\hat H_A = - h_{\rm eff}^2 \nabla^2 + h_{\rm eff}^2 U_J,
$$

$$
\hat H_\Omega =
- h_{\rm eff}^2 \nabla^2 + \lambda_\Omega + h_{\rm eff}^2 U_J,
$$

$$
\hat H_{\rm qn} =
- h_{\rm eff}^2 \nabla^2 + h_{\rm eff}^2 U_{\tilde c_{\rm qn}}.
$$

The associated homogeneous spectral problems are

$$
\hat H_X |n,X\rangle = E_n^{(X)} |n,X\rangle.
$$

For a basis \(\{|\chi_i\rangle\}\), the matrix elements are

$$
(H_X)_{ij}
= \langle \chi_i | \hat H_X | \chi_j \rangle
= h_{\rm eff}^2 \int_\Omega \nabla \chi_i^* \cdot \nabla \chi_j \, d\mathbf x
+ \int_\Omega \chi_i^* V_X(\mathbf x) \chi_j \, d\mathbf x.
$$

Then the forced problem becomes

$$
H_X \mathbf c_X = \mathbf s_X.
$$

## Interpretation

This produces a clean classification:

- Ampere: kinetic term plus geometric potential
- Ohm: kinetic term plus geometric potential plus a physical reaction
  potential
- quasi-neutrality: kinetic term plus a polarization-induced effective
  potential

In short:

- `Ampere` is a purely geometric Schrödinger-type system
- `Ohm` is a geometric plus reactive Schrödinger-type system
- `quasineutrality` is a variable-medium Schrödinger-type system

## What can be analyzed theoretically in this language

This common Hamiltonian form opens several standard routes.

### 1. Well-posedness and coercivity

Study positivity, self-adjointness, null spaces, and boundary-condition
compatibility through the quadratic form

$$
\mathcal E[\psi]
= h_{\rm eff}^2 \|\nabla \psi\|^2
+ \langle \psi, V_X \psi \rangle
- 2 \Re \langle \psi, S_X \rangle.
$$

### 2. Spectral analysis

Study low eigenvalues and eigenfunctions of \(\hat H_X\) to understand:

- soft modes
- spectral gaps
- localization
- geometry-induced trapping

### 3. Semiclassical analysis

Since \(h_{\rm eff} = \rho_{\rm ref}/L_{\rm ref}\) is a natural small
parameter, one can investigate:

- WKB structure
- boundary layers
- turning points
- asymptotic localization near minima of \(V_X\)

### 4. Perturbation theory

Expand around simple backgrounds such as

$$
J = 1 + \varepsilon j_1,
\qquad
\lambda_\Omega = \lambda_0 + \varepsilon \lambda_1,
\qquad
\tilde c_{\rm qn} = c_0 + \varepsilon c_1,
$$

to compute eigenvalue drift, mode-shape corrections, and sensitivity to
geometry or profiles.

### 5. Green's functions and response kernels

The solution can be written formally as

$$
\psi_X = \hat H_X^{-1} S_X.
$$

This allows analysis of nonlocal response, decay length scales, and source
propagation in the three systems.

### 6. Conditioning and solver behavior

The spectral viewpoint translates directly into conditioning estimates for the
discrete systems and helps explain why certain preconditioners or multigrid
strategies behave differently for the three operators.

## Can the three field equations be solved together?

Yes, but there are two different meanings.

### Operator-level statement

At the level of the transformed field equations alone, they can be assembled
into a block system

$$
\mathcal H
\begin{pmatrix}
\psi_{\rm qn} \\
\psi_A \\
\psi_\Omega
\end{pmatrix}
=
\begin{pmatrix}
S_{\rm qn} \\
S_A \\
S_\Omega
\end{pmatrix}.
$$

With the current implementation, this is effectively block diagonal once the
sources and coefficients are frozen, so solving them together gives little
numerical benefit over three separate solves.

### Full physics statement

At the full coupled level, the right-hand sides and coefficients are generated
from moments of the kinetic distribution function, and those moments depend in
turn on the electromagnetic fields through the kinetic evolution.

Therefore the larger coupled problem is not just a three-field elliptic system.
It is a block system involving at least

$$
(f, \phi, A_\parallel, E_\parallel),
$$

and a fully monolithic solve would have to include that kinetic-field coupling.

The present `GENE-X` implementation uses a staggered strategy instead:

- compute moments and coefficients
- solve quasi-neutrality
- solve parallel Ampere
- solve Ohm separately

That is mathematically consistent, but it is not a monolithic block solve.

## Bottom line

The three `GENE-X` field operators admit a common Schrödinger-type
representation with the same effective semiclassical parameter

$$
h_{\rm eff} = \frac{\rho_{\rm ref}}{L_{\rm ref}}.
$$

This reformulation is useful because it makes it possible to compare all three
systems using one shared language:

- kinetic term
- effective potential
- source state
- Hamiltonian spectrum
- conditioning and localization properties

That common language is a practical bridge between geometric plasma modeling,
elliptic solvers, and spectral or semiclassical analysis.

## Dirichlet operator analysis

In this section we restrict attention to homogeneous Dirichlet boundary
conditions and adopt the working assumption

$$
\lambda_\Omega(x) \ge 0.
$$

We also assume:

- \(\Omega\) is bounded and sufficiently smooth
- \(J(x) > 0\) and \(\tilde c_{\rm qn}(x) > 0\)
- \(J\), \(\tilde c_{\rm qn}\), and \(\lambda_\Omega\) are real-valued and
  sufficiently regular

Under these assumptions, all three transformed field operators are self-adjoint
on \(L^2(\Omega)\) with a natural form domain \(H_0^1(\Omega)\).

### Factorization identity

For a positive scalar field \(f\), define

$$
a_f := \nabla \ln \sqrt f = \frac{1}{2}\nabla \ln f,
\qquad
D_f := \nabla - a_f.
$$

Then

$$
-\Delta + U_f = D_f^\dagger D_f,
\qquad
U_f = \frac{\Delta \sqrt f}{\sqrt f}.
$$

It is also useful to note that

$$
D_f \psi = \sqrt f \, \nabla\left(\frac{\psi}{\sqrt f}\right),
$$

and therefore

$$
\|D_f \psi\|_{L^2}^2
=
\int_\Omega f \left|\nabla\left(\frac{\psi}{\sqrt f}\right)\right|^2 dx.
$$

This gives the exact factorizations

$$
H_A = h_{\rm eff}^2 D_J^\dagger D_J,
$$

$$
H_{\rm qn} = h_{\rm eff}^2 D_{\tilde c_{\rm qn}}^\dagger
D_{\tilde c_{\rm qn}},
$$

$$
H_\Omega = h_{\rm eff}^2 D_J^\dagger D_J + \lambda_\Omega.
$$

### Ampere operator

The bilinear form associated with \(H_A\) is

$$
a_A(\psi,\varphi)
=
h_{\rm eff}^2
\int_\Omega \overline{D_J \psi}\cdot D_J \varphi \, dx.
$$

Its quadratic form is

$$
a_A(\psi,\psi)
=
h_{\rm eff}^2 \|D_J\psi\|_{L^2}^2
=
h_{\rm eff}^2 \int_\Omega
J \left|\nabla\left(\frac{\psi}{\sqrt J}\right)\right|^2 dx.
$$

The corresponding energy functional is

$$
\mathcal E_A[\psi]
=
\frac{1}{2} a_A(\psi,\psi)
- \Re \int_\Omega \overline{\psi}\, S_A \, dx.
$$

Because of the factorization, \(H_A\) is nonnegative:

$$
a_A(\psi,\psi) \ge 0.
$$

Under homogeneous Dirichlet boundary conditions, Poincare's inequality and the
positivity of \(J\) imply coercivity:

$$
a_A(\psi,\psi)
\ge
c_A h_{\rm eff}^2 \|\psi\|_{H_0^1(\Omega)}^2
$$

for some constant \(c_A > 0\).

### Quasi-neutrality operator

The bilinear form associated with \(H_{\rm qn}\) is

$$
a_{\rm qn}(\psi,\varphi)
=
h_{\rm eff}^2
\int_\Omega
\overline{D_{\tilde c_{\rm qn}} \psi}
\cdot
D_{\tilde c_{\rm qn}} \varphi
\,
dx.
$$

Its quadratic form is

$$
a_{\rm qn}(\psi,\psi)
=
h_{\rm eff}^2
\|D_{\tilde c_{\rm qn}}\psi\|_{L^2}^2
=
h_{\rm eff}^2
\int_\Omega
\tilde c_{\rm qn}
\left|\nabla\left(\frac{\psi}{\sqrt{\tilde c_{\rm qn}}}\right)\right|^2 dx.
$$

The corresponding energy functional is

$$
\mathcal E_{\rm qn}[\psi]
=
\frac{1}{2} a_{\rm qn}(\psi,\psi)
- \Re \int_\Omega \overline{\psi}\, S_{\rm qn} \, dx.
$$

Again,

$$
a_{\rm qn}(\psi,\psi) \ge 0,
$$

and homogeneous Dirichlet boundary conditions yield coercivity:

$$
a_{\rm qn}(\psi,\psi)
\ge
c_{\rm qn} h_{\rm eff}^2 \|\psi\|_{H_0^1(\Omega)}^2
$$

for some constant \(c_{\rm qn} > 0\).

### Ohm operator

Assuming \(\lambda_\Omega \ge 0\), the bilinear form associated with
\(H_\Omega\) is

$$
a_\Omega(\psi,\varphi)
=
h_{\rm eff}^2
\int_\Omega \overline{D_J \psi}\cdot D_J \varphi \, dx
+
\int_\Omega \lambda_\Omega \overline{\psi}\,\varphi \, dx.
$$

Its quadratic form is

$$
a_\Omega(\psi,\psi)
=
h_{\rm eff}^2 \|D_J \psi\|_{L^2}^2
+
\int_\Omega \lambda_\Omega |\psi|^2 dx.
$$

The corresponding energy functional is

$$
\mathcal E_\Omega[\psi]
=
\frac{1}{2} a_\Omega(\psi,\psi)
- \Re \int_\Omega \overline{\psi}\, S_\Omega \, dx.
$$

Since \(\lambda_\Omega \ge 0\),

$$
a_\Omega(\psi,\psi) \ge a_A(\psi,\psi) \ge 0.
$$

Hence \(H_\Omega\) is also nonnegative and coercive:

$$
a_\Omega(\psi,\psi)
\ge
c_A h_{\rm eff}^2 \|\psi\|_{H_0^1(\Omega)}^2.
$$

If in addition \(\lambda_\Omega(x) \ge \lambda_* > 0\), then one has the
stronger estimate

$$
a_\Omega(\psi,\psi)
\ge
c_A h_{\rm eff}^2 \|\nabla\psi\|_{L^2}^2
+
\lambda_* \|\psi\|_{L^2}^2.
$$

### Template coercivity bound

Let \(f \in \{J, \tilde c_{\rm qn}\}\) and assume

$$
0 < f_{\min} \le f(x) \le f_{\max}.
$$

Multiplication by \(\sqrt f\) is an isomorphism on \(H_0^1(\Omega)\), so there
exists a constant \(C_f > 0\) such that

$$
\left\|\frac{\psi}{\sqrt f}\right\|_{H_0^1(\Omega)}
\ge
C_f^{-1}\|\psi\|_{H_0^1(\Omega)}.
$$

Combining this with Poincare's inequality gives the explicit template

$$
h_{\rm eff}^2
\int_\Omega
f
\left|\nabla\left(\frac{\psi}{\sqrt f}\right)\right|^2 dx
\ge
\frac{h_{\rm eff}^2 f_{\min}}{(1+C_P^2) C_f^2}
\|\psi\|_{H_0^1(\Omega)}^2,
$$

where \(C_P\) is the Poincare constant of the domain.

This is a convenient generic lower bound for the coercivity constants of the
Ampere and quasi-neutrality operators.

### Summary of the Dirichlet case

Under the assumptions above:

- \(H_A\) is self-adjoint, nonnegative, and coercive
- \(H_{\rm qn}\) is self-adjoint, nonnegative, and coercive
- \(H_\Omega\) is self-adjoint, nonnegative, and coercive whenever
  \(\lambda_\Omega \ge 0\)

Thus all three problems are covered by the standard variational framework based
on Lax-Milgram and by the usual spectral theory for lower-bounded
self-adjoint elliptic operators.

## Lower bounds on the lowest eigenvalue

Still in the homogeneous Dirichlet setting, let

$$
\lambda_1(H_X)
$$

denote the lowest eigenvalue of \(H_X\). Since the form domains are
\(H_0^1(\Omega)\) and the operators are coercive, the spectrum is discrete and
\(\lambda_1(H_X) > 0\).

### Ampere

Because

$$
H_A = h_{\rm eff}^2 D_J^\dagger D_J,
$$

the whole operator factors out a factor \(h_{\rm eff}^2\). Therefore there is a
geometry-dependent constant \(\nu_1^A > 0\), independent of \(h_{\rm eff}\),
such that

$$
\lambda_1(H_A) = h_{\rm eff}^2 \nu_1^A.
$$

Equivalently, using the coercivity estimate,

$$
\lambda_1(H_A) \ge c_A h_{\rm eff}^2.
$$

### Quasi-neutrality

Similarly,

$$
H_{\rm qn}
=
h_{\rm eff}^2 D_{\tilde c_{\rm qn}}^\dagger D_{\tilde c_{\rm qn}},
$$

so there exists \(\nu_1^{\rm qn} > 0\), independent of \(h_{\rm eff}\), such
that

$$
\lambda_1(H_{\rm qn}) = h_{\rm eff}^2 \nu_1^{\rm qn},
$$

and hence

$$
\lambda_1(H_{\rm qn}) \ge c_{\rm qn} h_{\rm eff}^2.
$$

### Ohm

For Ohm's law,

$$
H_\Omega = h_{\rm eff}^2 D_J^\dagger D_J + \lambda_\Omega.
$$

Let

$$
\lambda_* := \operatorname*{ess\,inf}_{x\in\Omega} \lambda_\Omega(x) \ge 0.
$$

Then, by the min-max principle,

$$
\lambda_1(H_\Omega)
\ge
\lambda_* + \lambda_1(H_A)
\ge
\lambda_* + c_A h_{\rm eff}^2.
$$

Hence two important regimes appear:

- if \(\lambda_* > 0\), then the spectral gap of \(H_\Omega\) remains
  \(O(1)\) as \(h_{\rm eff}\to 0\)
- if \(\lambda_* = 0\), then the lowest eigenvalue is still bounded below by
  an \(O(h_{\rm eff}^2)\) contribution from the elliptic part

This is the key distinction between the Ohm operator and the other two field
operators.

## Condition number scaling with \(h_{\rm eff}\)

Condition numbers are meaningful only after discretization, since the
continuous operators are unbounded above. Therefore let \(A_X(h_{\rm eff})\)
denote any conforming finite-dimensional SPD discretization of \(H_X\) on a
fixed spatial grid or fixed finite-element space.

The discussion below isolates the dependence on \(h_{\rm eff}\). Usual
mesh-dependent growth, such as \(O(\Delta x^{-2})\), is a separate issue.

### Ampere

The discrete matrix has the form

$$
A_A(h_{\rm eff}) = h_{\rm eff}^2 K_A,
$$

where \(K_A\) does not depend on \(h_{\rm eff}\) if the geometry and spatial
discretization are fixed.

Therefore

$$
\kappa_2(A_A(h_{\rm eff}))
=
\kappa_2(K_A),
$$

so the spectral condition number is independent of \(h_{\rm eff}\).

In particular, shrinking \(h_{\rm eff}\) rescales all eigenvalues by the same
factor \(h_{\rm eff}^2\), but does not worsen the condition number by itself.

### Quasi-neutrality

Exactly the same argument gives

$$
A_{\rm qn}(h_{\rm eff}) = h_{\rm eff}^2 K_{\rm qn},
$$

and hence

$$
\kappa_2(A_{\rm qn}(h_{\rm eff}))
=
\kappa_2(K_{\rm qn}),
$$

again independent of \(h_{\rm eff}\) for fixed geometry and fixed spatial
discretization.

### Ohm

For Ohm's law, the discrete matrix has the form

$$
A_\Omega(h_{\rm eff}) = h_{\rm eff}^2 K_A + M_\lambda,
$$

where \(M_\lambda\) is the reaction matrix associated with \(\lambda_\Omega\).

Let \(\lambda_{\min}(M_\lambda)\) and \(\lambda_{\max}(M_\lambda)\) denote the
extreme eigenvalues of the discrete reaction part. Then

$$
\lambda_{\min}(A_\Omega)
\ge
h_{\rm eff}^2 \lambda_{\min}(K_A)
+
\lambda_{\min}(M_\lambda),
$$

$$
\lambda_{\max}(A_\Omega)
\le
h_{\rm eff}^2 \lambda_{\max}(K_A)
+
\lambda_{\max}(M_\lambda).
$$

Hence

$$
\kappa_2(A_\Omega(h_{\rm eff}))
\le
\frac{h_{\rm eff}^2 \lambda_{\max}(K_A) + \lambda_{\max}(M_\lambda)}
{h_{\rm eff}^2 \lambda_{\min}(K_A) + \lambda_{\min}(M_\lambda)}.
$$

This leads to the following cases.

#### Case 1: strictly positive reaction floor

If \(\lambda_\Omega(x) \ge \lambda_* > 0\), then
\(\lambda_{\min}(M_\lambda)\) is bounded below by a positive constant
independent of \(h_{\rm eff}\), and therefore

$$
\kappa_2(A_\Omega(h_{\rm eff})) = O(1)
\qquad
\text{as } h_{\rm eff}\to 0
$$

on a fixed spatial discretization.

So the Ohm operator becomes reaction-dominated at small \(h_{\rm eff}\), but
not ill-conditioned because of that.

#### Case 2: nonnegative but vanishing reaction

If \(\lambda_\Omega \ge 0\) but \(\lambda_{\min}(M_\lambda)=0\), then for small
\(h_{\rm eff}\) the elliptic part still controls coercivity on the Dirichlet
space, and the matrix behaves like

$$
A_\Omega(h_{\rm eff}) \approx h_{\rm eff}^2 K_A
$$

on the weakest modes.

In that case the \(h_{\rm eff}^2\) prefactor again cancels from the condition
number, so there is still no singular dependence on \(h_{\rm eff}\) alone:

$$
\kappa_2(A_\Omega(h_{\rm eff}))
\sim
\kappa_2(K_A)
$$

up to \(h_{\rm eff}\)-independent constants.

## Spectral scaling summary

With Dirichlet boundary conditions and \(\lambda_\Omega \ge 0\), the lowest
eigenvalues satisfy

$$
\lambda_1(H_A) \asymp h_{\rm eff}^2,
\qquad
\lambda_1(H_{\rm qn}) \asymp h_{\rm eff}^2,
$$

$$
\lambda_1(H_\Omega) \gtrsim \lambda_* + h_{\rm eff}^2.
$$

Thus:

- Ampere and quasi-neutrality are soft at small \(h_{\rm eff}\), with spectral
  gaps proportional to \(h_{\rm eff}^2\)
- Ohm is potentially much stiffer if \(\lambda_\Omega\) has an \(O(1)\)
  positive lower bound

For a fixed spatial discretization, the \(h_{\rm eff}\)-dependence of the
condition number is correspondingly simple:

- Ampere: \(\kappa_2\) independent of \(h_{\rm eff}\)
- quasi-neutrality: \(\kappa_2\) independent of \(h_{\rm eff}\)
- Ohm: \(\kappa_2 = O(1)\) as \(h_{\rm eff}\to 0\), with no singular
  \(h_{\rm eff}\)-dependence

The practical conclusion is that \(h_{\rm eff}\) controls the spectral scale of
the operators, but by itself it does not create an intrinsic condition-number
blow-up. Any strong ill-conditioning must come from geometry, coefficient
contrast, reaction-profile variation, or spatial discretization, not from the
overall \(h_{\rm eff}^2\) prefactor alone.

## WKB analysis of Ohm's law

Among the three field equations, Ohm's law is the most natural target for a
semiclassical or WKB analysis because it contains a genuine nonnegative
reaction term. In the transformed variables,

$$
\left[
-h_{\rm eff}^2 \Delta
+ \lambda_\Omega(x)
+ h_{\rm eff}^2 U_J(x)
\right]\psi
=
S,
\qquad
\psi = \sqrt{J}\,E_\parallel.
$$

Assume for this section that

$$
\lambda_\Omega(x) \ge \lambda_* > 0,
$$

with \(\lambda_\Omega\) and \(J\) smooth enough for the formal asymptotics.

### Homogeneous WKB ansatz

For the homogeneous equation

$$
\left[
-h_{\rm eff}^2 \Delta
+ \lambda_\Omega(x)
+ h_{\rm eff}^2 U_J(x)
\right]\psi
=
0,
$$

one may introduce the exponential ansatz

$$
\psi(x)
\sim
e^{-\Phi(x)/h_{\rm eff}}
\left(
a_0(x) + h_{\rm eff} a_1(x) + \cdots
\right).
$$

Substituting into the equation and collecting orders of \(h_{\rm eff}\) gives
the eikonal equation

$$
|\nabla \Phi(x)|^2 = \lambda_\Omega(x),
$$

and, at the next order, the transport equation

$$
2 \nabla \Phi \cdot \nabla a_0
+
(\Delta \Phi) a_0
=
0.
$$

The leading-order geometric term \(U_J\) does not enter the eikonal equation;
it contributes only at subleading order.

### Interpretation

Because \(\lambda_\Omega \ge 0\), the WKB behavior is not oscillatory. The
dominant semiclassical picture is exponential attenuation or localization,
rather than wave propagation. The local decay length is therefore

$$
\ell_\Omega(x)
\sim
\frac{h_{\rm eff}}{\sqrt{\lambda_\Omega(x)}}.
$$

This scale is the main practical quantity produced by the WKB analysis:

- where \(\lambda_\Omega\) is large, the response is short-range and strongly
  localized
- where \(\lambda_\Omega\) is small, diffusion becomes more important and the
  coupling becomes longer-range

If \(\lambda_\Omega \approx \lambda_0\) is approximately constant, then

$$
\ell_\Omega \sim \frac{h_{\rm eff}}{\sqrt{\lambda_0}}.
$$

### Forced problem and localized response

For the forced problem,

$$
\left[
-h_{\rm eff}^2 \Delta
+ \lambda_\Omega(x)
+ h_{\rm eff}^2 U_J(x)
\right]\psi
=
S,
$$

the same semiclassical picture suggests that the Green's function is short
ranged. More precisely, the response away from the support of the source is
expected to decay like

$$
\exp\left(
- \frac{d_\lambda(x, \operatorname{supp} S)}{h_{\rm eff}}
\right),
$$

where \(d_\lambda\) is the Agmon distance induced by
\(\sqrt{\lambda_\Omega(x)}\).

This means that the Ohm operator behaves much more like a local
reaction-diffusion operator than a globally coupled Poisson operator.

### Leading-order approximation

When \(\lambda_\Omega\) dominates the diffusion term, the leading balance is
purely local:

$$
\psi_0(x) \approx \frac{S(x)}{\lambda_\Omega(x)}.
$$

Equivalently, in the original field variable,

$$
E_{\parallel,0}(x)
\approx
\frac{1}{\sqrt{J(x)}}\,
\frac{S(x)}{\lambda_\Omega(x)}.
$$

Diffusion and geometry then enter as \(O(h_{\rm eff}^2)\) corrections.

### Numerical implications

The WKB picture gives several direct numerical conclusions.

- A natural preconditioner for Ohm's law is the reaction-diffusion principal
  part
  $$
  P_\Omega
  =
  -h_{\rm eff}^2 \Delta + \lambda_\Omega,
  $$
  or, in the original variable,
  $$
  -\frac{h_{\rm eff}^2}{J}\nabla\cdot(J\nabla\cdot) + \lambda_\Omega,
  $$
  with \(U_J\) treated as a lower-order correction.
- A natural Krylov initial guess in reaction-dominated regions is
  $$
  \psi^{(0)} = \frac{S}{\lambda_\Omega}.
  $$
- The mesh spacing should resolve the smallest local decay scale,
  $$
  \Delta x \ll \min_x \frac{h_{\rm eff}}{\sqrt{\lambda_\Omega(x)}},
  $$
  in regions where \(\lambda_\Omega\) is large.
- Domain-decomposition and Schwarz-type methods are especially compatible with
  Ohm's law because the operator is short-ranged when \(\lambda_\Omega\) has a
  positive floor.

The main message is that, for Ohm's law, the WKB analysis is not primarily a
closed-form approximation tool. Its main value is to identify local decay
scales, explain why the response is short-ranged, and guide mesh design,
initial guesses, and preconditioner construction.

## Semiclassical analysis of quasi-neutrality

The quasi-neutrality equation behaves very differently from Ohm's law in the
semiclassical limit. After extracting a common
\(h_{\rm eff}^2 = (\rho_{\rm ref}/L_{\rm ref})^2\) from the polarization
coefficient, the transformed equation takes the form

$$
\left[
-h_{\rm eff}^2 \Delta
+ h_{\rm eff}^2 U_{\tilde c_{\rm qn}}(x)
\right]\psi
=
S_{\rm qn},
\qquad
\psi = \sqrt{\tilde c_{\rm qn}}\,\phi.
$$

Equivalently,

$$
H_{\rm qn}
=
h_{\rm eff}^2
\left(
-\Delta + U_{\tilde c_{\rm qn}}
\right).
$$

This means that the small parameter factors out of the entire operator. After
dividing by \(h_{\rm eff}^2\), one obtains

$$
\left(
-\Delta + U_{\tilde c_{\rm qn}}
\right)\psi
=
\frac{1}{h_{\rm eff}^2} S_{\rm qn}.
$$

Therefore the spatial structure of the homogeneous problem is governed by the
variable-coefficient elliptic operator
\(-\Delta + U_{\tilde c_{\rm qn}}\), not by a nontrivial eikonal equation of
the Ohm type.

### Consequence for WKB analysis

Unlike Ohm's law, quasi-neutrality has no \(O(1)\) reaction term. A standard
WKB ansatz does not produce a leading-order equation of the form

$$
|\nabla \Phi|^2 = V(x)
$$

with a nontrivial positive potential \(V\). There is therefore no natural
semiclassical decay length of the form
\(h_{\rm eff}/\sqrt{V(x)}\).

The main role of the Schrödinger reformulation for quasi-neutrality is instead:

- to expose the self-adjoint structure
- to identify the operator as a variable-coefficient diffusion problem
- to separate spectral scaling from coefficient-induced stiffness

### Diffusion viewpoint

In the original field variable, the equation is

$$
-\frac{h_{\rm eff}^2}{J}\nabla\cdot
\left(
\tilde c_{\rm qn}\nabla \phi
\right)
=
b_{\rm qn}.
$$

Thus the difficult numerical features are controlled primarily by the
diffusion coefficient \(\tilde c_{\rm qn}\), not by a semiclassical
localization mechanism. In the linear-polarization regime,

$$
\tilde c_{\rm qn} \sim \frac{J}{B^2},
$$

so strong spatial variation in geometry or magnetic-field strength directly
translates into coefficient contrast in the quasi-neutrality operator.

### Numerical implications

For quasi-neutrality, the most useful numerical interpretation is that of a
strongly variable-coefficient elliptic solve.

- Preconditioners should approximate
  $$
  -\nabla\cdot(\tilde c_{\rm qn}\nabla\cdot)
  $$
  rather than only a constant-coefficient Laplacian.
- Solver difficulty is driven mainly by coefficient contrast, anisotropy, and
  geometry, not by \(h_{\rm eff}\) itself.
- In nonlinear polarization regimes, \(\tilde c_{\rm qn}\) becomes state
  dependent, so frozen-coefficient or lagged-coefficient preconditioners are a
  natural practical strategy.

The quasi-neutrality equation is therefore best viewed as a variable-coefficient
diffusion problem whose eigenvalues scale like \(h_{\rm eff}^2\), but whose
mode shapes are determined by geometry and polarization structure rather than
by a reaction-dominated WKB mechanism.

## Semiclassical analysis of Ampere's law

Ampere's law is even closer to a pure weighted Poisson problem. In transformed
variables,

$$
\left[
-h_{\rm eff}^2 \Delta
+ h_{\rm eff}^2 U_J(x)
\right]\psi_A
=
S_A,
\qquad
\psi_A = \sqrt{J}\,A_\parallel,
$$

or equivalently

$$
H_A
=
h_{\rm eff}^2
\left(
-\Delta + U_J
\right).
$$

Exactly as for quasi-neutrality, the factor \(h_{\rm eff}^2\) multiplies the
entire operator. After removing this common factor, the spatial structure is
controlled by \(-\Delta + U_J\), and no Ohm-like eikonal equation appears.

### Consequence for WKB analysis

There is no \(O(1)\) reaction potential in Ampere's law, so the standard
semiclassical ansatz does not lead to a nontrivial local decay scale. The
Schrödinger form is useful mainly because it reveals the geometric potential

$$
U_J = \frac{\Delta \sqrt{J}}{\sqrt{J}},
$$

but the operator remains fundamentally elliptic rather than
reaction-dominated.

### Weighted-Poisson viewpoint

In the original field variable, Ampere's law is

$$
-\frac{h_{\rm eff}^2}{J}\nabla\cdot(J\nabla A_\parallel) = b_A,
$$

or equivalently

$$
-\nabla\cdot(J\nabla A_\parallel)
=
\frac{J}{h_{\rm eff}^2} b_A.
$$

Thus the natural numerical interpretation is that of a weighted Poisson solve
with geometry-driven coefficients.

### Numerical implications

For Ampere's law, the dominant numerical issues are geometric weighting and
mesh quality rather than semiclassical localization.

- Preconditioners should target the weighted diffusion operator
  $$
  -\nabla\cdot(J\nabla\cdot).
  $$
- The main source of stiffness is spatial variation in \(J\), especially in
  strongly curved or nonuniform coordinates.
- Relative to quasi-neutrality, the coefficient structure is simpler because
  the operator depends only on the Jacobian factor and not on the additional
  \(B^{-2}\) polarization weighting.

Ampere's law is therefore best regarded as a geometry-weighted Poisson problem:
its eigenvalues scale like \(h_{\rm eff}^2\), but the shape and difficulty of
the solve are controlled by geometry rather than by a true WKB mechanism.
