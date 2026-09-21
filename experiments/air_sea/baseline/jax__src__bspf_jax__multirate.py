"""Fourth-order explicit MRI-GARK-ERK45a with classical RK4 inner integration.

Solves y'=fast(t,y)+slow(t,y), for arbitrary matching JAX pytrees. The five
slow RHS evaluations per macro step are coupled to five fast intervals of
length H/5; each interval contains `inner_steps` RK4 microsteps. This is an
additive multirate RK method, not frozen-state subcycling or extrapolation.

Method: A. Sandu, SIAM J. Numer. Anal. 57 (2019), 2300--2327,
arXiv:1808.02759. Rational coefficients cross-checked against the primary
table ARKODE_MRI_GARK_ERK45a in SUNDIALS v7.2.1:
https://github.com/LLNL/sundials/blob/v7.2.1/src/arkode/arkode_mri_tables.def
Only the principal method is used, not its third-order embedding.
"""

import jax
import jax.numpy as jnp
import numpy as np


# Rows correspond to intervals [0,.2], [.2,.4], ..., [.8,1].
# gamma_ij(theta) = GAMMA0[i,j] + theta*GAMMA1[i,j], 0<=theta<=1.
# The physical-time fast ODE uses gamma / Delta-c = 5*gamma.
GAMMA0 = np.array(
    [
        [1 / 5, 0, 0, 0, 0],
        [-53 / 16, 281 / 80, 0, 0, 0],
        [-36562993 / 71394880, 34903117 / 17848720, -88770499 / 71394880, 0, 0],
        [
            -7631593 / 71394880,
            -166232021 / 35697440,
            6068517 / 1519040,
            8644289 / 8924360,
            0,
        ],
        [
            277061 / 303808,
            -209323 / 1139280,
            -1360217 / 1139280,
            -148789 / 56964,
            147889 / 45120,
        ],
    ],
    dtype=np.float64,
)
GAMMA1 = np.array(
    [
        [0, 0, 0, 0, 0],
        [503 / 80, -503 / 80, 0, 0, 0],
        [-1365537 / 35697440, 4963773 / 7139488, -1465833 / 2231090, 0, 0],
        [66974357 / 35697440, 21445367 / 7139488, -3, -8388609 / 4462180, 0],
        [-18227 / 7520, 2, 1, 5, -41933 / 7520],
    ],
    dtype=np.float64,
)
GAMMA0.setflags(write=False)
GAMMA1.setflags(write=False)


def mri_gark_erk45a_step(state, time, macro_dt, fast_rhs, slow_rhs, *, inner_steps=1):
    """One fourth-order MRI macro step, with `20*inner_steps` fast RHS calls.

    `inner_steps` must be a positive static Python integer. RHS callbacks have
    signature (physical_time, state)->derivative. Budgets can be included as
    extra state leaves, preserving any linear invariant of each split RHS.
    RK4 finite inner solves preserve fourth order at fixed positive step ratio;
    explicit stability and smoothness restrictions still apply.
    """
    if (
        not isinstance(inner_steps, int)
        or isinstance(inner_steps, bool)
        or inner_steps < 1
    ):
        raise ValueError("inner_steps must be a positive static integer")
    gamma0, gamma1 = jnp.asarray(GAMMA0), jnp.asarray(GAMMA1)
    history = jax.tree.map(lambda y: jnp.zeros((5,) + y.shape, dtype=y.dtype), state)
    micro_dt = macro_dt / (5 * inner_steps)

    def add(a, b, scale):
        return jax.tree.map(lambda x, y: x + scale * y, a, b)

    def interval(i, carry):
        current, derivatives = carry
        slow = slow_rhs(time + macro_dt * (i / 5), current)
        derivatives = jax.tree.map(
            lambda values, value: values.at[i].set(value), derivatives, slow
        )
        forcing0 = jax.tree.map(
            lambda values: jnp.tensordot(5 * gamma0[i], values, axes=1), derivatives
        )
        forcing1 = jax.tree.map(
            lambda values: jnp.tensordot(5 * gamma1[i], values, axes=1), derivatives
        )

        def microstep(k, value):
            def rhs(local, stage):
                theta = (k + local) / inner_steps
                fast = fast_rhs(time + macro_dt * (i + theta) / 5, stage)
                return jax.tree.map(
                    lambda a, b, c: a + b + theta * c, fast, forcing0, forcing1
                )

            k1 = rhs(0.0, value)
            k2 = rhs(0.5, add(value, k1, micro_dt / 2))
            k3 = rhs(0.5, add(value, k2, micro_dt / 2))
            k4 = rhs(1.0, add(value, k3, micro_dt))
            return jax.tree.map(
                lambda y, a, b, c, d: y + micro_dt / 6 * (a + 2 * b + 2 * c + d),
                value,
                k1,
                k2,
                k3,
                k4,
            )

        current = jax.lax.fori_loop(0, inner_steps, microstep, current)
        return current, derivatives

    return jax.lax.fori_loop(0, 5, interval, (state, history))[0]
