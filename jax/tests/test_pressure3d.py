"""Three-axis direct inversion, boundary recovery and compressed batching."""

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import bspf_jax as b


def test_rectangular_lifted_and_compatible():
    grids = [np.linspace(0, 1, n) for n in (9, 10, 11)]
    dense = b.plan_pressure_poisson3d(
        *grids,
        q=2,
        n_basis=5,
        degree=4,
        baseline_points=6,
        chebyshev_modes=5,
    )
    compressed = b.compress_pressure_plan3d(dense, leaf_size=4, protected_modes=2)
    p = jnp.asarray(np.random.default_rng(7).normal(size=(9, 10, 11)))
    rhs = b.pressure_action3d(dense, p)
    for plan in (dense, compressed):
        run = jax.jit(partial(b.solve_pressure_poisson3d, lifted=True, batch_size=7))
        result = run(plan, rhs)
        assert bool(result.converged)
        np.testing.assert_allclose(result.pressure, p, atol=2e-9, rtol=2e-9)
        untiled = b.solve_pressure_poisson3d(plan, rhs, lifted=True, batch_size=None)
        np.testing.assert_allclose(result.pressure, untiled.pressure, atol=1e-10)
        compatible = b.solve_pressure_poisson3d(plan, b.pressure_schur3d(dense, p))
        assert bool(compatible.converged)
        assert not bool(
            b.solve_pressure_poisson3d(
                plan, jnp.zeros_like(p).at[0, 0, 0].set(1)
            ).converged
        )


def test_cube_shares_line_factors():
    x = np.linspace(0, 1, 35)
    plan = b.plan_pressure_poisson3d(x, x, x)
    assert plan.lines[0] is plan.lines[1] is plan.lines[2]
    compressed = b.compress_pressure_plan3d(plan)
    assert compressed.lines[0] is compressed.lines[1] is compressed.lines[2]
    assert compressed.lines[0].vectors is None
