"""MHD work exchange, conducting walls, and an independent resistive eigenmode."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bspf_models.plasma.mhd_cavity import plan_mhd_cavity
from bspf_models.plasma.mhd_cavity import island_pair
from bspf_models.plasma.mhd_cavity import magnetic_fields
from bspf_models.plasma.mhd_cavity import mhd_exchange
from bspf_models.plasma.mhd_cavity import mhd_rhs
from bspf_models.plasma.mhd_cavity import mhd_step
from bspf_models.plasma.mhd_cavity import mhd_budget
from bspf_models.fluids.cavity import plan_cavity_stepper
from bspf_models.fluids.cavity import cavity_step
from bspf_models.fluids.stream_navier_stokes import stream_ns_velocity

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def plan():
    pytest.importorskip("gmpy2")
    return plan_mhd_cavity(n=33, lid_speed=0.0)


def test_magnetic_wall_and_continuous_divergence(plan):
    p, m = plan, plan.magnetic
    b = island_pair(p, peak_field=2.0)
    flux, magnetic, current = map(np.asarray, magnetic_fields(p, b, nodes=True))
    assert max(abs(flux[[0, -1]]).max(), abs(flux[:, [0, -1]]).max()) < 1e-11
    assert (
        max(abs(magnetic[[0, -1], :, 0]).max(), abs(magnetic[:, [0, -1], 1]).max())
        < 1e-10
    )
    div = (m.g @ b) @ m.g.T - m.g @ (b @ m.g.T)
    assert np.max(abs(div)) < 1e-10
    # Do not accidentally clamp both A and its normal derivative (B_t).
    assert m.b.shape[1] == p.fluid.spatial.x.b.shape[1] + 2
    assert abs(current[[0, -1]]).max() < 1e-8


def test_exchange_and_closed_box_energy_law(plan):
    p, f = plan, plan.fluid.spatial
    rng = np.random.default_rng(7)
    a = jnp.asarray(rng.normal(size=f.denominator.shape) * 1e-6)
    b = island_pair(p, peak_field=2.0)
    lorentz, induction = mhd_exchange(p, a, b, 0.0)
    fluid_work = jnp.sum(a * lorentz)
    magnetic_work = jnp.sum(p.magnetic_laplacian * b * induction)
    np.testing.assert_allclose(fluid_work, -magnetic_work, atol=2e-13, rtol=2e-12)
    budget = np.asarray(mhd_budget(p, a, b, 0.0))
    assert budget[3] > 0 and budget[4] > 0
    assert abs(budget[-1]) < 1e-10


def test_no_magnetic_field_recovers_ns(plan):
    p, f = plan, plan.fluid.spatial
    a = jnp.asarray(np.random.default_rng(8).normal(size=f.denominator.shape) * 1e-7)
    b = jnp.zeros_like(p.magnetic_laplacian)
    stepper = plan_cavity_stepper(p.fluid, 0.001)
    aa, bb = jax.jit(mhd_step)(p, stepper, a, b, 0.0)
    expected = cavity_step(p.fluid, stepper, a, 0.0)
    np.testing.assert_allclose(aa, expected, atol=1e-14)
    np.testing.assert_array_equal(bb, 0.0)


def test_exact_resistive_mode_and_pressure_balance(plan):
    p, m = plan, plan.magnetic
    x, y = m.points[:, None], m.points[None, :]
    exact = jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
    w = m.weights[:, None] * m.weights[None, :]
    b = m.b.T @ (w * exact) @ m.b
    a = jnp.zeros_like(p.fluid.spatial.denominator)
    flux, _, current = magnetic_fields(p, b)
    np.testing.assert_allclose(flux, exact, atol=2e-8)
    np.testing.assert_allclose(current, 2 * jnp.pi**2 * exact, atol=2e-5)
    da, db = mhd_rhs(p, a, b, 0.0)
    # j*grad(A)=grad(pi²*A²), hence incompressible pressure cancels it.
    assert np.max(abs(stream_ns_velocity(p.fluid.spatial, da))) < 2e-5
    residual = m.b @ (db + 2 * jnp.pi**2 * p.resistivity * b) @ m.b.T
    assert np.max(abs(residual)) < 2e-7


def test_magnetic_release_and_time_refinement(plan):
    p, f = plan, plan.fluid.spatial
    initial_b = island_pair(p, peak_field=2.0)
    initial_a = jnp.zeros_like(f.denominator)
    energy0 = float(mhd_budget(p, initial_a, initial_b, 0.0)[:2].sum())
    solutions = []
    for steps in (5, 10, 20):
        dt = 0.01 / steps
        stepper = plan_cavity_stepper(p.fluid, dt)

        @jax.jit
        def run():
            return jax.lax.fori_loop(
                0,
                steps,
                lambda k, state: mhd_step(p, stepper, *state, k * dt),
                (initial_a, initial_b),
            )

        a, b = run()
        budget = np.asarray(mhd_budget(p, a, b, 0.01))
        assert budget[0] > 1e-5  # motion was generated from magnetic energy
        assert budget[:2].sum() < energy0
        solutions.append((a, b))

    def difference(s1, s2):
        return float(
            jnp.sqrt(
                jnp.sum(f.denominator * (s1[0] - s2[0]) ** 2)
                + jnp.sum(p.magnetic_laplacian * (s1[1] - s2[1]) ** 2)
            )
        )

    assert (
        difference(solutions[0], solutions[1]) / difference(solutions[1], solutions[2])
        > 3.3
    )
