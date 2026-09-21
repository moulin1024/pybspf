import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bspf_jax.air_sea import (
    plan_air_sea,
    plan_air_sea_stepper,
    initial_air_sea_state,
    air_sea_fast_rhs,
    air_sea_slow_rhs,
    air_sea_step,
)
from bspf_jax.air_sea_audit import (
    PROCESSES,
    QUANTITIES,
    process_rhs,
    zero_ledger,
    kinetic_energy,
    audited_step,
)
from bspf_jax.surface_exchange import SurfaceExchangeConfig

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def plan():
    return plan_air_sea(n=33, quadrature_order=20)


@pytest.mark.parametrize("fast", [False, True])
def test_process_decomposition_and_budgets(plan, fast):
    s = initial_air_sea_state(plan)
    s = s._replace(
        ocean=0.001 * plan.flow.denominator**-1, air_mean_wind=jnp.array(0.2)
    )
    ds, ledger = process_rhs(plan, s, fast=fast)
    expected, _ = (air_sea_fast_rhs if fast else air_sea_slow_rhs)(plan, s)
    for a, b in zip(ds, expected):
        np.testing.assert_allclose(a, b, rtol=3e-12, atol=1e-14)
    for name in ("ocean_advection", "ocean_rotation", "air_advection", "air_rotation"):
        row = np.asarray(ledger[PROCESSES.index(name)])
        assert abs(row[QUANTITIES.index("ocean_kinetic")]) < 1e-9
        assert abs(row[QUANTITIES.index("air_kinetic")]) < 1e-9
    row = np.asarray(ledger[PROCESSES.index("interface")])
    assert abs(sum(row[:3])) < 1e-10
    assert abs(sum(row[3:5])) < 1e-14
    assert abs(row[13] + row[14] + row[21]) < 1e-10
    for name in ("ocean_viscosity", "air_viscosity", "bottom_drag"):
        assert np.sum(ledger[PROCESSES.index(name), 13:15]) <= 1e-12


def test_audited_step_reproduces_original_and_kinetic_balance(plan):
    stepper = plan_air_sea_stepper(dt_air=30, dt_ocean=150, window=300)
    s = initial_air_sea_state(plan)
    expected, _ = jax.jit(lambda s: air_sea_step(plan, stepper, s))(s)
    actual, ledger, observation = jax.jit(
        lambda s: audited_step(plan, stepper, s, zero_ledger(), 1200.0)
    )(s)
    assert observation.code == 0
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=2e-11, atol=2e-12)
    defect = (
        kinetic_energy(plan, actual)
        - kinetic_energy(plan, s)
        - jnp.sum(ledger[:, 13:15], axis=0)
    )
    assert np.max(abs(defect)) < 1e-6


def test_failure_preserves_first_stage_state_and_global_time(plan):
    s = initial_air_sea_state(plan)._replace(
        humidity=-initial_air_sea_state(plan).humidity
    )
    stepper = plan_air_sea_stepper(dt_air=1, dt_ocean=5, window=5)
    _, _, failure = jax.jit(
        lambda s: audited_step(plan, stepper, s, zero_ledger(), 7200.0)
    )(s)
    assert failure.code == 2
    assert failure.time == 7200.0
    for a, b in zip(s, failure.state):
        np.testing.assert_array_equal(a, b)


def test_coare_coupled_interface_is_conservative(plan):
    from dataclasses import replace

    p = replace(plan, surface=SurfaceExchangeConfig(method="coare35"))
    s = initial_air_sea_state(p)
    _, ledger = jax.jit(lambda s: process_rhs(p, s, fast=True))(s)
    row = np.asarray(ledger[PROCESSES.index("interface")])
    assert abs(sum(row[:3])) < 1e-10
    assert abs(sum(row[3:5])) < 1e-14
    assert abs(row[13] + row[14] + row[21]) < 1e-10


def test_stage_failure_location_catalog(plan):
    from bspf_jax.air_sea_audit import StageObservation, observe_stage

    s = initial_air_sea_state(plan)
    empty = StageObservation(
        jnp.array(0), jnp.array(0.0), jnp.array(0), jnp.array(0), s, jnp.array(0)
    )
    bad = s._replace(sst=s.sst - 500 * plan.unit_scalar)
    failure = observe_stage(plan, 123.0, (bad, zero_ledger()), empty)
    assert failure.code == 4
    assert failure.field_index == len(s) + 3  # SST quadrature
    assert failure.event == 1
    bad = s._replace(air_mean_wind=jnp.array(float("nan")))
    failure = observe_stage(plan, 123.0, (bad, zero_ledger()), empty)
    assert failure.code == 1
    assert failure.field_index == 5
    assert failure.flat_index == 0


def test_kinetic_discrete_defect_converges(plan):
    initial = initial_air_sea_state(plan)
    energy0 = kinetic_energy(plan, initial)
    defects = []
    for h in (2400.0, 1200.0, 600.0):
        stepper = plan_air_sea_stepper(dt_ocean=h, dt_air=h / 5, window=h)

        def integrate():
            def step(i, value):
                state, ledger, failure = audited_step(
                    plan, stepper, value[0], value[1], i * h
                )
                return state, ledger

            return jax.lax.fori_loop(0, round(9600 / h), step, (initial, zero_ledger()))

        state, ledger = jax.jit(integrate)()
        defects.append(
            np.abs(
                np.asarray(
                    kinetic_energy(plan, state)
                    - energy0
                    - jnp.sum(ledger[:, 13:15], axis=0)
                )
            )
        )
    ratios = np.asarray(defects[:-1]) / defects[1:]
    assert np.all(ratios > 10), (defects, ratios)


@pytest.mark.parametrize("field", ["sst", "air_temperature", "humidity"])
def test_thermodynamic_feedback_changes_stress_only_with_coare(plan, field):
    """Temperature/moisture must feed back into momentum through the closure."""
    from dataclasses import replace
    from bspf_jax.air_sea import scalar_project

    state = initial_air_sea_state(plan)
    if field == "humidity":
        modified = state._replace(humidity=state.humidity * 1.05)
    else:
        unit = scalar_project(plan, jnp.ones_like(plan.weight), air=field != "sst")
        modified = state._replace(**{field: getattr(state, field) + unit})
    for method in ("constant", "coare35"):
        p = replace(plan, surface=SurfaceExchangeConfig(method=method))
        evaluate = jax.jit(lambda s: process_rhs(p, s, fast=True))
        before, _ = evaluate(state)
        after, ledger = evaluate(modified)
        if method == "constant":
            np.testing.assert_array_equal(before.ocean, after.ocean)
        else:
            relative = np.linalg.norm(after.ocean - before.ocean) / np.linalg.norm(
                before.ocean
            )
            assert relative > 1e-5
        interface = np.asarray(ledger[PROCESSES.index("interface")])
        assert abs(np.sum(interface[:3])) < 1e-10
        assert abs(np.sum(interface[3:5])) < 1e-14
