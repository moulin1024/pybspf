"""Physical budgets and independent operator checks for the coupled example."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bspf_models.air_sea.air_sea import AirSeaConfig
from bspf_models.air_sea.air_sea import air_sea_fields
from bspf_models.air_sea.air_sea import air_sea_fast_rhs
from bspf_models.air_sea.air_sea import air_sea_slow_rhs
from bspf_models.air_sea.air_sea import air_sea_step
from bspf_models.air_sea.air_sea import exchange_loads
from bspf_models.air_sea.air_sea import flow_rhs
from bspf_models.air_sea.air_sea import heat_content
from bspf_models.air_sea.air_sea import initial_air_sea_state
from bspf_models.air_sea.air_sea import _rk4_with_budget
from bspf_models.air_sea.air_sea import bulk_flux
from bspf_models.air_sea.air_sea import plan_air_sea
from bspf_models.air_sea.air_sea import plan_air_sea_stepper
from bspf_models.air_sea.air_sea import scalar_mean
from bspf_models.air_sea.air_sea import scalar_project
from bspf_models.air_sea.air_sea import scalar_transport
from bspf_models.air_sea.air_sea import scalar_values
from bspf_models.air_sea.air_sea import saturation_specific_humidity
from bspf_models.fluids.stream_navier_stokes import stream_ns_velocity

jax.config.update("jax_enable_x64", True)


def test_bulk_exchange_signs_galilean_invariance_and_drag_work():
    c = AirSeaConfig()
    ocean = jnp.array([[1.0, 2.0], [-2.0, 1.0]])
    air = jnp.array([[8.0, -1.0], [4.0, 5.0]])
    ts = jnp.array([294.0, 290.0])
    ta = ts - 2
    q = 0.6 * saturation_specific_humidity(ta)
    stress, heat, water = bulk_flux(c, ocean, air, ts, ta, q)
    assert np.all(heat > 0) and np.all(water > 0)
    rel = np.asarray(air - ocean)
    np.testing.assert_allclose(
        stress,
        c.rho_air * c.drag_coefficient * np.linalg.norm(rel, axis=-1)[:, None] * rel,
    )
    shifted = bulk_flux(c, ocean + 17, air + 17, ts, ta, q)
    for a, b in zip((stress, heat, water), shifted):
        np.testing.assert_allclose(a, b, atol=1e-14)
    # Equal/opposite momentum exchange dissipates relative kinetic energy.
    assert np.all(np.sum(stress * (ocean - air), axis=-1) < 0)
    equilibrium = bulk_flux(c, ocean, ocean, ts, ta, q)
    for flux in equilibrium:
        np.testing.assert_array_equal(flux, jnp.zeros_like(flux))


def test_condensation_reverses_water_and_latent_exchange():
    c = AirSeaConfig()
    qs = 0.98 * saturation_specific_humidity(jnp.array(290.0))
    _, sensible, water = bulk_flux(
        c,
        jnp.zeros(2),
        jnp.array([8.0, 0.0]),
        jnp.array(290.0),
        jnp.array(292.0),
        qs * 1.1,
    )
    assert sensible < 0 and water < 0


def test_rk_stage_source_accounting_with_nonlinear_exchange():
    # Independent zero-dimensional heat/water exchange, no spatial machinery.
    # Atmospheric enthalpy receives sensible heat + latent energy in vapor;
    # ocean loses exactly this amount, not latent heat counted twice.
    c = replace(AirSeaConfig(), radiation=0)
    initial = jnp.array([294.0, 292.0, 0.008])

    def rhs(s):
        _, h, e = bulk_flux(c, jnp.zeros(2), jnp.array([8.0, 0.0]), s[0], s[1], s[2])
        return jnp.array(
            [
                -(h + c.latent_heat * e) / c.ocean_capacity,
                h / c.air_capacity,
                e / c.air_mass,
            ]
        ), jnp.array([h, e])

    final, integrated = _rk4_with_budget(initial, 600.0, rhs)
    capacity = jnp.array([c.ocean_capacity, c.air_capacity, c.latent_heat * c.air_mass])
    assert abs(float(capacity @ (final - initial))) < 1e-5
    np.testing.assert_allclose(
        c.air_mass * (final[2] - initial[2]), integrated[1], atol=1e-14
    )
    np.testing.assert_allclose(
        c.air_capacity * (final[1] - initial[1]), integrated[0], atol=1e-7
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt_air": 0},
        {"window": np.nan},
        {"dt_ocean": -1},
        {"dt_air": 17},
        {"dt_ocean": 400},
    ],
)
def test_invalid_time_hierarchies(kwargs):
    with pytest.raises(ValueError):
        plan_air_sea_stepper(method="lagged", **kwargs)


def test_integer_subcycling():
    s = plan_air_sea_stepper(dt_air=10, dt_ocean=100, window=300)
    assert s.air_steps == 30 and s.ocean_steps == 3


@pytest.fixture(scope="module")
def plan():
    pytest.importorskip("gmpy2")
    return plan_air_sea(n=33)


def test_initial_wind_and_boundaries(plan):
    s = initial_air_sea_state(plan)
    f = air_sea_fields(plan, s)
    y = np.asarray(plan.scalar.x)
    expected = np.broadcast_to(
        -plan.config.wind_speed * np.cos(2 * np.pi * y), (33, 33)
    )
    np.testing.assert_allclose(f["air_velocity"][..., 0], expected, atol=2e-12)
    np.testing.assert_allclose(f["air_velocity"][..., 1], 0, atol=2e-12)
    # Test boundary subspaces with unrelated random coefficients, not only seed.
    rng = np.random.default_rng(2)
    ap = plan.air_flow
    a = jnp.asarray(rng.normal(size=ap.denominator.shape) * 1e-4)
    velocity = np.asarray(stream_ns_velocity(ap, a, nodes=True))
    np.testing.assert_allclose(velocity[0], velocity[-1], atol=2e-12)
    np.testing.assert_allclose(velocity[:, [0, -1], 1], 0, atol=2e-12)
    du_dy = ap.x.bn @ a @ ap.y.hn.T
    np.testing.assert_allclose(du_dy[:, [0, -1]], 0, atol=1e-10)
    coeff = jnp.asarray(rng.normal(size=s.humidity.shape))
    values = scalar_values(plan, coeff, nodes=True, air=True)
    np.testing.assert_allclose(values[0], values[-1], atol=2e-12)
    qy = ap.x.bn @ coeff @ plan.air_scalar_y.gn.T
    np.testing.assert_allclose(qy[:, [0, -1]], 0, atol=5e-10)
    op = plan.flow
    ocean = jnp.asarray(rng.normal(size=op.denominator.shape) * 1e-4)
    ov = np.asarray(stream_ns_velocity(op, ocean, nodes=True))
    assert max(abs(ov[[0, -1]]).max(), abs(ov[:, [0, -1]]).max()) < 1e-10


def test_analytic_periodic_scalar_advection_diffusion(plan):
    # Analytic instantaneous PDE for cos(2*pi*x)*cos(pi*y), constant zonal wind.
    x, y = plan.scalar.points[:, None], plan.scalar.points[None, :]
    value = jnp.cos(2 * jnp.pi * x) * jnp.cos(jnp.pi * y)
    coeff = scalar_project(plan, value, air=True)
    velocity = jnp.broadcast_to(jnp.array([7.0, 0.0]), value.shape + (2,))
    kappa, length = 15000.0, plan.config.length
    actual = scalar_values(
        plan, scalar_transport(plan, coeff, velocity, kappa, air=True), air=True
    )
    exact = 7 / length * 2 * jnp.pi * jnp.sin(2 * jnp.pi * x) * jnp.cos(jnp.pi * y)
    exact -= kappa / length**2 * 5 * jnp.pi**2 * value
    np.testing.assert_allclose(actual, exact, atol=2e-16, rtol=1e-8)


def test_scalar_mass_constant_preservation_and_variance_dissipation(plan):
    s = initial_air_sea_state(plan)
    for air, flow, a, diffusivity in (
        (True, plan.air_flow, s.atmosphere, 20000.0),
        (
            False,
            plan.flow,
            jnp.sin(jnp.arange(s.ocean.size).reshape(s.ocean.shape)) * 1e-4,
            1000.0,
        ),
    ):
        velocity = stream_ns_velocity(flow, a)
        unit = scalar_project(plan, jnp.ones_like(plan.weight), air=air)
        constant_rhs = scalar_transport(plan, unit, velocity, diffusivity, air=air)
        assert np.max(abs(constant_rhs)) < 2e-13
        coeff = scalar_project(
            plan,
            jnp.cos(3 * plan.scalar.points[:, None])
            * jnp.cos(2 * plan.scalar.points[None, :]),
            air=air,
        )
        rhs = scalar_transport(plan, coeff, velocity, diffusivity, air=air)
        assert abs(float(scalar_mean(plan, rhs, air=air))) < 2e-14
        assert float(jnp.sum(coeff * rhs)) < 0


def test_rotating_inviscid_flow_does_no_work(plan):
    rng = np.random.default_rng(18)
    for air, p in ((False, plan.flow), (True, plan.air_flow)):
        a = jnp.asarray(rng.normal(size=p.denominator.shape) * 1e-5)
        rhs = flow_rhs(plan, a, viscosity=0.0, air=air)
        assert abs(float(jnp.sum(p.denominator * a * rhs))) < 2e-15


def test_dual_space_flux_projection_and_two_way_sensitivity(plan):
    s = initial_air_sea_state(plan)
    f = air_sea_fields(plan, s, nodes=False)
    args = [
        f[k]
        for k in (
            "ocean_velocity",
            "sst",
            "air_velocity",
            "air_temperature",
            "humidity",
        )
    ]
    flux = exchange_loads(plan, *args)
    for sea, air in ((flux.sensible, flux.air_sensible), (flux.water, flux.air_water)):
        np.testing.assert_allclose(
            scalar_mean(plan, sea),
            scalar_mean(plan, air, air=True),
            rtol=2e-12,
            atol=1e-13,
        )
    # Warmer SST increases heat and moisture transfer to air.
    warm = list(args)
    warm[1] += 1.0
    warmer = exchange_loads(plan, *warm)
    assert scalar_mean(plan, warmer.air_sensible - flux.air_sensible, air=True) > 0
    assert scalar_mean(plan, warmer.air_water - flux.air_water, air=True) > 0
    # A current aligned with the wind reduces stress everywhere.
    moving = list(args)
    moving[0] = 0.2 * args[2]
    weaker = exchange_loads(plan, *moving)
    np.testing.assert_allclose(
        weaker.stress, 0.64 * flux.stress, rtol=2e-11, atol=1e-13
    )


@pytest.mark.parametrize("method", ["mri-gark4", "lagged"])
def test_multirate_heat_water_and_atmospheric_momentum_budgets(plan, method):
    stepper = plan_air_sea_stepper(dt_air=30, dt_ocean=150, window=300, method=method)
    initial = initial_air_sea_state(plan)
    initial = initial._replace(air_mean_wind=jnp.asarray(0.7))
    advance = jax.jit(lambda s: air_sea_step(plan, stepper, s))
    current = initial
    heat, water, stress, momentum = 0.0, 0.0, 0.0, 0.0
    for _ in range(3):
        current, budget = advance(current)
        heat += budget.external_heat
        water += scalar_mean(plan, budget.exchange.water) + budget.external_water
        stress += budget.exchange.stress_mean[0]
        momentum += budget.external_air_zonal_momentum
    assert (
        abs(float(heat_content(plan, current) - heat_content(plan, initial) - heat))
        < 2e-5
    )
    assert (
        abs(
            float(
                plan.config.air_mass
                * scalar_mean(plan, current.humidity - initial.humidity, air=True)
                - water
            )
        )
        < 2e-11
    )
    assert (
        abs(
            float(
                plan.config.air_mass * (current.air_mean_wind - initial.air_mean_wind)
                + stress
                - momentum
            )
        )
        < 2e-10
    )
    assert np.max(abs(current.ocean)) > 0
    assert np.max(abs(current.atmosphere - initial.atmosphere)) > 0
    assert np.max(abs(current.sst - initial.sst)) > 0
    assert np.max(abs(current.humidity - initial.humidity)) > 0


def test_closed_thermodynamic_budget_without_external_sources(plan):
    config = replace(
        plan.config,
        radiation=0.0,
        temperature_restore_rate=0.0,
        humidity_restore_rate=0.0,
    )
    p = replace(plan, config=config)
    initial = initial_air_sea_state(p)
    stepper = plan_air_sea_stepper()
    final, budget = jax.jit(lambda s: air_sea_step(p, stepper, s))(initial)
    assert abs(float(heat_content(p, final) - heat_content(p, initial))) < 2e-5
    assert budget.external_heat == 0 and budget.external_water == 0


def test_mri_individual_splits_obey_linear_budget_laws(plan):
    state = initial_air_sea_state(plan)._replace(air_mean_wind=jnp.asarray(0.6))
    for rhs in (air_sea_fast_rhs, air_sea_slow_rhs):
        rate, budget = rhs(plan, state)
        assert abs(float(heat_content(plan, rate) - budget.external_heat)) < 1e-8
        assert (
            abs(
                float(
                    plan.config.air_mass * scalar_mean(plan, rate.humidity, air=True)
                    - scalar_mean(plan, budget.exchange.water)
                    - budget.external_water
                )
            )
            < 1e-12
        )
        assert (
            abs(
                float(
                    plan.config.air_mass * rate.air_mean_wind
                    + budget.exchange.stress_mean[0]
                    - budget.external_air_zonal_momentum
                )
            )
            < 1e-12
        )


def test_mri_microstep_alignment_and_bad_method():
    stepper = plan_air_sea_stepper(dt_air=17, dt_ocean=150, window=300)
    assert stepper.method == "mri-gark4"
    assert stepper.inner_steps == 2 and stepper.effective_dt_air == 15
    assert stepper.air_steps == 20
    with pytest.raises(ValueError):
        plan_air_sea_stepper(method="RK4-independent")


def test_mri_synchronization_window_is_only_grouping(plan):
    short = plan_air_sea_stepper(dt_air=30, dt_ocean=150, window=300)
    long = plan_air_sea_stepper(dt_air=30, dt_ocean=150, window=600)
    initial = initial_air_sea_state(plan)
    step = jax.jit(lambda y: air_sea_step(plan, short, y))
    intermediate, b1 = step(initial)
    twice, b2 = step(intermediate)
    once, total = jax.jit(lambda y: air_sea_step(plan, long, y))(initial)
    for a, b in zip(twice, once):
        np.testing.assert_allclose(a, b, atol=2e-13, rtol=2e-13)
    for a, b in zip(
        jax.tree.leaves(total), jax.tree.leaves(jax.tree.map(jnp.add, b1, b2))
    ):
        np.testing.assert_allclose(a, b, atol=2e-9, rtol=2e-12)


def test_seeded_multijet_initialization(plan):
    p = plan_air_sea(
        n=33,
        quadrature_order=20,
        config=replace(plan.config, wind_jet_count=2, initial_air_perturbation=1.0),
    )
    s = initial_air_sea_state(p)
    velocity = np.asarray(stream_ns_velocity(p.air_flow, s.atmosphere, nodes=True))
    np.testing.assert_allclose(velocity[0], velocity[-1], atol=1e-12)
    np.testing.assert_allclose(velocity[:, (0, -1), 1], 0.0, atol=1e-12)
    assert np.max(abs(velocity[..., 1])) > 0.9
    target = np.asarray(stream_ns_velocity(p.air_flow, p.wind_target, nodes=True))
    np.testing.assert_allclose(target[..., 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(
        target[0, :, 0],
        -p.config.wind_speed * np.cos(4 * np.pi * np.asarray(p.scalar.x)),
        atol=1e-11,
    )
