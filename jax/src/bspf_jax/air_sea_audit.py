"""Process-resolved budgets and guarded MRI stages for the 2-D research model.

All extensive diagnostics are per unit horizontal area. Constraint reaction
means the difference between physical forcing and its admissible velocity
projection; it is not an independently reconstructed wall pressure field.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .air_sea import (
    AirSeaState,
    scalar_values,
    scalar_mean,
    scalar_project,
    scalar_transport,
    bulk_flux,
)
from .stream_navier_stokes import (
    stream_ns_velocity,
    stream_ns_vorticity,
    stream_ns_load,
)
from .surface_exchange import coare35
from .multirate import mri_gark_erk45a_step

PROCESSES = (
    "ocean_advection",
    "ocean_rotation",
    "ocean_viscosity",
    "bottom_drag",
    "sst_advection",
    "sst_diffusion",
    "radiation",
    "air_advection",
    "air_rotation",
    "air_viscosity",
    "wind_restore",
    "temperature_advection",
    "temperature_diffusion",
    "temperature_restore",
    "humidity_advection",
    "humidity_diffusion",
    "humidity_restore",
    "interface",
)
QUANTITIES = (
    "ocean_heat",
    "air_heat",
    "air_latent",
    "air_water",
    "ocean_water",
    "ocean_momentum_x",
    "ocean_momentum_y",
    "air_momentum_x",
    "air_momentum_y",
    "ocean_constraint_x",
    "ocean_constraint_y",
    "air_constraint_x",
    "air_constraint_y",
    "ocean_kinetic",
    "air_kinetic",
    "sst_variance",
    "temperature_variance",
    "humidity_variance",
    "absolute_heat",
    "absolute_water",
    "absolute_work",
    "interface_dissipation",
)


def zero_ledger():
    return jnp.zeros((len(PROCESSES), len(QUANTITIES)), dtype=jnp.float64)


def kinetic_energy(p, s):
    c = p.config
    return jnp.array(
        [
            0.5 * c.ocean_mass * jnp.sum(p.flow.denominator * s.ocean**2),
            0.5
            * c.air_mass
            * (jnp.sum(p.air_flow.denominator * s.atmosphere**2) + s.air_mean_wind**2),
        ]
    )


def mean_velocity(flow, coeff):
    bx, gx = flow.x.weights @ flow.x.b, flow.x.weights @ flow.x.g
    by, gy = flow.y.weights @ flow.y.b, flow.y.weights @ flow.y.g
    return jnp.array([bx @ coeff @ gy, -gx @ coeff @ by])


def viscous_mean(flow, coeff, viscosity, length):
    """Integral of nu Laplacian(u) from endpoint Hessian traces (Gauss theorem)."""
    x, y = flow.x, flow.y
    bx, gx = x.weights @ x.b, x.weights @ x.g
    by, gy = y.weights @ y.b, y.weights @ y.g
    dxg, dxh = x.gn[-1] - x.gn[0], x.hn[-1] - x.hn[0]
    dyg, dyh = y.gn[-1] - y.gn[0], y.hn[-1] - y.hn[0]
    return (
        viscosity
        / length**2
        * jnp.array(
            [
                dxg @ coeff @ gy + bx @ coeff @ dyh,
                -dxh @ coeff @ by - gx @ coeff @ dyg,
            ]
        )
    )


def _flow_terms(p, coeff, velocity, air):
    c, f = p.config, p.air_flow if air else p.flow
    omega = stream_ns_vorticity(f, coeff) / c.length
    viscosity = c.air_viscosity if air else c.ocean_viscosity
    adv_force = jnp.stack((velocity[..., 1] * omega, -velocity[..., 0] * omega), -1)
    rot_force = jnp.stack(
        (velocity[..., 1] * p.coriolis, -velocity[..., 0] * p.coriolis), -1
    )
    adv = stream_ns_load(f, adv_force) / f.denominator
    rot = stream_ns_load(f, rot_force) / f.denominator
    visc = (
        -viscosity
        / c.length**2
        * (
            f.x.bending @ coeff
            + coeff @ f.y.bending.T
            + 2 * f.x.lam[:, None] * coeff * f.y.lam[None, :]
        )
        / f.denominator
    )

    def mean(force):
        return jnp.sum(p.weight[..., None] * force, axis=(0, 1))

    return (
        (adv, mean(adv_force)),
        (rot, mean(rot_force)),
        (visc, viscous_mean(f, coeff, viscosity, c.length)),
    )


def _row(
    p, s, ds, raw_o, raw_a, water_o=0.0, absolute_heat=0.0, absolute_water=0.0, diss=0.0
):
    c = p.config
    ho = c.ocean_capacity * scalar_mean(p, ds.sst)
    ha = c.air_capacity * scalar_mean(p, ds.air_temperature, air=True)
    wa = c.air_mass * scalar_mean(p, ds.humidity, air=True)
    mo = c.ocean_mass * mean_velocity(p.flow, ds.ocean)
    ma = c.air_mass * (
        mean_velocity(p.air_flow, ds.atmosphere) + jnp.array([ds.air_mean_wind, 0.0])
    )
    ko = c.ocean_mass * jnp.sum(p.flow.denominator * s.ocean * ds.ocean)
    ka = c.air_mass * (
        jnp.sum(p.air_flow.denominator * s.atmosphere * ds.atmosphere)
        + s.air_mean_wind * ds.air_mean_wind
    )

    def variance(a, da, air=False):
        return jnp.sum(a * da) - scalar_mean(p, a, air=air) * scalar_mean(
            p, da, air=air
        )

    return jnp.concatenate(
        (
            jnp.array([ho, ha, c.latent_heat * wa, wa, water_o]),
            mo,
            ma,
            mo - raw_o,
            ma - raw_a,
            jnp.array(
                [
                    ko,
                    ka,
                    variance(s.sst, ds.sst),
                    variance(s.air_temperature, ds.air_temperature, True),
                    variance(s.humidity, ds.humidity, True),
                    absolute_heat,
                    absolute_water,
                    abs(ko) + abs(ka),
                    diss,
                ]
            ),
        )
    )


def process_rhs(p, s, *, fast):
    """Return exactly the model's split tendency and its process budget rates."""
    c = p.config
    zero = jax.tree.map(jnp.zeros_like, s)
    total, ledger = zero, zero_ledger()
    z = jnp.zeros(2)

    def add(name, ds, raw_o=z, raw_a=z, **kw):
        nonlocal total, ledger
        total = jax.tree.map(lambda a, b: a + b, total, ds)
        ledger = ledger.at[PROCESSES.index(name)].set(
            _row(p, s, ds, raw_o, raw_a, **kw)
        )

    if not fast:
        u = stream_ns_velocity(p.flow, s.ocean)
        for name, (term, raw) in zip(PROCESSES[:3], _flow_terms(p, s.ocean, u, False)):
            add(name, zero._replace(ocean=term), raw_o=c.ocean_mass * raw)
        add(
            "bottom_drag",
            zero._replace(ocean=-c.bottom_drag * s.ocean),
            raw_o=-c.ocean_mass * c.bottom_drag * mean_velocity(p.flow, s.ocean),
        )
        adv = scalar_transport(p, s.sst, u, 0.0)
        diff = (
            -c.ocean_diffusivity
            / c.length**2
            * (p.scalar.lam[:, None] + p.scalar.lam[None, :])
            * s.sst
        )
        add("sst_advection", zero._replace(sst=adv))
        add("sst_diffusion", zero._replace(sst=diff))
        add(
            "radiation",
            zero._replace(sst=c.radiation / c.ocean_capacity * p.unit_scalar),
            absolute_heat=abs(c.radiation),
        )
        return total, ledger

    uo = stream_ns_velocity(p.flow, s.ocean)
    ua = stream_ns_velocity(p.air_flow, s.atmosphere) + jnp.array(
        [s.air_mean_wind, 0.0]
    )
    for name, (term, raw) in zip(
        PROCESSES[7:10], _flow_terms(p, s.atmosphere, ua, True)
    ):
        add(name, zero._replace(atmosphere=term), raw_a=c.air_mass * raw)
    wind = zero._replace(
        atmosphere=c.wind_restore_rate * (p.wind_target - s.atmosphere),
        air_mean_wind=-c.wind_restore_rate * s.air_mean_wind,
    )
    add(
        "wind_restore",
        wind,
        raw_a=c.air_mass
        * (
            mean_velocity(p.air_flow, wind.atmosphere)
            + jnp.array([wind.air_mean_wind, 0.0])
        ),
    )
    for name, field, target, rate in (
        (
            "temperature",
            "air_temperature",
            p.temperature_target,
            c.temperature_restore_rate,
        ),
        ("humidity", "humidity", p.humidity_target, c.humidity_restore_rate),
    ):
        v = getattr(s, field)
        adv = scalar_transport(p, v, ua, 0.0, air=True)
        diff = (
            -c.air_diffusivity
            / c.length**2
            * (p.air_flow.x.lam[:, None] + p.air_scalar_y.lam[None, :])
            * v
        )
        restore = rate * (target - v)
        add(name + "_advection", zero._replace(**{field: adv}))
        add(name + "_diffusion", zero._replace(**{field: diff}))
        absolute = jnp.sum(p.weight * abs(scalar_values(p, restore, air=True)))
        add(
            name + "_restore",
            zero._replace(**{field: restore}),
            absolute_heat=absolute
            * (c.air_capacity if name == "temperature" else c.air_mass * c.latent_heat),
            absolute_water=absolute * c.air_mass if name == "humidity" else 0.0,
        )
    ts = scalar_values(p, s.sst) + c.reference_temperature
    ta = scalar_values(p, s.air_temperature, air=True) + c.reference_temperature
    q = scalar_values(p, s.humidity, air=True)
    tau, h, e = bulk_flux(c, uo, ua, ts, ta, q, surface=p.surface)
    mean_stress = jnp.sum(p.weight[..., None] * tau, axis=(0, 1))
    exchange = zero._replace(
        ocean=stream_ns_load(p.flow, tau) / (c.ocean_mass * p.flow.denominator),
        atmosphere=-stream_ns_load(p.air_flow, tau)
        / (c.air_mass * p.air_flow.denominator),
        air_mean_wind=-mean_stress[0] / c.air_mass,
        sst=-scalar_project(p, h + c.latent_heat * e) / c.ocean_capacity,
        air_temperature=scalar_project(p, h, air=True) / c.air_capacity,
        humidity=scalar_project(p, e, air=True) / c.air_mass,
    )
    add(
        "interface",
        exchange,
        raw_o=mean_stress,
        raw_a=-mean_stress,
        water_o=-jnp.sum(p.weight * e),
        absolute_heat=jnp.sum(p.weight * (abs(h) + c.latent_heat * abs(e))),
        absolute_water=jnp.sum(p.weight * abs(e)),
        diss=jnp.sum(p.weight * jnp.sum(tau * (ua - uo), axis=-1)),
    )
    return total, ledger


class StageObservation(NamedTuple):
    code: jax.Array
    time: jax.Array
    field_index: jax.Array
    flat_index: jax.Array
    state: AirSeaState
    event: jax.Array


def observe_stage(p, t, augmented, previous):
    s = augmented[0]
    # Both physical quadrature and nodal values: neither replaces the other.
    qs = (
        scalar_values(p, s.humidity, air=True),
        scalar_values(p, s.humidity, air=True, nodes=True),
    )
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in s]))
    negative = jnp.any(qs[0] < 0) | jnp.any(qs[1] < 0)
    invalid_q = jnp.any(qs[0] >= 1) | jnp.any(qs[1] >= 1)
    ta = scalar_values(p, s.air_temperature, air=True) + p.config.reference_temperature
    ts = scalar_values(p, s.sst) + p.config.reference_temperature
    temperatures = (
        ta,
        ts,
        scalar_values(p, s.air_temperature, air=True, nodes=True)
        + p.config.reference_temperature,
        scalar_values(p, s.sst, nodes=True) + p.config.reference_temperature,
    )
    finite = finite & jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(v)) for v in (*qs, *temperatures)])
    )
    invalid_temp = jnp.any(jnp.stack([jnp.any(v <= 0) for v in temperatures]))
    valid_exchange = jnp.asarray(True)
    if p.surface.method == "coare35":
        ua = stream_ns_velocity(p.air_flow, s.atmosphere) + jnp.array(
            [s.air_mean_wind, 0.0]
        )
        uo = stream_ns_velocity(p.flow, s.ocean)
        flux = coare35(
            ua - uo,
            ta,
            qs[0],
            ts,
            pressure=p.config.pressure,
            boundary_layer_height=p.config.atmosphere_depth,
            surface=p.surface,
            rho_air=p.config.rho_air,
            cp_air=p.config.cp_air,
        )
        valid_exchange = jnp.all(flux.valid)
    code = jnp.where(
        ~finite,
        1,
        jnp.where(
            negative,
            2,
            jnp.where(
                invalid_q,
                3,
                jnp.where(invalid_temp, 4, jnp.where(~valid_exchange, 5, 0)),
            ),
        ),
    )
    # Indexed evidence catalog: coefficient arrays, q quadrature/nodes,
    # Ta/Ts quadrature then nodes, followed by closure validity.
    evidence = (*s, *qs, *temperatures)
    masks = [~jnp.isfinite(v) for v in evidence]
    offset = len(s)
    for k, v in enumerate(qs):
        masks[offset + k] |= jnp.where(code == 2, v < 0, (code == 3) & (v >= 1))
    for k, v in enumerate(temperatures):
        masks[offset + 2 + k] |= (code == 4) & (v <= 0)
    masks.append(~flux.valid if p.surface.method == "coare35" else jnp.array([False]))
    present = jnp.stack([jnp.any(v) for v in masks])
    field_index = jnp.argmax(present)
    flat_index = jnp.stack([jnp.argmax(v.reshape(-1)) for v in masks])[field_index]
    return jax.lax.cond(
        previous.code != 0,
        lambda: previous,
        lambda: StageObservation(
            code, t, field_index, flat_index, s, previous.event + 1
        ),
    )


def audited_step(p, stepper, s, ledger, time):
    """First failing physical stage retained exactly, with global stage time."""
    if stepper.method != "mri-gark4":
        raise ValueError(
            "Audited production stepping requires mri-gark4; legacy remains in air_sea_step"
        )
    observation = StageObservation(
        jnp.asarray(0),
        jnp.asarray(time),
        jnp.asarray(0),
        jnp.asarray(0),
        jax.tree.map(jnp.zeros_like, s),
        jnp.asarray(0),
    )

    def macro(k, carry):
        value, obs = carry
        return mri_gark_erk45a_step(
            value,
            time + k * stepper.dt_ocean,
            stepper.dt_ocean,
            lambda t, y: process_rhs(p, y[0], fast=True),
            lambda t, y: process_rhs(p, y[0], fast=False),
            inner_steps=stepper.inner_steps,
            stage_observer=lambda t, y, obs: observe_stage(p, t, y, obs),
            observation=obs,
        )

    (s, ledger), observation = jax.lax.fori_loop(
        0, stepper.ocean_steps, macro, ((s, ledger), observation)
    )
    return s, ledger, observation


def scalar_variances(p, s):
    return jnp.array(
        [
            0.5 * (jnp.sum(v * v) - scalar_mean(p, v, air=air) ** 2)
            for v, air in (
                (s.sst, False),
                (s.air_temperature, True),
                (s.humidity, True),
            )
        ]
    )


def spectral_tails(p, s):
    """Energy fraction in top 20% of each 1D stiffness spectrum (union)."""
    values = []
    for coeff, x, y, mass in (
        (s.ocean, p.flow.x.lam, p.flow.y.lam, p.flow.denominator),
        (s.atmosphere, p.air_flow.x.lam, p.air_flow.y.lam, p.air_flow.denominator),
        (s.sst, p.scalar.lam, p.scalar.lam, 1.0),
        (s.air_temperature, p.air_flow.x.lam, p.air_scalar_y.lam, 1.0),
        (s.humidity, p.air_flow.x.lam, p.air_scalar_y.lam, 1.0),
    ):
        tail = (x[:, None] >= jnp.sort(x)[int(0.8 * x.size)]) | (
            y[None, :] >= jnp.sort(y)[int(0.8 * y.size)]
        )
        energy = mass * coeff**2
        # Exclude scalar constant mode via derivative weighting, so mean T/q
        # cannot hide unresolved structure. Velocity mass already does this.
        if isinstance(mass, float):
            energy = energy * (x[:, None] + y[None, :])
        values.append(
            jnp.sum(jnp.where(tail, energy, 0)) / jnp.maximum(jnp.sum(energy), 1e-30)
        )
    return jnp.array(values)
