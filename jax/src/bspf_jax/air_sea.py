"""Idealized 2D double-gyre / moist atmospheric mixed-layer coupling.

The ocean uses clamped, divergence-free BSPF streamfunctions on [0,1]^2.
The atmosphere uses a zonally periodic real Fourier Galerkin space and
meridional sine/cosine modes (impermeable free-slip walls), plus a harmonic
mean zonal velocity. Physical length
is supplied explicitly; velocities are m/s,
time is seconds, scalar temperatures are anomalies about reference_temperature.
Ocean scalars have natural zero diffusive flux. Atmospheric scalars are periodic
in x and have zero normal derivative at the north/south walls. Both use beta-plane
rotation. This is a horizontal, vertically averaged model, not an x-z section.

The default coupler uses fourth-order MRI-GARK-ERK45a with RK4 inner solves.
Slow terms are ocean interior dynamics; fast terms include atmospheric dynamics
AND both sides of interface exchange, evaluated on the same evolving stage.
The first-order frozen-ocean scheme remains available as method="lagged".
No clipping, condensation, salinity evolution,
free surface, baroclinic ocean feedback, or vertical turbulence is implemented.
"""

from dataclasses import dataclass, fields
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la

from .stream_navier_stokes import (
    StreamNavierStokes2DPlan,
    _stream_line,
    stream_ns_load,
    stream_ns_velocity,
    stream_ns_vorticity,
)
from .multirate import mri_gark_erk45a_step
from .surface_exchange import SurfaceExchangeConfig, coare35


@dataclass(frozen=True)
class AirSeaConfig:
    length: float = 1.0e6
    ocean_depth: float = 1000.0
    mixed_layer_depth: float = 50.0
    atmosphere_depth: float = 1000.0
    rho_ocean: float = 1025.0
    rho_air: float = 1.2
    cp_ocean: float = 3990.0
    cp_air: float = 1004.0
    latent_heat: float = 2.5e6
    reference_temperature: float = 290.0
    pressure: float = 101325.0
    f0: float = 1.0e-4
    beta: float = 2.0e-11
    ocean_viscosity: float = 2000.0
    air_viscosity: float = 20000.0
    ocean_diffusivity: float = 1000.0
    air_diffusivity: float = 20000.0
    bottom_drag: float = 1.0 / (100 * 86400)
    drag_coefficient: float = 1.3e-3
    heat_coefficient: float = 1.2e-3
    moisture_coefficient: float = 1.2e-3
    wind_speed: float = 8.0
    wind_jet_count: int = 1
    initial_air_perturbation: float = 0.0
    wind_restore_rate: float = 1.0 / 21600
    temperature_restore_rate: float = 1.0 / 86400
    humidity_restore_rate: float = 1.0 / 172800
    radiation: float = 100.0
    initial_air_temperature_offset: float = -2.0
    initial_relative_humidity: float = 0.75

    @property
    def ocean_mass(self):
        return self.rho_ocean * self.ocean_depth

    @property
    def air_mass(self):
        return self.rho_air * self.atmosphere_depth

    @property
    def ocean_capacity(self):
        return self.rho_ocean * self.cp_ocean * self.mixed_layer_depth

    @property
    def air_capacity(self):
        return self.air_mass * self.cp_air


class AirSeaState(NamedTuple):
    ocean: jax.Array
    atmosphere: jax.Array
    sst: jax.Array
    air_temperature: jax.Array
    humidity: jax.Array
    air_mean_wind: jax.Array


class AirState(NamedTuple):
    velocity: jax.Array
    temperature: jax.Array
    humidity: jax.Array
    mean_wind: jax.Array


class Exchange(NamedTuple):
    """Projected physical stress (Pa), sensible heat (W/m²), water (kg/m²/s).

    After time integration these become impulses. stress_mean is the actual
    quadrature mean stress, not a pressure-projected net momentum tendency.
    """

    stress: jax.Array
    sensible: jax.Array
    water: jax.Array
    stress_mean: jax.Array
    air_stress: jax.Array
    air_sensible: jax.Array
    air_water: jax.Array


class WindowBudget(NamedTuple):
    exchange: Exchange
    external_heat: jax.Array
    external_water: jax.Array
    external_air_zonal_momentum: jax.Array


@dataclass(frozen=True)
class AirSeaPlan:
    config: AirSeaConfig
    flow: StreamNavierStokes2DPlan
    scalar: object
    weight: jax.Array
    mean_mode: jax.Array
    coriolis: jax.Array
    wind_target: jax.Array
    temperature_target: jax.Array
    humidity_target: jax.Array
    unit_scalar: jax.Array
    air_flow: StreamNavierStokes2DPlan
    air_mean_mode: tuple
    initial_sst: jax.Array
    air_scalar_y: object
    surface: SurfaceExchangeConfig = SurfaceExchangeConfig()


class AirSeaStepper(NamedTuple):
    dt_air: float
    dt_ocean: float
    window: float
    air_steps: int
    ocean_steps: int
    method: str
    inner_steps: int

    @property
    def effective_dt_air(self):
        return (
            self.dt_ocean / (5 * self.inner_steps)
            if self.method == "mri-gark4"
            else self.dt_air
        )


def saturation_specific_humidity(temperature, pressure=101325.0):
    """Liquid-water Bolton-form saturation; T in K, pressure in Pa.

    Intended for ordinary marine surface temperatures, not ice or extremes.
    Sea salinity's vapor-pressure reduction is applied separately in bulk_flux.
    """
    tc = temperature - 273.15
    vapor = 611.2 * jnp.exp(17.67 * tc / (tc + 243.5))
    return 0.622 * vapor / (pressure - 0.378 * vapor)


def bulk_flux(
    config,
    ocean_velocity,
    air_velocity,
    sst,
    air_temperature,
    humidity,
    *,
    surface=None,
):
    """Stress toward ocean; sensible heat and evaporation toward atmosphere."""
    relative = air_velocity - ocean_velocity
    if surface is not None and surface.method == "coare35":
        result = coare35(
            relative,
            air_temperature,
            humidity,
            sst,
            pressure=config.pressure,
            boundary_layer_height=config.atmosphere_depth,
            surface=surface,
            rho_air=config.rho_air,
            cp_air=config.cp_air,
        )
        return result.stress, result.sensible, result.water
    speed = jnp.linalg.norm(relative, axis=-1)
    stress = config.rho_air * config.drag_coefficient * speed[..., None] * relative
    sensible = (
        config.rho_air
        * config.cp_air
        * config.heat_coefficient
        * speed
        * (sst - air_temperature)
    )
    water = (
        config.rho_air
        * config.moisture_coefficient
        * speed
        * (0.98 * saturation_specific_humidity(sst, config.pressure) - humidity)
    )
    return stress, sensible, water


def _clamp_line(line):
    """Restrict the existing mass-orthonormal full BSPF line to psi=psi'=0."""
    traces = np.vstack((np.asarray(line.bn)[[0, -1]], np.asarray(line.gn)[[0, -1]]))
    traces /= la.norm(traces, axis=1)[:, None]
    z = la.null_space(traces)
    stiffness = np.asarray(line.g).T @ (np.asarray(line.weights)[:, None] * line.g)
    lam, vectors = la.eigh(z.T @ stiffness @ z)
    rotation = jnp.asarray(z @ vectors)
    h = line.h @ rotation
    return line._replace(
        b=line.b @ rotation,
        g=line.g @ rotation,
        h=h,
        bn=line.bn @ rotation,
        gn=line.gn @ rotation,
        hn=line.hn @ rotation,
        lam=jnp.asarray(lam),
        bending=h.T @ (line.weights[:, None] * h),
        rotation=line.rotation @ rotation,
    )


def _periodic_line(template, modes):
    """Mass-orthonormal real Fourier basis on shared BSPF quadrature points.

    2*modes+1 real functions, without a duplicated Nyquist degree of freedom.
    Endpoint-inclusive nodes are for output only; endpoints are not independent.
    """
    k = np.arange(1, modes + 1) * 2 * np.pi

    def evaluate(points):
        phase = np.asarray(points)[:, None] * k[None, :]
        co, si = np.sqrt(2) * np.cos(phase), np.sqrt(2) * np.sin(phase)
        b = np.column_stack((np.ones(len(points)), co, si))
        g = np.column_stack((np.zeros(len(points)), -si * k, co * k))
        h = np.column_stack((np.zeros(len(points)), -co * k**2, -si * k**2))
        return map(jnp.asarray, (b, g, h))

    b, g, h = evaluate(template.points)
    bn, gn, hn = evaluate(template.x)
    lam = jnp.asarray(np.r_[0.0, k**2, k**2])
    return template._replace(
        b=b,
        g=g,
        h=h,
        bn=bn,
        gn=gn,
        hn=hn,
        lam=lam,
        bending=jnp.diag(lam**2),
        rotation=None,
    )


def _meridional_line(template, modes, *, scalar=False):
    """Sines for streamfunction (psi=psi_yy=0), cosines for scalar Neumann BC."""
    k = np.arange(1, modes + 1) * np.pi

    def evaluate(points):
        phase = np.asarray(points)[:, None] * k[None, :]
        co, si = np.sqrt(2) * np.cos(phase), np.sqrt(2) * np.sin(phase)
        if scalar:
            b = np.column_stack((np.ones(len(points)), co))
            g = np.column_stack((np.zeros(len(points)), -si * k))
            h = np.column_stack((np.zeros(len(points)), -co * k**2))
        else:
            b, g, h = si, co * k, -si * k**2
        return map(jnp.asarray, (b, g, h))

    b, g, h = evaluate(template.points)
    bn, gn, hn = evaluate(template.x)
    lam = jnp.asarray(np.r_[0.0, k**2] if scalar else k**2)
    return template._replace(
        b=b,
        g=g,
        h=h,
        bn=bn,
        gn=gn,
        hn=hn,
        lam=lam,
        bending=jnp.diag(lam**2),
        rotation=None,
    )


def plan_air_sea(*, n=33, config=None, quadrature_order=None, surface=None):
    """Setup on host. Runtime is float64 JAX, only 1D spatial factors stored."""
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 before setup")
    if not isinstance(n, int) or isinstance(n, bool) or n < 33:
        raise ValueError("n must be an integer >=33")
    config = AirSeaConfig() if config is None else config
    for field in fields(config):
        if not np.isfinite(getattr(config, field.name)):
            raise ValueError(f"{field.name} must be finite")
    signed = {"f0", "beta", "radiation", "initial_air_temperature_offset"}
    positive = {
        "length",
        "ocean_depth",
        "mixed_layer_depth",
        "atmosphere_depth",
        "rho_ocean",
        "rho_air",
        "cp_ocean",
        "cp_air",
        "latent_heat",
        "reference_temperature",
        "pressure",
    }
    for field in fields(config):
        value = getattr(config, field.name)
        if (field.name in positive and value <= 0) or (
            field.name not in signed | positive and value < 0
        ):
            raise ValueError(f"Invalid {field.name}")
    if config.mixed_layer_depth > config.ocean_depth:
        raise ValueError("mixed_layer_depth must not exceed ocean_depth")
    if (
        not isinstance(config.wind_jet_count, int)
        or isinstance(config.wind_jet_count, bool)
        or config.wind_jet_count < 1
    ):
        raise ValueError("wind_jet_count must be a positive integer")
    # Degree-13 basis needs >=14 points just for quadratic spline products.
    # Keep >=20 for the nonlinear demonstration; default is existing overintegration.
    if quadrature_order is not None and (
        not isinstance(quadrature_order, int) or quadrature_order < 20
    ):
        raise ValueError("quadrature_order must be an integer >=20")
    scalar = _stream_line(
        np.linspace(0, 1, n), clamped=False, quadrature_order=quadrature_order
    )
    line = _clamp_line(scalar)
    flow = StreamNavierStokes2DPlan(
        line, line, line.lam[:, None] + line.lam[None, :], jnp.asarray(0.0)
    )
    periodic = _periodic_line(scalar, (n - 1) // 2)
    meridional = _meridional_line(scalar, n - 1)
    air_scalar_y = _meridional_line(scalar, n - 1, scalar=True)
    air_mass = periodic.lam[:, None] + meridional.lam[None, :]
    air_flow = StreamNavierStokes2DPlan(
        periodic, meridional, air_mass, jnp.asarray(0.0)
    )
    weight = scalar.weights[:, None] * scalar.weights[None, :]
    mean = scalar.b.T @ scalar.weights
    x, y = scalar.points[:, None], scalar.points[None, :]
    zonal_wind = jnp.broadcast_to(
        -config.wind_speed * jnp.cos(2 * jnp.pi * config.wind_jet_count * y),
        weight.shape,
    )
    wind = jnp.stack((zonal_wind, jnp.zeros_like(zonal_wind)), axis=-1)
    # Smooth at the atmospheric x seam, with a meridional temperature gradient.
    # Ocean basin evolution need not preserve matching values at its x walls.
    target_sst = (
        config.reference_temperature
        + 4 * jnp.cos(jnp.pi * y)
        + 0.5 * jnp.cos(2 * jnp.pi * x)
    )
    target_air = target_sst + config.initial_air_temperature_offset
    target_q = config.initial_relative_humidity * saturation_specific_humidity(
        target_air, config.pressure
    )

    def project(value, bx=scalar.b, by=scalar.b):
        return bx.T @ (weight * value) @ by

    return AirSeaPlan(
        config,
        flow,
        scalar,
        weight,
        mean,
        config.f0 + config.beta * config.length * (y - 0.5),
        stream_ns_load(air_flow, wind) / air_flow.denominator,
        project(target_air - config.reference_temperature, periodic.b, air_scalar_y.b),
        project(target_q, periodic.b, air_scalar_y.b),
        project(jnp.ones_like(target_sst)),
        air_flow,
        (periodic.b.T @ periodic.weights, air_scalar_y.b.T @ air_scalar_y.weights),
        project(target_sst - config.reference_temperature),
        air_scalar_y,
        SurfaceExchangeConfig() if surface is None else surface,
    )


def plan_air_sea_stepper(
    *, dt_air=30.0, dt_ocean=150.0, window=300.0, method="mri-gark4"
):
    """Plan fixed multirate steps; window is the output/synchronization interval.

    MRI: dt_ocean is macro H. Each H/5 interval has ceil(H/(5*dt_air))
    RK4 microsteps, making dt_air an upper bound. Changing window alone while
    H and microsteps are fixed does not change the discretization anymore.
    Legacy lagged mode requires both nominal steps to divide the window.
    """
    values = (dt_air, dt_ocean, window)
    if not all(np.isfinite(v) and v > 0 for v in values):
        raise ValueError("Time steps and window must be finite and positive")
    if method not in ("mri-gark4", "lagged"):
        raise ValueError("method must be 'mri-gark4' or 'lagged'")
    ocean_steps = int(round(window / dt_ocean))
    if ocean_steps < 1 or not np.isclose(
        ocean_steps * dt_ocean, window, rtol=1e-12, atol=0
    ):
        raise ValueError("window must be an integer multiple of dt_ocean")
    inner_steps = max(1, int(np.ceil(dt_ocean / (5 * dt_air) - 1e-12)))
    air_steps = 5 * inner_steps * ocean_steps
    if method == "lagged":
        air_steps = int(round(window / dt_air))
        if air_steps < 1 or not np.isclose(
            air_steps * dt_air, window, rtol=1e-12, atol=0
        ):
            raise ValueError("lagged window must be an integer multiple of dt_air")
    return AirSeaStepper(
        float(dt_air),
        float(dt_ocean),
        float(window),
        air_steps,
        ocean_steps,
        method,
        inner_steps,
    )


def scalar_values(plan, coefficients, *, nodes=False, air=False):
    x, y = (plan.air_flow.x, plan.air_scalar_y) if air else (plan.scalar, plan.scalar)
    bx, by = (x.bn, y.bn) if nodes else (x.b, y.b)
    return bx @ coefficients @ by.T


def scalar_project(plan, values, *, air=False):
    x, y = (plan.air_flow.x, plan.air_scalar_y) if air else (plan.scalar, plan.scalar)
    return x.b.T @ (plan.weight * values) @ y.b


def scalar_mean(plan, coefficients, *, air=False):
    mx, my = plan.air_mean_mode if air else (plan.mean_mode, plan.mean_mode)
    return mx @ coefficients @ my


def initial_air_sea_state(plan):
    atmosphere = plan.wind_target
    amplitude = plan.config.initial_air_perturbation
    if amplitude:
        # Curl of a smooth streamfunction, periodic x / free-slip y.
        # The deterministic seed is initial data only, never a restoring target.
        x, y = plan.scalar.points[:, None], plan.scalar.points[None, :]
        u = amplitude * (
            0.5 * jnp.cos(jnp.pi * y) * jnp.cos(2 * jnp.pi * x)
            + 0.1875 * jnp.cos(3 * jnp.pi * y) * jnp.cos(4 * jnp.pi * x + 0.37)
        )
        v = amplitude * (
            jnp.sin(jnp.pi * y) * jnp.sin(2 * jnp.pi * x)
            + 0.25 * jnp.sin(3 * jnp.pi * y) * jnp.sin(4 * jnp.pi * x + 0.37)
        )
        atmosphere = (
            atmosphere
            + stream_ns_load(plan.air_flow, jnp.stack((u, v), axis=-1))
            / plan.air_flow.denominator
        )
    return AirSeaState(
        jnp.zeros_like(plan.flow.denominator),
        atmosphere,
        plan.initial_sst,
        plan.temperature_target,
        plan.humidity_target,
        jnp.asarray(0.0),
    )


def scalar_transport(plan, coefficients, velocity, diffusivity, *, air=False):
    """Conservative weak advection and natural-Neumann diffusion, per second."""
    x, y = (plan.air_flow.x, plan.air_scalar_y) if air else (plan.scalar, plan.scalar)
    length = plan.config.length
    value = scalar_values(plan, coefficients, air=air)
    adv = (
        x.g.T @ (plan.weight * velocity[..., 0] * value) @ y.b
        + x.b.T @ (plan.weight * velocity[..., 1] * value) @ y.g
    ) / length
    # Keep the assembled stiffness rather than modifying its constant eigenmode.
    diffusion = (
        diffusivity / length**2 * (x.lam[:, None] + y.lam[None, :]) * coefficients
    )
    return adv - diffusion


def flow_rhs(plan, coefficients, viscosity, drag=0.0, velocity=None, *, air=False):
    p, length = (plan.air_flow if air else plan.flow), plan.config.length
    u = stream_ns_velocity(p, coefficients) if velocity is None else velocity
    omega = stream_ns_vorticity(p, coefficients) / length
    absolute_omega = omega + plan.coriolis
    load = stream_ns_load(
        p, jnp.stack((u[..., 1] * absolute_omega, -u[..., 0] * absolute_omega), axis=-1)
    )
    diffusion = (
        p.x.bending @ coefficients
        + coefficients @ p.y.bending.T
        + 2 * p.x.lam[:, None] * coefficients * p.y.lam[None, :]
    )
    result = (
        load - viscosity / length**2 * diffusion
    ) / p.denominator - drag * coefficients
    return result


def exchange_loads(plan, ocean_velocity, sst, air_velocity, air_temperature, humidity):
    stress, sensible, water = bulk_flux(
        plan.config,
        ocean_velocity,
        air_velocity,
        sst,
        air_temperature,
        humidity,
        surface=plan.surface,
    )
    return Exchange(
        stream_ns_load(plan.flow, stress),
        scalar_project(plan, sensible),
        scalar_project(plan, water),
        jnp.sum(plan.weight[..., None] * stress, axis=(0, 1)),
        stream_ns_load(plan.air_flow, stress),
        scalar_project(plan, sensible, air=True),
        scalar_project(plan, water, air=True),
    )


def _rk4_with_budget(state, dt, rhs):
    """Pytree RK4 with the identical stage quadrature for all source budgets."""

    def add(a, b, scale):
        return jax.tree.map(lambda x, y: x + scale * y, a, b)

    k1, b1 = rhs(state)
    k2, b2 = rhs(add(state, k1, dt / 2))
    k3, b3 = rhs(add(state, k2, dt / 2))
    k4, b4 = rhs(add(state, k3, dt))

    def weighted(a, b, c, d):
        return dt / 6 * (a + 2 * b + 2 * c + d)

    increment = jax.tree.map(weighted, k1, k2, k3, k4)
    return add(state, increment, 1.0), jax.tree.map(weighted, b1, b2, b3, b4)


def zero_air_sea_budget(state):
    return WindowBudget(
        Exchange(
            jnp.zeros_like(state.ocean),
            jnp.zeros_like(state.sst),
            jnp.zeros_like(state.sst),
            jnp.zeros(2),
            jnp.zeros_like(state.atmosphere),
            jnp.zeros_like(state.air_temperature),
            jnp.zeros_like(state.humidity),
        ),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
    )


def air_sea_fast_rhs(plan, state):
    """Atmosphere plus BOTH interface tendencies, all at one common stage.

    Fast ocean exchange is essential: splitting the equal/opposite fluxes
    between independently weighted slow and fast parts would lose conservation.
    Ocean convection/diffusion/Coriolis remain in the slow RHS.
    """
    c = plan.config
    ocean_velocity = stream_ns_velocity(plan.flow, state.ocean)
    velocity = stream_ns_velocity(plan.air_flow, state.atmosphere) + jnp.array(
        [state.air_mean_wind, 0.0]
    )
    flux = exchange_loads(
        plan,
        ocean_velocity,
        scalar_values(plan, state.sst) + c.reference_temperature,
        velocity,
        scalar_values(plan, state.air_temperature, air=True) + c.reference_temperature,
        scalar_values(plan, state.humidity, air=True),
    )
    restore_t = c.temperature_restore_rate * (
        plan.temperature_target - state.air_temperature
    )
    restore_q = c.humidity_restore_rate * (plan.humidity_target - state.humidity)
    derivative = AirSeaState(
        flux.stress / (c.ocean_mass * plan.flow.denominator),
        flow_rhs(plan, state.atmosphere, c.air_viscosity, velocity=velocity, air=True)
        + c.wind_restore_rate * (plan.wind_target - state.atmosphere)
        - flux.air_stress / (c.air_mass * plan.air_flow.denominator),
        -(flux.sensible + c.latent_heat * flux.water) / c.ocean_capacity,
        scalar_transport(
            plan, state.air_temperature, velocity, c.air_diffusivity, air=True
        )
        + flux.air_sensible / c.air_capacity
        + restore_t,
        scalar_transport(plan, state.humidity, velocity, c.air_diffusivity, air=True)
        + flux.air_water / c.air_mass
        + restore_q,
        -c.wind_restore_rate * state.air_mean_wind - flux.stress_mean[0] / c.air_mass,
    )
    water = c.air_mass * scalar_mean(plan, restore_q, air=True)
    heat = (
        c.air_capacity * scalar_mean(plan, restore_t, air=True) + c.latent_heat * water
    )
    return derivative, WindowBudget(
        flux, heat, water, -c.air_mass * c.wind_restore_rate * state.air_mean_wind
    )


def air_sea_slow_rhs(plan, state):
    """Ocean interior dynamics and radiation, called five times per MRI macro step."""
    c = plan.config
    velocity = stream_ns_velocity(plan.flow, state.ocean)
    zero = jax.tree.map(jnp.zeros_like, state)
    derivative = zero._replace(
        ocean=flow_rhs(plan, state.ocean, c.ocean_viscosity, c.bottom_drag, velocity),
        sst=scalar_transport(plan, state.sst, velocity, c.ocean_diffusivity)
        + c.radiation * plan.unit_scalar / c.ocean_capacity,
    )
    return derivative, zero_air_sea_budget(state)._replace(
        external_heat=jnp.asarray(c.radiation)
    )


def air_sea_step(plan, stepper, state):
    """Advance one synchronization window; return state and integrated budgets.

    Default MRI-GARK4 evaluates all interface fluxes at evolving common stages.
    Select method='lagged' in the planner to reproduce the original first order.
    """
    if stepper.method == "lagged":
        return _air_sea_step_lagged(plan, stepper, state)

    def macro_step(k, augmented):
        return mri_gark_erk45a_step(
            augmented,
            k * stepper.dt_ocean,
            stepper.dt_ocean,
            lambda t, y: air_sea_fast_rhs(plan, y[0]),
            lambda t, y: air_sea_slow_rhs(plan, y[0]),
            inner_steps=stepper.inner_steps,
        )

    # Budgets are extra ODE components, so every RK coupling weight also acts
    # on its associated source integral. No post-step clipping/correction.
    return jax.lax.fori_loop(
        0, stepper.ocean_steps, macro_step, (state, zero_air_sea_budget(state))
    )


def _air_sea_step_lagged(plan, stepper, state):
    """Advance both fluids through one window; return state and budget impulses.

    Close over plan and stepper in jax.jit. The plan's config is static.
    External budgets include radiative and air-restoring sources only. Water
    removed from the fixed-volume ocean is a diagnostic reservoir, not salinity.
    """
    c, p = plan.config, plan.flow
    ocean_velocity = stream_ns_velocity(p, state.ocean)
    sst = scalar_values(plan, state.sst) + c.reference_temperature
    zero_exchange = Exchange(
        jnp.zeros_like(state.ocean),
        jnp.zeros_like(state.sst),
        jnp.zeros_like(state.sst),
        jnp.zeros(2),
        jnp.zeros_like(state.atmosphere),
        jnp.zeros_like(state.air_temperature),
        jnp.zeros_like(state.humidity),
    )
    zero_budget = WindowBudget(
        zero_exchange, jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.0)
    )

    def atmosphere_rhs(a):
        velocity = stream_ns_velocity(plan.air_flow, a.velocity) + jnp.array(
            [a.mean_wind, 0.0]
        )
        flux = exchange_loads(
            plan,
            ocean_velocity,
            sst,
            velocity,
            scalar_values(plan, a.temperature, air=True) + c.reference_temperature,
            scalar_values(plan, a.humidity, air=True),
        )
        restore_t = c.temperature_restore_rate * (
            plan.temperature_target - a.temperature
        )
        restore_q = c.humidity_restore_rate * (plan.humidity_target - a.humidity)
        rhs = AirState(
            flow_rhs(plan, a.velocity, c.air_viscosity, velocity=velocity, air=True)
            + c.wind_restore_rate * (plan.wind_target - a.velocity)
            - flux.air_stress / (c.air_mass * plan.air_flow.denominator),
            scalar_transport(plan, a.temperature, velocity, c.air_diffusivity, air=True)
            + flux.air_sensible / c.air_capacity
            + restore_t,
            scalar_transport(plan, a.humidity, velocity, c.air_diffusivity, air=True)
            + flux.air_water / c.air_mass
            + restore_q,
            -c.wind_restore_rate * a.mean_wind - flux.stress_mean[0] / c.air_mass,
        )
        external_water = c.air_mass * scalar_mean(plan, restore_q, air=True)
        external_heat = (
            c.air_capacity * scalar_mean(plan, restore_t, air=True)
            + c.latent_heat * external_water
        )
        return rhs, WindowBudget(
            flux,
            external_heat,
            external_water,
            -c.air_mass * c.wind_restore_rate * a.mean_wind,
        )

    def air_substep(_, carry):
        air, budget = carry
        air, increment = _rk4_with_budget(air, stepper.dt_air, atmosphere_rhs)
        return air, jax.tree.map(jnp.add, budget, increment)

    air, budget = jax.lax.fori_loop(
        0,
        stepper.air_steps,
        air_substep,
        (
            AirState(
                state.atmosphere,
                state.air_temperature,
                state.humidity,
                state.air_mean_wind,
            ),
            zero_budget,
        ),
    )
    flux = jax.tree.map(lambda v: v / stepper.window, budget.exchange)

    def ocean_rhs(o):
        a, temperature = o
        velocity = stream_ns_velocity(p, a)
        return (
            flow_rhs(plan, a, c.ocean_viscosity, c.bottom_drag, velocity)
            + flux.stress / (c.ocean_mass * p.denominator),
            scalar_transport(plan, temperature, velocity, c.ocean_diffusivity)
            + (
                c.radiation * plan.unit_scalar
                - flux.sensible
                - c.latent_heat * flux.water
            )
            / c.ocean_capacity,
        ), ()

    ocean, temperature = jax.lax.fori_loop(
        0,
        stepper.ocean_steps,
        lambda _, o: _rk4_with_budget(o, stepper.dt_ocean, ocean_rhs)[0],
        (state.ocean, state.sst),
    )
    return AirSeaState(
        ocean, air.velocity, temperature, air.temperature, air.humidity, air.mean_wind
    ), budget._replace(
        external_heat=budget.external_heat + c.radiation * stepper.window
    )


def heat_content(plan, state):
    """Mean sensible anomalies + atmospheric vapor latent energy, J/m².

    This is the model's thermodynamic budget, not total mechanical energy.
    Momentum dissipation is not returned as heat in this first version.
    """
    c = plan.config
    return c.ocean_capacity * scalar_mean(plan, state.sst) + scalar_mean(
        plan,
        c.air_capacity * state.air_temperature
        + c.latent_heat * c.air_mass * state.humidity,
        air=True,
    )


def air_sea_fields(plan, state, *, nodes=True):
    c, p = plan.config, plan.flow
    b = p.x.bn if nodes else p.x.b
    ax, ay = (
        (plan.air_flow.x.bn, plan.air_flow.y.bn)
        if nodes
        else (plan.air_flow.x.b, plan.air_flow.y.b)
    )
    yy = plan.air_flow.y.x if nodes else plan.air_flow.y.points
    return {
        "ocean_velocity": stream_ns_velocity(p, state.ocean, nodes=nodes),
        "air_velocity": stream_ns_velocity(plan.air_flow, state.atmosphere, nodes=nodes)
        + jnp.array([state.air_mean_wind, 0.0]),
        "ocean_streamfunction": c.length * (b @ state.ocean @ b.T),
        "air_streamfunction": c.length
        * (ax @ state.atmosphere @ ay.T + state.air_mean_wind * yy[None, :]),
        "sst": scalar_values(plan, state.sst, nodes=nodes) + c.reference_temperature,
        "air_temperature": scalar_values(
            plan, state.air_temperature, nodes=nodes, air=True
        )
        + c.reference_temperature,
        "humidity": scalar_values(plan, state.humidity, nodes=nodes, air=True),
    }
