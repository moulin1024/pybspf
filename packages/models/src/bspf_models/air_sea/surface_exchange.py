"""JAX COARE 3.5 surface exchange, skin SST / no rain / no wave input.

Adapted from NOAA-PSL/COARE-algorithm bd1cac80d2dae454e699f9ef46204dfd88086a71
(MIT license; upstream copyright and license in tests/reference/coare35/LICENSE).
The published ten iterations and zetu>50 first-iteration branch are retained.
No cool-skin or warm-layer model is included. Layer means are explicit proxies,
not a diagnosed vertical boundary-layer profile. Upward H and E; tau to ocean.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

REFERENCE_COMMIT = "bd1cac80d2dae454e699f9ef46204dfd88086a71"


@dataclass(frozen=True)
class SurfaceExchangeConfig:
    method: str = "constant"
    wind_height: float = 10.0
    temperature_height: float = 10.0
    humidity_height: float = 10.0
    latitude: float = 45.0
    stability: bool = True
    state_proxy: str = "layer_mean_as_surface_proxy"

    def __post_init__(self):
        if self.method not in ("constant", "coare35"):
            raise ValueError("Unknown surface exchange method")
        if self.state_proxy != "layer_mean_as_surface_proxy":
            raise ValueError(
                "Only the explicit layer-mean surface proxy is implemented"
            )
        if not isinstance(self.stability, bool):
            raise ValueError("stability must be boolean")
        for height in (self.wind_height, self.temperature_height, self.humidity_height):
            if not np.isfinite(height) or height <= 0:
                raise ValueError(
                    "Surface reference heights must be finite and positive"
                )
        if not np.isfinite(self.latitude) or abs(self.latitude) > 90:
            raise ValueError("Invalid latitude")


class SurfaceFlux(NamedTuple):
    stress: jax.Array
    sensible: jax.Array
    water: jax.Array
    drag: jax.Array
    heat: jax.Array
    moisture: jax.Array
    stability: jax.Array
    friction_velocity: jax.Array
    gustiness: jax.Array
    valid: jax.Array
    thin_stable_branch: jax.Array
    iteration_change: jax.Array


def vapor_pressure(temperature_c, pressure_hpa):
    return (
        6.1121
        * jnp.exp(17.502 * temperature_c / (240.97 + temperature_c))
        * (1.0007 + pressure_hpa * 3.46e-6)
    )


def specific_humidity_from_rh(temperature_c, pressure_hpa, rh_percent):
    e = vapor_pressure(temperature_c, pressure_hpa) * rh_percent / 100
    return 0.62197 * e / (pressure_hpa - 0.378 * e)


def gravity(latitude):
    x = jnp.sin(jnp.deg2rad(latitude)) ** 2
    return 9.7803267715 * (
        1
        + 0.0052790414 * x
        + 0.0000232718 * x**2
        + 0.0000001262 * x**3
        + 0.0000000007 * x**4
    )


def _psi(zeta, *, thermal=False, initial=False):
    # Safe inactive branches avoid NaNs under JAX where/autodiff. This does not
    # clamp the selected physical branch; upstream's stable exponent cap stays.
    stable = jnp.maximum(zeta, 0)
    unstable = jnp.minimum(zeta, 0)
    dzet = jnp.minimum(0.35 * stable, 50.0)
    if thermal:
        ps = -(
            (1 + 0.6667 * stable) ** 1.5
            + 0.6667 * (stable - 14.28) * jnp.exp(-dzet)
            + 8.525
        )
        x = (1 - 15 * unstable) ** 0.5
        pk = 2 * jnp.log((1 + x) / 2)
        x = (1 - 34.15 * unstable) ** 0.3333
    else:
        ps = -(
            (1.0 if initial else 0.7) * stable
            + 0.75 * (stable - 5 / 0.35) * jnp.exp(-dzet)
            + 0.75 * 5 / 0.35
        )
        x = (1 - (18 if initial else 15) * unstable) ** 0.25
        pk = (
            2 * jnp.log((1 + x) / 2)
            + jnp.log((1 + x * x) / 2)
            - 2 * jnp.arctan(x)
            + jnp.pi / 2
        )
        x = (1 - (10 if initial else 10.15) * unstable) ** 0.3333
    pc = (
        1.5 * jnp.log((1 + x + x * x) / 3)
        - jnp.sqrt(3.0) * jnp.arctan((1 + 2 * x) / jnp.sqrt(3.0))
        + jnp.pi / jnp.sqrt(3.0)
    )
    f = unstable**2 / (1 + unstable**2)
    return jnp.where(zeta < 0, (1 - f) * pk + f * pc, ps)


def coare35(
    velocity,
    air_temperature,
    humidity,
    sst,
    *,
    pressure=101325.0,
    boundary_layer_height=1000.0,
    surface=SurfaceExchangeConfig(method="coare35"),
    rho_air=None,
    cp_air=None,
):
    """Vector relative wind (m/s), absolute T (K), q (kg/kg), pressure (Pa).

    rho_air/cp_air=None reproduces official thermodynamic prefactors. Supplying
    model constants adapts only flux prefactors; transfer physics stays COARE.
    latent heat is deliberately not applied here: model L_v * water is unique.
    """
    u = jnp.linalg.norm(velocity, axis=-1)
    t, ts = air_temperature - 273.15, sst - 273.15
    p = pressure / 100
    ta = t + 273.16  # Preserve upstream offset for exact reference comparisons.
    es = 0.98 * vapor_pressure(ts, p)
    qs = 0.622 * es / (p - 0.378 * es)
    dt = ts - t - 0.0098 * surface.temperature_height
    dq = qs - humidity
    g = gravity(surface.latitude)
    rho = (
        (p * 100 / (287.1 * ta * (1 + 0.61 * humidity))) if rho_air is None else rho_air
    )
    cp = 1004.67 if cp_air is None else cp_air
    visa = 1.326e-5 * (1 + 6.542e-3 * t + 8.301e-6 * t * t - 4.84e-9 * t**3)
    zu, zt, zq = (
        surface.wind_height,
        surface.temperature_height,
        surface.humidity_height,
    )
    ut = jnp.sqrt(u * u + 0.5**2)
    u10 = ut * jnp.log(10 / 1e-4) / jnp.log(zu / 1e-4)
    usr = 0.035 * u10
    zo10 = 0.011 * usr**2 / g + 0.11 * visa / usr
    cd10 = (0.4 / jnp.log(10 / zo10)) ** 2
    zot10 = 10 * jnp.exp(-0.4 / (0.00115 / jnp.sqrt(cd10)))
    cd = (0.4 / jnp.log(zu / zo10)) ** 2
    ct = 0.4 / jnp.log(zt / zot10)
    cc = 0.4 * ct / cd
    rib = -g * zu / ta * (dt + 0.61 * ta * dq) / ut**2
    ribc = -zu / boundary_layer_height / 0.004 / 1.2**3
    # Avoid dividing the inactive stable branch by a possibly zero denominator.
    negative_rib = jnp.minimum(rib, 0)
    zet0 = jnp.where(
        rib < 0,
        cc * negative_rib / (1 + negative_rib / ribc),
        cc * rib * (1 + 3 * rib / cc),
    )
    # Upstream records k50 BEFORE replacing the unstable initial estimate.
    # Retain even its unusual strong-unstable/weak-wind consequence for parity.
    thin = (cc * rib * (1 + 3 * rib / cc) > 50) & surface.stability

    def psi(z, **kw):
        return _psi(z, **kw) if surface.stability else jnp.zeros_like(z)

    usr = ut * 0.4 / (jnp.log(zu / zo10) - psi(zet0, initial=True))
    tsr = -dt * 0.4 / (jnp.log(zt / zot10) - psi(zt / zu * zet0, thermal=True))
    qsr = -dq * 0.4 / (jnp.log(zq / zot10) - psi(zq / zu * zet0, thermal=True))
    charn = 0.0017 * jnp.minimum(u10, 19.0) - 0.005
    zeros = jnp.zeros_like(u)

    def iteration(i, carry):
        usr, tsr, qsr, ut, charn, first, _, _, _, _, _ = carry
        zet = 0.4 * g * zu / ta * (tsr + 0.61 * ta * qsr) / usr**2
        zo = charn * usr**2 / g + 0.11 * visa / usr
        rr = zo * usr / visa
        zoq = jnp.minimum(1.6e-4, 5.8e-5 / rr**0.72)
        cdu = 0.4 / (jnp.log(zu / zo) - psi(zet))
        cth = 0.4 / (jnp.log(zt / zoq) - psi(zt / zu * zet, thermal=True))
        cqh = 0.4 / (jnp.log(zq / zoq) - psi(zq / zu * zet, thermal=True))
        new_usr, new_tsr, new_qsr = ut * cdu, -dt * cth, -dq * cqh
        bf = -g / ta * new_usr * (new_tsr + 0.61 * ta * new_qsr)
        ug = jnp.where(
            bf > 0, 1.2 * (jnp.maximum(bf, 0) * boundary_layer_height) ** 0.333, 0.2
        )
        new_ut = jnp.sqrt(u * u + ug * ug)
        first = jnp.where(i == 0, jnp.stack((new_usr, new_tsr, new_qsr, zet)), first)
        # usr/von/gf, expressed without division by zero resolved wind.
        u10n = new_usr / 0.4 * (u / new_ut) * jnp.log(10 / zo)
        new_charn = 0.0017 * jnp.minimum(u10n, 19.0) - 0.005
        change = jnp.maximum(
            abs(new_usr - usr) / jnp.maximum(abs(usr), 1e-15),
            abs(new_ut - ut) / jnp.maximum(ut, 1e-15),
        )
        return (
            new_usr,
            new_tsr,
            new_qsr,
            new_ut,
            new_charn,
            first,
            zet,
            ug,
            cth,
            cqh,
            change,
        )

    result = jax.lax.fori_loop(
        0,
        10,
        iteration,
        (
            usr,
            tsr,
            qsr,
            ut,
            charn,
            jnp.stack((zeros, zeros, zeros, zeros)),
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
        ),
    )
    usr, tsr, qsr, ut, _, first, zet, ug, cth, cqh, change = result
    usr, tsr, qsr, zet = [
        jnp.where(thin, first[i], a) for i, a in enumerate((usr, tsr, qsr, zet))
    ]
    drag = (usr / ut) ** 2
    stress = (rho * usr**2 / ut)[..., None] * velocity
    h, water = -rho * cp * usr * tsr, -rho * usr * qsr
    # Finite diagnostics at zero scalar contrast, with exact finite limit.
    ch = jnp.where(
        abs(dt) > 1e-15,
        -usr * tsr / (ut * jnp.where(abs(dt) > 1e-15, dt, 1)),
        usr * cth / ut,
    )
    ce = jnp.where(
        abs(dq) > 1e-15,
        -usr * qsr / (ut * jnp.where(abs(dq) > 1e-15, dq, 1)),
        usr * cqh / ut,
    )
    valid = (
        jnp.all(jnp.isfinite(stress), axis=-1)
        & jnp.isfinite(ch)
        & jnp.isfinite(ce)
        & jnp.isfinite(usr)
        & jnp.isfinite(ug)
        & jnp.isfinite(change)
        & jnp.isfinite(h)
        & jnp.isfinite(water)
        & jnp.isfinite(drag)
        & jnp.isfinite(zet)
        & (drag >= 0)
        & (ch >= 0)
        & (ce >= 0)
        & (humidity >= 0)
        & (humidity < 1)
        & (air_temperature > 0)
        & (sst > 0)
    )
    return SurfaceFlux(
        stress, h, water, drag, ch, ce, zet, usr, ug, valid, thin, change
    )
