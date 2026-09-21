"""Independent NOAA reference, vector symmetry and zero-wind limits."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bspf_models.air_sea.surface_exchange import SurfaceExchangeConfig
from bspf_models.air_sea.surface_exchange import coare35
from bspf_models.air_sea.surface_exchange import specific_humidity_from_rh

jax.config.update("jax_enable_x64", True)
REFERENCE = Path(__file__).parent / "reference/coare35"


def reference():
    manifest = json.loads((REFERENCE / "manifest.json").read_text())
    for path, sha in manifest["files"].items():
        assert (
            hashlib.sha256((REFERENCE / Path(path).name).read_bytes()).hexdigest()
            == sha
        )
    for name in ("util", "meteo", "coare35vn"):
        spec = importlib.util.spec_from_file_location(name, REFERENCE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module.coare35vn


@pytest.mark.parametrize("height", [5.0, 10.0, 20.0])
def test_official_fluxes(height):
    fn = reference()
    # Deliberately include zero/weak wind, stable/unstable, and thin-stable branch.
    speed = np.array([0.0, 0.01, 0.2, 1.0, 3.0, 8.0, 15.0, 25.0, 0.1, 5.0])
    t = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 10.0, 25.0, 25.0, 20.0])
    ts = t + np.array([2.0, 2.0, -0.5, -2.0, 0.0, 2.0, -1.0, 1.0, -10.0, 0.098])
    rh = np.array([75.0, 75.0, 90.0, 80.0, 100.0, 60.0, 85.0, 90.0, 90.0, 95.0])
    q = specific_humidity_from_rh(t, 1013.25, rh)
    surface = SurfaceExchangeConfig(
        method="coare35",
        wind_height=height,
        temperature_height=height,
        humidity_height=height,
    )
    with np.errstate(all="ignore"):
        expected = fn(
            speed,
            t,
            rh,
            ts,
            P=1013.25,
            zu=height,
            zt=height,
            zq=height,
            lat=45.0,
            zi=1000.0,
            jcool=0,
        )
    result = jax.jit(lambda u, a, q, s: coare35(u, a, q, s, surface=surface))(
        jnp.stack((jnp.asarray(speed), jnp.zeros_like(q)), axis=-1),
        t + 273.15,
        q,
        ts + 273.15,
    )
    np.testing.assert_allclose(
        result.stress[:, 0], expected[:, 1], rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(result.sensible, expected[:, 2], rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(
        result.water * (2.501 - 0.00237 * ts) * 1e6,
        expected[:, 3],
        rtol=1e-6,
        atol=1e-5,
    )
    assert np.all(result.valid)
    assert result.stress[0, 0] == 0


def test_vector_rotation_and_constant_prefactors():
    u = jnp.array([[3.0, 4.0], [0.0, 0.0]])
    args = (
        jnp.array([290.0, 290.0]),
        jnp.array([0.008, 0.008]),
        jnp.array([292.0, 292.0]),
    )
    a = coare35(u, *args, rho_air=1.2, cp_air=1004.0)
    b = coare35(u[:, ::-1] * jnp.array([-1.0, 1.0]), *args, rho_air=1.2, cp_air=1004.0)
    np.testing.assert_allclose(b.stress, a.stress[:, ::-1] * np.array([-1.0, 1.0]))
    np.testing.assert_allclose(a.sensible, b.sensible)
    assert np.all(np.sum(a.stress * u, axis=-1) >= 0)
    assert np.isfinite(np.asarray(a.drag)).all()
    assert np.all(a.water > 0)


def test_invalid_config_and_humidity():
    with pytest.raises(ValueError):
        SurfaceExchangeConfig(wind_height=0)
    with pytest.raises(ValueError):
        SurfaceExchangeConfig(method="unknown")
    assert not bool(coare35(jnp.array([1.0, 0.0]), 290.0, -0.01, 292.0).valid)
