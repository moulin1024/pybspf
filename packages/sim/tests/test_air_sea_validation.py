import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from bspf_models.air_sea.air_sea import plan_air_sea
from bspf_models.air_sea.air_sea import initial_air_sea_state
from bspf_sim.air_sea.platform import ValidationConfig
from bspf_sim.air_sea.validation import transfer_quadrature_basis
from bspf_sim.air_sea.validation import compare
from bspf_models.air_sea.surface_exchange import coare35
from bspf_models.air_sea.surface_exchange import SurfaceExchangeConfig
from pybspf.multirate import mri_gark_erk45a_step

jax.config.update("jax_enable_x64", True)


def test_common_quadrature_preserves_physical_trial_function():
    a, b = [plan_air_sea(n=33, quadrature_order=q) for q in (20, 24)]
    s = initial_air_sea_state(a)
    s = s._replace(
        ocean=jnp.sin(jnp.arange(s.ocean.size).reshape(s.ocean.shape)) * 0.001
    )
    mapped = transfer_quadrature_basis(a, b, s)
    for x, y, lx, ly in (
        (s.ocean, mapped.ocean, a.flow.x, b.flow.x),
        (s.sst, mapped.sst, a.scalar, b.scalar),
    ):
        np.testing.assert_allclose(lx.bn @ x @ lx.bn.T, ly.bn @ y @ ly.bn.T, atol=3e-12)


def test_temperature_error_retains_mean_and_uses_anomaly_scale():
    weight = np.ones((2, 2)) / 4
    b = {"sst": np.array([[289.0, 291.0], [289.0, 291.0]])}
    a = {"sst": b["sst"] + 2.0}
    r = compare(weight, a, b)["sst"]
    assert r["rms"] == r["relative_rms"] == 2.0


def test_smooth_coare_multirate_fourth_order():
    # Fixed wind and evolving thermodynamic states: an actual nonlinear COARE
    # exchange coupled to slow radiation/restoring, away from branch changes.
    surface = SurfaceExchangeConfig(method="coare35")

    @jax.jit
    def fast(t, s):
        f = coare35(
            jnp.array([8.0, 0.0]),
            s[1],
            s[2],
            s[0],
            surface=surface,
            rho_air=1.2,
            cp_air=1004.0,
        )
        return jnp.array(
            [
                -(f.sensible + 2.5e6 * f.water) / (1025 * 3990 * 2),
                f.sensible / (1.2 * 1004 * 200),
                f.water / (1.2 * 200),
            ]
        )

    @jax.jit
    def slow(t, s):
        return jnp.array(
            [50 / (1025 * 3990 * 2), (290.0 - s[1]) / 21600.0, (0.007 - s[2]) / 21600.0]
        )

    initial = np.array([294.0, 290.0, 0.007])
    duration = 4800.0
    ref = solve_ivp(
        lambda t, y: np.asarray(fast(t, y) + slow(t, y)),
        (0.0, duration),
        initial,
        method="DOP853",
        rtol=3e-13,
        atol=1e-14,
    ).y[:, -1]
    errors = []
    for h in (1200.0, 600.0, 300.0, 150.0):
        step = jax.jit(
            lambda s, t: mri_gark_erk45a_step(s, t, h, fast, slow, inner_steps=2)
        )
        s = jnp.asarray(initial)
        for k in range(round(duration / h)):
            s = step(s, k * h)
        errors.append(
            np.linalg.norm((np.asarray(s) - ref) / np.array([1.0, 1.0, 0.01]))
        )
    orders = np.log2(np.asarray(errors[:-1]) / errors[1:])
    limits = ValidationConfig()
    assert np.all(
        (orders > limits.smooth_order_min) & (orders < limits.smooth_order_max)
    ), (errors, orders)
