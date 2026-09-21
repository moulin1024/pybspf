"""Report flux-history time convergence across COARE branches; no order promise."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad_vec

from bspf_jax.surface_exchange import coare35, SurfaceExchangeConfig
from bspf_jax.multirate import mri_gark_erk45a_step


def validate(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_enable_x64", True)
    duration = 3600.0
    surface = SurfaceExchangeConfig(method="coare35")
    result = {
        "scope": "Prescribed nonautonomous surface flux histories; scaled integrals, not a coupled climate trajectory. Branch crossing orders are observations, not pass/fail criteria.",
        "duration_seconds": duration,
        "cases": {},
    }
    for case in ("weak_wind_reversal", "stable_unstable_transition"):

        @jax.jit
        def forcing(t):
            u = (
                0.01 + jnp.cos(2 * jnp.pi * t / duration)
                if case == "weak_wind_reversal"
                else 8.0
            )
            ta = (
                291.0
                if case == "weak_wind_reversal"
                else 293.0 + 4 * jnp.cos(2 * jnp.pi * t / duration)
            )
            f = coare35(
                jnp.array([u, 0.0]),
                ta,
                0.009,
                293.0,
                surface=surface,
                rho_air=1.2,
                cp_air=1004.0,
            )
            return jnp.array([f.sensible, 2.5e6 * f.water, 1000 * f.stress[0]])

        reference, error = quad_vec(
            lambda t: np.asarray(forcing(t)),
            0.0,
            duration,
            epsabs=1e-7,
            epsrel=1e-11,
            points=[duration / 4, duration / 2, 3 * duration / 4],
            limit=2000,
        )
        values, errors = [], []
        steps = [600.0, 300.0, 150.0, 75.0, 37.5]
        for h in steps:

            def integrate():
                return jax.lax.fori_loop(
                    0,
                    round(duration / h),
                    lambda k, value: mri_gark_erk45a_step(
                        value,
                        k * h,
                        h,
                        lambda t, y: forcing(t),
                        lambda t, y: jnp.zeros_like(y),
                    ),
                    jnp.zeros(3),
                )

            value = np.asarray(jax.jit(integrate)())
            values.append(value.tolist())
            errors.append(float(np.linalg.norm(value - reference)))
        orders = np.log2(np.array(errors[:-1]) / errors[1:])
        result["cases"][case] = {
            "macro_steps": steps,
            "reference": reference.tolist(),
            "reference_error_estimate": float(error),
            "integrals": values,
            "errors_l2": errors,
            "observed_orders": orders.tolist(),
        }
    (out / "convergence.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.out), indent=2))
