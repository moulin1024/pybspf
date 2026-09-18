"""JAX x64 RK4 check against the semidiscrete matrix-exponential reference."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--out", type=Path, default=Path("build/bspf_advection_diffusion_1d")
)
root = parser.parse_args().out
data = np.load(root / "operators_n160_w16.npz")
a, f, initial = map(jnp.asarray, (data["weak"], data["load"], data["initial"]))
records = []
for dt in [0.008, 0.004, 0.002, 0.001]:

    @jax.jit
    def advance(u):
        def step(i, y):
            t = i * dt

            def rhs(v, s):
                return a @ v + jnp.exp(-s) * f

            k1 = rhs(y, t)
            k2 = rhs(y + dt / 2 * k1, t + dt / 2)
            k3 = rhs(y + dt / 2 * k2, t + dt / 2)
            k4 = rhs(y + dt * k3, t + dt)
            return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return jax.lax.fori_loop(0, round(3 / dt), step, u)

    result = np.asarray(advance(initial))
    records.append(
        dict(
            dt=dt,
            reference_linf=float(abs(result - data["reference"]).max()),
            pde_linf=float(abs(result - np.exp(-3) * data["initial"]).max()),
        )
    )
(root / "jax_validation.json").write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
assert records[-1]["reference_linf"] < 5e-13
