"""GPU-resident IMEX channel evolution after host geometry/MPFR assembly.

All matrices are uploaded once. State, nonlinear quadrature, outlet loads,
Cholesky factors and both stage solves stay on the selected GPU. Only explicit
output/checkpoint calls should download state. Optional loads must be JAX
traceable and return a device vector in the reduced velocity space.
"""
from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl

from pybspf.time_integration import imex_midpoint


def _explicit(d, state):
    u, v, xy, yy, minus_xx = [
        o @ state + lift for o, lift in zip(d["ops"], d["lift"])
    ]
    bu, bv = d["ops"][:2]
    rhs = -bu.T @ (d["weights"] * (u * xy + v * yy))
    rhs -= bv.T @ (d["weights"] * (u * minus_xx - v * xy))
    outu, outv = [o @ state + lift for o, lift in zip(d["out_ops"], d["out_lift"])]
    incoming = jnp.minimum(outu, 0) * d["out_weights"]
    return (rhs + d["out_ops"][0].T @ (incoming * outu)
            + d["out_ops"][1].T @ (incoming * outv) - d["linear_lift"])


@partial(jax.jit, static_argnames=("load",))
def _step(d, state, time, dt, load=None):
    def explicit(a, t):
        value = _explicit(d, a)
        return value if load is None else value + load(t)
    return imex_midpoint(
        state, time, dt, lambda a: d["mass"] @ a,
        lambda a: d["linear"] @ a, explicit,
        lambda b: jl.cho_solve((d["factor"], True), b),
    )


@jax.jit
def _diagnostics(d, state):
    u, v, xy, uy, vx = [o @ state + lift for o, lift in zip(d["ops"], d["lift"])]
    outu = d["out_ops"][0] @ state + d["out_lift"][0]
    derivative = jl.cho_solve(
        (d["mass_factor"], True), _explicit(d, state) - d["linear"] @ state
    )
    du = u - d["base_u"]
    return dict(
        kinetic_energy=jnp.sum(d["weights"] * (u*u + v*v))/2,
        max_speed=jnp.max(jnp.hypot(u, v)), min_outlet_u=jnp.min(outu),
        flux_in=d["flux_in"], flux_out=d["out_weights"] @ outu,
        acceleration_l2=jnp.sqrt(jnp.maximum(derivative @ d["mass"] @ derivative, 0)),
        dissipation=d["nu"] * jnp.sum(d["weights"] * (2*xy*xy + uy*uy + vx*vx)),
        sponge_perturbation_dissipation=jnp.sum(d["weights"] * d["sigma"] * (du*du + v*v)),
    )


class GPUImmersedFlowStepper:
    """Use plan.stepper(dt, device=jax.devices('gpu')[0]).

    Host assembly remains available for independent verification and rendering.
    This runtime owns a device copy of every array needed for evolution.
    """
    def __init__(self, plan, dt, device):
        if device.platform != "gpu":
            raise ValueError("GPUImmersedFlowStepper requires a GPU device")
        self.device, self.dt = device, jax.device_put(dt, device)
        self.shape = (plan.dofs,)
        d = dict(
            mass=plan.mass, linear=plan.linear, linear_lift=plan.linear_lift,
            ops=tuple(plan.operators_fluid), lift=tuple(plan.lift_fields[1:]),
            weights=plan.weights, out_ops=tuple(plan.out_ops),
            out_lift=tuple(plan.out_lift), out_weights=plan.out_weights,
            sigma=plan.sigma, nu=plan.nu,
            base_u=plan.peak * (1 - plan.points[:, 1]**2 / plan.bounds[2]**2),
            flux_in=4 * plan.peak * plan.bounds[2] / 3,
        )
        start = perf_counter()
        self.data = jax.device_put(d, device)
        self.initial_state = jax.device_put(plan.stokes_state, device)
        jax.block_until_ready((self.data, self.initial_state))
        self.setup_timings = {"upload_seconds": perf_counter() - start}
        start = perf_counter()
        with jax.default_device(device):
            self.data["factor"] = jnp.linalg.cholesky(
                self.data["mass"] + dt / 2 * self.data["linear"]
            )
            self.data["mass_factor"] = jnp.linalg.cholesky(self.data["mass"])
        jax.block_until_ready((self.data["factor"], self.data["mass_factor"]))
        self.setup_timings["factorization_with_compile_seconds"] = perf_counter() - start
        for key in ("factor", "mass_factor"):
            if not bool(jnp.all(jnp.isfinite(self.data[key]))):
                raise ValueError("Non-finite GPU Cholesky factor: " + key)

    def step(self, state, time=0.0, load=None):
        if not isinstance(state, jax.Array) or state.devices() != {self.device}:
            raise ValueError("Upload state to the stepper GPU before stepping")
        if state.shape != self.shape:
            raise ValueError("State shape does not match the flow plan")
        return _step(self.data, state, jax.device_put(time, self.device), self.dt, load)

    def diagnostics(self, state):
        """Device scalar diagnostics; callers explicitly download for output."""
        return _diagnostics(self.data, state)
