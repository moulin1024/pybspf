"""Compatibility aliases for the exploratory scripts; implementation is in JAX."""

import numpy as np
from bspf_jax.stream_navier_stokes import (
    StreamLine,
    StreamNavierStokes2DPlan as StreamPlan,
    plan_stream_navier_stokes2d,
    stream_ns_velocity as velocity,
    stream_ns_vorticity as vorticity,
    stream_ns_load as load,
    stream_ns_rhs as rhs,
    stream_ns_rk4_step as rk4,
    stream_kh_initial as kh_seed,
    stream_evaluate_line as evaluate_line,
)


def plan(
    nx=64,
    ny=64,
    domain=((-3, 3), (-1, 1)),
    nu=0.002,
    layers=(),
    x_boundary="fixed",
    boundary_D0=1.0,
):
    return plan_stream_navier_stokes2d(
        np.linspace(*domain[0], nx),
        np.linspace(*domain[1], ny),
        viscosity=nu,
        x_layers=layers,
        x_boundary=x_boundary,
        boundary_D0=boundary_D0,
    )


__all__ = [
    "StreamLine",
    "StreamPlan",
    "plan",
    "velocity",
    "vorticity",
    "load",
    "rhs",
    "rk4",
    "kh_seed",
    "evaluate_line",
]
