"""Reuse saved BSPF one-dimensional factors without repeating MPFR setup."""

from pathlib import Path
import json
import numpy as np
import jax.numpy as jnp
from bspf_jax.stream_navier_stokes import StreamLine, StreamNavierStokes2DPlan


def load_open_plan(directory):
    directory = Path(directory)
    f = np.load(directory / "basis.npz")
    summary = json.loads((directory / "summary.json").read_text())
    if not summary["boundary"].startswith(("Open", "Dynamic")):
        raise ValueError("Saved factors must have open vertical faces")
    x, y = [
        StreamLine(*(jnp.asarray(f[axis + "_" + key]) for key in StreamLine._fields))
        for axis in ("x", "y")
    ]
    return StreamNavierStokes2DPlan(
        x, y, x.lam[:, None] + y.lam[None, :], jnp.asarray(summary["viscosity"]), 1.0
    )
