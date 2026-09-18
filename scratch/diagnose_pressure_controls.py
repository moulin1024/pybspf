"""Controlled endpoint-window changes; all pressure solves use zero refinement."""

import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from bspf_jax import (
    plan_pressure_poisson2d,
    pressure_schur,
    pressure_gradient,
    solve_pressure_poisson2d,
    pressure_remove_mean,
)

jax.config.update("jax_enable_x64", True)

out = Path("build/pressure_accuracy_controls")
out.mkdir(parents=True, exist_ok=True)
n = 2048
grid = np.linspace(0, 1, n)
records = []
for label, options in [
    ("wider_window", dict(q=9, baseline_points=32)),
    ("fewer_jets", dict(q=7, baseline_points=16)),
]:
    print(label + " building plan", flush=True)
    plan = plan_pressure_poisson2d(
        grid, grid, endpoint_method="chebyshev", chebyshev_modes=12, **options
    )
    line = plan.x
    row = dict(
        label=label,
        N=n,
        options=options,
        cond_V=float(np.linalg.cond(np.asarray(line.vectors))),
        projector_norm=float(jnp.linalg.norm(line.projector)),
        first_eigenvalues=np.asarray(line.eigenvalues[:8]).real.tolist(),
        cases={},
    )
    print(json.dumps(row), flush=True)
    rng = np.random.default_rng(71)
    rng.standard_normal((n, n, 2))
    xx, yy = jnp.meshgrid(line.x, line.x, indexing="ij")
    fields = {
        "smooth": jnp.exp(xx + 0.5 * yy)
        + jnp.sin(3 * jnp.pi * xx) * jnp.cos(2 * jnp.pi * yy),
        "random": jnp.asarray(rng.standard_normal((n, n))),
    }
    schur = jax.jit(pressure_schur)
    grad = jax.jit(pressure_gradient)
    solve = jax.jit(
        lambda plan, b, g: solve_pressure_poisson2d(
            plan, b, wall_gradient=g, refinement_steps=0
        )
    )
    for name, p in fields.items():
        b = schur(plan, p)
        g = grad(plan, p)
        result = solve(plan, b, g)
        expected = pressure_remove_mean(plan, p)
        metrics = dict(
            converged=bool(result.converged),
            relative_pressure_error=float(
                jnp.linalg.norm(result.pressure - expected) / jnp.linalg.norm(expected)
            ),
            residual_l2=float(result.schur_residual_l2),
            residual_linf=float(result.schur_residual_linf),
            threshold=float(1e-9 + 1e-10 * jnp.linalg.norm(b)),
        )
        row["cases"][name] = metrics
        print(name + " " + json.dumps(metrics), flush=True)
    records.append(row)
    (out / "results.json").write_text(json.dumps(records, indent=2))
    jax.clear_caches()
