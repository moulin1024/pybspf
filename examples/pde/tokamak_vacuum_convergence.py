"""Historical BSPF/FEM comparator: equilibrium, FEM mesh and plasma cutoff checks."""

import json
from pathlib import Path
import jax
import numpy as np
from bspf_jax.tokamak_equilibrium import (
    plan_axisymmetric_bspf,
    fit_fixed_coils,
    solve_equilibrium,
)
from bspf_jax.tokamak_vacuum import assemble_plasma_vacuum


def main():
    jax.config.update("jax_enable_x64", True)
    out = Path("build/tokamak_vacuum")
    out.mkdir(parents=True, exist_ok=True)
    coils, offset, _ = fit_fixed_coils(quadrupole=-0.004, vertical=0.03, offset=-0.2)
    results = []
    for n in (33, 41, 49):
        p = plan_axisymmetric_bspf(n)
        eq = solve_equilibrium(p, coils, offset=offset, max_iterations=700)
        m = assemble_plasma_vacuum(
            p,
            eq,
            coils,
            offset,
            modes=14,
            angles=256,
            radial_quadrature=64,
            vacuum_layers=48,
            vacuum_method="fem",
        )
        values, _ = m.modes()
        row = dict(n=n, gamma=float(np.sqrt(-values[0])))
        results.append(row)
        print(row, flush=True)
    (out / "equilibrium_refinement.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    results = []
    settings = [
        (10, 128, 20, 36, 1e-10),
        (12, 128, 20, 36, 1e-10),
        (12, 192, 32, 48, 1e-10),
        (14, 192, 32, 48, 1e-10),
        (16, 192, 32, 48, 1e-10),
        (14, 256, 48, 64, 1e-10),
        (14, 192, 32, 48, 1e-9),
        (14, 192, 32, 48, 1e-11),
    ]
    for modes, angles, layers, radial, cutoff in settings:
        m = assemble_plasma_vacuum(
            p,
            eq,
            coils,
            offset,
            modes=modes,
            angles=angles,
            radial_quadrature=radial,
            vacuum_layers=layers,
            mass_cutoff=cutoff,
            vacuum_method="fem",
        )
        values, _ = m.modes()
        row = dict(**m.diagnostics, gamma=float(np.sqrt(-values[0])))
        results.append(row)
        print(row, flush=True)
    (out / "refinement.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
