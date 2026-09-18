"""Compare trusted local FEM/BSPF results and audit actual interface trajectories."""

import json
import pickle
from pathlib import Path
import jax
import numpy as np


def main():
    jax.config.update("jax_enable_x64", True)
    root = Path("build")
    old = json.loads((root / "tokamak_vacuum/summary.json").read_text())
    new_path = root / "tokamak_vacuum_bspf"
    new = json.loads((new_path / "summary.json").read_text())
    m = pickle.loads((new_path / "model.pkl").read_bytes())
    data = np.load(new_path / "evolution.npz")
    exact = data["displacement_coefficients"] @ m.trace.T
    projected = data["vacuum_flux"][:, m.vacuum.inner]
    trace_error = float(np.linalg.norm(exact - projected) / np.linalg.norm(exact))
    new["interface_trace_relative_error"] = trace_error
    (new_path / "summary.json").write_text(json.dumps(new, indent=2) + "\n")
    old_close = json.loads((root / "tokamak_confined/summary.json").read_text())
    close_path = root / "tokamak_confined_bspf"
    close = json.loads((close_path / "summary.json").read_text())
    m = pickle.loads((close_path / "model.pkl").read_bytes())
    data = np.load(close_path / "evolution.npz")
    exact = data["displacement_coefficients"] @ m.trace.T
    projected = data["vacuum_flux"][:, m.vacuum.inner]
    close_error = float(np.linalg.norm(exact - projected) / np.linalg.norm(exact))
    close["trajectory"]["interface_trajectory_relative_error"] = close_error
    (close_path / "summary.json").write_text(json.dumps(close, indent=2) + "\n")
    wold = old_close["wall_scan"][1]["dominant_vertical_omega"]
    wnew = close["wall_scan"][1]["dominant_vertical_omega"]
    result = dict(
        far_growth_fem=old["gamma"],
        far_growth_bspf=new["gamma"],
        relative_growth_difference=abs(new["gamma"] / old["gamma"] - 1),
        close_frequency_fem=wold,
        close_frequency_bspf=wnew,
        relative_frequency_difference=abs(wnew / wold - 1),
        growing_mode_interface_relative_error=trace_error,
        close_trajectory_interface_relative_error=close_error,
        close_energy_drift=close["trajectory"]["relative_energy_drift"],
        note="Differences between two discretizations, not certified errors against an exact solution",
    )
    (close_path / "bspf_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
