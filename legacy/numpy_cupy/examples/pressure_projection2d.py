"""Run with: PYTHONPATH=src python examples/pressure_projection2d.py."""

import numpy as np

from pybspf import PressurePoisson2D


def main():
    solver = PressurePoisson2D(np.linspace(0, 1, 64), np.linspace(0, 1.5, 72))
    x, y = np.meshgrid(solver.x, solver.y)
    pressure = np.exp(x + 0.5 * y)

    # A divergence-free, no-slip field from psi = h(x) h(y/1.5).
    def h(s):
        return s * s * (1 - s) ** 2

    def dh(s):
        return 2 * s * (1 - s) * (1 - 2 * s)

    solenoidal = np.stack([h(x) * dh(y / 1.5) / 1.5, -dh(x) * h(y / 1.5)], axis=-1)
    exact_gradient = np.stack([pressure, 0.5 * pressure], axis=-1)
    raw = solenoidal + exact_gradient
    projected, result = solver.project(raw)
    expected_pressure = solver.remove_mean(pressure)
    print(
        "Relative pressure error:",
        np.linalg.norm(result.pressure - expected_pressure)
        / np.linalg.norm(expected_pressure),
    )
    print("Max projected-field error:", abs(projected - solenoidal).max())
    print("Max divergence:", abs(solver.divergence(projected)).max())
    print("Max wall value:", abs(projected[solver.walls]).max())
    print("Max Schur residual:", result.schur_residual_linf)
    print("Max wall-gradient fit residual:", result.wall_gradient_fit_linf)


if __name__ == "__main__":
    main()
