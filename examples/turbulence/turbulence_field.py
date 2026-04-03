#!/usr/bin/env python3
"""
Generate a 2D synthetic turbulence-like field from random sine modes.

Idea
----
1. Baseline field:
   A sum of pure sine modes
       sin(kx * x) * sin(ky * y)
   which is exactly zero on all boundaries.

2. Perturbed field:
   Add a small random phase shift to each mode
       sin(kx * x + phix) * sin(ky * y + phiy)
   so the field deviates slightly from the zero-boundary condition.

This is useful for testing Poisson solvers with:
- exact homogeneous Dirichlet boundary data
- slightly non-homogeneous boundary data

Usage
-----
python synthetic_turbulence.py

Optional:
python synthetic_turbulence.py --nx 128 --ny 128 --nmodes 80 --phase 0.08 --seed 42
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class FieldData:
    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    field_zero_bc: np.ndarray
    field_shifted_bc: np.ndarray
    boundary_zero_bc: dict
    boundary_shifted_bc: dict


def rms_normalize(f: np.ndarray, target_rms: float = 1.0) -> np.ndarray:
    rms = np.sqrt(np.mean(f**2))
    if rms == 0.0:
        return f.copy()
    return f * (target_rms / rms)


def get_boundaries(f: np.ndarray) -> dict:
    return {
        "left": f[:, 0].copy(),
        "right": f[:, -1].copy(),
        "bottom": f[0, :].copy(),
        "top": f[-1, :].copy(),
    }


def generate_synthetic_turbulence(
    nx: int = 1024,
    ny: int = 1024,
    lx: float = 1.0,
    ly: float = 1.0,
    nmodes: int = 128,
    phase_amplitude: float = 0.05,
    seed: int | None = 1234,
    decay_power: float = 5/3,
) -> FieldData:
    """
    Build two fields on [0, lx] x [0, ly]:

    field_zero_bc:
        Sum of sine modes with exact zero boundary values.

    field_shifted_bc:
        Same type of field, but with small random phase shifts so that
        the boundary is no longer exactly zero.

    Parameters
    ----------
    nx, ny
        Number of grid points in x and y.
    lx, ly
        Domain lengths.
    nmodes
        Number of random sine components.
    phase_amplitude
        Maximum magnitude of random phase shift, in radians.
    seed
        RNG seed.
    decay_power
        Spectral amplitude decay exponent. Larger means smoother field.
    """
    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, lx, nx)
    y = np.linspace(0.0, ly, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")

    field_zero_bc = np.zeros((ny, nx), dtype=np.float64)
    field_shifted_bc = np.zeros((ny, nx), dtype=np.float64)

    # Build random sine modes.
    # Pure sine basis ensures exact zero at x=0,lx and y=0,ly.
    for _ in range(nmodes):
        mx = rng.integers(1, nmodes//2)
        my = rng.integers(1, nmodes//2)

        kx = mx * np.pi / lx
        ky = my * np.pi / ly

        # Random amplitude with spectral decay
        amp = rng.normal() / ((mx**2 + my**2) ** (0.5 * decay_power))

        # Baseline: exact zero boundary
        mode_zero = np.sin(kx * X) * np.sin(ky * Y)

        # Perturbed: small random phase shift breaks exact zero boundary
        phix = rng.uniform(-phase_amplitude, phase_amplitude)
        phiy = rng.uniform(-phase_amplitude, phase_amplitude)
        mode_shifted = np.sin(kx * X + phix) * np.sin(ky * Y + phiy)

        field_zero_bc += amp * mode_zero
        field_shifted_bc += amp * mode_shifted

    # Normalize both for easier comparison
    field_zero_bc = rms_normalize(field_zero_bc, target_rms=1.0)
    field_shifted_bc = rms_normalize(field_shifted_bc, target_rms=1.0)

    return FieldData(
        x=x,
        y=y,
        X=X,
        Y=Y,
        field_zero_bc=field_zero_bc,
        field_shifted_bc=field_shifted_bc,
        boundary_zero_bc=get_boundaries(field_zero_bc),
        boundary_shifted_bc=get_boundaries(field_shifted_bc),
    )


def boundary_report(name: str, bd: dict) -> None:
    print(f"\n{name}")
    for side, values in bd.items():
        print(
            f"  {side:>6s}: "
            f"min={values.min(): .3e}, "
            f"max={values.max(): .3e}, "
            f"rms={np.sqrt(np.mean(values**2)): .3e}"
        )


def plot_fields(data: FieldData) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    im0 = axes[0].imshow(
        data.field_zero_bc,
        origin="lower",
        extent=[data.x[0], data.x[-1], data.y[0], data.y[-1]],
        aspect="auto",
    )
    axes[0].set_title("Strict zero boundary field")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        data.field_shifted_bc,
        origin="lower",
        extent=[data.x[0], data.x[-1], data.y[0], data.y[-1]],
        aspect="auto",
    )
    axes[1].set_title("Shifted-phase field")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1])

    diff = data.field_shifted_bc - data.field_zero_bc
    im2 = axes[2].imshow(
        diff,
        origin="lower",
        extent=[data.x[0], data.x[-1], data.y[0], data.y[-1]],
        aspect="auto",
    )
    axes[2].set_title("Difference")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2])

    plt.show()


def save_outputs(data: FieldData, prefix: str) -> None:
    np.savez(
        f"{prefix}.npz",
        x=data.x,
        y=data.y,
        field_zero_bc=data.field_zero_bc,
        field_shifted_bc=data.field_shifted_bc,
        left_zero=data.boundary_zero_bc["left"],
        right_zero=data.boundary_zero_bc["right"],
        bottom_zero=data.boundary_zero_bc["bottom"],
        top_zero=data.boundary_zero_bc["top"],
        left_shifted=data.boundary_shifted_bc["left"],
        right_shifted=data.boundary_shifted_bc["right"],
        bottom_shifted=data.boundary_shifted_bc["bottom"],
        top_shifted=data.boundary_shifted_bc["top"],
    )
    print(f"\nSaved data to {prefix}.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic turbulence-like fields.")
    parser.add_argument("--nx", type=int, default=128, help="Number of grid points in x.")
    parser.add_argument("--ny", type=int, default=128, help="Number of grid points in y.")
    parser.add_argument("--lx", type=float, default=1.0, help="Domain length in x.")
    parser.add_argument("--ly", type=float, default=1.0, help="Domain length in y.")
    parser.add_argument("--nmodes", type=int, default=64, help="Number of random sine modes.")
    parser.add_argument(
        "--phase",
        type=float,
        default=0.05,
        help="Maximum phase shift magnitude in radians.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument(
        "--decay",
        type=float,
        default=2.0,
        help="Spectral decay exponent; larger gives smoother fields.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synthetic_turbulence",
        help="Output prefix for .npz data.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = generate_synthetic_turbulence(
        nx=args.nx,
        ny=args.ny,
        lx=args.lx,
        ly=args.ly,
        nmodes=args.nmodes,
        phase_amplitude=args.phase,
        seed=args.seed,
        decay_power=args.decay,
    )

    boundary_report("Zero-BC field boundary statistics", data.boundary_zero_bc)
    boundary_report("Shifted-phase field boundary statistics", data.boundary_shifted_bc)

    save_outputs(data, args.output)

    if not args.no_plot:
        plot_fields(data)


if __name__ == "__main__":
    main()