# test_bspf2d_heat_neumann.py
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

try:
    from pybspf import BSPF2D, integrate_rk4
except ModuleNotFoundError:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    from pybspf import BSPF2D, integrate_rk4

try:
    from tqdm import tqdm
    def _iter(it, total): return tqdm(it, total=total)
except Exception:
    def _iter(it, total): return it


def exact_one_mode_neumann(x: np.ndarray, y: np.ndarray, t: float, nu: float,
                           Lx: float, Ly: float, A: float, mean: float = 1.0) -> np.ndarray:
    """
    Exact solution for IC: u(x,y,0) = mean + A cos(pi x/Lx) cos(pi y/Ly) with zero-flux BCs.
    u(x,y,t) = mean + A * exp(-nu*((pi/Lx)^2 + (pi/Ly)^2)*t) * cos(pi x/Lx) cos(pi y/Ly)
    """
    kx = np.pi / Lx
    ky = np.pi / Ly
    decay = np.exp(-nu * ((kx**2) + (ky**2)) * t)
    X, Y = np.meshgrid(x, y)  # y rows, x cols (ny, nx)
    return mean + A * decay * np.cos(kx * X) * np.cos(ky * Y)


def create_heat_rhs_2d(
    op: BSPF2D,
    nu: float,
    *,
    flux_x: tuple[float | None, float | None] = (0.0, 0.0),
    flux_y: tuple[float | None, float | None] = (0.0, 0.0),
    lam_x: float = 0.0,
    lam_y: float = 0.0,
):
    """Build the heat-equation RHS from the package BSPF2D operator."""

    def rhs(_t: float, field: np.ndarray) -> np.ndarray:
        return nu * op.laplacian(
            field,
            lam_x=lam_x,
            lam_y=lam_y,
            neumann_bc_x=flux_x,
            neumann_bc_y=flux_y,
        )

    return rhs


def main():
    # ------------------ grid & params ------------------
    Lx, Ly = 1.0, 1.0
    nx, ny = 64, 64
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    nu = 1e-2
    Tfinal = 0.2
    
    dt = 0.1 * (min(dx, dy) ** 2) / nu
    nsteps = int(np.ceil(Tfinal / dt))
    dt = Tfinal / nsteps  # end exactly at Tfinal

    # ------------------ initial condition (one mode) ------------------
    A = 0.5
    mean = 1.0
    X, Y = np.meshgrid(x, y)  # (ny, nx)
    U0 = mean + A * np.cos(np.pi * X / Lx) * np.cos(np.pi * Y / Ly)

    # ------------------ operator & RHS ------------------
    # Use degree high enough so model.end.order >= 2 (Neumann requires order>=2).
    # bspf1d defaults order = degree-1, so degree >= 3 is fine; we pick 8 for comfort.
    op = BSPF2D.from_grids(x=x, y=y, degree_x=8, degree_y=8, correction="spectral")

    # Create heat-equation RHS from the package operator.
    rhs_func = create_heat_rhs_2d(op, nu, flux_x=(0.0, 0.0), flux_y=(0.0, 0.0))

    # ------------------ time integration ------------------
    flux_tol = 1e-8
    check_stride = max(1, nsteps // 20)

    # Mass (heat) should be conserved for zero-flux BCs
    cell_area = dx * dy
    mass0 = float(np.sum(U0) * cell_area)

    t0 = time.time()
    t_eval = np.linspace(0.0, Tfinal, nsteps + 1)
    history = integrate_rk4(rhs_func, U0, t_eval, dt=dt)

    for step in _iter(range(1, nsteps + 1), total=nsteps):
        U = history[step]

        # check boundary flux occasionally
        if (step % check_stride == 0) or (step == nsteps):
            Ux, _ = op.partial_x(U, order=1, lam=0.0, neumann_bc=(0.0, 0.0))
            Uy, _ = op.partial_y(U, order=1, lam=0.0, neumann_bc=(0.0, 0.0))
            # endpoints: x-direction -> first/last columns; y-direction -> first/last rows
            max_flux = max(
                float(np.max(np.abs(Ux[:, 0]))),
                float(np.max(np.abs(Ux[:, -1]))),
                float(np.max(np.abs(Uy[0, :]))),
                float(np.max(np.abs(Uy[-1, :]))),
            )
            if max_flux > flux_tol:
                print(f"[warn] flux check @ step {step}: |grad·n|_max = {max_flux:.2e} > {flux_tol:.2e}")

    wall = time.time() - t0

    # ------------------ diagnostics ------------------
    mass_final = float(np.sum(U) * cell_area)
    mass_err = abs(mass_final - mass0)

    U_exact = exact_one_mode_neumann(x, y, Tfinal, nu, Lx, Ly, A, mean)
    L2 = float(np.sqrt(np.mean((U - U_exact) ** 2)))
    Linf = float(np.max(np.abs(U - U_exact)))

    print("\n--- run summary ---")
    print(f"grid           : ny={ny}, nx={nx}")
    print(f"dt, steps      : {dt:.3e}, {nsteps}")
    print(f"runtime        : {wall:.2f} s")
    print(f"mass initial   : {mass0:.12e}")
    print(f"mass final     : {mass_final:.12e}")
    print(f"|Δmass|        : {mass_err:.3e}")
    print(f"L2 error       : {L2:.3e}")
    print(f"L∞ error       : {Linf:.3e}")

    # quick plot (optional)
    try:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        im0 = axs[0].imshow(U0, origin='lower', extent=[0, Lx, 0, Ly]); axs[0].set_title('U0')
        im1 = axs[1].imshow(U, origin='lower', extent=[0, Lx, 0, Ly]); axs[1].set_title('U (num)')
        im2 = axs[2].imshow(U_exact, origin='lower', extent=[0, Lx, 0, Ly]); axs[2].set_title('U (exact)')
        for ax in axs:
            ax.set_xlabel('x'); ax.set_ylabel('y')
        for im in (im0, im1, im2):
            fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.85)
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
