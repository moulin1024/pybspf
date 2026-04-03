from pathlib import Path
import sys
import time

cwd = Path.cwd().resolve()
REPO_ROOT = next((parent for parent in (cwd, *cwd.parents) if (parent / "pyproject.toml").exists()), cwd)
SRC_ROOT = REPO_ROOT / "src"

for candidate in (SRC_ROOT, REPO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import matplotlib.pyplot as plt
import numpy as np
import pyamg
from scipy.fft import dstn, idstn
from scipy import sparse
from scipy.sparse import linalg as spla

from pybspf import Poisson2DDirichletSolver

NX = 1024
NY = 1024
DOMAIN_X = [0.0, 2.0 * np.pi]
DOMAIN_Y = [0.0, 2.0 * np.pi]

TURB_N_MODES = 20
TURB_POWER_LAW = -5.0 / 3.0
TURB_SEED = 123
PHASE_MAGNITUDE = 0.5

SHOCK_CENTER_X = np.pi
SHOCK_CENTER_Y = np.pi
SHOCK_RADIUS = 1.0
SHOCK_AMPLITUDE = 1.0
SHOCK_WIDTH = 0.05

CG_TOL = 1.0e-10
CG_MAXITER = 1000

BSPF_PRECOND_DEGREE = 3
BSPF_PRECOND_N_BASIS = 2 * BSPF_PRECOND_DEGREE
BSPF_SECOND_NORMAL_WEIGHT = 100.0
BSPF_STRIP_WEIGHT = 1.0


_turbulence_params_cache = {}


def _get_turbulence_parameters(Lx, Ly, n_modes, seed, power_law=TURB_POWER_LAW):
    cache_key = (Lx, Ly, n_modes, seed, power_law)
    if cache_key in _turbulence_params_cache:
        return _turbulence_params_cache[cache_key]

    rng = np.random.default_rng(seed)
    n = np.arange(1, n_modes + 1, dtype=np.float64)
    m = np.arange(1, n_modes + 1, dtype=np.float64)
    kx = n * np.pi / Lx
    ky = m * np.pi / Ly
    k_mag = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    sigma = k_mag ** (0.5 * power_law)

    n_total = n_modes * n_modes
    indices = rng.choice(n_total, size=min(n_modes, n_total), replace=False)
    i_flat, j_flat = np.unravel_index(indices, (n_modes, n_modes))

    params = {
        "kx": kx[i_flat],
        "ky": ky[j_flat],
        "amplitude": sigma[i_flat, j_flat] * rng.standard_normal(len(i_flat)),
        "phase_x": PHASE_MAGNITUDE * rng.standard_normal(len(i_flat)),
        "phase_y": PHASE_MAGNITUDE * rng.standard_normal(len(i_flat)),
    }
    _turbulence_params_cache[cache_key] = params
    return params


def evaluate_turbulence_field_on_grid(x, y, Lx, Ly, n_modes, seed, power_law=TURB_POWER_LAW):
    params = _get_turbulence_parameters(Lx, Ly, n_modes, seed, power_law=power_law)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    kx = params["kx"]
    ky = params["ky"]
    amplitude = params["amplitude"]
    phase_x = params["phase_x"]
    phase_y = params["phase_y"]

    sin_kx_x = np.sin(np.outer(kx, x) + phase_x[:, None])
    sin_ky_y = np.sin(np.outer(ky, y) + phase_y[:, None])

    u = np.einsum("m,mx,my->yx", amplitude, sin_kx_x, sin_ky_y, optimize=True)
    u_xx = np.einsum("m,mx,my->yx", -amplitude * kx**2, sin_kx_x, sin_ky_y, optimize=True)
    u_yy = np.einsum("m,mx,my->yx", -amplitude * ky**2, sin_kx_x, sin_ky_y, optimize=True)
    return u, u_xx, u_yy


def add_circular_shock_wave(X, Y, u, u_xx, u_yy, center_x, center_y, radius, amplitude, width):
    r = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    step = np.tanh((r - radius) / width)
    shock = amplitude * (step + 1.0) / 2.0
    u_shock = shock

    r_safe = np.where(r > 1.0e-10, r, 1.0e-10)
    sech2 = 1.0 / np.cosh((r - radius) / width) ** 2
    factor = amplitude / (2.0 * width) * sech2

    d_factor_dx = -amplitude / (width**2) * sech2 * step * (X - center_x) / r_safe
    d_factor_dy = -amplitude / (width**2) * sech2 * step * (Y - center_y) / r_safe

    shock_xx = factor * (1.0 / r_safe - (X - center_x) ** 2 / r_safe**3) + d_factor_dx * (X - center_x) / r_safe
    shock_yy = factor * (1.0 / r_safe - (Y - center_y) ** 2 / r_safe**3) + d_factor_dy * (Y - center_y) / r_safe

    return u + u_shock, u_xx + shock_xx, u_yy + shock_yy


def build_fd_matrix(nx_int, ny_int, hx, hy):
    main_x = 2.0 / hx**2 * np.ones(nx_int)
    off_x = -1.0 / hx**2 * np.ones(nx_int - 1)
    tx = sparse.diags([off_x, main_x, off_x], [-1, 0, 1], format="csr")

    main_y = 2.0 / hy**2 * np.ones(ny_int)
    off_y = -1.0 / hy**2 * np.ones(ny_int - 1)
    ty = sparse.diags([off_y, main_y, off_y], [-1, 0, 1], format="csr")

    ix = sparse.eye(nx_int, format="csr")
    iy = sparse.eye(ny_int, format="csr")
    return sparse.kron(iy, tx, format="csr") + sparse.kron(ty, ix, format="csr")


def build_fd_rhs(rhs_grid, boundary_grid, hx, hy):
    b = -rhs_grid[1:-1, 1:-1].copy()
    b[:, 0] += boundary_grid[1:-1, 0] / hx**2
    b[:, -1] += boundary_grid[1:-1, -1] / hx**2
    b[0, :] += boundary_grid[0, 1:-1] / hy**2
    b[-1, :] += boundary_grid[-1, 1:-1] / hy**2
    return b.reshape(-1)


def make_residual_callback(A, b, history):
    b_norm = float(np.linalg.norm(b))
    denom = b_norm if b_norm > 0.0 else 1.0

    def _callback(xk):
        rk = b - A @ xk
        history.append(float(np.linalg.norm(rk) / denom))

    return _callback


def build_fd_dst_inverse_diagonal(nx_int, ny_int, hx, hy):
    kx = np.arange(1, nx_int + 1, dtype=np.float64)
    ky = np.arange(1, ny_int + 1, dtype=np.float64)
    lam_x = 4.0 * np.sin(0.5 * np.pi * kx / (nx_int + 1)) ** 2 / hx**2
    lam_y = 4.0 * np.sin(0.5 * np.pi * ky / (ny_int + 1)) ** 2 / hy**2
    return lam_y[:, None] + lam_x[None, :]


def apply_fd_dst_inverse(rhs_vec, denom, nx_int, ny_int):
    rhs_grid = rhs_vec.reshape(ny_int, nx_int)
    rhs_hat = dstn(rhs_grid, type=1, norm="ortho")
    sol_hat = rhs_hat / denom
    return idstn(sol_hat, type=1, norm="ortho").reshape(-1)


def run_cg_case(name, A, b, boundary_grid, nx_int, ny_int, *, M=None):
    history = []
    start = time.perf_counter()
    u_int, info = spla.cg(
        A,
        b,
        rtol=CG_TOL,
        atol=0.0,
        maxiter=CG_MAXITER,
        M=M,
        callback=make_residual_callback(A, b, history),
    )
    solve_time = time.perf_counter() - start
    u_full = boundary_grid.copy()
    u_full[1:-1, 1:-1] = u_int.reshape(ny_int, nx_int)
    return {
        "name": name,
        "u": u_full,
        "info": info,
        "iterations": len(history),
        "history": history,
        "solve_time_s": solve_time,
    }


x = np.linspace(DOMAIN_X[0], DOMAIN_X[1], NX, dtype=np.float64)
y = np.linspace(DOMAIN_Y[0], DOMAIN_Y[1], NY, dtype=np.float64)
xx, yy = np.meshgrid(x, y, indexing="xy")
Lx = DOMAIN_X[1] - DOMAIN_X[0]
Ly = DOMAIN_Y[1] - DOMAIN_Y[0]
hx = x[1] - x[0]
hy = y[1] - y[0]

u_exact, u_xx_exact, u_yy_exact = evaluate_turbulence_field_on_grid(
    x,
    y,
    Lx,
    Ly,
    TURB_N_MODES,
    TURB_SEED,
)
u_exact, u_xx_exact, u_yy_exact = add_circular_shock_wave(
    xx,
    yy,
    u_exact,
    u_xx_exact,
    u_yy_exact,
    center_x=SHOCK_CENTER_X,
    center_y=SHOCK_CENTER_Y,
    radius=SHOCK_RADIUS,
    amplitude=SHOCK_AMPLITUDE,
    width=SHOCK_WIDTH,
)
lap_exact = u_xx_exact + u_yy_exact

boundary_grid = np.zeros_like(u_exact)
boundary_grid[:, 0] = u_exact[:, 0]
boundary_grid[:, -1] = u_exact[:, -1]
boundary_grid[0, :] = u_exact[0, :]
boundary_grid[-1, :] = u_exact[-1, :]

nx_int = NX - 2
ny_int = NY - 2
A = build_fd_matrix(nx_int, ny_int, hx, hy)
b = build_fd_rhs(lap_exact, boundary_grid, hx, hy)

start = time.perf_counter()
precond_solver = Poisson2DDirichletSolver.from_grids(
    x=x,
    y=y,
    degree_x=BSPF_PRECOND_DEGREE,
    degree_y=BSPF_PRECOND_DEGREE,
    n_basis_x=BSPF_PRECOND_N_BASIS,
    n_basis_y=BSPF_PRECOND_N_BASIS,
)
precond_n_strip = max(1, NX // 20)
bspf_setup_time = time.perf_counter() - start


def apply_bspf_preconditioner(residual_vec):
    residual_grid = np.zeros((NY, NX), dtype=np.float64)
    residual_grid[1:-1, 1:-1] = residual_vec.reshape(ny_int, nx_int)
    z_grid, _, _, _, _ = precond_solver.solve_dst_corrected_02(
        -residual_grid,
        left=np.zeros(NY, dtype=np.float64),
        right=np.zeros(NY, dtype=np.float64),
        bottom=np.zeros(NX, dtype=np.float64),
        top=np.zeros(NX, dtype=np.float64),
        second_normal_weight=BSPF_SECOND_NORMAL_WEIGHT,
        n_strip=precond_n_strip,
        strip_weight=BSPF_STRIP_WEIGHT,
    )
    z_grid[:, 0] = 0.0
    z_grid[:, -1] = 0.0
    z_grid[0, :] = 0.0
    z_grid[-1, :] = 0.0
    return z_grid[1:-1, 1:-1].reshape(-1)


M = spla.LinearOperator((nx_int * ny_int, nx_int * ny_int), matvec=apply_bspf_preconditioner, dtype=np.float64)
bspf_result = run_cg_case("FD2+BSPF-PCG", A, b, boundary_grid, nx_int, ny_int, M=M)
bspf_result["setup_time_s"] = bspf_setup_time
bspf_result["total_time_s"] = bspf_setup_time + bspf_result["solve_time_s"]

start = time.perf_counter()
dst_denom = build_fd_dst_inverse_diagonal(nx_int, ny_int, hx, hy)
dst_setup_time = time.perf_counter() - start
M_dst = spla.LinearOperator(
    (nx_int * ny_int, nx_int * ny_int),
    matvec=lambda r: apply_fd_dst_inverse(r, dst_denom, nx_int, ny_int),
    dtype=np.float64,
)
dst_result = run_cg_case(
    "FD2+DST-PCG",
    A,
    b,
    boundary_grid,
    nx_int,
    ny_int,
    M=M_dst,
)
dst_result["setup_time_s"] = dst_setup_time
dst_result["total_time_s"] = dst_setup_time + dst_result["solve_time_s"]

start = time.perf_counter()
rs_ml = pyamg.ruge_stuben_solver(A)
rs_setup_time = time.perf_counter() - start
rs_result = run_cg_case(
    "FD2+PyAMG-RS-PCG",
    A,
    b,
    boundary_grid,
    nx_int,
    ny_int,
    M=rs_ml.aspreconditioner(),
)
rs_result["setup_time_s"] = rs_setup_time
rs_result["total_time_s"] = rs_setup_time + rs_result["solve_time_s"]

results = [bspf_result, dst_result, rs_result]
error_bspf = bspf_result["u"] - u_exact
error_dst = dst_result["u"] - u_exact
error_rs = rs_result["u"] - u_exact
exact_norm = np.sqrt(np.mean(u_exact**2))

for result in results:
    err = result["u"] - u_exact
    err_rms = np.sqrt(np.mean(err**2))
    print(f"{result['name']} error L2 rms      : {err_rms:.6e}")
    print(f"{result['name']} relative L2 err : {err_rms / exact_norm:.6e}")
    print(f"{result['name']} iterations      : {result['iterations']}")
    print(f"{result['name']} info            : {result['info']}")
    print(f"{result['name']} setup time [s]  : {result['setup_time_s']:.6e}")
    print(f"{result['name']} solve time [s]  : {result['solve_time_s']:.6e}")
    print(f"{result['name']} total time [s]  : {result['total_time_s']:.6e}")

extent = [x[0], x[-1], y[0], y[-1]]
fig1, axes1 = plt.subplots(2, 4, figsize=(18, 8.0), constrained_layout=True)
for ax, field, title in (
    (axes1[0, 0], u_exact, "Exact Solution"),
    (axes1[0, 1], bspf_result["u"], "FD2+BSPF-PCG Solution"),
    (axes1[0, 2], error_bspf, "FD2+BSPF-PCG Error"),
    (axes1[0, 3], error_dst, "FD2+DST-PCG Error"),
    (axes1[1, 0], dst_result["u"], "FD2+DST-PCG Solution"),
    (axes1[1, 1], rs_result["u"], "FD2+PyAMG-RS-PCG Solution"),
    (axes1[1, 2], error_rs, "FD2+PyAMG-RS-PCG Error"),
    (axes1[1, 3], bspf_result["u"] - rs_result["u"], "BSPF-PCG vs PyAMG-RS"),
):
    im = ax.imshow(field, origin="lower", extent=extent, aspect="auto", cmap="coolwarm")
    fig1.colorbar(im, ax=ax, shrink=0.85, format="%.2e")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
plt.show()

fig2, ax2 = plt.subplots(1, 1, figsize=(7, 4.5), constrained_layout=True)
for result in results:
    if result["history"]:
        ax2.semilogy(
            np.arange(1, len(result["history"]) + 1),
            result["history"],
            label=result["name"],
        )
ax2.set_xlabel("CG iteration")
ax2.set_ylabel(r"$\|r_k\|_2 / \|b\|_2$")
ax2.set_title("Normalized Residual Convergence")
ax2.grid(True, which="both", alpha=0.3)
ax2.legend()
plt.show()
