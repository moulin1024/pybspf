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
from ngsolve import CoefficientFunction, VoxelCoefficient, cosh, sin, sinh, sqrt, x, y

from pybspf import Poisson2DDirichletSolver
from pybspf.solvers.poisson2d_ngsolve import NGSolvePoisson2DBaselineSolver

NX = 32
NY = 32
DOMAIN_X = [0.0, 2.0 * np.pi]
DOMAIN_Y = [0.0, 2.0 * np.pi]

ORDER = 4

TURB_N_MODES = 32
TURB_POWER_LAW = -5.0 / 3.0
TURB_SEED = 123
PHASE_MAGNITUDE = 0.5

SHOCK_CENTER_X = np.pi
SHOCK_CENTER_Y = np.pi
SHOCK_RADIUS = 1.0
SHOCK_AMPLITUDE = 1.0
SHOCK_WIDTH = 0.05

PRECONDITIONER = "multigrid"
CG_TOL = 1.0e-10
CG_MAXSTEPS = 500
RESIDUAL_SAMPLE_NX = NX + 1
RESIDUAL_SAMPLE_NY = NY + 1

BSPF_DEGREE = 5
BSPF_N_BASIS = 50
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


def evaluate_turbulence_field_on_grid(x_grid, y_grid, Lx, Ly, n_modes, seed, power_law=TURB_POWER_LAW):
    params = _get_turbulence_parameters(Lx, Ly, n_modes, seed, power_law=power_law)

    x_grid = np.asarray(x_grid, dtype=np.float64)
    y_grid = np.asarray(y_grid, dtype=np.float64)
    kx = params["kx"]
    ky = params["ky"]
    amplitude = params["amplitude"]
    phase_x = params["phase_x"]
    phase_y = params["phase_y"]

    sin_kx_x = np.sin(np.outer(kx, x_grid) + phase_x[:, None])
    sin_ky_y = np.sin(np.outer(ky, y_grid) + phase_y[:, None])

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


def build_exact_coefficient_functions(Lx, Ly, n_modes, seed, power_law=TURB_POWER_LAW):
    params = _get_turbulence_parameters(Lx, Ly, n_modes, seed, power_law=power_law)
    exact_terms = []
    laplace_terms = []
    for amp, kx, ky, phase_x, phase_y in zip(
        params["amplitude"],
        params["kx"],
        params["ky"],
        params["phase_x"],
        params["phase_y"],
    ):
        mode = sin(float(kx) * x + float(phase_x)) * sin(float(ky) * y + float(phase_y))
        exact_terms.append(float(amp) * mode)
        laplace_terms.append(float(-amp * (kx**2 + ky**2)) * mode)

    eps = 1.0e-10
    rx = x - SHOCK_CENTER_X
    ry = y - SHOCK_CENTER_Y
    r = sqrt(rx * rx + ry * ry)
    r_safe = sqrt(rx * rx + ry * ry + eps**2)
    step = sinh((r - SHOCK_RADIUS) / SHOCK_WIDTH) / cosh((r - SHOCK_RADIUS) / SHOCK_WIDTH)
    shock = SHOCK_AMPLITUDE * (step + 1.0) / 2.0
    sech2 = 1.0 / cosh((r - SHOCK_RADIUS) / SHOCK_WIDTH) ** 2
    factor = SHOCK_AMPLITUDE / (2.0 * SHOCK_WIDTH) * sech2

    d_factor_dx = -SHOCK_AMPLITUDE / (SHOCK_WIDTH**2) * sech2 * step * rx / r_safe
    d_factor_dy = -SHOCK_AMPLITUDE / (SHOCK_WIDTH**2) * sech2 * step * ry / r_safe

    shock_xx = factor * (1.0 / r_safe - rx * rx / r_safe**3) + d_factor_dx * rx / r_safe
    shock_yy = factor * (1.0 / r_safe - ry * ry / r_safe**3) + d_factor_dy * ry / r_safe

    exact_terms.append(shock)
    laplace_terms.append(shock_xx + shock_yy)
    return CoefficientFunction(sum(exact_terms)), CoefficientFunction(sum(laplace_terms))


def sample_laplacian_on_grid(solution, solver, x_grid, y_grid):
    hesse = solution.Operator("hesse")
    lap = np.empty((y_grid.size, x_grid.size), dtype=np.float64)
    for j, yj in enumerate(y_grid):
        for i, xi in enumerate(x_grid):
            h_xx, _, _, h_yy = hesse(solver.mesh(float(xi), float(yj)))
            lap[j, i] = h_xx + h_yy
    return lap


ngsolve_solver = NGSolvePoisson2DBaselineSolver.from_domain(
    domain_x=(DOMAIN_X[0], DOMAIN_X[1]),
    domain_y=(DOMAIN_Y[0], DOMAIN_Y[1]),
    n_elem_x=NX,
    n_elem_y=NY,
    order=ORDER,
)

Lx = DOMAIN_X[1] - DOMAIN_X[0]
Ly = DOMAIN_Y[1] - DOMAIN_Y[0]
x_eval = np.linspace(DOMAIN_X[0], DOMAIN_X[1], ORDER * NX + 1)
y_eval = np.linspace(DOMAIN_Y[0], DOMAIN_Y[1], ORDER * NY + 1)
xx, yy = np.meshgrid(x_eval, y_eval, indexing="xy")

u_exact, u_xx_exact, u_yy_exact = evaluate_turbulence_field_on_grid(
    x_eval,
    y_eval,
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

u_exact_cf, lap_exact_cf = build_exact_coefficient_functions(
    Lx,
    Ly,
    TURB_N_MODES,
    TURB_SEED,
)

start = time.perf_counter()
solution_cold, meta_cold = ngsolve_solver.solve(
    rhs_cf=lap_exact_cf,
    dirichlet_cf=u_exact_cf,
    preconditioner=PRECONDITIONER,
    tol=CG_TOL,
    maxsteps=CG_MAXSTEPS,
)
meta_cold["wall_time_s"] = time.perf_counter() - start
u_num_cold = ngsolve_solver.sample_on_grid(solution_cold, x_eval, y_eval)

bspf_solver = Poisson2DDirichletSolver.from_grids(
    x=x_eval,
    y=y_eval,
    degree_x=BSPF_DEGREE,
    degree_y=BSPF_DEGREE,
    n_basis_x=BSPF_N_BASIS,
    n_basis_y=BSPF_N_BASIS,
)
bspf_n_strip = max(1, x_eval.size // 20)
u_bspf, _, _, _, _ = bspf_solver.solve_dst_corrected_02(
    lap_exact,
    left=u_exact[:, 0],
    right=u_exact[:, -1],
    bottom=u_exact[0, :],
    top=u_exact[-1, :],
    second_normal_weight=BSPF_SECOND_NORMAL_WEIGHT,
    n_strip=bspf_n_strip,
    strip_weight=BSPF_STRIP_WEIGHT,
)
bspf_voxel = VoxelCoefficient(
    (DOMAIN_X[0], DOMAIN_Y[0]),
    (DOMAIN_X[1], DOMAIN_Y[1]),
    u_bspf,
    linear=True,
)

start = time.perf_counter()
solution_warm, meta_warm = ngsolve_solver.solve(
    rhs_cf=lap_exact_cf,
    dirichlet_cf=u_exact_cf,
    initial_guess_cf=bspf_voxel,
    preconditioner=PRECONDITIONER,
    tol=CG_TOL,
    maxsteps=CG_MAXSTEPS,
)
meta_warm["wall_time_s"] = time.perf_counter() - start
u_num_warm = ngsolve_solver.sample_on_grid(solution_warm, x_eval, y_eval)

error_bspf = u_bspf - u_exact
error_cold = u_num_cold - u_exact
error_warm = u_num_warm - u_exact
exact_norm = np.sqrt(np.mean(u_exact**2))

x_res = np.linspace(DOMAIN_X[0], DOMAIN_X[1], RESIDUAL_SAMPLE_NX)
y_res = np.linspace(DOMAIN_Y[0], DOMAIN_Y[1], RESIDUAL_SAMPLE_NY)
xx_res, yy_res = np.meshgrid(x_res, y_res, indexing="xy")
_u_res, u_xx_res, u_yy_res = evaluate_turbulence_field_on_grid(
    x_res,
    y_res,
    Lx,
    Ly,
    TURB_N_MODES,
    TURB_SEED,
)
_u_res, u_xx_res, u_yy_res = add_circular_shock_wave(
    xx_res,
    yy_res,
    _u_res,
    u_xx_res,
    u_yy_res,
    center_x=SHOCK_CENTER_X,
    center_y=SHOCK_CENTER_Y,
    radius=SHOCK_RADIUS,
    amplitude=SHOCK_AMPLITUDE,
    width=SHOCK_WIDTH,
)
lap_exact_res = u_xx_res + u_yy_res
lap_cold_res = sample_laplacian_on_grid(solution_cold, ngsolve_solver, x_res, y_res)
lap_warm_res = sample_laplacian_on_grid(solution_warm, ngsolve_solver, x_res, y_res)
residual_cold = lap_exact_res - lap_cold_res
residual_warm = lap_exact_res - lap_warm_res

print(f"BSPF warm-start error L2 rms      : {np.sqrt(np.mean(error_bspf**2)):.6e}")
print(f"BSPF warm-start relative L2 error : {np.sqrt(np.mean(error_bspf**2)) / exact_norm:.6e}")
print(f"NGSolve cold error L2 rms         : {np.sqrt(np.mean(error_cold**2)):.6e}")
print(f"NGSolve cold relative L2 error    : {np.sqrt(np.mean(error_cold**2)) / exact_norm:.6e}")
print(f"NGSolve warm error L2 rms         : {np.sqrt(np.mean(error_warm**2)):.6e}")
print(f"NGSolve warm relative L2 error    : {np.sqrt(np.mean(error_warm**2)) / exact_norm:.6e}")
print(f"NGSolve cold CG steps             : {meta_cold['cg_steps']}")
print(f"NGSolve warm CG steps             : {meta_warm['cg_steps']}")
print(f"NGSolve cold initial residual     : {meta_cold['initial_residual_norm']:.6e}")
print(f"NGSolve warm initial residual     : {meta_warm['initial_residual_norm']:.6e}")
print(f"NGSolve cold final residual       : {meta_cold['final_residual_norm']:.6e}")
print(f"NGSolve warm final residual       : {meta_warm['final_residual_norm']:.6e}")
print(f"NGSolve cold wall time [s]        : {meta_cold['wall_time_s']:.6e}")
print(f"NGSolve warm wall time [s]        : {meta_warm['wall_time_s']:.6e}")

extent = [x_eval[0], x_eval[-1], y_eval[0], y_eval[-1]]
fig1, axes1 = plt.subplots(2, 3, figsize=(15, 8.0), constrained_layout=True)
for ax, field, title in (
    (axes1[0, 0], u_exact, "Exact Solution"),
    (axes1[0, 1], u_bspf, "BSPF Warm Start"),
    (axes1[0, 2], error_bspf, "BSPF Warm-Start Error"),
    (axes1[1, 0], u_num_cold, "NGSolve Cold Solution"),
    (axes1[1, 1], u_num_warm, "NGSolve Warm Solution"),
    (axes1[1, 2], error_warm, "NGSolve Warm Error"),
):
    im = ax.imshow(field, origin="lower", extent=extent, aspect="auto", cmap="coolwarm")
    fig1.colorbar(im, ax=ax, shrink=0.85, format="%.2e")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
plt.show()

extent_res = [x_res[0], x_res[-1], y_res[0], y_res[-1]]
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for ax, field, title in (
    (axes2[0], residual_cold, "NGSolve Cold Residual"),
    (axes2[1], residual_warm, "NGSolve Warm Residual"),
):
    im = ax.imshow(field, origin="lower", extent=extent_res, aspect="auto", cmap="coolwarm")
    fig2.colorbar(im, ax=ax, shrink=0.85, format="%.2e")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
plt.show()
