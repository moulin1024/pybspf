from pathlib import Path
import sys

cwd = Path.cwd().resolve()
REPO_ROOT = next((parent for parent in (cwd, *cwd.parents) if (parent / "pyproject.toml").exists()), cwd)
SRC_ROOT = REPO_ROOT / "src"

for candidate in (SRC_ROOT, REPO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dstn, idstn

from pybspf import Poisson2DDirichletSolver

NX = 512
NY = 512
DOMAIN_X = [0.0, 2.0 * np.pi]
DOMAIN_Y = [0.0, 2.0 * np.pi]

DEGREE = 6
N_BASIS = 60

TURB_N_MODES = 10
TURB_POWER_LAW = -5.0 / 3.0
TURB_SEED = 123
PHASE_MAGNITUDE = 0.5

SHOCK_CENTER_X = np.pi
SHOCK_CENTER_Y = np.pi
SHOCK_RADIUS = 1.0
SHOCK_AMPLITUDE = 1.0
SHOCK_WIDTH = 0.05


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
    u_shock = 0.0 # shock

    r_safe = np.where(r > 1.0e-10, r, 1.0e-10)
    sech2 = 1.0 / np.cosh((r - radius) / width) ** 2
    factor = amplitude / (2.0 * width) * sech2

    d_factor_dx = -amplitude / (width**2) * sech2 * step * (X - center_x) / r_safe
    d_factor_dy = -amplitude / (width**2) * sech2 * step * (Y - center_y) / r_safe

    shock_xx = 0.0 # factor * (1.0 / r_safe - (X - center_x) ** 2 / r_safe**3) + d_factor_dx * (X - center_x) / r_safe
    shock_yy = 0.0 # factor * (1.0 / r_safe - (Y - center_y) ** 2 / r_safe**3) + d_factor_dy * (Y - center_y) / r_safe

    return u + u_shock, u_xx + shock_xx, u_yy + shock_yy

x = np.linspace(DOMAIN_X[0], DOMAIN_X[1], NX)
y = np.linspace(DOMAIN_Y[0], DOMAIN_Y[1], NY)
xx, yy = np.meshgrid(x, y)
Lx = DOMAIN_X[1] - DOMAIN_X[0]
Ly = DOMAIN_Y[1] - DOMAIN_Y[0]

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
rhs = lap_exact

solver = Poisson2DDirichletSolver.from_grids(
    x=x,
    y=y,
    degree_x=DEGREE,
    degree_y=DEGREE,
    n_basis_x=N_BASIS,
    n_basis_y=N_BASIS,
)

# Use the spline boundary corrector plus zero-Dirichlet DST remainder solve.
N_STRIP = NX // 10
STRIP_WEIGHT = 1.0

u_num, lifting_sol, _dst_remainder, lap_lift, _ = solver.solve_dst_corrected_02(
    rhs,
    left=u_exact[:, 0],
    right=u_exact[:, -1],
    bottom=u_exact[0, :],
    top=u_exact[-1, :],
    n_strip=N_STRIP,
    strip_weight=STRIP_WEIGHT,
)
error = u_num - u_exact
error_norm = np.sqrt(np.mean(error**2))
exact_norm = np.sqrt(np.mean(u_exact**2))
print(f"B-spline+DST error L2 rms     : {error_norm:.6e}")
print(f"B-spline+DST relative L2 error: {error_norm / exact_norm:.6e}")

# --- Direct DST with bilinear lift ---
hx = x[1] - x[0]
hy = y[1] - y[0]
nx_int = NX - 2
ny_int = NY - 2
lx_dst = hx * (nx_int + 1)
ly_dst = hy * (ny_int + 1)
kx_dst = np.arange(1, nx_int + 1, dtype=np.float64)
ky_dst = np.arange(1, ny_int + 1, dtype=np.float64)
eig_x = (np.pi * kx_dst / lx_dst) ** 2
eig_y = (np.pi * ky_dst / ly_dst) ** 2
denom_dst = eig_y[:, None] + eig_x[None, :]

# Bilinear lift: g = (1-s)*L(y) + s*R(y) + (1-t)*B(x) + t*T(x) - corner blend
# s = (x - x0) / Lx,  t = (y - y0) / Ly
s = (xx - x[0]) / (x[-1] - x[0])   # shape (NY, NX)
t = (yy - y[0]) / (y[-1] - y[0])
bc_left   = u_exact[:, 0]   # shape (NY,)
bc_right  = u_exact[:, -1]
bc_bottom = u_exact[0, :]   # shape (NX,)
bc_top    = u_exact[-1, :]
# Corner blend removes double-counting at corners
corner_bl = bc_left[0];  corner_br = bc_right[0]
corner_tl = bc_left[-1]; corner_tr = bc_right[-1]
corner_blend = ((1 - s) * (1 - t) * corner_bl + s * (1 - t) * corner_br
              + (1 - s) * t       * corner_tl + s * t       * corner_tr)
lift = ((1 - s) * bc_left[:, None].T.reshape(NY, 1)  # broadcasts over x
      + s       * bc_right[:, None].T.reshape(NY, 1)
      + (1 - t) * bc_bottom[None, :]
      + t       * bc_top[None, :]
      - corner_blend)

# Laplacian of lift: Δg = (1-s)*L''(y) + s*R''(y) + (1-t)*B''(x) + t*T''(x)
left_yy   = np.gradient(np.gradient(bc_left,   hy), hy)
right_yy  = np.gradient(np.gradient(bc_right,  hy), hy)
bottom_xx = np.gradient(np.gradient(bc_bottom, hx), hx)
top_xx    = np.gradient(np.gradient(bc_top,    hx), hx)
lap_lift_dst = ((1 - s) * left_yy[:, None].T.reshape(NY, 1)
              + s       * right_yy[:, None].T.reshape(NY, 1)
              + (1 - t) * bottom_xx[None, :]
              + t       * top_xx[None, :])

# Corrected RHS: Δv = f - Δg  (since Δu = f and u = v + g)
rhs_corrected_dst = rhs - lap_lift_dst
rhs_hat = dstn(rhs_corrected_dst[1:-1, 1:-1], type=1, norm="ortho")
v_dst_interior = idstn(-rhs_hat / denom_dst, type=1, norm="ortho")
u_dst = lift.copy()
u_dst[1:-1, 1:-1] = v_dst_interior + lift[1:-1, 1:-1]

error_dst = u_dst - u_exact
error_dst_norm = np.sqrt(np.mean(error_dst**2))
print(f"Direct DST error L2 rms       : {error_dst_norm:.6e}")
print(f"Direct DST relative L2 error  : {error_dst_norm / exact_norm:.6e}")
# --- End direct DST with bilinear lift ---

d2u_dx2_result = solver.x_model.derivatives_batched(u_num.T, orders=2)
d2u_dx2 = d2u_dx2_result[2].T
d2u_dy2_result = solver.y_model.derivatives_batched(u_num, orders=2)
d2u_dy2 = d2u_dy2_result[2]
lap_num = d2u_dx2 + d2u_dy2

extent = [x[0], x[-1], y[0], y[-1]]

fig1, axes1 = plt.subplots(2, 3, figsize=(15, 8.0), constrained_layout=True)
for ax, (field, title) in zip(axes1.flat, [
    (u_exact,             "Exact Solution"),
    (u_num,               "B-spline+DST Solution"),
    (error,               "B-spline+DST Error"),
    (u_dst,               "Direct DST Solution"),
    (error_dst,           "Direct DST Error"),
    (u_num - u_dst,       "B-spline+DST vs Direct DST"),
]):
    im = ax.imshow(field, origin="lower", extent=extent, aspect="auto", cmap="coolwarm")
    fig1.colorbar(im, ax=ax, shrink=0.85, format="%.2e")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
plt.show()

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
for ax, (field, title) in zip(axes2.flat, [
    (lifting_sol, "B-spline Lifting"),
    (lap_lift,     "Lifting Laplacian"),
]):
    im = ax.imshow(field, origin="lower", extent=extent, aspect="auto", cmap="coolwarm")
    fig2.colorbar(im, ax=ax, shrink=0.85, format="%.2e")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
plt.show()
