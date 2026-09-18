"""Shared BSPF flow kernels, usable with NumPy or JAX arrays.

These operations were factored out of the existing stream NS / pressure
solvers. Geometry-specific loads and boundary conditions stay with their plans.
"""


def tensor_product(left, right, coefficients=None, *, paired=False):
    if paired:
        if coefficients is None:
            return (left[:, :, None] * right[:, None, :]).reshape(len(left), -1)
        return ((left @ coefficients) * right).sum(axis=1)
    if coefficients is None:
        raise ValueError("Tensor-grid evaluation requires coefficients")
    return left @ coefficients @ right.T


def curl_from_gradient(dx, dy, *, radius=None):
    """Cartesian NS curl or axisymmetric poloidal curl, with common derivatives."""
    if radius is None:
        return dy, -dx
    return -dy / radius, dx / radius


def tensor_elliptic_solve(load, denominator, left=None, right=None):
    """Generalized symmetric tensor Poisson inverse in mass-normalized modes.

    Rotations have columns of generalized eigenvectors. Identity rotations
    recover the original stream NS diagonal inertia solve exactly.
    """
    transformed = load if left is None else left.T @ load
    if right is not None:
        transformed = transformed @ right
    result = transformed / denominator
    if left is not None:
        result = left @ result
    if right is not None:
        result = result @ right.T
    return result


def rk4_stages(state, dt, rhs):
    """Original NS RK4 stages; rhs returns (derivative, stage diagnostics)."""
    a, da = rhs(state)
    b, db = rhs(state + dt / 2 * a)
    c, dc = rhs(state + dt / 2 * b)
    d, dd = rhs(state + dt * c)
    return state + dt / 6 * (a + 2 * b + 2 * c + d), (da, db, dc, dd)


def imex_midpoint(state, time, dt, mass, diffusion, explicit, solve):
    """Cavity IMEX midpoint, with caller-supplied linear operators/factorization.

    diffusion is the positive viscous operator including viscosity; solve
    inverts M + dt/2 D. Works with either NumPy or JAX arrays.
    """
    momentum = mass(state)
    half = solve(momentum + dt / 2 * explicit(state, time))
    return solve(
        momentum - dt / 2 * diffusion(state) + dt * explicit(half, time + dt / 2)
    )
