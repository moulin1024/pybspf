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
    """Shared flow RK4 for array/PyTree states and per-stage diagnostics.

    Tree mapping preserves NumPy leaves for host callers and JAX leaves for
    compiled callers; coupled fields need not share a shape or be packed.
    """
    from jax.tree_util import tree_map

    def add(y, derivative, h):
        return tree_map(lambda v, k: v+h*k, y, derivative)

    a, da = rhs(state)
    b, db = rhs(add(state, a, dt/2))
    c, dc = rhs(add(state, b, dt/2))
    d, dd = rhs(add(state, c, dt))
    result = tree_map(lambda y, ka, kb, kc, kd: y+dt/6*(ka+2*kb+2*kc+kd),
                      state, a, b, c, d)
    return result, (da, db, dc, dd)


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
