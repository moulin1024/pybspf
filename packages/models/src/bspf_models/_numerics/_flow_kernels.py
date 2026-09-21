"""Shared BSPF flow kernels, usable with NumPy or JAX arrays.

These operations were factored out of the existing stream NS / pressure
solvers. Geometry-specific loads and boundary conditions stay with their plans.
"""


def curl_from_gradient(dx, dy, *, radius=None):
    """Cartesian NS curl or axisymmetric poloidal curl, with common derivatives."""
    if radius is None:
        return dy, -dx
    return -dy / radius, dx / radius
