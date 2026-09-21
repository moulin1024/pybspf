"""Local Chebyshev endpoint estimation, assembled once into a linear map."""

import jax.numpy as jnp
import jax.scipy.linalg as jl


def chebyshev_boundary_blocks(x, *, order, points, modes, alpha, penalty_power):
    """Estimate jets 0,...,order-1 at both ends of a uniform closed grid.

    Fit on each local window with a normalized modal penalty. Augmented QR
    implements regularized least squares without forming normal equations.
    Endpoint values are copied exactly from samples, as in the archive.
    Returns (2, order, points) weights for the left/right sample windows.
    Geometry and parameters are validated by the plan constructor.
    """
    if order == 0:
        return jnp.zeros((2, 0, points), dtype=x.dtype)
    xi = jnp.linspace(-1., 1., points, dtype=x.dtype)
    columns = [jnp.ones_like(xi)]
    if modes > 1:
        columns.append(xi)
    for k in range(2, modes):
        columns.append(2*xi*columns[-1] - columns[-2])
    v = jnp.stack(columns, axis=1)
    index = jnp.arange(modes, dtype=x.dtype)
    penalty = (index/max(1, modes-1))**penalty_power
    penalty = penalty.at[:min(2, modes)].set(0)
    augmented = jnp.concatenate((v, jnp.sqrt(alpha)*jnp.diag(penalty)))
    Q, R = jnp.linalg.qr(augmented, mode="reduced")
    projector = jl.solve_triangular(R, Q[:points].T, lower=False)

    # T_j^(k)(1) = product_{l=0}^{k-1}(j^2-l^2)/(2k-1)!!.
    right = jnp.ones(modes, dtype=x.dtype)
    rows = []
    width = x[points-1] - x[0]
    for k in range(order):
        if k:
            right = right*(index**2-(k-1)**2)/(2*k-1)
        left = (-1.)**(index-k)*right
        rows.append((2/width)**k*(left @ projector))
    left = jnp.stack(rows).at[0].set(jnp.zeros(points, dtype=x.dtype).at[0].set(1))
    right = left[:, ::-1]*(-1.)**jnp.arange(order)[:, None]
    return jnp.stack((left, right))
