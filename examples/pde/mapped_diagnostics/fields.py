"""Independent Cartesian evaluation of a mapped streamfunction on GPU.

Invert the analytic geometry along a ray, then differentiate the scalar field
in physical coordinates. No production velocity/gradient matrices are used.
"""
from functools import partial
import jax
import jax.numpy as jnp

from bspf_jax.mapped_navier_stokes import _spline_jets


def local_coefficients(plan, state):
    coefficients = state*plan.scale
    return jnp.where(plan.ids >= 0, coefficients[jnp.maximum(plan.ids, 0)], 0.)


def coordinates(x, geometry):
    vertices, center, axes = geometry
    dx, dy = x-center
    left, bottom = vertices[0]
    right, top = vertices[2]
    # Avoid zero division in unused branches, including differentiation.
    sx = jnp.where(jnp.abs(dx) > 1e-30, dx, 1e-30)
    sy = jnp.where(jnp.abs(dy) > 1e-30, dy, 1e-30)
    ratios = jnp.array(((bottom-center[1])/sy, (right-center[0])/sx,
                        (top-center[1])/sy, (left-center[0])/sx))
    ratios = jnp.where(ratios > 0, ratios, jnp.inf)
    patch = jnp.argmin(ratios)
    distance = ratios[patch]
    outer = center+distance*(x-center)
    ts = jnp.array(((outer[0]-left)/(right-left), (outer[1]-bottom)/(top-bottom),
                   (right-outer[0])/(right-left), (top-outer[1])/(top-bottom)))
    q = jnp.sqrt(jnp.sum(((x-center)/axes)**2))
    radial = (q-1)/(distance*q-1)
    return patch, radial, ts[patch]


def stream_at(x, local, geometry, knots, degree, lift):
    patch, r, t = coordinates(x, geometry)
    br = _spline_jets(knots[0], jnp.reshape(r, (1,)), degree=degree)[0][0]
    bt = _spline_jets(knots[1], jnp.reshape(t, (1,)), degree=degree)[0][0]
    value = br @ local[patch] @ bt
    return value if lift is None else value+lift(x)


@partial(jax.jit, static_argnames=('degree', 'lift', 'order'))
def evaluate(points, local, geometry, knots, *, degree, lift=None, order=2):
    def psi(x): return stream_at(x, local, geometry, knots, degree, lift)
    def omega(x): return -jnp.trace(jax.hessian(psi)(x))
    def single(x):
        grad = jax.grad(psi)(x)
        h = jax.hessian(psi)(x)
        result = dict(velocity=jnp.stack((grad[1], -grad[0])),
                      gradient=jnp.stack((h[1], -h[0])), vorticity=-jnp.trace(h))
        if order >= 3:
            result['vorticity_gradient'] = jax.grad(omega)(x)
        return result
    return jax.vmap(single)(points)
