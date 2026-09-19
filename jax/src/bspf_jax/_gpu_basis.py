"""Opt-in FP64 GPU basis evaluation, without the MPFR cancellation guard.

This trades numerical precision for setup speed; it is not equivalent to the
113-bit reference for ill-conditioned projectors or enriched transforms.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np



def _spline_rows(knots, points, degree, orders):
    """Fixed-width recurrence avoids unrolling a new spline graph per degree."""
    width = knots.size-1
    indices = jnp.arange(width)
    xx = points[:, None]
    seed = ((xx >= knots[:-1]) & (xx < knots[1:])).astype(points.dtype)
    last = jnp.max(jnp.where(knots[:-1] < knots[-1], indices, -1))
    seed = jnp.where(xx == knots[-1], indices == last, seed)
    initial = jnp.zeros((orders, points.size, width), dtype=points.dtype).at[0].set(seed)

    def recurrence(p, previous):
        end = knots[jnp.minimum(indices+p+1, width)]
        dl = knots[jnp.minimum(indices+p, width)]-knots[:-1]
        dr = end-knots[1:]
        il = jnp.where(dl != 0, 1/jnp.where(dl != 0, dl, 1.), 0.)
        ir = jnp.where(dr != 0, 1/jnp.where(dr != 0, dr, 1.), 0.)
        shifted = jnp.pad(previous[:, :, 1:], ((0, 0), (0, 0), (0, 1)))
        values = (xx-knots[:-1])*il*previous[0] + (end-xx)*ir*shifted[0]
        derivatives = p*il*previous[:-1]-p*ir*shifted[:-1]
        result = jnp.concatenate((values[None], derivatives), axis=0)
        return jnp.where(indices < width-p, result, 0.)

    return jax.lax.fori_loop(1, degree+1, recurrence, initial)[:, :, :width-degree]


@partial(jax.jit, static_argnames=("degree", "second", "values_only"))
def _trial_values(x, knots, projector, points, transform, layers, *, degree,
                  second, values_only):
    m = x.size - 1
    length = x[-1] - x[0]
    frequency = jnp.pi / length
    nodes = x[0] + length * jnp.arange(m, dtype=x.dtype) / m
    bn = _spline_rows(knots, nodes, degree, 1)[0]
    delta = (points[:, None] - x[0]) / length - jnp.arange(m, dtype=x.dtype) / m
    delta = delta - jnp.rint(delta)
    near = jnp.abs(delta) < 1e-12
    u = jnp.pi * delta
    su, cu = jnp.sin(u), jnp.cos(u)
    # Keep the unselected branch finite at cardinal nodes.
    denominator = jnp.where(near, 1., su)
    sm, cm = jnp.sin(m*u), jnp.cos(m*u)
    f = jnp.where(near, 1-(m*m-1)*u*u/6, sm/(m*denominator))
    g = jnp.where(near, -(m*m-1)*jnp.pi**2*delta/(3*length),
                  (cm/denominator-sm*cu/(m*denominator**2))*frequency)
    h = jnp.where(near,
                  (-(m*m-1)/3+(3*m**4-10*m*m+7)*u*u/30)*frequency**2,
                  -(m*m-1)*frequency**2*f-2*cu/denominator*frequency*g)
    if m % 2 == 0:
        h = h*cu-2*g*su*frequency-f*cu*frequency**2
        g = g*cu-f*su*frequency
        f = f*cu
    fourier = (f,) if values_only else (f, g, h) if second else (f, g)
    splines = _spline_rows(knots, points, degree, len(fourier))
    result = []
    for order, cardinal in enumerate(fourier):
        b = splines[order]
        values = (b-cardinal @ bn) @ projector
        values = values.at[:, :m].add(cardinal)
        # Match the MPFR enrichment column order: left/right for each width.
        endpoints = jnp.stack((x[0], x[-1]))
        signs = jnp.array([-1., 1.], dtype=x.dtype)
        rates = signs[None, :]/layers[:, None]
        enrichment = jnp.exp((points[:, None, None]-endpoints)*rates)*rates**order
        values = jnp.concatenate((values, enrichment.reshape(points.size, 2*layers.size)), axis=1)
        result.append(values if transform is None else values @ transform)
    return tuple(result)


def gpu_trial_values(line, spline, points, *, device, second=False,
                     transform=None, layers=(), values_only=False):
    """Evaluate in float64 on the selected GPU, returning host-plan arrays."""
    if device is None or device.platform != "gpu":
        raise ValueError("float64 basis evaluation requires a GPU device")
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 for float64 basis evaluation")
    if values_only and second:
        raise ValueError("values_only cannot request second derivatives")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 1:
        raise ValueError("points must be one-dimensional")
    orders = 1 if values_only else 3 if second else 2
    cols = np.shape(line.P)[1]+2*len(layers) if transform is None else np.shape(transform)[1]
    if not len(points):
        return tuple(np.empty((0, cols)) for _ in range(orders))
    fixed = jax.device_put(tuple(np.asarray(a, dtype=np.float64)
                                for a in (line.x, spline.t, line.P)), device)
    tr, widths = jax.device_put((None if transform is None else np.asarray(transform, dtype=np.float64),
                                np.asarray(layers, dtype=np.float64)), device)
    output = [[] for _ in range(orders)]
    # Reuse compiled kernels across nodes, quadrature, boundary and output grids.
    batch_size = 256
    for start in range(0, len(points), batch_size):
        batch = points[start:start+batch_size]
        padded = np.pad(batch, (0, batch_size-len(batch)), mode="edge")
        arrays = _trial_values(*fixed, jax.device_put(padded, device), tr, widths,
                               degree=spline.k, second=second, values_only=values_only)
        for destination, array in zip(output, jax.device_get(arrays)):
            destination.append(array[:len(batch)])
    return tuple(np.concatenate(parts, axis=0) for parts in output)
