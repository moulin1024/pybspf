"""Experimental consistent curl-residual augmentation of the BSPF momentum form.

Research only. Add ell^2 (curl v, curl(momentum residual)) to every equation,
including time, viscosity and sponge. This unforced channel prototype does
not yet implement external forcing/curl-forcing loads. The trial space and
physical boundary constraints are unchanged. No output filtering or added
viscosity. The augmented test is not a proven stable discretization.
"""
from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np

from bspf_jax._gpu_basis import _spline_rows
from bspf_jax._flow_kernels import imex_midpoint
from bspf_jax.immersed_flow_gpu import _explicit
from bspf_jax.stream_navier_stokes import _smooth_step


@jax.jit
def _line_jets(x, knots, projector, transform, rotation, points):
    """Fourier sum avoids singular quotient derivatives at cardinal nodes."""
    m = x.size-1
    k = jnp.arange(-(m//2), m//2+1)
    weight = jnp.ones_like(k, dtype=x.dtype)
    if m % 2 == 0:
        weight = weight.at[0].set(.5).at[-1].set(.5)
    frequency = 2*jnp.pi*k/(x[-1]-x[0])
    phase = jnp.exp(1j*(points[:, None]-x[0])*frequency)
    node_phase = jnp.exp(-2j*jnp.pi*k[:, None]*jnp.arange(m)[None, :]/m)
    nodes = x[0]+(x[-1]-x[0])*jnp.arange(m)/m
    node_spline = _spline_rows(knots, nodes, 13, 1)[0]
    splines = _spline_rows(knots, points, 13, 5)
    result = []
    for order in range(5):
        cardinal = jnp.real((phase*(weight*(1j*frequency)**order)[None, :]) @ node_phase)/m
        raw = (splines[order]-cardinal @ node_spline) @ projector
        raw = raw.at[:, :m].add(cardinal)
        result.append((raw @ transform) @ rotation)
    return tuple(result)


def line_jets(plan, axis, points):
    line = plan.x if axis == 0 else plan.y
    if line.layers.size:
        raise ValueError('Prototype does not implement enriched exponential line bases')
    x = np.asarray(line.x)
    knots = np.r_[np.repeat(x[0], 14), np.linspace(x[0], x[-1], 20)[1:-1], np.repeat(x[-1], 14)]
    rotation = line.rotation if axis else line.rotation @ jax.device_put(plan.x_rotation, plan.assembly_device)
    fixed = jax.device_put((line.x, knots, line.projector, line.transform, rotation), plan.assembly_device)
    values = [[] for _ in range(5)]
    for start in range(0, len(points), 256):
        part = points[start:start+256]
        padded = np.pad(part, (0, 256-len(part)), mode='edge')
        batch = _line_jets(*fixed, jax.device_put(padded, plan.assembly_device))
        for result, value in zip(values, batch):
            result.append(value[:len(part)])
    return tuple(jnp.concatenate(value) for value in values)


@jax.jit
def _rational_curl_rows(z, values):
    dd = values[2]
    b = jnp.column_stack((-4*dd, jnp.zeros_like(dd)))
    o = 4/z**2
    zero = jnp.zeros_like(z.real)
    wx = jnp.column_stack((b.imag, o.imag, zero, b.real, o.real, zero))
    wy = jnp.column_stack((b.real, o.real, zero, -b.imag, -o.imag, zero))
    removed = 2*dd.shape[1]+1
    return tuple(jnp.concatenate((v[:, :removed], v[:, removed+1:]), axis=1) for v in (wx, wy))


@jax.jit
def _bulk_curl_derivatives(bx, by, transform):
    def tensor(i, j):
        return (bx[i][:, :, None]*by[j][:, None, :]).reshape(len(bx[0]), -1)
    return ((-tensor(3, 0)-tensor(1, 2)) @ transform,
            (-tensor(2, 1)-tensor(0, 3)) @ transform,
            (-tensor(4, 0)-2*tensor(2, 2)-tensor(0, 4)) @ transform)


def assemble(plan, step, strength=1., *, boundary_compatible=False):
    start = perf_counter()
    device = step.device
    points = plan.points
    transform = jax.device_put(plan.transform, device)
    factors = []
    for axis in (0, 1):
        coordinate, indices = np.unique(points[:, axis], return_inverse=True)
        values = line_jets(plan, axis, coordinate)
        index = jax.device_put(indices, device)
        factors.append(tuple(value[index] for value in values))
    wx, wy, lap = _bulk_curl_derivatives(*factors, transform)
    del factors
    # Rational vorticity is harmonic; only its first derivatives are needed.
    rational_coeff = jax.device_put(plan.rational_modes, device) @ (jax.device_put(plan.rational_map, device) @ transform)
    rational_coeff = jnp.column_stack((rational_coeff, jax.device_put(plan.rational_lift, device)))
    chunks = [[], []]
    for first in range(0, len(points), 512):
        part = points[first:first+512]
        z = part[:, 0]+1j*part[:, 1]-plan.rational.center
        shifted = jax.device_put(np.pad(z, (0, 512-len(z)), mode='edge'), device)
        values = plan.rational.basis.evaluate_gpu(shifted, device)
        rows = _rational_curl_rows(shifted, values)
        for out, row in zip(chunks, rows):
            out.append((row @ rational_coeff)[:len(part)])
    rx, ry = (jnp.concatenate(value) for value in chunks)
    wx, wy = wx+rx[:, :-1], wy+ry[:, :-1]
    lift_wx, lift_wy = rx[:, -1], ry[:, -1]+2*plan.peak/plan.bounds[2]**2
    d = dict(step.data)
    omega = d['ops'][4]-d['ops'][3]
    lift_omega = d['lift'][4]-d['lift'][3]
    if plan.buffer_length:
        s = jax.device_put((points[:, 0]-plan.buffer_start)/plan.buffer_length, device)
        sigma_x = plan.buffer_strength/plan.buffer_length*_smooth_step(s)[1]
    else:
        sigma_x = jnp.zeros_like(d['weights'])
    linear_curl = -plan.nu*lap+d['sigma'][:, None]*omega+sigma_x[:, None]*d['ops'][1]
    reference_omega = jax.device_put(2*plan.peak*points[:, 1]/plan.bounds[2]**2, device)
    lift_linear_curl = d['sigma']*(lift_omega-reference_omega)+sigma_x*d['lift'][1]
    kx = np.pi*(plan.nx-1)/(plan.bounds[1]-plan.bounds[0])
    ky = np.pi*(plan.ny-1)/(2*plan.bounds[2])
    base_length_squared = 1/(kx*kx+ky*ky)
    length_squared = strength*base_length_squared
    beta = jnp.ones_like(d['weights'])
    if boundary_compatible:
        xy = jax.device_put(points, device)
        left, right, half = plan.bounds
        center, axes = jax.device_put((np.asarray(plan.hole.center), np.asarray(plan.hole.axes)), device)
        relative = xy-center
        q = jnp.sqrt(jnp.sum((relative/axes)**2, axis=-1))
        grad_q = jnp.sqrt(jnp.sum((relative/axes**2)**2, axis=-1))/q
        hole_distance = (q-1)/grad_q
        distances = jnp.stack((xy[:, 0]-left, right-xy[:, 0], half-xy[:, 1],
                               half+xy[:, 1], hole_distance), axis=-1)
        boundary_length_squared = length_squared if strength > 0 else base_length_squared
        beta = jnp.prod(distances**2/(distances**2+boundary_length_squared), axis=-1)
    test = length_squared*omega.T*(d['weights']*beta)[None, :]
    mass = d['mass']+test @ omega
    linear = d['linear']+test @ linear_curl
    lift_linear = d['linear_lift']+test @ lift_linear_curl
    d.update(curl_test=test, omega=omega, omega_x=wx, omega_y=wy,
             omega_laplacian=lap, curl_linear=linear_curl, curl_linear_lift=lift_linear_curl,
             curl_weight=beta,
             omega_lift=lift_omega, omega_x_lift=lift_wx, omega_y_lift=lift_wy,
             augmented_mass=mass, augmented_linear=linear, augmented_lift=lift_linear,
             augmented_factor=jl.lu_factor(mass+step.dt/2*linear),
             augmented_mass_factor=jl.cho_factor(mass))
    jax.block_until_ready(d)
    return d, dict(length_squared=length_squared, strength=strength,
                   boundary_compatible=boundary_compatible, curl_setup_seconds=perf_counter()-start)


@jax.jit
def explicit(d, state):
    u, v = [op @ state+b for op, b in zip(d['ops'][:2], d['lift'][:2])]
    wx = d['omega_x'] @ state+d['omega_x_lift']
    wy = d['omega_y'] @ state+d['omega_y_lift']
    return (_explicit(d, state)+d['linear_lift']-d['augmented_lift']
            -d['curl_test'] @ (u*wx+v*wy))


@partial(jax.jit, static_argnames=('steps',))
def advance(d, state, dt, *, steps):
    def body(_, a):
        return imex_midpoint(a, 0., dt, lambda b: d['augmented_mass'] @ b,
                             lambda b: d['augmented_linear'] @ b,
                             lambda b, t: explicit(d, b),
                             lambda b: jl.lu_solve(d['augmented_factor'], b))
    return jax.lax.fori_loop(0, steps, body, state)
