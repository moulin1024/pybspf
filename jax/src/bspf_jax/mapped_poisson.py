"""Body-fitted tensor B-spline Poisson prototype, with a resident GPU solve.

Four H1-conforming patches cover a rectangle minus an eccentric ellipse. Each
patch connects one exact ellipse arc to a straight outer side. The analytic
geometry map is independent of the solution splines (not a NURBS geometry fit).
Shared coefficients give C0 continuity across patch seams. Scalar pullbacks
are the first building block of a compatible mapped spline complex; this
module does not yet implement an H(div) velocity/pressure discretization.
"""
from dataclasses import dataclass
from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from .immersed_poisson import EllipticHole


@partial(jax.jit, static_argnames=('degree',))
def spline_values(knots, points, *, degree):
    """Open B-spline values and first derivatives, including endpoint limits."""
    b = ((points[:, None] >= knots[:-1]) & (points[:, None] < knots[1:])).astype(points.dtype)
    last = knots.size-degree-2
    b = jnp.where((points == knots[-1])[:, None],
                  jax.nn.one_hot(last, knots.size-1, dtype=points.dtype)[None, :], b)
    for p in range(1, degree+1):
        lower = b
        n = knots.size-p-1
        left, right = knots[p:p+n]-knots[:n], knots[p+1:p+1+n]-knots[1:1+n]
        li = jnp.where(left > 0, 1/jnp.where(left > 0, left, 1), 0)
        ri = jnp.where(right > 0, 1/jnp.where(right > 0, right, 1), 0)
        b = ((points[:, None]-knots[:n])*li*lower[:, :n]
             + (knots[p+1:p+1+n]-points[:, None])*ri*lower[:, 1:n+1])
    derivative = degree*(li*lower[:, :n]-ri*lower[:, 1:n+1])
    return b, derivative


@jax.jit
def patch_geometry(vertices, center, axes, radial, tangent):
    """Exact map and Jacobian, shape (patch, radial, tangent, physical[, ref])."""
    edge = jnp.roll(vertices, -1, axis=0)-vertices
    d = vertices[:, None, :]-center + tangent[None, :, None]*edge[:, None, :]
    radius = jnp.sum((d/axes)**2, axis=-1)
    rho = radius**-.5
    drho = -jnp.sum(d*edge[:, None, :]/axes**2, axis=-1)*radius**-1.5
    scale = rho[:, None, :]+radial[None, :, None]*(1-rho[:, None, :])
    points = center+scale[..., None]*d[:, None, :, :]
    fr = jnp.broadcast_to(((1-rho)[..., None]*d)[:, None, :, :], points.shape)
    ft = (scale[..., None]*edge[:, None, None, :]
          + ((1-radial)[None, :, None]*drho[:, None, :])[..., None]*d[:, None, :, :])
    jacobian = jnp.stack((fr, ft), axis=-1)
    det = fr[..., 0]*ft[..., 1]-ft[..., 0]*fr[..., 1]
    inverse = jnp.stack((ft[..., 1], -ft[..., 0], -fr[..., 1], fr[..., 0]), axis=-1)
    inverse = inverse.reshape(*det.shape, 2, 2)/det[..., None, None]
    return points, jacobian, det, inverse


def _local(data, coefficients):
    padded = jnp.pad(coefficients, ((1, 1), (0, 0)))
    return jnp.moveaxis(padded[:, data['indices']], 1, 0)


def _assemble(data, contributions):
    nr = contributions.shape[1]
    joined = jnp.zeros((nr, 4*(contributions.shape[2]-1)), dtype=contributions.dtype)
    joined = joined.at[:, data['indices'].ravel()].add(
        jnp.moveaxis(contributions, 1, 0).reshape(nr, -1))
    return joined[1:-1]


@jax.jit
def apply_stiffness(data, coefficients):
    c = _local(data, coefficients)
    br, dr, bt, dt = (data[k] for k in ('br', 'dr', 'bt', 'dt'))
    ur, ut = dr @ c @ bt.T, br @ c @ dt.T
    gr, gt = data['g00']*ur+data['g01']*ut, data['g01']*ur+data['g11']*ut
    return _assemble(data, dr.T @ gr @ bt + br.T @ gt @ dt)


@jax.jit
def _diagonal(data):
    br, dr, bt, dt = (data[k] for k in ('br', 'dr', 'bt', 'dt'))
    local = ((dr*dr).T @ data['g00'] @ (bt*bt)
             + 2*(br*dr).T @ data['g01'] @ (bt*dt)
             + (br*br).T @ data['g11'] @ (dt*dt))
    return _assemble(data, local)


@jax.jit
def _load(data, forcing, lift_gradient):
    br, dr, bt, dt = (data[k] for k in ('br', 'dr', 'bt', 'dt'))
    flux = jnp.einsum('...ij,...j->...i', data['inverse'], lift_gradient)*data['measure'][..., None]
    return _assemble(data, br.T @ (data['measure']*forcing) @ bt
                     - dr.T @ flux[..., 0] @ bt - br.T @ flux[..., 1] @ dt)


@jax.jit
def _pcg(data, rhs, diagonal, rtol, atol, maxiter):
    x = jnp.zeros_like(rhs)
    z = rhs/diagonal
    rz = jnp.vdot(rhs, z).real
    threshold = jnp.maximum(atol, rtol*jnp.linalg.norm(rhs))
    def condition(state):
        k, _, r, _, _ = state
        return (k < maxiter) & (jnp.linalg.norm(r) > threshold)
    def body(state):
        k, x, r, direction, rz = state
        action = apply_stiffness(data, direction)
        alpha = rz/jnp.vdot(direction, action).real
        x, r = x+alpha*direction, r-alpha*action
        z = r/diagonal
        next_rz = jnp.vdot(r, z).real
        direction = z+(next_rz/rz)*direction
        return k+1, x, r, direction, next_rz
    k, x, _, _, _ = jax.lax.while_loop(condition, body, (jnp.int32(0), x, rhs, z, rz))
    residual = jnp.linalg.norm(rhs-apply_stiffness(data, x))
    return x, k, residual, threshold


@jax.jit
def _evaluate(data, coefficients, br, dr, bt, dt, inverse):
    c = _local(data, coefficients)
    value = br @ c @ bt.T
    reference_gradient = jnp.stack((dr @ c @ bt.T, br @ c @ dt.T), axis=-1)
    gradient = jnp.einsum('...ji,...j->...i', inverse, reference_gradient)
    return value, gradient


def _axis(elements, degree, order):
    breaks = np.linspace(0, 1, elements+1)
    knots = np.r_[np.repeat(0., degree), breaks, np.repeat(1., degree)]
    q, w = np.polynomial.legendre.leggauss(order)
    points = ((np.arange(elements)[:, None]+(q+1)/2)/elements).ravel()
    weights = np.tile(w/(2*elements), elements)
    return knots, points, weights


@partial(jax.jit, static_argnums=(0,))
def _sample(function, points):
    return jax.vmap(function)(points.reshape(-1, 2)).reshape(points.shape[:-1])


@partial(jax.jit, static_argnums=(0,))
def _sample_gradient(function, points):
    return jax.vmap(jax.grad(function))(points.reshape(-1, 2)).reshape(points.shape)


@dataclass
class MappedPoissonSolution:
    coefficients: jax.Array
    iterations: int
    residual_norm: float
    rhs_norm: float
    solve_seconds: float
    lift: object = None


class MappedPoissonPlan:
    """Solve -Delta u=f with prescribed physical Dirichlet lift on both walls.

    forcing(xy) and optional lift(xy) must return JAX-traceable scalar values.
    lift is an extension to the whole fluid domain, not just boundary samples.
    Basis evaluation, mapping, weak operators, loads, preconditioner, and PCG
    execute on the explicitly selected GPU. Only small knot/quadrature metadata
    originate on the host; no assembled global matrix or CPU solve is used.
    """
    def __init__(self, *, elements=(8, 8), degree=3, bounds=(-1., 3., -1., 1.),
                 hole=None, quadrature_order=None, device=None):
        start = perf_counter()
        if not jax.config.x64_enabled:
            raise ValueError('Enable jax_enable_x64 before constructing the plan')
        if device is None or device.platform != 'gpu':
            raise ValueError('An explicit GPU device is required')
        if not isinstance(degree, int) or degree < 1:
            raise ValueError('degree must be a positive integer')
        if len(elements) != 2 or any(not isinstance(n, int) or n < 1 for n in elements):
            raise ValueError('elements must contain two positive integers')
        order = degree+2 if quadrature_order is None else quadrature_order
        if not isinstance(order, int) or order < degree+1:
            raise ValueError('quadrature_order must be at least degree+1')
        if len(bounds) != 4 or not np.all(np.isfinite(bounds)):
            raise ValueError('bounds must contain four finite values')
        left, right, bottom, top = bounds
        hole = hole or EllipticHole()
        cx, cy = hole.center
        a, b = hole.axes
        if not (left < cx-a < cx+a < right and bottom < cy-b < cy+b < top):
            raise ValueError('Ellipse must lie strictly inside an increasing rectangle')
        self.device, self.degree, self.elements = device, degree, tuple(elements)
        self.bounds, self.hole, self.quadrature_order = tuple(bounds), hole, order
        self.geometry = jax.device_put((np.array(((left,bottom),(right,bottom),(right,top),(left,top))),
                                        np.array(hole.center), np.array(hole.axes)), device)
        radial, tangent = (_axis(n, degree, order) for n in elements)
        self.knots = jax.device_put((radial[0], tangent[0]), device)
        r, wr, t, wt = jax.device_put((radial[1],radial[2],tangent[1],tangent[2]), device)
        br, dr = spline_values(self.knots[0], r, degree=degree)
        bt, dt = spline_values(self.knots[1], t, degree=degree)
        self.shape = (br.shape[1]-2, 4*(bt.shape[1]-1))
        if min(self.shape) < 1:
            raise ValueError('No interior radial basis functions; increase elements or degree')
        self.dofs = int(np.prod(self.shape))
        points, _, det, inverse = patch_geometry(*self.geometry, r, t)
        measure = det*wr[None, :, None]*wt[None, None, :]
        metric = inverse @ jnp.swapaxes(inverse, -1, -2)*measure[..., None, None]
        indices = (np.arange(4)[:, None]*(bt.shape[1]-1)+np.arange(bt.shape[1])) % self.shape[1]
        self.data = dict(br=br,dr=dr,bt=bt,dt=dt,indices=jax.device_put(indices,device),
                         points=points,inverse=inverse,measure=measure,
                         g00=metric[...,0,0],g01=metric[...,0,1],g11=metric[...,1,1])
        self.diagonal = _diagonal(self.data)
        self.min_jacobian = float(jnp.min(det))
        self.area = float(jnp.sum(measure))
        if self.min_jacobian <= 0 or not bool(jnp.all(jnp.isfinite(self.diagonal) & (self.diagonal > 0))):
            raise ValueError('Invalid geometry or nonpositive operator diagonal')
        jax.block_until_ready(self.data)
        self.setup_seconds = perf_counter()-start

    def solve(self, forcing, *, lift=None, rtol=1e-11, atol=1e-13, maxiter=3000):
        if not (0 < rtol < 1 and np.isfinite(atol) and atol >= 0):
            raise ValueError('Require 0 < rtol < 1 and finite atol >= 0')
        if not isinstance(maxiter, int) or maxiter < 1:
            raise ValueError('maxiter must be a positive integer')
        start = perf_counter()
        points = self.data['points']
        f = _sample(forcing, points)
        gradient = (jnp.zeros_like(points) if lift is None else
                    _sample_gradient(lift, points))
        rhs = _load(self.data, f, gradient)
        rtol_d, atol_d, maxiter_d = jax.device_put((rtol, atol, maxiter), self.device)
        coeff, iterations, residual, threshold = _pcg(self.data,rhs,self.diagonal,rtol_d,atol_d,maxiter_d)
        iterations, residual, threshold, norm = jax.device_get((iterations,residual,threshold,jnp.linalg.norm(rhs)))
        if not np.isfinite(residual) or residual > 10*threshold:
            raise RuntimeError(f'GPU PCG did not converge: residual={residual:.3e}, target={threshold:.3e}, iterations={iterations}')
        return MappedPoissonSolution(coeff,int(iterations),float(residual),float(norm),perf_counter()-start,lift)

    def evaluate(self, solution, radial, tangent):
        """Evaluate all patches on a reference tensor grid, returning GPU arrays."""
        for values in (radial, tangent):
            a = np.asarray(values)
            if a.ndim != 1 or not np.all(np.isfinite(a)) or np.any((a < 0) | (a > 1)):
                raise ValueError('Reference coordinates must be finite 1D arrays in [0,1]')
        r,t = jax.device_put((np.asarray(radial,dtype=float),np.asarray(tangent,dtype=float)),self.device)
        br,dr = spline_values(self.knots[0],r,degree=self.degree)
        bt,dt = spline_values(self.knots[1],t,degree=self.degree)
        points,_,det,inverse = patch_geometry(*self.geometry,r,t)
        value,gradient = _evaluate(self.data,solution.coefficients,br,dr,bt,dt,inverse)
        if solution.lift is not None:
            value += _sample(solution.lift, points)
            gradient += _sample_gradient(solution.lift, points)
        return dict(points=points,value=value,gradient=gradient,jacobian_determinant=det)
