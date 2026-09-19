"""GPU Navier--Stokes in the divergence-free kernel of mapped spline spaces.

Velocity is the physical curl of a patchwise tensor spline streamfunction.
Shared traces make its normal component continuous (H(div)); tangential jumps
are coupled by symmetric interior penalty viscosity. Physical tangential
Dirichlet data can use exact coefficient constraints or Nitsche's method.
A free obstacle streamfunction constant
retains the circulation degree of freedom on this multiply connected domain.
This initial implementation assembles dense reduced operators on the GPU.
"""
from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np

from ._flow_kernels import imex_midpoint
from .mapped_poisson import MappedPoissonPlan, _axis


@partial(jax.jit, static_argnames=('degree',))
def _spline_jets(knots, points, *, degree):
    b = ((points[:, None] >= knots[:-1]) & (points[:, None] < knots[1:])).astype(points.dtype)
    last = knots.size-degree-2
    b = jnp.where((points == knots[-1])[:, None],
                  jax.nn.one_hot(last, knots.size-1, dtype=points.dtype)[None], b)
    d, h = jnp.zeros_like(b), jnp.zeros_like(b)
    for p in range(1, degree+1):
        n = knots.size-p-1
        left, right = knots[p:p+n]-knots[:n], knots[p+1:p+1+n]-knots[1:1+n]
        li = jnp.where(left > 0, 1/jnp.where(left > 0, left, 1), 0)
        ri = jnp.where(right > 0, 1/jnp.where(right > 0, right, 1), 0)
        h = p*(li*d[:, :n]-ri*d[:, 1:n+1])
        d = p*(li*b[:, :n]-ri*b[:, 1:n+1])
        b = ((points[:, None]-knots[:n])*li*b[:, :n]
             + (knots[p+1:p+1+n]-points[:, None])*ri*b[:, 1:n+1])
    return b, d, h


def _map(z, vertex, edge, center, axes):
    r, t = z
    d = vertex-center+t*edge
    rho = jnp.sum((d/axes)**2)**-.5
    return center+(rho+r*(1-rho))*d


@jax.jit
def _geometry_jets(geometry, radial, tangent):
    vertices, center, axes = geometry
    edges = jnp.roll(vertices, -1, axis=0)-vertices
    z = jnp.stack(jnp.meshgrid(radial, tangent, indexing='ij'), axis=-1).reshape(-1, 2)
    def patch(vertex, edge):
        def evaluate(x):
            args = (x, vertex, edge, center, axes)
            return _map(*args), jax.jacfwd(_map)(*args), jax.jacfwd(jax.jacfwd(_map))(*args)
        return jax.vmap(evaluate)(z)
    points, jac, hessian = jax.vmap(patch)(vertices, edges)
    det = jac[..., 0, 0]*jac[..., 1, 1]-jac[..., 0, 1]*jac[..., 1, 0]
    inv = jnp.stack((jac[..., 1, 1], -jac[..., 0, 1], -jac[..., 1, 0], jac[..., 0, 0]), axis=-1)
    inv = inv.reshape(*det.shape, 2, 2)/det[..., None, None]
    return points, jac, hessian, det, inv


@partial(jax.jit, static_argnames=('degree', 'dofs'))
def _operators(geometry, knots, ids, radial, tangent, *, degree, dofs):
    br, dr, hr = _spline_jets(knots[0], radial, degree=degree)
    bt, dt, ht = _spline_jets(knots[1], tangent, degree=degree)
    c = jax.nn.one_hot(ids, dofs, dtype=radial.dtype)
    def tensor(a, b):
        return jnp.einsum('ri,pijn,tj->prtn', a, c, b).reshape(4, -1, dofs)
    psi = tensor(br, bt)
    ref_g = jnp.stack((tensor(dr, bt), tensor(br, dt)), axis=-2)
    rr, rt, tt = tensor(hr, bt), tensor(dr, dt), tensor(br, ht)
    ref_h = jnp.stack((jnp.stack((rr, rt), axis=-2), jnp.stack((rt, tt), axis=-2)), axis=-3)
    points, jac, map_h, det, inv = _geometry_jets(geometry, radial, tangent)
    grad = jnp.einsum('pqji,pqjn->pqin', inv, ref_g)
    corrected = ref_h-jnp.einsum('pqkn,pqkij->pqijn', grad, map_h)
    hessian = jnp.einsum('pqai,pqabn,pqbj->pqijn', inv, corrected, inv)
    velocity = jnp.stack((grad[..., 1, :], -grad[..., 0, :]), axis=-2)
    gradient = jnp.stack((hessian[..., 1, :, :], -hessian[..., 0, :, :]), axis=-3)
    return dict(points=points, jacobian=jac, inverse=inv, determinant=det,
                stream=psi, velocity=velocity, gradient=gradient)


@partial(jax.jit, static_argnums=(0,))
def _stream_fields(function, points):
    def sample(x):
        g = jax.grad(function)(x)
        h = jax.hessian(function)(x)
        return jnp.array((g[1], -g[0])), jnp.stack((h[1], -h[0]))
    u, g = jax.vmap(sample)(points.reshape(-1, 2))
    return u.reshape(*points.shape[:-1], 2), g.reshape(*points.shape[:-1], 2, 2)


@jax.jit
def _gram(a, b, weights):
    return a.reshape(-1, a.shape[-1]).T @ (b*weights.reshape((-1,)+(1,)*(b.ndim-1))).reshape(-1, b.shape[-1])


@jax.jit
def _load(op, field, weights):
    return op.reshape(-1, op.shape[-1]).T @ (field*weights.reshape((-1,)+(1,)*(field.ndim-1))).ravel()


@jax.jit
def _convection(d, state):
    v, g, w = d['v'], d['g'], d['w']
    u = jnp.einsum('qin,n->qi', v, state)+d['lift_v']
    grad = jnp.einsum('qijn,n->qij', g, state)+d['lift_g']
    adv = jnp.einsum('qij,qj->qi', grad, u)
    result = .5*(_load(v, adv, w)-_load(g, u[:, :, None]*u[:, None, :], w))
    minus = jnp.einsum('qin,n->qi', d['im'], state)+d['lift_i']
    plus = jnp.einsum('qin,n->qi', d['ip'], state)+d['lift_i']
    normal = .5*jnp.sum((minus+plus)*d['inormal'], axis=-1)
    result += .5*(_load(d['im'], plus, d['iw']*normal)-_load(d['ip'], minus, d['iw']*normal))
    boundary = jnp.einsum('qin,n->qi', d['bv'], state)+d['lift_b']
    normal = jnp.sum(boundary*d['bnormal'], axis=-1)
    return result+.5*_load(d['bv'], boundary, d['bw']*normal)


@partial(jax.jit, static_argnames=('forcing',))
def _step(d, state, time, dt, forcing=None):
    def explicit(a, t):
        result = d['nu']*d['lift_load']-_convection(d, a)
        if forcing is not None:
            f = jax.vmap(forcing, in_axes=(0, None))(d['points'], t)
            result += _load(d['v'], f, d['w'])
        return result
    return imex_midpoint(state, time, dt, lambda a: d['mass'] @ a,
                         lambda a: d['nu']*(d['stiffness'] @ a), explicit,
                         lambda b: jl.cho_solve((d['factor'], True), b))


@partial(jax.jit, static_argnames=('steps', 'forcing'))
def _advance(d, state, time, dt, *, steps, forcing=None):
    def body(k, a):
        return _step(d, a, time+k*dt, dt, forcing)
    return jax.lax.fori_loop(0, steps, body, state)


@jax.jit
def _diagnostics(d, state):
    u = jnp.einsum('qin,n->qi', d['v'], state)+d['lift_v']
    g = jnp.einsum('qijn,n->qij', d['g'], state)+d['lift_g']
    b = jnp.einsum('qin,n->qi', d['bv'], state)+d['lift_b']
    jump = jnp.einsum('qin,n->qi', d['im']-d['ip'], state)
    normal_jump = jnp.sum(jump*d['inormal'], axis=-1)
    error = b-d['lift_b']
    flux = jnp.sum(b*d['bnormal'], axis=-1)*d['bw']
    return dict(kinetic_energy=.5*jnp.sum(d['w']*jnp.sum(u*u, axis=-1)),
                max_speed=jnp.max(jnp.linalg.norm(u, axis=-1)),
                divergence_linf=jnp.max(jnp.abs(g[:, 0, 0]+g[:, 1, 1])),
                normal_jump_linf=jnp.max(jnp.abs(normal_jump)),
                tangent_jump_l2=jnp.sqrt(jnp.sum(d['iw']*jnp.sum(jump*jump, axis=-1))),
                boundary_normal_error=jnp.max(jnp.abs(jnp.sum(error*d['bnormal'], axis=-1))),
                boundary_velocity_error=jnp.max(jnp.linalg.norm(error, axis=-1)),
                net_boundary_flux=jnp.sum(flux),
                boundary_fluxes=jnp.sum(flux.reshape(4, 2, -1), axis=-1),
                viscous_quadratic=state @ d['stiffness'] @ state,
                convection_power=state @ _convection(d, state))


def channel_streamfunction(bounds, hole, peak=1.):
    """Smooth divergence-free parabolic throughflow lift, zero velocity on hole.

    The ellipse cutoff finishes strictly before any outer boundary. It changes
    only the lift/initial guess; the complete Navier--Stokes equations evolve
    the total velocity, without sponge or relaxation terms.
    """
    left, right, bottom, top = bounds
    cx, cy = hole.center
    a, b = hole.axes
    radius = .9*min((cx-left)/a, (right-cx)/a, (cy-bottom)/b, (top-cy)/b)
    if radius <= 1:
        raise ValueError('Channel lift requires space for its cutoff around the ellipse')
    mid, half = (bottom+top)/2, (top-bottom)/2
    def base(y):
        z = (y-mid)/half
        return peak*half*(z-z**3/3)
    constant = base(cy)
    def stream(x):
        q = jnp.sqrt(((x[0]-cx)/a)**2+((x[1]-cy)/b)**2)
        s = jnp.clip((q-1)/(radius-1), 0, 1)
        cutoff = s**3*(10+s*(-15+6*s))
        return constant+cutoff*(base(x[1])-constant)
    return stream


class MappedNavierStokesPlan:
    """FP64 GPU compatible curl-space Navier--Stokes with static boundary data.

    `lift` is a physical streamfunction whose curl prescribes boundary velocity.
    boundary="strong" (default) enforces its full curl exactly by coefficient
    constraints; boundary="nitsche" imposes the tangential component weakly.
    The streamfunction trace on each wall must be representable by this lift;
    variations have homogeneous normal velocity. Pressure is eliminated by
    testing only with divergence-free functions; no pressure field is returned.
    Degree >= 2 gives H1 velocities inside each patch. All four outer sides
    have prescribed velocity, including the channel outlet in the example.
    """
    def __init__(self, *, elements=(8, 8), degree=3, bounds=(-1., 3., -1., 1.),
                 hole=None, viscosity=.01, dt=.005, lift=None,
                 quadrature_order=None, boundary="strong", device=None):
        start = perf_counter()
        if degree < 2:
            raise ValueError('Mapped flow requires degree >= 2')
        if not (np.isfinite(viscosity) and viscosity > 0 and np.isfinite(dt) and dt > 0):
            raise ValueError('viscosity and dt must be finite and positive')
        if boundary not in ('strong', 'nitsche'):
            raise ValueError("boundary must be 'strong' or 'nitsche'")
        order = degree+3 if quadrature_order is None else quadrature_order
        base = MappedPoissonPlan(elements=elements, degree=degree, bounds=bounds,
                                 hole=hole, quadrature_order=order, device=device)
        self.base, self.device, self.degree = base, device, degree
        self.elements, self.bounds, self.hole = base.elements, base.bounds, base.hole
        self.lift, self.viscosity, self.dt = lift, viscosity, dt
        self.boundary, self.quadrature_order = boundary, order
        nr, nt = elements[0]+degree, elements[1]+degree
        wall_rows = 2 if boundary == "strong" else 1
        free_rows = nr-2*wall_rows
        if free_rows < 0:
            raise ValueError("Increase radial elements or degree for exact no-slip constraints")
        self.dofs = free_rows*(4*(nt-1))+1
        ids = np.full((4, nr, nt), -1, dtype=np.int32)
        for p in range(4):
            ids[p, wall_rows:nr-wall_rows] = np.arange(free_rows)[:, None]*(4*(nt-1))+(p*(nt-1)+np.arange(nt))[None, :] % (4*(nt-1))
        # Repeated endpoint coefficients set both the value and radial derivative.
        # Since the boundary trace is constant, this kills the full physical curl.
        ids[:, :wall_rows, :] = self.dofs-1
        self.ids = jax.device_put(ids, device)
        _, r, wr = _axis(elements[0], degree, order)
        _, t, wt = _axis(elements[1], degree, order)
        r, t, wr, wt, ends = jax.device_put((r, t, wr, wt, np.array((0., 1.))), device)
        volume = self._operators(r, t)
        boundary = self._operators(ends, t)
        interface = self._operators(r, ends)
        self.setup_timings = dict(geometry_and_basis_seconds=perf_counter()-start)
        start_ops = perf_counter()
        v = volume['velocity'].reshape(-1, 2, self.dofs)
        g = volume['gradient'].reshape(-1, 2, 2, self.dofs)
        w = (volume['determinant'].reshape(4, len(r), len(t))*wr[None, :, None]*wt[None, None, :]).ravel()
        def boundary_shape(a): return a.reshape(4, 2, len(t), *a.shape[2:])
        bj, bi = boundary_shape(boundary['jacobian']), boundary_shape(boundary['inverse'])
        signs = jax.device_put(np.array((-1., 1.)), device)
        bn = bi[..., 0, :]*signs[None, :, None, None]
        bn = bn/jnp.linalg.norm(bn, axis=-1, keepdims=True)
        bw = (jnp.linalg.norm(bj[..., :, 1], axis=-1)*wt[None, None, :]).ravel()
        bsigma = (8*(degree+1)**2*elements[0]*jnp.linalg.norm(bi[..., 0, :], axis=-1)).ravel()
        bv = boundary['velocity'].reshape(-1, 2, self.dofs)
        bg = boundary['gradient'].reshape(-1, 2, 2, self.dofs)
        bn = bn.reshape(-1, 2)
        bdn = jnp.einsum('qijn,qj->qin', bg, bn)
        def interface_shape(a): return a.reshape(4, len(r), 2, *a.shape[2:])
        ij, ii = interface_shape(interface['jacobian']), interface_shape(interface['inverse'])
        normal = ii[:, :, 1, 1, :]
        normal = normal/jnp.linalg.norm(normal, axis=-1, keepdims=True)
        iw = (jnp.linalg.norm(ij[:, :, 1, :, 0], axis=-1)*wr[None, :]).ravel()
        ih_minus = jnp.linalg.norm(ii[:, :, 1, 1, :], axis=-1)
        ih_plus = jnp.roll(jnp.linalg.norm(ii[:, :, 0, 1, :], axis=-1), -1, axis=0)
        isigma = (8*(degree+1)**2*elements[1]*jnp.maximum(ih_minus, ih_plus)).ravel()
        iv, ig = interface_shape(interface['velocity']), interface_shape(interface['gradient'])
        im, ip = iv[:, :, 1].reshape(-1, 2, self.dofs), jnp.roll(iv[:, :, 0], -1, axis=0).reshape(-1, 2, self.dofs)
        average = .5*(ig[:, :, 1]+jnp.roll(ig[:, :, 0], -1, axis=0))
        idn = jnp.einsum('pqijn,pqj->pqin', average, normal).reshape(-1, 2, self.dofs)
        jump = im-ip
        mass = _gram(v, v, w)
        stiffness = _gram(g, g, w)
        cross = _gram(jump, idn, iw)+_gram(bv, bdn, bw)
        stiffness += -cross-cross.T+_gram(jump, jump, iw*isigma)+_gram(bv, bv, bw*bsigma)
        scale = 1/jnp.sqrt(jnp.diag(mass))
        self.scale = scale
        mass = mass*scale[:, None]*scale[None, :]
        stiffness = stiffness*scale[:, None]*scale[None, :]
        def lift_fields(points):
            if lift is None:
                zero = jnp.zeros_like(points)
                return zero, jnp.broadcast_to(zero[..., :, None], (*points.shape[:-1], 2, 2))
            return _stream_fields(lift, points)
        points = volume['points'].reshape(-1, 2)
        lv, lg = lift_fields(points)
        lb, lbg = lift_fields(boundary['points'].reshape(-1, 2))
        li, lig = lift_fields(interface_shape(interface['points'])[:, :, 1].reshape(-1, 2))
        lift_load = -_load(g, lg, w)+_load(jump, jnp.einsum('qij,qj->qi', lig, normal.reshape(-1, 2)), iw)
        lift_load += _load(bv, jnp.einsum('qij,qj->qi', lbg, bn), bw)
        d = dict(v=v*scale, g=g*scale, w=w, points=points,
                 im=im*scale, ip=ip*scale, inormal=normal.reshape(-1, 2), iw=iw,
                 bv=bv*scale, bnormal=bn, bw=bw,
                 lift_v=lv, lift_g=lg, lift_b=lb, lift_i=li, lift_load=lift_load*scale,
                 mass=(mass+mass.T)/2, stiffness=(stiffness+stiffness.T)/2,
                 nu=jax.device_put(viscosity, device))
        jax.block_until_ready(d)
        self.setup_timings['assembly_seconds'] = perf_counter()-start_ops
        factor_start = perf_counter()
        d['mass_factor'] = jnp.linalg.cholesky(d['mass'])
        d['stiffness_factor'] = jnp.linalg.cholesky(d['stiffness'])
        d['factor'] = jnp.linalg.cholesky(d['mass']+dt/2*viscosity*d['stiffness'])
        for name in ('mass_factor', 'stiffness_factor', 'factor'):
            if not bool(jnp.all(jnp.isfinite(d[name]))):
                raise RuntimeError('GPU factorization failed: '+name)
        self.data = d
        self.dt_device = jax.device_put(dt, device)
        self.stokes_state = jl.cho_solve((d['stiffness_factor'], True), d['lift_load'])
        jax.block_until_ready(self.stokes_state)
        self.setup_timings['factorization_seconds'] = perf_counter()-factor_start
        self.setup_seconds = perf_counter()-start

    def _operators(self, radial, tangent):
        return _operators(self.base.geometry, self.base.knots, self.ids, radial, tangent,
                          degree=self.degree, dofs=self.dofs)

    def project(self, velocity):
        """GPU L2 projection of a physical velocity, subtracting the static lift."""
        values = jax.jit(jax.vmap(velocity))(self.data['points'])-self.data['lift_v']
        return jl.cho_solve((self.data['mass_factor'], True), _load(self.data['v'], values, self.data['w']))

    def advance(self, state, *, time=0., steps=1, forcing=None):
        if not isinstance(steps, int) or steps < 0:
            raise ValueError('steps must be a nonnegative integer')
        if not isinstance(state, jax.Array) or state.devices() != {self.device} or state.shape != (self.dofs,):
            raise ValueError('state must have the plan shape and reside on its GPU')
        return _advance(self.data, state, jax.device_put(time, self.device), self.dt_device,
                        steps=steps, forcing=forcing)

    def diagnostics(self, state):
        """GPU diagnostics; the caller chooses when to download them."""
        return _diagnostics(self.data, state)

    def evaluate(self, state, radial, tangent):
        for values in (radial, tangent):
            a = np.asarray(values)
            if a.ndim != 1 or not np.all(np.isfinite(a)) or np.any((a < 0) | (a > 1)):
                raise ValueError('Reference coordinates must be finite 1D arrays in [0,1]')
        r, t = jax.device_put((np.asarray(radial, dtype=float), np.asarray(tangent, dtype=float)), self.device)
        op = self._operators(r, t)
        coeff = state*self.scale
        velocity = jnp.einsum('pqin,n->pqi', op['velocity'], coeff)
        gradient = jnp.einsum('pqijn,n->pqij', op['gradient'], coeff)
        stream = op['stream'] @ coeff
        if self.lift is not None:
            v, g = _stream_fields(self.lift, op['points'])
            velocity, gradient = velocity+v, gradient+g
            stream += jax.jit(jax.vmap(self.lift))(op['points'].reshape(-1, 2)).reshape(stream.shape)
        result = dict(points=op['points'], velocity=velocity, gradient=gradient,
                      stream=stream, vorticity=gradient[..., 1, 0]-gradient[..., 0, 1],
                      determinant=op['determinant'])
        return {k: v.reshape(4, len(r), len(t), *v.shape[2:]) for k, v in result.items()}
