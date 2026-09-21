"""FFT + compact spline BSPF weak axes, without N-by-N axis operators.

Composite Gauss points are shifted uniform grids. Each shift uses one FFT;
B-splines are evaluated with degree+1 local weights. The quadrature mass is
h*I plus a correction spanning at most 2*n_basis+quadrature_order+1 vectors.
An orthonormal low-rank solve replaces a full-axis mass factorization.
Keep n_basis and quadrature_order bounded for O(N log N) axis applications.
"""
from dataclasses import dataclass, replace
from functools import partial
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np

from .basis import basis_matrix
from .plans import Plan1D


@partial(jax.tree_util.register_dataclass, data_fields=[
    'x', 'points', 'weights', 'omega', 'phase', 'coefficient_map',
    'node_index', 'node_value', 'quad_index', 'quad_value', 'quad_derivative',
    'mass_vectors', 'mass_factor', 'mass_whitened', 'transport_cross', 'transport_core',
    'gap_values', 'gap_derivatives', 'gap_weights'], meta_fields=['spacing', 'quadrature_order'])
@dataclass(frozen=True)
class FastAxis:
    x: jax.Array
    points: jax.Array
    weights: jax.Array
    omega: jax.Array
    phase: jax.Array
    coefficient_map: jax.Array
    node_index: jax.Array
    node_value: jax.Array
    quad_index: jax.Array
    quad_value: jax.Array
    quad_derivative: jax.Array
    mass_vectors: jax.Array
    mass_factor: jax.Array
    mass_whitened: jax.Array
    transport_cross: jax.Array
    transport_core: jax.Array
    gap_values: jax.Array
    gap_derivatives: jax.Array
    gap_weights: jax.Array
    spacing: float
    quadrature_order: int


def _local(values, index, weights):
    shape = weights.shape+(1,)*(values.ndim-1)
    return jnp.sum(weights.reshape(shape)*values[index], axis=1)


def _fourier(axis, samples, derivative=0):
    n, q = axis.x.size, axis.quadrature_order
    spectrum = jnp.fft.fft(samples, axis=0)
    multiplier = axis.phase*(1j*axis.omega[:, None])**derivative
    shifted = jnp.fft.ifft(spectrum[:, None]*multiplier.reshape((n,q)+(1,)*(samples.ndim-1)), axis=0).real
    return shifted[:-1].reshape((axis.points.size,)+samples.shape[1:])


def axis_values(axis, samples, derivative=0):
    """Apply Q or dQ/dx, sample-first, at uniform-cell Gauss points."""
    if derivative not in (0, 1):
        raise ValueError('FastAxis supports derivative 0 or 1')
    c = jnp.tensordot(axis.coefficient_map, samples, axes=(1,0))
    residual = samples-_local(c, axis.node_index, axis.node_value)
    weights = axis.quad_value if derivative == 0 else axis.quad_derivative
    return _local(c, axis.quad_index, weights)+_fourier(axis, residual, derivative)


def axis_adjoint(axis, values, derivative=0):
    """Exact real transpose of the implemented FFT/local-spline evaluation."""
    zero = jnp.zeros((axis.x.size,)+values.shape[1:], dtype=values.dtype)
    return jax.linear_transpose(lambda f: axis_values(axis, f, derivative), zero)(values)[0]


def axis_mass(axis, samples):
    weights = axis.weights.reshape((-1,)+(1,)*(samples.ndim-1))
    return axis_adjoint(axis, weights*axis_values(axis, samples))


def axis_solve(axis, load):
    """Mass inverse: h^-1 off the correction space, whitened factors inside."""
    shape = load.shape
    flat = load.reshape((shape[0],-1))
    v = axis.mass_vectors
    reduced = v.T@flat
    w = axis.mass_whitened
    return ((flat-v@reduced)/axis.spacing+w@(w.T@flat)).reshape(shape)


def axis_project(axis, values):
    weights = axis.weights.reshape((-1,)+(1,)*(values.ndim-1))
    return axis_solve(axis, axis_adjoint(axis, weights*values))


def axis_transport(axis, samples):
    """M^-1 S, S=.5*(G^T W Q-Q^T W G+endpoint surface)."""
    shape = samples.shape
    f = samples.reshape((shape[0],-1))
    c = axis.coefficient_map@f
    spectral = jnp.fft.ifft(1j*axis.omega[:,None]*jnp.fft.fft(f,axis=0),axis=0).real
    extra = axis.transport_cross@c-axis.coefficient_map.T@(axis.transport_cross.T@f)
    extra += axis.coefficient_map.T@(axis.transport_core@c)
    extra += axis.gap_values.T@(axis.gap_weights[:,None]*(axis.gap_derivatives@f))
    extra -= axis.gap_derivatives.T@(axis.gap_weights[:,None]*(axis.gap_values@f))
    extra = extra.at[0].add(-f[0]).at[-1].add(f[-1])
    return axis_solve(axis,(-axis.spacing*spectral+.5*extra).reshape(shape))


def axis_multiply(axis, samples, multiplier):
    values = axis_values(axis, samples)
    return axis_project(axis, multiplier.reshape((-1,)+(1,)*(samples.ndim-1))*values)


def axis_lift(axis, left, right):
    """M^-1(e_left*left - e_right*right), accepting incoming coordinate fluxes."""
    load = jnp.zeros((axis.x.size,)+left.shape, dtype=left.dtype)
    return axis_solve(axis, load.at[0].set(left).at[-1].add(-right))


def plan_fast_axis(plan, *, quadrature_order=8):
    """Build from a clean Plan1D, without probing an N-by-N identity.

    Quadrature splits uniform sample cells, NOT extra spline knots. Increase
    quadrature_order and check convergence for knots inside cells. Even grids
    include the real Nyquist mass correction. Require 2*m+q+(N even) < N so the
    correction remains strictly smaller than a full axis matrix. No fallback
    to dense axis assembly occurs. Small spline KKT and correction matrices
    remain; growing m with N forfeits near-linear complexity.
    """
    if not isinstance(plan, Plan1D) or plan.noise is not None:
        raise ValueError('a clean Plan1D is required')
    if isinstance(quadrature_order, bool) or not isinstance(quadrature_order, (int,np.integer)) or quadrature_order < 2:
        raise ValueError('quadrature_order must be an integer >= 2')
    n, m = plan.x.size, plan.knots.size-plan.degree-1
    q = int(quadrature_order)
    if 2*m+q+(n % 2 == 0) >= n:
        raise ValueError('FastAxis requires 2*n_basis+quadrature_order < n_samples; reduce the spline core or refine the grid')
    x, knots = np.asarray(plan.x), np.asarray(plan.knots)
    h = float(x[1]-x[0])
    nodes, weights = np.polynomial.legendre.leggauss(q)
    theta, weights = (nodes+1)/2, weights*h/2
    points = (x[:-1,None]+h*theta[None,:]).reshape(-1)

    def local(points, derivative=0):
        span = np.clip(np.searchsorted(knots,points,side='right')-1,plan.degree,m-1)
        index = span[:,None]-plan.degree+np.arange(plan.degree+1)[None,:]
        # Temporary N-by-m arrays only, not an N-by-N sample operator.
        dense = np.asarray(basis_matrix(plan.knots,jnp.asarray(points),degree=plan.degree,derivative=derivative))
        return jnp.asarray(index), jnp.asarray(np.take_along_axis(dense,index,axis=1))

    ni, nv = local(x)
    qi, qv = local(points)
    _, qd = local(points,1)
    jets = np.zeros((2*plan.constraint_order,n))
    width = plan.boundary_blocks.shape[-1]
    jets[:plan.constraint_order,:width] = np.asarray(plan.boundary_blocks[0])
    jets[plan.constraint_order:,-width:] = np.asarray(plan.boundary_blocks[1])
    # Scale endpoint rows before the small KKT solve. This prevents high
    # derivatives (h^-k) from destroying scaling at large N.
    scale=jnp.linalg.norm(plan.constraint,axis=1)
    scale=jnp.where(scale>0,scale,1)
    constraint=plan.constraint/scale[:,None]
    system=jnp.block([[2*(plan.gram+plan.regularization*jnp.eye(m)), -constraint.T],
                      [constraint,jnp.zeros((constraint.shape[0],constraint.shape[0]))]])
    rhs = jnp.concatenate((2*plan.weighted_basis,jnp.asarray(jets)/scale[:,None]))
    c = jnp.linalg.solve(system,rhs)[:m]
    axis = FastAxis(plan.x,jnp.asarray(points),jnp.asarray(np.tile(weights,n-1)),
        plan.omega,jnp.exp(1j*plan.omega[:,None]*h*jnp.asarray(theta)[None,:]),c,
        ni,nv,qi,qv,qd,jnp.empty((n,0)),jnp.empty((0,0)),jnp.empty((n,0)),
        jnp.empty((n,m)),jnp.empty((m,m)),jnp.empty((q,n)),jnp.empty((q,n)),jnp.asarray(weights),h,q)
    # Q = F + (Bq - F Bn) C. The full periodic shifted-Gauss Fourier
    # Gram is h I for odd N; even N has a rank-one real-Nyquist defect.
    # Removing the final cell contributes q additional vectors.
    bn = _local(jnp.eye(m,dtype=plan.x.dtype),ni,nv)
    bq = _local(jnp.eye(m,dtype=plan.x.dtype),qi,qv)
    correction = bq-_fourier(axis,bn)
    zero = jnp.zeros((n,m),dtype=plan.x.dtype)
    a = jax.linear_transpose(lambda f:_fourier(axis,f),zero)(axis.weights[:,None]*correction)[0]
    delta = (n-1+theta[:,None])-np.arange(n)[None,:]
    gap = np.sinc(delta)/np.sinc(delta/n)
    if n % 2 == 0:
        gap *= np.cos(np.pi*delta/n)
    blocks = [np.asarray(c.T),np.asarray(a),gap.T]
    if n % 2 == 0:
        blocks.append(((-1.)**np.arange(n))[:,None]/np.sqrt(n))
    u = np.concatenate(blocks,axis=1)
    norms = np.linalg.norm(u,axis=0)
    u = u/np.where(norms>0,norms,1)[None,:]
    vectors, _ = np.linalg.qr(u, mode='reduced')
    vectors = jnp.asarray(vectors)
    evaluated = axis_values(axis,vectors)
    core = evaluated.T@(axis.weights[:,None]*evaluated)
    factor = jnp.linalg.cholesky(.5*(core+core.T))
    if not np.isfinite(np.asarray(factor)).all():
        raise ValueError('FastAxis mass correction is not positive definite')
    t0 = correction
    t1 = _local(jnp.eye(m,dtype=plan.x.dtype),qi,qd)-_fourier(axis,bn,1)
    wt0,wt1 = axis.weights[:,None]*t0,axis.weights[:,None]*t1
    cross = jax.linear_transpose(lambda f:_fourier(axis,f,1),zero)(wt0)[0]
    cross -= jax.linear_transpose(lambda f:_fourier(axis,f),zero)(wt1)[0]
    small = t1.T@wt0-t0.T@wt1
    # Analytic derivative of the odd-grid Dirichlet cardinal function.
    if n % 2:
        gd = gap*(np.pi/np.tan(np.pi*delta)-np.pi/n/np.tan(np.pi*delta/n))/h
    else:
        angle=np.pi*delta/n
        gd=(np.pi/n*np.cos(np.pi*delta)/np.tan(angle)
            -np.pi/n**2*np.sin(np.pi*delta)/np.sin(angle)**2)/h
    return replace(axis,mass_vectors=vectors,mass_factor=factor,
        mass_whitened=jl.solve_triangular(factor,vectors.T,lower=True).T,
        transport_cross=cross,transport_core=small,
        gap_values=jnp.asarray(gap),gap_derivatives=jnp.asarray(gd))


@partial(jax.tree_util.register_dataclass,
         data_fields=['symbol','cross','core'],meta_fields=[])
@dataclass(frozen=True)
class FastMultiplier:
    symbol: jax.Array
    cross: jax.Array
    core: jax.Array


def plan_axis_multiplier(axis, multiplier):
    """Compress Q^T W a Q to a Fourier Toeplitz convolution + spline terms.

    a is a fixed real value at each Gauss point. No full axis multiplication
    matrix is formed. Fourier moments use shifted FFTs; runtime uses a padded
    FFT convolution, irrespective of how complicated the sampled a is.
    """
    a=np.asarray(multiplier)
    if a.shape != axis.points.shape or np.iscomplexobj(a) or not np.isfinite(a).all():
        raise ValueError('multiplier must be finite real quadrature values')
    n,q=axis.x.size,axis.quadrature_order
    theta=(np.asarray(axis.points[:q])-float(axis.x[0]))/axis.spacing
    weighted=(np.asarray(axis.weights)*a).reshape((n-1,q))
    spectrum=np.fft.fft(np.pad(weighted,((0,1),(0,0))),axis=0)
    def moment(k):
        return np.sum(spectrum[k % n]*np.exp(-2j*np.pi*k[:,None]*theta[None,:]/n),axis=1)
    modes=n+int(n % 2 == 0)
    length=1 << (2*modes-2).bit_length()
    embedding=np.zeros(length,dtype=np.complex128)
    embedding[:modes]=moment(np.arange(modes))
    embedding[-(modes-1):]=moment(np.arange(-(modes-1),0))
    m=axis.coefficient_map.shape[0]
    identity=jnp.eye(m,dtype=axis.x.dtype)
    bn=_local(identity,axis.node_index,axis.node_value)
    t0=_local(identity,axis.quad_index,axis.quad_value)-_fourier(axis,bn)
    weighted=axis.weights[:,None]*jnp.asarray(a)[:,None]*t0
    cross=jax.linear_transpose(lambda f:_fourier(axis,f),jnp.zeros((n,m),dtype=axis.x.dtype))(weighted)[0]
    return FastMultiplier(jnp.asarray(np.fft.fft(embedding)),cross,t0.T@weighted)


def axis_apply_multiplier(axis, operator, samples):
    """Apply projected fixed multiplication using FFT convolution and low rank."""
    shape=samples.shape
    f=samples.reshape((shape[0],-1))
    n=axis.x.size
    c=axis.coefficient_map@f
    coefficients=jnp.fft.fftshift(jnp.fft.fft(f,axis=0),axes=0)/n
    if n % 2 == 0:
        coefficients=jnp.concatenate((coefficients[:1]/2,coefficients[1:],coefficients[:1]/2),axis=0)
    modes=coefficients.shape[0]
    padded=jnp.pad(coefficients,((0,operator.symbol.size-modes),(0,0)))
    transformed=jnp.fft.ifft(operator.symbol[:,None]*jnp.fft.fft(padded,axis=0),axis=0)[:modes]
    if n % 2 == 0:
        transformed=jnp.concatenate(((transformed[:1]+transformed[-1:])/2,transformed[1:-1]),axis=0)
    load=jnp.fft.ifft(jnp.fft.ifftshift(transformed,axes=0),axis=0).real
    load+=operator.cross@c+axis.coefficient_map.T@(operator.cross.T@f+operator.core@c)
    return axis_solve(axis,load.reshape(shape))


def sample_aligned_knots(x, *, degree, n_basis, endpoint_blend=0.):
    """Clamped spline knots selected from the sample grid.

    Then composite shifted Gauss cells also split every spline knot, preserving
    the original split-knot weak quadrature without nonuniform Fourier sums.
    endpoint_blend in [0,1] blends uniform breaks with cosine-spaced breaks;
    samples remain uniform. Zero preserves the original knot placement.
    """
    x=np.asarray(x)
    spans=n_basis-degree
    if spans<1 or spans>x.size-1:
        raise ValueError('require degree < n_basis <= degree+n_samples-1')
    if not 0 <= endpoint_blend <= 1:
        raise ValueError('endpoint_blend must lie in [0,1]')
    s=np.linspace(0,1,spans+1)
    fractions=(1-endpoint_blend)*s+endpoint_blend*(1-np.cos(np.pi*s))/2
    index=np.rint((x.size-1)*fractions).astype(int)
    if np.any(np.diff(index)<=0):
        raise ValueError('endpoint clustering needs more samples for distinct breaks')
    return jnp.asarray(np.concatenate((np.repeat(x[0],degree),x[index],np.repeat(x[-1],degree))))



def boundary_clustered_knots(x, *, degree=7):
    """Fixed 3*degree+1 spline core, with degree cells at each boundary.

    Constraining high endpoint jets on globally spaced knots becomes badly
    conditioned as N increases. Local boundary knots scale with grid spacing;
    the broad interior residual is represented by FFT. This keeps the core
    fixed (22 for degree 7) while retaining high-order endpoint constraints.
    """
    x=np.asarray(x)
    if x.size<2*(degree+1):
        raise ValueError('too few samples for disjoint boundary knot blocks')
    breaks=np.concatenate((x[:degree+1],x[-degree-1:]))
    return jnp.asarray(np.concatenate((np.repeat(x[0],degree),breaks,np.repeat(x[-1],degree))))
