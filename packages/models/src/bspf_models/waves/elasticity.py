"""Accurate modal evolution of small constant-load linear elastic weak systems."""
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


def _factor_modes(weak, density, rigidity):
    lower = jnp.linalg.cholesky(density*weak.mass)
    factor = jnp.sqrt(rigidity*weak.quadrature_weights[:, None])*weak.derivative_values
    scaled = solve_triangular(lower, factor.T, lower=True).T
    # SVD of the derivative factor avoids squaring its condition number in K.
    _, frequencies, vectors = jnp.linalg.svd(scaled, full_matrices=False)
    frequencies, vectors = frequencies[::-1], vectors[::-1]
    modes = solve_triangular(lower.T, vectors.T, lower=False)
    projection = vectors@lower.T
    return frequencies, modes, projection


def elastic_modes(weak, *, density=1., rigidity=1.):
    """Return ascending angular frequencies and mass-normalized nodal modes.

    ``weak`` must use resolved quadrature for high accuracy. Mass is density*M;
    stiffness is rigidity*G*WG. SVD of sqrt(rigidity*W) G L^{-*}, with M=LL*,
    avoids the loss of small eigenvalues from forming a squared derivative
    factor. Caller supplies positive real scalar density and rigidity. This
    dense O(Nquad*N²) primitive is JIT compatible. Zero frequencies are allowed.
    """
    frequencies, modes, _ = _factor_modes(weak, density, rigidity)
    return frequencies, modes


def integrate_elastic(weak, displacement, velocity, times, *, force=None,
                      density=1., rigidity=1.):
    """Solve density*M q_tt + rigidity*K q = force with constant coefficients.

    Initial displacement/velocity and force are vectors of free nodal
    coefficients / dual loads. Histories include initial at times[0], and
    each has shape (ntimes, Nfree). Return (displacement, velocity); reconstruct
    physical samples with weak.extension. Force defaults to zero and is constant
    in time. For a uniform physical load q, use q*(weak.values.T@weights).

    Exact modal sine/cosine evolution removes time-step error. Stable sinc
    formulas include the zero-frequency rigid-motion/constant-acceleration
    limit. Spatial, quadrature, and floating-point errors remain. No damping,
    nonlinearity, or time-dependent forcing is implied. Caller supplies positive
    density/rigidity and finite increasing real output times. Supports JIT.
    """
    displacement, velocity, times = map(jnp.asarray, (displacement, velocity, times))
    n = weak.mass.shape[0]
    if displacement.shape != (n,) or velocity.shape != (n,):
        raise ValueError('displacement and velocity must be vectors of free coefficients')
    if times.ndim != 1 or times.size < 1 or jnp.iscomplexobj(times):
        raise ValueError('times must be a nonempty real 1D array')
    force = jnp.zeros_like(displacement) if force is None else jnp.asarray(force)
    if force.shape != (n,):
        raise ValueError('force must be a vector matching the free coefficients')
    frequencies, modes, projection = _factor_modes(weak, density, rigidity)
    q0, v0, load = projection@displacement, projection@velocity, modes.T@force
    elapsed = (times-times[0])[:, None]
    phase = elapsed*frequencies
    cosine = jnp.cos(phase)
    sine_over_frequency = elapsed*jnp.sinc(phase/jnp.pi)
    one_minus_cosine_over_frequency_squared = elapsed**2/2*jnp.sinc(phase/(2*jnp.pi))**2
    q = (cosine*q0+sine_over_frequency*v0+one_minus_cosine_over_frequency_squared*load)@modes.T
    v = (-frequencies*jnp.sin(phase)*q0+cosine*v0+sine_over_frequency*load)@modes.T
    return q.at[0].set(displacement), v.at[0].set(velocity)
