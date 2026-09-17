"""Independent analytical benchmark solutions; not numerical BSPF propagators."""
from numbers import Integral
import jax.numpy as jnp


def cantilever_spectrum(mode_count):
    """Dimensionless roots of cos(b)+sech(b)=0 and cantilever mode coefficients.

    Newton iteration starts at the asymptotic half-integer pi roots. Scaled
    exponentials avoid cosh overflow for large mode counts. mode_count is static.
    """
    if isinstance(mode_count, bool) or not isinstance(mode_count, Integral) or mode_count < 1:
        raise ValueError('mode_count must be a positive static integer')
    roots = (jnp.arange(mode_count, dtype=jnp.float64)+.5)*jnp.pi
    for _ in range(10):
        exponential = jnp.exp(-roots)
        sech = 2*exponential/(1+exponential**2)
        roots = roots+(jnp.cos(roots)+sech)/(jnp.sin(roots)+sech*jnp.tanh(roots))
    exponential = jnp.exp(-roots)
    denominator = 1+2*jnp.sin(roots)*exponential-exponential**2
    sigma = (1+2*jnp.cos(roots)*exponential+exponential**2)/denominator
    return roots, sigma


def cantilever_modes(x, *, mode_count=256, length=1.):
    """Unit-normalized dimensionless cantilever shapes; integral phi² dx = length.

    Equivalent to cosh(bz)-cos(bz)-sigma*(sinh(bz)-sin(bz)), but avoids
    subtracting exponentially large terms. x lies in [0,length].
    """
    roots, sigma = cantilever_spectrum(mode_count)
    exponential = jnp.exp(-roots)
    denominator = 1+2*jnp.sin(roots)*exponential-exponential**2
    z = jnp.asarray(x)[None, :]/length
    b = roots[:, None]
    right = (jnp.sin(roots)-jnp.cos(roots)-exponential)/denominator
    return (right[:, None]*jnp.exp(-b*(1-z))
            +.5*(1+sigma[:, None])*jnp.exp(-b*z)
            -jnp.cos(b*z)+sigma[:, None]*jnp.sin(b*z))


def cantilever_step_response(x, times, *, mode_count=256, length=1.,
                              density=1., rigidity=1., load=1.):
    """Initially resting cantilever under a uniform load switched on at t=0.

    rhoA*w_tt + EI*w_xxxx = load, clamp at x=0, free end at x=length.
    The modal force integral is exactly 2*sigma/b and the dimensionless mode
    norm is one. No projection quadrature or BSPF matrices enter this reference.
    Stable 2*sin²(omega*t/2) enforces zero displacement at t=0. Caller supplies
    positive length/density/rigidity and nonnegative times. Truncation must be
    checked by increasing mode_count. Returns (ntimes, nx).
    """
    roots, sigma = cantilever_spectrum(mode_count)
    frequency = roots**2*jnp.sqrt(rigidity/density)/length**2
    coefficient = 2*sigma*load*length**4/(rigidity*roots**5)
    phase = jnp.asarray(times)[:, None]*frequency
    return (2*jnp.sin(phase/2)**2*coefficient)@cantilever_modes(x, mode_count=mode_count, length=length)
