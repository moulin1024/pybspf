"""BSPF interpolation and integration of the same spline + Fourier interpolant.

FFT period is n*dx, while the physical closed interval is [x[0],x[-1]].
All operations preserve trailing batch/component dimensions and complex data.
"""
from __future__ import annotations

from numbers import Integral

import jax.numpy as jnp

from pybspf.basis import basis_matrix
from pybspf.basis import spline_primitive
from pybspf.operators import _axis
from pybspf.operators import _axes
from pybspf.operators import _field
from pybspf.operators import _split_axis
from pybspf.operators import _contract


def _real_if_needed(value, reference):
    return value if jnp.iscomplexobj(reference) else value.real


def _evaluate_axis(p, f, points, boundary=None, derivative=0):
    c, _, r = _split_axis(p, f, boundary)
    points = jnp.atleast_1d(points)
    if points.ndim != 1:
        raise ValueError("evaluation coordinates must be 1D")
    spline = _contract(basis_matrix(p.knots, points, degree=p.degree, derivative=derivative), c)
    modes = jnp.exp(1j*(points[:, None] - p.x[0])*p.omega[None, :])
    modes = modes*(1j*p.omega[None, :])**derivative
    correction = _contract(modes, jnp.fft.fft(r, axis=0)) / p.x.size
    values = spline + _real_if_needed(correction, f)
    valid = (points >= p.x[0]) & (points <= p.x[-1])
    return jnp.where(valid.reshape((-1,) + (1,)*(f.ndim-1)), values, jnp.nan)


def interpolate(plan, f, points, *, axis=0, boundary=None, derivative=0):
    """Evaluate along one axis; invalid/out-of-domain points return NaN under jit.

    Uses trigonometric residual interpolation, not piecewise-linear residuals.
    ``derivative`` evaluates that derivative of the same fitted interpolant,
    without refitting differentiated samples; it is a static nonnegative
    integer no greater than the plan max_derivative.
    For a new tensor mesh, call interpolate_grid instead.
    """
    p, f = _axis(plan, axis), _field(plan, f)
    if isinstance(derivative, bool) or not isinstance(derivative, Integral) or not 0 <= derivative <= p.max_derivative:
        raise ValueError("derivative must be a supported nonnegative static integer")
    return jnp.moveaxis(_evaluate_axis(p, jnp.moveaxis(f, axis, 0), points, boundary, derivative), 0, axis)


def interpolate_grid(plan, f, coordinates):
    """Separable interpolation to a tensor mesh specified in physical axis order."""
    axes, result = _axes(plan), _field(plan, f)
    if len(coordinates) != len(axes):
        raise ValueError("provide one coordinate vector per spatial axis")
    for i, (p, x) in enumerate(zip(axes, coordinates)):
        result = jnp.moveaxis(_evaluate_axis(p, jnp.moveaxis(result, i, 0), x), 0, i)
    return result


def _integrate_axis(p, f, a, b):
    c, _, r = _split_axis(p, f)
    ends = spline_primitive(p.knots, c, jnp.stack((a, b)), degree=p.degree)
    # Integral exp(i*w*(x-x0)) = length*exp(i*w*midpoint)*sinc(w*length/(2*pi)).
    # This expression includes the zero mode and is differentiable at a==b.
    length = b-a
    kernel = length*jnp.exp(1j*p.omega*((a+b)/2-p.x[0]))*jnp.sinc(p.omega*length/(2*jnp.pi))
    tail = _contract(kernel[None, :], jnp.fft.fft(r, axis=0))[0] / p.x.size
    result = ends[1] - ends[0] + _real_if_needed(tail, f)
    valid = (a >= p.x[0]) & (a <= p.x[-1]) & (b >= p.x[0]) & (b <= p.x[-1])
    return jnp.where(valid, result, jnp.nan)


def integrate(plan, f, *, axis=0, a=None, b=None):
    """Definite integral along one axis, removing that array axis.

    Bounds may be traced scalar values. Reversed bounds reverse the sign;
    bounds outside the physical domain return NaN (no implicit extrapolation).
    """
    p, f = _axis(plan, axis), _field(plan, f)
    a = p.x[0] if a is None else jnp.asarray(a)
    b = p.x[-1] if b is None else jnp.asarray(b)
    return _integrate_axis(p, jnp.moveaxis(f, axis, 0), a, b)


def integrate_box(plan, f, bounds=None):
    """Tensor volume integral, preserving trailing batch dimensions."""
    axes, result = _axes(plan), _field(plan, f)
    bounds = tuple((p.x[0], p.x[-1]) for p in axes) if bounds is None else bounds
    if len(bounds) != len(axes):
        raise ValueError("provide one (a,b) pair per spatial axis")
    for i in reversed(range(len(axes))):
        result = _integrate_axis(axes[i], jnp.moveaxis(result, i, 0), *bounds[i])
    return result


def antiderivative(plan, f, *, axis=0, order=1, left_value=0.0, left_slope=0.0):
    """First/second primitive along an axis, with explicit integration constants.

    For order=2, value and first derivative are fixed at the left endpoint.
    Constants broadcast to the other axes/batches. No conflicting right-end
    condition is silently substituted. order and axis must be static under jit.
    """
    if order not in (1, 2):
        raise ValueError("primitive order must be 1 or 2")
    p, f = _axis(plan, axis), _field(plan, f)
    f = jnp.moveaxis(f, axis, 0)
    c, _, r = _split_axis(p, f)
    spline = spline_primitive(p.knots, c, p.x, degree=p.degree, order=order)
    spectrum = jnp.fft.fft(r, axis=0)
    omega = p.omega.reshape((-1,) + (1,)*(f.ndim-1))
    safe = jnp.where(omega != 0, 1j*omega, 1)
    primitive_hat = jnp.where(omega != 0, spectrum / safe**order, 0)
    periodic = jnp.fft.ifft(primitive_hat, axis=0)
    periodic = _real_if_needed(periodic, f)
    dx = (p.x-p.x[0]).reshape((-1,) + (1,)*(f.ndim-1))
    mean = jnp.mean(r, axis=0)
    value = spline + periodic - periodic[0]
    if order == 1:
        value = value + mean*dx
    else:
        first_hat = jnp.where(omega != 0, spectrum / safe, 0)
        slope = _real_if_needed(jnp.fft.ifft(first_hat, axis=0)[0], f)
        value = value - slope*dx + mean*dx**2/2 + left_slope*dx
    return jnp.moveaxis(value + left_value, 0, axis)
