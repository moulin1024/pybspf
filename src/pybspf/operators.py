"""Pure constrained fits and separable BSPF differential operators.

A scalar field has leading spatial axes (x, y, z). All trailing dimensions are
batches/components. The same functions work for Plan1D and TensorPlan.
"""
from __future__ import annotations

from numbers import Integral
from typing import NamedTuple

from jax import Array
import jax.numpy as jnp
import jax.scipy.linalg as jl

from pybspf.plans import Plan1D
from pybspf.plans import TensorPlan
from pybspf.noise import select_noise
from pybspf.noise import apply_noise


class Split(NamedTuple):
    """Explicit f = B c + r along one selected spatial axis."""
    coefficients: Array
    spline: Array
    residual: Array


def _axes(plan):
    return (plan,) if isinstance(plan, Plan1D) else plan.axes


def _field(plan, f):
    f = jnp.asarray(f)
    shape = tuple(p.x.size for p in _axes(plan))
    if f.shape[:len(shape)] != shape:
        raise ValueError(f"expected leading spatial shape {shape}, got {f.shape}")
    return f.astype(jnp.result_type(f, _axes(plan)[0].x))


def _axis(plan, axis):
    axes = _axes(plan)
    if isinstance(axis, bool) or not isinstance(axis, Integral) or not 0 <= axis < len(axes):
        raise ValueError(f"axis must be a static integer in [0, {len(axes)})")
    return axes[axis]


def _order(p, order):
    if isinstance(order, bool) or not isinstance(order, Integral) or not 0 <= order <= p.max_derivative:
        raise ValueError(f"order must be a static integer in [0, {p.max_derivative}]")


def _contract(matrix, data):
    return jnp.tensordot(matrix, data, axes=(1, 0))


def _endpoint_jets(p, f):
    """Apply compact blocks to the endpoint windows of sample-first data."""
    points = p.boundary_blocks.shape[-1]
    return jnp.stack((_contract(p.boundary_blocks[0], f[:points]),
                      _contract(p.boundary_blocks[1], f[-points:])))


def _split_axis(p, f, boundary=None):
    """f uses sample-first layout. Boundary is (2,q,...) or inferred from f."""
    if p.noise is not None:
        raise ValueError("decomposition/interpolation/integration of noisy plans is not supported; use differentiate(order=0) for fitted samples")
    q = p.constraint_order
    if boundary is None:
        jets = _endpoint_jets(p, f).reshape((2*q,) + f.shape[1:])
    else:
        # A pair of q-jets (value, first derivative, ...) can vary across slices.
        bc = jnp.asarray(boundary, dtype=f.dtype)
        if bc.shape[:2] != (2, q):
            raise ValueError(f"boundary must have shape (2, {q}, ...) for endpoint jets")
        bc = bc.reshape(bc.shape + (1,) * (f.ndim + 1 - bc.ndim))
        jets = jnp.broadcast_to(bc, (2, q) + f.shape[1:]).reshape((2*q,) + f.shape[1:])
    rhs = jnp.concatenate((2*_contract(p.weighted_basis, f), jets), axis=0)
    shape = rhs.shape
    # JAX LU requires compatible dtypes for a real matrix and complex RHS.
    lu, piv = p.lu
    sol = jl.lu_solve((lu.astype(rhs.dtype), piv), rhs.reshape((shape[0], -1)))
    c = sol[:p.gram.shape[0]].reshape((p.gram.shape[0],) + f.shape[1:])
    spline = _contract(p.basis[0], c)
    return Split(c, spline, f - spline)


def endpoint_jets(plan, f, *, axis=0):
    """Estimate endpoint derivatives as (2, q, *other_axes_and_batches).

    These are the default spline-fit constraints. Callers can replace selected
    entries with ``jets.at[:, k].set(values)`` before passing them as boundary
    data to decompose/differentiate. This does not impose a strong PDE boundary
    condition on the final Fourier-corrected field.
    """
    p, f = _axis(plan, axis), _field(plan, f)
    if p.noise is not None:
        raise ValueError("noisy plans do not impose endpoint jets")
    f = jnp.moveaxis(f, axis, 0)
    return _endpoint_jets(p, f)


def decompose(plan, f, *, axis=0, boundary=None) -> Split:
    """Constrained fit along an axis; coefficients replace that axis by n_basis.

    Optional boundary jets have shape (2,q,*other_axes_and_batches), ordered
    left/right and then derivative order 0,...,q-1. They are affine data, not a
    claim that the final Fourier-corrected derivatives exactly satisfy the jets.
    """
    p, f = _axis(plan, axis), _field(plan, f)
    split = _split_axis(p, jnp.moveaxis(f, axis, 0), boundary)
    return Split(*(jnp.moveaxis(a, 0, axis) for a in split))


def derivatives(plan, f, *, orders=(1, 2), axis=0, boundary=None, correction=True):
    """Return {order: array}; clean plans share one KKT solve and FFT per axis.

    Noisy plans share selections from the original input across all orders.

    orders, axis and correction are static under jit. Real outputs use the
    real trigonometric interpolant, whose odd Nyquist derivatives vanish.
    """
    p, f = _axis(plan, axis), _field(plan, f)
    if not orders:
        raise ValueError("orders cannot be empty")
    for order in orders:
        _order(p, order)
    if p.noise is not None:
        if boundary is not None or not correction:
            raise ValueError("noisy differentiation requires the joint fit: boundary and correction=False are unsupported")
        selections = noise_diagnostics(plan, f)
        return {k: _noise_partial(plan, f, tuple(k if i == axis else 0 for i in range(len(_axes(plan)))), selections)
                for k in orders}
    f = jnp.moveaxis(f, axis, 0)
    c, spline, residual = _split_axis(p, f, boundary)
    spectrum = jnp.fft.fft(residual, axis=0) if correction else None
    multiplier = (1j*p.omega).reshape((-1,) + (1,) * (f.ndim - 1))
    values = {}
    for k in orders:
        value = _contract(p.basis[k], c)
        if correction:
            tail = jnp.fft.ifft(spectrum * multiplier**k, axis=0)
            value = value + (tail if jnp.iscomplexobj(f) else tail.real)
        values[k] = jnp.moveaxis(value, 0, axis)
    return values


def differentiate(plan, f, *, order=1, axis=0, boundary=None, correction=True):
    """One partial derivative; order=0 reconstructs clean data or returns noisy fitted data.

    Noisy plans use a tensor product of regularized fits, including smoothing
    along transverse axes. Selection is based on the original input field.
    """
    return derivatives(plan, f, orders=(order,), axis=axis, boundary=boundary,
                       correction=correction)[order]


def mixed_partial(plan, f, orders):
    """Tensor-product partial D_x^a D_y^b D_z^c with inferred endpoint jets."""
    axes = _axes(plan)
    if len(orders) != len(axes):
        raise ValueError("orders must have one entry per spatial axis")
    result = _field(plan, f)
    for p, k in zip(axes, orders):
        _order(p, k)
    if axes[0].noise is not None:
        return _noise_partial(plan, result, orders, noise_diagnostics(plan, result))
    for axis, (p, k) in enumerate(zip(axes, orders)):
        _order(p, k)
        if k:
            result = differentiate(plan, result, order=k, axis=axis)
    return result


def gradient(plan, f):
    """Stack Cartesian partials on a new leading component axis."""
    if _axes(plan)[0].noise is not None:
        f = _field(plan, f)
        selections = noise_diagnostics(plan, f)
        d = len(_axes(plan))
        return jnp.stack([_noise_partial(plan, f, tuple(int(i == j) for j in range(d)), selections) for i in range(d)])
    return jnp.stack([differentiate(plan, f, axis=i) for i in range(len(_axes(plan)))])


def divergence(plan, vector):
    """Divergence for vector.shape == (dimension, *spatial_shape, *batch_shape)."""
    vector = jnp.asarray(vector)
    if vector.shape[0] != len(_axes(plan)):
        raise ValueError("vector must have one leading component per spatial axis")
    return sum(differentiate(plan, vector[i], axis=i) for i in range(len(_axes(plan))))


def laplacian(plan, f):
    """Sum of direct second partials (not two successive first derivatives)."""
    if _axes(plan)[0].noise is not None:
        f = _field(plan, f)
        selections = noise_diagnostics(plan, f)
        d = len(_axes(plan))
        return sum(_noise_partial(plan, f, tuple(2*int(i == j) for j in range(d)), selections) for i in range(d))
    return sum(differentiate(plan, f, order=2, axis=i) for i in range(len(_axes(plan))))


def hessian(plan, f):
    """H[i,j,...] = partial_i partial_j f; diagonal uses direct second derivatives."""
    d = len(_axes(plan))
    if _axes(plan)[0].noise is not None:
        f = _field(plan, f)
        selections = noise_diagnostics(plan, f)
        return jnp.stack([jnp.stack([_noise_partial(plan, f, tuple(int(k == i)+int(k == j) for k in range(d)), selections)
                                     for j in range(d)]) for i in range(d)])
    return jnp.stack([jnp.stack([mixed_partial(plan, f, tuple(int(k == i)+int(k == j)
                                                             for k in range(d)))
                                  for j in range(d)]) for i in range(d)])


def curl(plan, vector):
    """Scalar 2D curl or vector 3D curl; components follow physical axis order."""
    d = len(_axes(plan))
    vector = jnp.asarray(vector)
    if d not in (2, 3) or vector.shape[0] != d:
        raise ValueError("curl needs a 2D or 3D vector field")
    if d == 2:
        return differentiate(plan, vector[1], axis=0) - differentiate(plan, vector[0], axis=1)
    return jnp.stack([differentiate(plan, vector[2], axis=1) - differentiate(plan, vector[1], axis=2),
                      differentiate(plan, vector[0], axis=2) - differentiate(plan, vector[2], axis=0),
                      differentiate(plan, vector[1], axis=0) - differentiate(plan, vector[0], axis=1)])


def tensor_decompose(plan, f):
    """Return all 2**d tensor spline/residual components, keyed by S/F strings.

    SS/SF/FS/FF in 2D, eight components in 3D. Every component is stored on the
    original grid. Their sum reconstructs f; this is not a compressed storage API.
    """
    components = {"": _field(plan, f)}
    for axis in range(len(_axes(plan))):
        next_components = {}
        for key, value in components.items():
            split = decompose(plan, value, axis=axis)
            next_components[key + "S"] = split.spline
            next_components[key + "F"] = split.residual
        components = next_components
    return components


def noise_diagnostics(plan, f):
    """Return per-axis selection (index, alpha, residual_ratio, at_search_edge, noise_std).

    Uses one alpha per axis for the complete field, including trailing batches;
    all components must share the supplied a priori noise level. That estimate
    is reported as noise_std; it is not inferred from the field. Ratios near one
    meet the discrepancy target. Selection at a grid edge calls for reviewing
    the alpha range or noise estimate. Noisy plans only; JIT-compatible.
    """
    axes, f = _axes(plan), _field(plan, f)
    if any(p.noise is None for p in axes):
        raise ValueError("noise_diagnostics requires positive noise_std on every axis")
    if not f.size:
        raise ValueError("noisy differentiation requires nonempty data")
    return tuple(select_noise(p, jnp.moveaxis(f, i, 0)) for i, p in enumerate(axes))


def _noise_partial(plan, f, orders, selections):
    # Freeze choices made on raw samples; never reselect on derivative data.
    result = f
    for axis, (p, k, selection) in enumerate(zip(_axes(plan), orders, selections)):
        _order(p, k)
        result = jnp.moveaxis(apply_noise(p, jnp.moveaxis(result, axis, 0), selection.index, k), 0, axis)
    return result
