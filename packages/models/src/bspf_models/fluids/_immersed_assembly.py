"""GPU construction and reduction of dense volume operators.

Use the original float64 Galerkin formulas. Tensor operators and rational
corrections can be assembled directly on device and retained through scaling
and Gram products. Host-constructed operators are uploaded once. Gram matrices,
reduced matrices and normalized operators return to the host plan API only
when needed. Large operators are processed sequentially to bound workspace.
"""
from functools import partial

import jax


# Keep fused kernels for small plans; avoid multi-operator scratch allocations
# once the resident input operators exceed two GiB. This changes only scheduling.
_FUSED_OPERATOR_BYTES = 2 * 1024**3


@jax.jit
def _gram(ops, weights):
    u, v, xy, yy, minus_xx = ops
    w = weights[:, None]
    return (u.T @ (w*u) + v.T @ (w*v),
            2*xy.T @ (w*xy) + yy.T @ (w*yy) + minus_xx.T @ (w*minus_xx))


@jax.jit
def _transform(ops, normalization):
    return tuple(o @ normalization for o in ops)


@jax.jit
def _transform_and_reduce(ops, weights, normalization, mass, stiffness, sigma):
    transformed = _transform(ops, normalization)
    u, v = transformed[:2]
    sw = (weights * sigma)[:, None]
    sponge = u.T @ (sw * u) + v.T @ (sw * v)
    return (transformed, normalization.T @ mass @ normalization,
            normalization.T @ stiffness @ normalization, sponge)


@jax.jit
def _transform_one(op, normalization):
    return op @ normalization


@jax.jit
def _weighted_gram_one(op, weights):
    return op.T @ (weights[:, None]*op)


@jax.jit
def _reduce(normalization, mass, stiffness):
    return normalization.T @ mass @ normalization, normalization.T @ stiffness @ normalization


class GPUVolumeAssembly:
    def __init__(self, ops, weights, device):
        self.device = device
        self.ops, self.weights = jax.device_put((ops, weights), device)

    def gram(self):
        if sum(op.nbytes for op in self.ops) <= _FUSED_OPERATOR_BYTES:
            return jax.device_get(_gram(self.ops, self.weights))
        mass = stiffness = None
        for k, op in enumerate(self.ops):
            term = _weighted_gram_one(op, self.weights)
            if k < 2:
                mass = term if mass is None else mass+term
                mass.block_until_ready()
            else:
                term = 2*term if k == 2 else term
                stiffness = term if stiffness is None else stiffness+term
                stiffness.block_until_ready()
        return jax.device_get((mass, stiffness))

    def transform(self, normalization):
        return jax.device_get(_transform(self.ops, jax.device_put(normalization, self.device)))

    def transform_and_reduce(self, normalization, mass, stiffness, sigma):
        """Reuse resident quadrature operators for all dense reduced products."""
        data = jax.device_put((normalization, mass, stiffness, sigma), self.device)
        if sum(op.nbytes for op in self.ops) <= _FUSED_OPERATOR_BYTES:
            return jax.device_get(_transform_and_reduce(self.ops, self.weights, *data))
        normalization, mass, stiffness, sigma = data
        # Download each final operator before forming the next. A fused call
        # retains all five inputs, outputs and GEMM temporaries simultaneously.
        transformed = []
        sponge = None
        for k, op in enumerate(self.ops):
            value = _transform_one(op, normalization)
            transformed.append(jax.device_get(value))
            if k < 2:
                term = _weighted_gram_one(value, self.weights*sigma)
                sponge = term if sponge is None else sponge+term
                sponge.block_until_ready()
            del value
        reduced = jax.device_get(_reduce(normalization, mass, stiffness))
        return tuple(transformed), *reduced, jax.device_get(sponge)


@jax.jit
def _apply_rational_rows(rows, coefficients):
    return tuple(row @ coefficients for row in rows)


@jax.jit
def _join_rational_chunks(chunks):
    import jax.numpy as jnp
    return tuple(jnp.concatenate(parts, axis=0) for parts in zip(*chunks))


@jax.jit
def _tensor_operators(factors, correction, mapping, base):
    (x, dx, xx), (y, dy, yy) = factors
    def pair(a, b):
        return (a[:, :, None]*b[:, None, :]).reshape(a.shape[0], -1)
    operators = (pair(x, y), pair(x, dy), -pair(dx, y), pair(dx, dy),
                 pair(x, yy), -pair(xx, y))
    if correction is not None:
        operators = tuple(o + r[:, :mapping.shape[0]] @ mapping
                          for o, r in zip(operators, correction))
        if base is not None:
            base = tuple(b + r[:, -1] for b, r in zip(base, correction))
    return operators if base is None else (operators, base)


@partial(jax.jit, static_argnames=("sign",))
def _tensor_operator(a, b, sign, correction, mapping):
    result = sign*(a[:, :, None]*b[:, None, :]).reshape(a.shape[0], -1)
    if correction is not None:
        result += correction[:, :mapping.shape[0]] @ mapping
    return result


def gpu_tensor_operators(factors, correction, mapping, base, device):
    """Assemble sequentially to bound temporary memory to one full operator."""
    factors, correction, mapping, base = jax.device_put(
        (factors, correction, mapping, base), device)
    (x, dx, xx), (y, dy, yy) = factors
    if 6*x.shape[0]*x.shape[1]*y.shape[1]*x.dtype.itemsize <= _FUSED_OPERATOR_BYTES:
        return _tensor_operators(factors, correction, mapping, base)
    pairs = ((x,y,1.), (x,dy,1.), (dx,y,-1.), (dx,dy,1.), (x,yy,1.), (xx,y,-1.))
    operators = []
    for k, (a,b,sign) in enumerate(pairs):
        r = None if correction is None else correction[k]
        operators.append(_tensor_operator(a,b,sign,r,mapping).block_until_ready())
    if base is not None and correction is not None:
        base = tuple(b+r[:, -1] for b,r in zip(base, correction))
    return tuple(operators) if base is None else (tuple(operators), base)


@jax.jit
def _prepare_volume(operators, base, lift, scale, constraints):
    fields = tuple(o @ lift+b for o, b in zip(operators, base))
    raw = (tuple(o*scale for o in operators[1:]) if constraints is None
           else tuple(o @ constraints for o in operators[1:]))
    return fields, raw


@partial(jax.jit, donate_argnums=(0,))
def _prepare_operator(operator, base, lift, scale, constraints):
    field = operator @ lift + base
    raw = operator*scale if constraints is None else operator @ constraints
    return field, raw


_prepare_operator_allocating = jax.jit(_prepare_operator.__wrapped__)


def prepare_gpu_volume(operators, base, lift, scale, constraints, device):
    """Consume operator storage after computing the fixed lift fields."""
    base, lift, scale, constraints = jax.device_put((base, lift, scale, constraints), device)
    fields = [jax.device_get(operators[0] @ lift + base[0])]
    # Streamfunction is needed only for its lift, not the volume weak form.
    operators[0].delete()
    raw = []
    for o, b in zip(operators[1:], base[1:]):
        prepare = _prepare_operator if constraints is None else _prepare_operator_allocating
        field, scaled = prepare(o, b, lift, scale, constraints)
        fields.append(jax.device_get(field))
        raw.append(scaled.block_until_ready())
        if constraints is not None:
            o.delete()
    return tuple(fields), tuple(raw)
