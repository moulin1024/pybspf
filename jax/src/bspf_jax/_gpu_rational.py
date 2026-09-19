"""Complex128 GPU evaluation of the differentiated rational Arnoldi basis."""
from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def field_rows(z, basis_values):
    r, dr, ddr = basis_values
    cz = z.conj()[:, None]
    o = 1 / z
    logz = jnp.log(jnp.abs(z)) + 1j*jnp.angle(z)
    zero = jnp.zeros_like(r)
    # Complex coefficient order: [f rational, g rational, f log, g log].
    # g contains -conj(a)*(z log z-z), making velocities single-valued.
    ub = jnp.column_stack((cz * dr - r, dr))
    vb = jnp.column_stack((-cz * dr - r, -dr))
    pb = jnp.column_stack((4 * dr, zero))
    wb = jnp.column_stack((-4 * dr, zero))
    u = jnp.column_stack(
        (
            ub.real,
            (z.conj() * o - 2 * logz).real,
            o.real,
            -ub.imag,
            -(z.conj() * o).imag,
            -o.imag,
        )
    )
    v = jnp.column_stack(
        (
            vb.imag,
            (-z.conj() * o).imag,
            -o.imag,
            vb.real,
            (-z.conj() * o - 2 * logz).real,
            -o.real,
        )
    )
    pressure = jnp.column_stack(
        (
            pb.real,
            (4 * o).real,
            jnp.zeros(len(z)),
            -pb.imag,
            -(4 * o).imag,
            jnp.zeros(len(z)),
        )
    )
    omega = jnp.column_stack(
        (
            wb.imag,
            (-4 * o).imag,
            jnp.zeros(len(z)),
            wb.real,
            (-4 * o).real,
            jnp.zeros(len(z)),
        )
    )
    xb = jnp.column_stack((cz * ddr, ddr))
    yb = jnp.column_stack((-2 * dr - cz * ddr, -ddr))
    ux = jnp.column_stack(
        (
            xb.real,
            (-z.conj() * o**2 - o).real,
            (-(o**2)).real,
            -xb.imag,
            -(-z.conj() * o**2 + o).imag,
            -(-(o**2)).imag,
        )
    )
    vx = jnp.column_stack(
        (
            yb.imag,
            (-o + z.conj() * o**2).imag,
            (o**2).imag,
            yb.real,
            (-3 * o + z.conj() * o**2).real,
            (o**2).real,
        )
    )
    return u, v, pressure, omega, ux, vx


@jax.jit
def stream_rows(z, basis_values, gauge):
    def retained(a):
        removed = 2*basis_values[0].shape[1]+1
        return jnp.concatenate((a[:, :removed], a[:, removed+1:]), axis=1)
    r = basis_values[0]
    logz = jnp.log(jnp.abs(z)) + 1j*jnp.angle(z)
    cz = z.conj()
    base = jnp.column_stack((cz[:, None]*r, r))
    psi = retained(jnp.column_stack((base.imag, (cz*logz-z*logz+z).imag,
                           logz.imag, base.real, (cz*logz+z*logz-z).real,
                           logz.real))) - gauge
    u, v, _, omega, ux, vx = field_rows(z, basis_values)
    return (psi,) + tuple(retained(a) for a in (u, v, ux, vx-omega, vx))


@partial(jax.jit, static_argnames=("degree", "polynomial"))
def construct_block(z, poles, *, degree, polynomial):
    """Twice modified Gram–Schmidt, retaining the host algorithm's order."""
    q = jnp.zeros((degree+1, len(z)), dtype=jnp.complex128).at[0].set(1)
    h = jnp.zeros((degree+1, degree), dtype=jnp.complex128)

    if degree == 0:
        return h

    def step(k, state):
        q, h = state
        value = z*q[k] if polynomial else q[k]/(z-poles[k])
        hk = jnp.zeros(degree+1, dtype=jnp.complex128)

        def orthogonalize(_, state):
            def subtract(j, state):
                value, hk = state
                v = jnp.vdot(q[j], value)/len(z)
                return value-v*q[j], hk.at[j].add(v)
            return jax.lax.fori_loop(0, k+1, subtract, state)

        value, hk = jax.lax.fori_loop(0, 2, orthogonalize, (value, hk))
        norm = jnp.linalg.norm(value)/jnp.sqrt(float(len(z)))
        return q.at[k+1].set(value/norm), h.at[:, k].set(hk.at[k+1].set(norm))

    return jax.lax.fori_loop(0, degree, step, (q, h))[1]


@jax.jit
def evaluate_blocks(z, h, poles, polynomial):
    """Evaluate padded blocks together, sharing one compiled recurrence loop."""
    def block(h, poles, polynomial):
        n = h.shape[1]
        q = jnp.zeros((len(z), n+1), dtype=jnp.complex128).at[:, 0].set(1)
        d = jnp.zeros_like(q)
        dd = jnp.zeros_like(q)

        def step(k, values):
            q, d, dd = values
            inverse = 1/(z-poles[k])
            r = jnp.where(polynomial, z, inverse)
            dr = jnp.where(polynomial, 1., -inverse*inverse)
            ddr = jnp.where(polynomial, 0., 2*inverse*inverse*inverse)
            hk, scale = h[:, k], h[k+1, k].real
            qk = (r*q[:, k]-q @ hk)/scale
            dk = (r*d[:, k]+dr*q[:, k]-d @ hk)/scale
            ddk = (r*dd[:, k]+2*dr*d[:, k]+ddr*q[:, k]-dd @ hk)/scale
            return q.at[:, k+1].set(qk), d.at[:, k+1].set(dk), dd.at[:, k+1].set(ddk)

        return jax.lax.fori_loop(0, n, step, (q, d, dd))

    return jax.vmap(block, in_axes=(0, 0, 0))(h, poles, polynomial)


@partial(jax.jit, static_argnames=("sizes",))
def unpad_blocks(values, *, sizes):
    return tuple(jnp.concatenate([v[i, :, int(i != 0):n+1] for i, n in enumerate(sizes)], axis=1)
                 for v in values)
