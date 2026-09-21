"""JAX kernels for the mapped, derivative-closed BSPF slope discretization.

The caller supplies the retained Galerkin basis/metric arrays as a PyTree.
No rectangular pressure approximation or change of basis is introduced.
Configure JAX float64 before assembling the plan. Setup is host-side; these
matrix-free mass, RHS and RK4 kernels execute entirely on the JAX device.
"""
import jax
import jax.numpy as jnp
from pybspf.tensor import tensor_product
from pybspf.tensor import tensor_elliptic_solve
from pybspf.time_integration import rk4_stages
from bspf_models._numerics._tensor_pcg import plan_tensor_preconditioner
from bspf_models._numerics._tensor_pcg import tensor_pcg_device


def tensor(x, a, z):
    return tensor_product(x, z, a)


@jax.jit
def fields(plan, a):
    px, pz, metric = plan['Px'], plan['Pz'], plan['metric']
    keys = sorted({key for mp in metric for key in mp})
    dd = {key: tensor(px[key[0]], a, pz[key[1]]) for key in keys}
    return tuple(sum(v * dd[key] for key, v in mp.items()) for mp in metric)


@jax.jit
def adjoint(plan, values):
    rr = {}
    for f, mp in zip(values, plan['metric']):
        for key, v in mp.items():
            rr[key] = rr.get(key, 0.) + f*v
    return sum(plan['Px'][i].T @ (plan['W']*f) @ plan['Pz'][j]
               for (i, j), f in rr.items())


@jax.jit
def mass(plan, a):
    return sum(tensor(x, a, z) for x, z in plan['mass_terms'])


def plan_mass_preconditioner(mass_terms):
    """Reuse the stream/tokamak tensor inverse with actual depth coefficients.

    M0 = A(g/d) tensor Kz + Kx(g*d) tensor Mz is SPD. Terrain cross
    terms remain in the full mass operator used by PCG, never discarded.
    """
    mx, kz = mass_terms[0]
    kx, mz = mass_terms[2]
    pre = plan_tensor_preconditioner(mx, kx, mz, kz, 1., 1.)
    return tuple(jnp.asarray(v) for v in (pre.denominator, pre.left, pre.right))


@jax.jit
def solve_mass(plan, rhs):
    return tensor_pcg_device(
        lambda a: mass(plan, a), rhs,
        lambda r: tensor_elliptic_solve(r, *plan['mass_preconditioner']),
        rtol=1e-11, atol=1e-25, maxiter=250)


@jax.jit
def rhs(plan, a, bc):
    u, w, ux, uz, wx, wz = fields(plan, a)
    tx, tz = plan['Tx'], plan['Tz']
    d, k, W, n2 = (plan[key] for key in ('d', 'k', 'W', 'n2'))
    nu, kappa = plan['nu'], plan['kappa']
    b = tensor(tx[0], bc, tz[0])
    bs = tensor(tx[0], bc, tz[1])
    bx = tensor(tx[1], bc, tz[0])-k/d*bs
    bz = bs/d
    rr = adjoint(plan, (-.5*(u*ux+w*uz), -.5*(u*wx+w*wz)+b,
                        .5*u*u-nu*ux, .5*u*w-nu*uz,
                        .5*u*w-nu*wx, .5*w*w-nu*wz))
    at, info = solve_mass(plan, rr)
    f = -.5*(u*bx+w*bz)-w*n2
    gx = .5*u*b-kappa*bx
    gz = .5*w*b-kappa*(bz+n2)
    bt = (tx[0].T @ (W*f) @ tz[0] + tx[1].T @ (W*gx) @ tz[0]
          + tx[0].T @ (W*(gz-k*gx)/d) @ tz[1])
    return (at, bt), info


@jax.jit
def rk4(plan, a, b, dt):
    state, stages = rk4_stages((a, b), dt, lambda state: rhs(plan, *state))
    info = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *stages)
    return state, info


@jax.jit
def rk4_checked(plan, a, b, dt):
    """Keep fields on device and pack the step validation into one small buffer.

    Layout: four CG counts, four residuals, four success flags, state-finite
    flag. Counts/flags are exactly representable in the float64 buffer.
    """
    state, (counts, residuals, ok) = rk4(plan, a, b, dt)
    finite = jnp.all(jnp.isfinite(state[0])) & jnp.all(jnp.isfinite(state[1]))
    report = jnp.concatenate((counts.astype(residuals.dtype), residuals,
                              ok.astype(residuals.dtype),
                              finite.astype(residuals.dtype).reshape(1)))
    return state, report
