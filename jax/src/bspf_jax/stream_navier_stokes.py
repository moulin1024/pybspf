"""Pointwise-divergence-free BSPF streamfunction Galerkin NS in 2D.

Only 1D factors are assembled. Fixed velocity or open vertical boundaries. Pressure eliminated exactly by divergence-free test functions.
"""

from types import SimpleNamespace
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from .pressure import _line_projector
from ._weak_basis import evaluate_basis
from ._flow_kernels import (
    tensor_product,
    curl_from_gradient,
    tensor_elliptic_solve,
    rk4_stages,
)


class StreamLine(NamedTuple):
    x: jax.Array
    points: jax.Array
    weights: jax.Array
    b: jax.Array
    g: jax.Array
    h: jax.Array
    bn: jax.Array
    gn: jax.Array
    hn: jax.Array
    lam: jax.Array
    bending: jax.Array
    transform: jax.Array
    projector: jax.Array
    layers: jax.Array
    rotation: jax.Array | None = None


class StreamNavierStokes2DPlan(NamedTuple):
    x: StreamLine
    y: StreamLine
    denominator: jax.Array
    nu: jax.Array
    open_x: float = 0.0
    boundary_inertia: float = 0.0
    dong_backflow: float = 0.0
    inertia_transform: jax.Array | None = None
    inertia_denominator: jax.Array | None = None


def _stream_line(
    x,
    *,
    quadrature_order=None,
    layers=(),
    clamped=True,
    dirichlet=False,
    endpoint_method="chebyshev",
    endpoint_points=16,
    chebyshev_modes=12,
    endpoint_regularization=1e-12,
    basis_executor=None,
    basis_precision="mpfr",
    basis_device=None,
):
    import scipy.linalg as la
    from scipy.interpolate import BSpline
    from scipy.special import roots_legendre

    layers = tuple(float(d) for d in layers)
    x = np.asarray(x, dtype=float)
    if (
        x.ndim != 1
        or len(x) < 17
        or not np.all(np.isfinite(x))
        or x[-1] <= x[0]
        or not np.allclose(
            np.diff(x), (x[-1] - x[0]) / (len(x) - 1), rtol=1e-10, atol=1e-14
        )
    ):
        raise ValueError("Require >=17 finite increasing uniform BSPF nodes per axis")
    if not all(np.isfinite(d) and d > 0 for d in layers) or len(set(layers)) != len(
        layers
    ):
        raise ValueError("Layer lengths must be finite, positive and distinct")
    if quadrature_order is not None and (
        not isinstance(quadrature_order, int) or quadrature_order < 2
    ):
        raise ValueError("quadrature_order must be an integer >= 2")
    x = np.asarray(x)
    if endpoint_method not in ("chebyshev", "taylor"):
        raise ValueError("endpoint_method must be chebyshev or taylor")
    if not isinstance(
        endpoint_points, (int, np.integer)
    ) or not 9 <= endpoint_points <= len(x):
        raise ValueError("Require 9 <= endpoint_points <= number of grid nodes")
    if (
        not isinstance(chebyshev_modes, (int, np.integer))
        or not 9 <= chebyshev_modes <= endpoint_points
    ):
        raise ValueError("Require 9 <= chebyshev_modes <= endpoint_points")
    if not np.isfinite(endpoint_regularization) or endpoint_regularization < 0:
        raise ValueError("endpoint_regularization must be finite and nonnegative")
    projector, *_ = _line_projector(
        x,
        9,
        32,
        13,
        endpoint_points,
        endpoint_method,
        chebyshev_modes,
        endpoint_regularization,
    )
    host = SimpleNamespace(x=x, P=np.asarray(projector))
    breaks = np.linspace(x[0], x[-1], 20)
    knots = np.r_[np.repeat(x[0], 14), breaks[1:-1], np.repeat(x[-1], 14)]
    spline = BSpline(knots, np.eye(32), 13)
    order = quadrature_order or max(
        24, int(np.ceil(1.5 * np.pi * (len(x) - 1) / 19)) + 8
    )
    if layers:
        extra = [x[0] + min(layers) * a for a in (0.5, 1, 2, 4, 8, 16, 32, 64)]
        extra += [x[-1] - min(layers) * a for a in (0.5, 1, 2, 4, 8, 16, 32, 64)]
        breaks = np.unique(np.r_[breaks, [v for v in extra if x[0] < v < x[-1]]])
    q, w = roots_legendre(order)
    points = np.concatenate(
        [(a + b) / 2 + (b - a) / 2 * q for a, b in zip(breaks[:-1], breaks[1:])]
    )
    weights = np.concatenate([(b - a) / 2 * w for a, b in zip(breaks[:-1], breaks[1:])])
    # The first QR needs only values. Derivatives and nodal traces are
    # evaluated below after the sensitive normalization is available.
    (b,) = evaluate_basis(
        host, spline, points, values_only=True, executor=basis_executor,
        precision=basis_precision, device=basis_device,
    )
    if layers:
        enrichment = [
            np.exp(sign * (points - endpoint) / delta)
            for delta in layers
            for endpoint, sign in ((x[0], -1), (x[-1], 1))
        ]
        b = np.column_stack((b, np.array(enrichment).T))
    # Orthonormalize the entire enriched space BEFORE imposing the walls.
    # This keeps the boundary null-space calculation well conditioned, even
    # when an exponential is nearly represented by the original BSPF basis.
    root_w = np.sqrt(weights)
    _, rv = la.qr(root_w[:, None] * b, mode="economic")
    transform = la.solve_triangular(rv, np.eye(rv.shape[0]))
    bc, gc, hc = evaluate_basis(
        host, spline, points, second=True, transform=transform, layers=layers,
        executor=basis_executor, precision=basis_precision, device=basis_device,
    )
    bnc, gnc, hnc = evaluate_basis(
        host, spline, x, second=True, transform=transform, layers=layers,
        executor=basis_executor, precision=basis_precision, device=basis_device,
    )
    qv, rv = la.qr(root_w[:, None] * bc, mode="economic")
    correction = la.solve_triangular(rv, np.eye(rv.shape[0]))
    # A magnetic flux potential needs only its value fixed at the wall;
    # streamfunction velocity walls additionally constrain the normal derivative.
    traces = np.vstack((bnc[[0, -1]], gnc[[0, -1]])) if clamped else bnc[[0, -1]]
    constraints = traces @ correction
    constraints /= la.norm(constraints, axis=1)[:, None]
    z = (
        la.null_space(constraints)
        if clamped or dirichlet
        else np.eye(correction.shape[1])
    )
    constrained = correction @ z
    normalized_g = gc @ constrained
    stiff = normalized_g.T @ (weights[:, None] * normalized_g)
    lam, v = la.eigh(stiff)
    rotation = constrained @ v
    bc = (qv @ z @ v) / root_w[:, None]
    gc, hc = gc @ rotation, hc @ rotation
    bending = hc.T @ (weights[:, None] * hc)
    return StreamLine(
        *map(
            jnp.asarray,
            (
                x,
                points,
                weights,
                bc,
                gc,
                hc,
                bnc @ rotation,
                gnc @ rotation,
                hnc @ rotation,
                lam,
                bending,
                transform,
                host.P,
                np.asarray(layers),
                rotation,
            ),
        )
    )


def plan_stream_navier_stokes2d(
    x,
    y,
    *,
    viscosity=0.002,
    x_layers=(),
    y_layers=(),
    quadrature_order=None,
    x_boundary="fixed",
    boundary_D0=1.0,
):
    """Build only 1D factors, retaining degree-13/q9 BSPF with Cheb12/window16.

    Boundary-layer enrichment is optional and adds exp(-distance/delta) basis
    functions at both ends for every supplied delta. It adds degrees of freedom;
    it does not damp the solution or replace any original BSPF functions.
    With x_boundary="open", the vertical faces have natural traction with
    incoming-flow relaxation to the supplied base shear (zero if no lift).
    x_boundary="dynamic" adds boundary inertia nu*boundary_D0 and Dong
    backflow traction, with prescribed reference-flow traction compensation.
    Horizontal faces remain clamped; the default "fixed" clamps all faces.
    Pressure is
    eliminated by curl test functions. All stage solves are diagonal tensor
    Poisson solves. Runtime float64; host setup requires SciPy and gmpy2.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 before setup")
    if not np.isfinite(viscosity) or viscosity < 0:
        raise ValueError("viscosity must be finite and nonnegative")
    if x_boundary not in ("fixed", "open", "dynamic"):
        raise ValueError("x_boundary must be fixed, open, or dynamic")
    lx = _stream_line(
        x,
        layers=x_layers,
        quadrature_order=quadrature_order,
        clamped=x_boundary == "fixed",
    )
    ly = _stream_line(y, layers=y_layers, quadrature_order=quadrature_order)
    result = StreamNavierStokes2DPlan(
        lx,
        ly,
        lx.lam[:, None] + ly.lam[None, :],
        jnp.asarray(viscosity),
        float(x_boundary != "fixed"),
    )
    if x_boundary == "dynamic":
        result = with_stream_dynamic_boundary(result, D0=boundary_D0)
    return result


def with_stream_dynamic_boundary(p, *, D0=1.0, dong_backflow=True):
    """Attach constant boundary inertia beta=nu*D0 to an existing open plan.

    One generalized 1D eigendecomposition gives an exact direct inertia solve.
    The volume basis, physical stiffness, seed projection and viscosity are
    unchanged. Dong's sharp backflow switch is used; with a shear lift the
    prescribed boundary load subtracts E(U_ref) to maintain that base flow.
    D0=0 provides the static Dong-traction control; dong_backflow=False retains
    the prior incoming relaxation law while isolating the effect of inertia.
    """
    import scipy.linalg as la

    if not float(p.open_x):
        raise ValueError("Boundary inertia requires open vertical faces")
    if not np.isfinite(D0) or D0 < 0 or float(p.nu) <= 0:
        raise ValueError("Require finite D0>=0 and positive viscosity")
    beta = float(p.nu) * D0
    b, g = np.asarray(p.x.bn)[[0, -1]], np.asarray(p.x.gn)[[0, -1]]
    mass = np.eye(len(p.x.lam)) + beta * (b.T @ b)
    stiffness = np.diag(np.asarray(p.x.lam)) + beta * (g.T @ g)
    lam, rotation = la.eigh(stiffness, mass)
    denominator = lam[:, None] + np.asarray(p.y.lam)[None, :]
    if np.min(denominator) <= 0:
        raise ValueError("Augmented inertia is not positive definite")
    return p._replace(
        boundary_inertia=beta,
        dong_backflow=float(dong_backflow),
        inertia_transform=jnp.asarray(rotation),
        inertia_denominator=jnp.asarray(denominator),
    )


def stream_ns_inertia_apply(p, a):
    """Apply physical plus boundary kinetic mass without assembling a 2D matrix."""
    b, g = p.x.bn[jnp.array([0, -1])], p.x.gn[jnp.array([0, -1])]
    return p.denominator * a + p.boundary_inertia * (
        (b.T @ (b @ a)) * p.y.lam[None, :] + g.T @ (g @ a)
    )


def stream_ns_inertia_solve(p, load):
    """Direct tensor solve; original coordinates retained for all volume terms."""
    if p.inertia_transform is None:
        return tensor_elliptic_solve(load, p.denominator)
    r = p.inertia_transform
    return tensor_elliptic_solve(load, p.inertia_denominator, left=r)


def stream_ns_velocity(p, a, nodes=False, thickness=None):
    """Evaluate curl(psi), optionally adding the fixed tanh(y/thickness) lift."""
    x, y = p.x, p.y
    bx, gx = (x.bn, x.gn) if nodes else (x.b, x.g)
    by, gy = (y.bn, y.gn) if nodes else (y.b, y.g)
    u, v = curl_from_gradient(tensor_product(gx, by, a), tensor_product(bx, gy, a))
    if thickness is not None:
        yy = y.x if nodes else y.points
        u = u + jnp.tanh(yy[None, :] / thickness)
    return jnp.stack((u, v), axis=-1)


def stream_ns_vorticity(p, a, nodes=False, thickness=None):
    x, y = p.x, p.y
    bx, hx = (x.bn, x.hn) if nodes else (x.b, x.h)
    by, hy = (y.bn, y.hn) if nodes else (y.b, y.h)
    omega = -tensor_product(hx, by, a) - tensor_product(bx, hy, a)
    if thickness is not None:
        yy = y.x if nodes else y.points
        omega = omega - (1 - jnp.tanh(yy[None, :] / thickness) ** 2) / thickness
    return omega


def stream_ns_load(p, force):
    """Integrate a physical vector force against curl test functions."""
    x, y = p.x, p.y
    w = x.weights[:, None] * y.weights[None, :]
    return x.b.T @ (w * force[..., 0]) @ y.g - x.g.T @ (w * force[..., 1]) @ y.b


def stream_ns_boundary_load(p, traction):
    """Integrate vector traction at x endpoints / y quadrature points.

    Shape (2, len(p.y.points), 2); use physical traction nu*d_n(u)-p*n.
    This is an additional load; RHS already includes the kinetic-pressure
    conversion and incoming reservoir relaxation on open faces.
    """
    bx, gx = p.x.bn[jnp.array([0, -1])], p.x.gn[jnp.array([0, -1])]
    return p.open_x * (
        bx.T @ (traction[..., 0] * p.y.weights) @ p.y.g
        - gx.T @ (traction[..., 1] * p.y.weights) @ p.y.b
    )


def stream_ns_open_velocity(p, a, thickness=None):
    """Velocity on the two vertical faces at y quadrature points."""
    bx, gx = p.x.bn[jnp.array([0, -1])], p.x.gn[jnp.array([0, -1])]
    u, v = bx @ a @ p.y.g.T, -gx @ a @ p.y.b.T
    if thickness is not None:
        u = u + jnp.tanh(p.y.points[None, :] / thickness)
    return jnp.stack((u, v), axis=-1)


def _open_load(p, a, thickness):
    velocity = stream_ns_open_velocity(p, a, thickness)
    normal = jnp.array([-1.0, 1.0])[:, None]
    un = normal * velocity[..., 0]
    reference = jnp.zeros_like(velocity)
    if thickness is not None:
        reference = reference.at[..., 0].set(jnp.tanh(p.y.points / thickness))
    # Physical traction = min(un,0)*(u-u_ref). Outflow: zero traction.
    traction = jnp.minimum(un, 0)[..., None] * (velocity - reference)

    def dong_e(v):
        s = normal * v[..., 0]
        e = 0.5 * s[..., None] * v
        e = e.at[..., 0].add(0.5 * normal * jnp.sum(v * v, axis=-1))
        return jnp.where((s < 0)[..., None], e, 0.0)

    dong = dong_e(velocity) - dong_e(reference)
    traction = (1 - p.dong_backflow) * traction + p.dong_backflow * dong
    # Rotational formulation uses total pressure P=p+|u|^2/2.
    traction = traction.at[..., 0].add(-0.5 * normal * jnp.sum(velocity**2, axis=-1))
    return stream_ns_boundary_load(p, traction)


class StreamSponge2D(NamedTuple):
    mass: jax.Array
    stiffness: jax.Array
    sigma: jax.Array


def _smooth_step(s):
    # C-infinity ramp, exactly constant outside (0,1), safe at endpoints.
    z = jnp.clip(s, 1e-6, 1 - 1e-6)
    a, b = jnp.exp(-1 / z), jnp.exp(-1 / (1 - z))
    value = a / (a + b)
    deriv = value * (1 - value) * (1 / z**2 + 1 / (1 - z) ** 2)
    return (
        jnp.where(s <= 0, 0.0, jnp.where(s >= 1, 1.0, value)),
        jnp.where((s > 0) & (s < 1), deriv, 0.0),
    )


def plan_stream_sponge(p, *, interior=(-3.0, 3.0), strength=4.0):
    """External x-only relaxation to the lift, zero throughout interior.

    C-infinity ramp to strength at outer faces. Assemble only weighted 1D
    mass/stiffness; its load is -mass@a*lambda_y - stiffness@a.
    This adds a physical relaxation term in the extension, not viscosity or
    a filter. It acts on the perturbation; the supplied base lift is retained.
    """
    lo, hi = map(float, interior)
    left, right = float(p.x.x[0]), float(p.x.x[-1])
    if not left < lo < hi < right:
        raise ValueError("Absorbing extensions must lie outside the interior")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and nonnegative")
    sl = _smooth_step((lo - p.x.points) / (lo - left))[0]
    sr = _smooth_step((p.x.points - hi) / (right - hi))[0]
    sigma = strength * (sl + sr)
    w = p.x.weights * sigma
    return StreamSponge2D(
        p.x.b.T @ (w[:, None] * p.x.b), p.x.g.T @ (w[:, None] * p.x.g), sigma
    )


def stream_ns_sponge_load(p, a, sponge):
    """Curl-test load of -sigma*(u-u_ref), with nonpositive perturbation work."""
    return -(sponge.mass @ a) * p.y.lam[None, :] - sponge.stiffness @ a


def stream_ns_rhs(p, a, force=None, thickness=None, sponge=None):
    """Rotational-form convection, physical viscosity, and direct inertia solve.

    force is an integrated curl-test load (use stream_ns_load), not nodal
    acceleration. No pressure iteration, refinement, or filter. An optional
    sponge adds relaxation only in the explicitly configured extension.
    """
    if a.shape != p.denominator.shape or jnp.iscomplexobj(a):
        raise ValueError(
            "a must be a real modal streamfunction array matching the plan"
        )
    if force is not None and force.shape != a.shape:
        raise ValueError("force must be an integrated curl-test load matching a")
    u = stream_ns_velocity(p, a, thickness=thickness)
    omega = stream_ns_vorticity(p, a, thickness=thickness)
    conv = stream_ns_load(
        p, jnp.stack((u[..., 1] * omega, -u[..., 0] * omega), axis=-1)
    )
    diff = (
        p.x.bending @ a
        + a @ p.y.bending.T
        + 2 * p.x.lam[:, None] * a * p.y.lam[None, :]
    )
    result = conv - p.nu * diff + _open_load(p, a, thickness)
    if thickness is not None:
        yy = p.y.points[None, :]
        th = jnp.tanh(yy / thickness)
        upp = -2 * th * (1 - th * th) / thickness**2
        result = result + p.nu * (
            p.x.b.T @ (p.x.weights[:, None] * jnp.ones((len(p.x.weights), 1)))
        ) @ ((p.y.weights[None, :] * upp) @ p.y.g)
    if sponge is not None:
        result = result + stream_ns_sponge_load(p, a, sponge)
    if force is not None:
        result = result + force
    return stream_ns_inertia_solve(p, result)


def stream_ns_rk4_step(p, a, dt, force=None, thickness=None, sponge=None):
    """Explicit unsplit RK4; physical viscous/advective step limits still apply."""

    def f(v):
        return stream_ns_rhs(p, v, force, thickness, sponge), None

    return rk4_stages(a, dt, f)[0]


def stream_kh_initial(
    p, thickness=0.12, amplitude=0.03, wavelength=1.5, extension_cutoff=None
):
    """Kinetic projection of the analytic tapered KH seed on [-3,3] x [-1,1].

    On an extended domain, extension_cutoff=(3,4) preserves the exact original
    seed inside |x|<=3 and smoothly cuts its analytic continuation to zero
    outside |x|>=4. The taper is C-infinity and changes no interior values.
    Returns modal streamfunction coefficients. The base shear is supplied
    separately with thickness=... when evaluating velocity or advancing RHS.
    """
    x = p.x.points[:, None]
    y = p.y.points[None, :]
    k = 2 * jnp.pi / wavelength
    ex = (1 - (x / 3) ** 2) ** 4
    dex = -8 * x / 9 * (1 - (x / 3) ** 2) ** 3
    if extension_cutoff is not None:
        inner, outer = extension_cutoff
        if not 0 < inner < outer:
            raise ValueError("Require 0 < cutoff inner < outer")
        step, derivative = _smooth_step((jnp.abs(x) - inner) / (outer - inner))
        dex = dex * (1 - step) - ex * derivative * jnp.sign(x) / (outer - inner)
        ex = ex * (1 - step)
    ey = (1 - y * y) ** 4 * jnp.exp(-((y / (2 * thickness)) ** 2))
    dey = jnp.exp(-((y / (2 * thickness)) ** 2)) * (
        -8 * y * (1 - y * y) ** 3 - (1 - y * y) ** 4 * y / (2 * thickness**2)
    )
    u = amplitude * thickness * ex * jnp.cos(k * x) * dey
    v = -amplitude * thickness * (dex * jnp.cos(k * x) - k * ex * jnp.sin(k * x)) * ey
    return stream_ns_load(p, jnp.stack((u, v), axis=-1)) / p.denominator


def stream_evaluate_line(line, points, *, basis_executor=None,
                         basis_precision="mpfr", basis_device=None):
    """Evaluate the enriched basis and two derivatives at arbitrary host points."""
    from scipy.interpolate import BSpline

    x = np.asarray(line.x)
    host = SimpleNamespace(x=x, P=np.asarray(line.projector))
    knots = np.r_[
        np.repeat(x[0], 14), np.linspace(x[0], x[-1], 20)[1:-1], np.repeat(x[-1], 14)
    ]
    arrays = evaluate_basis(
        host,
        BSpline(knots, np.eye(32), 13),
        points,
        executor=basis_executor, precision=basis_precision, device=basis_device,
        second=True,
        transform=np.asarray(line.transform),
        layers=np.asarray(line.layers),
    )
    if line.rotation is None:
        return arrays
    return tuple(a @ np.asarray(line.rotation) for a in arrays)


def stream_ns_divergence(p, a, *, nodes=False):
    """Physical divergence via two separately contracted mixed derivatives.

    The curl representation cancels these at every physical point, not just
    against a finite pressure test space. Tiny nonzero values are roundoff.
    """
    gx = p.x.gn if nodes else p.x.g
    gy = p.y.gn if nodes else p.y.g
    return (gx @ a) @ gy.T - gx @ (a @ gy.T)


def stream_ns_energy(p, a, *, augmented=False):
    """Kinetic energy of the perturbation (excluding any lift)."""
    mass_a = stream_ns_inertia_apply(p, a) if augmented else p.denominator * a
    return 0.5 * jnp.sum(a * mass_a)
