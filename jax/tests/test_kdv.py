"""Nonperiodic KdV: dissipativity, boundary lifting, and nonautonomous forcing."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import bspf_jax as b


def setup():
    x = jnp.linspace(-1., 1., 17)
    spatial = b.plan_1d(x, degree=3, n_basis=8, boundary_points=5)
    return x, spatial, b.plan_kdv(spatial, quadrature_order=6)


def test_homogeneous_linear_operator_is_dissipative():
    x, spatial, kdv = setup()
    derivative = b.differentiate(spatial, jnp.eye(x.size))
    left, right = derivative[0, kdv.free], derivative[-1, kdv.free]
    np.testing.assert_allclose(kdv.dispersion+kdv.dispersion.T,
                               -jnp.outer(left, left)-jnp.outer(right, right), atol=1e-10)
    assert np.linalg.eigvalsh(np.asarray(kdv.mass)).min() > 0


def test_time_dependent_nonperiodic_cubic_linear_solution():
    # u=x^3-6t solves u_t+u_xxx=0. Its two values differ by 2, and its
    # nonzero right slope is 3. Omitting the mass lifting term breaks this.
    x, spatial, kdv = setup()
    times = jnp.linspace(0., .2, 5)
    def boundary(t): return jnp.array([-1.-6*t, 1.-6*t, 3.])
    out = jax.jit(lambda p: b.integrate_kdv(
        p, x**3, times, boundary=boundary, nonlinearity=0., substeps=2))(kdv)
    np.testing.assert_allclose(out, x[None, :]**3-6*times[:, None], atol=2e-10)
    np.testing.assert_allclose(b.differentiate(spatial, out.T)[-1], 3., atol=1e-9)
    np.testing.assert_array_equal(out[:, 0], -1.-6*times)
    np.testing.assert_array_equal(out[:, -1], 1.-6*times)


def test_initial_output_and_arguments():
    x, spatial, kdv = setup()
    def boundary(t): return jnp.array([-1., 1., 1.])
    out = b.integrate_kdv(kdv, x, jnp.array([0.]), boundary=boundary)
    np.testing.assert_array_equal(out[0], x)
    for steps in (0, True, 1.5):
        with pytest.raises(ValueError, match='substeps'):
            b.integrate_kdv(kdv, x, jnp.array([0., 1.]), boundary=boundary, substeps=steps)
    with pytest.raises(ValueError, match='real vector'):
        b.integrate_kdv(kdv, x+0j, jnp.array([0.]), boundary=boundary)
    with pytest.raises(ValueError, match='three real'):
        b.integrate_kdv(kdv, x, jnp.array([0.]), boundary=lambda t: jnp.zeros(2))
