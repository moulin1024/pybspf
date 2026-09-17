"""Weighted inertia, physical energy, driven boundaries and wave propagation."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import bspf_jax as b


def grid(n=33):
    x = jnp.linspace(0., 1., n)
    return x, b.plan_1d(x, degree=7, n_basis=16, boundary_points=9)


def test_variable_density_mass_and_energy():
    x, spatial = grid()
    p = b.plan_alfven(spatial, density=lambda z: 1+z, magnetic_field=2., permeability=2.)
    ones = jnp.ones_like(x)
    np.testing.assert_allclose(ones@p.mass@ones, 1.5, atol=1e-11)
    # xi=z, velocity=1: E=1/2 int (1+z+2) dz=1.75.
    np.testing.assert_allclose(b.alfven_energy(p, x, ones), 1.75, atol=1e-10)
    # xi=z², velocity=1: boundary power=2*(2-0)=4.
    np.testing.assert_allclose(b.alfven_boundary_power(p, x*x, ones), 4., atol=1e-9)
    assert np.linalg.eigvalsh(p.mass).min() > 0


def test_uniform_normal_mode_with_physical_scaling_and_jit():
    x, spatial = grid()
    # v_A=2/sqrt(2*2)=1; catches missing B0, mu0 or density factors.
    p = b.plan_alfven(spatial, density=2., magnetic_field=2., permeability=2.)
    times = jnp.array([0., .4, 1., 2.])
    initial = jnp.sin(jnp.pi*x).at[jnp.array([0, -1])].set(0.)
    run = jax.jit(lambda p, q: b.integrate_alfven(p, q, 0*q, times,
                    boundary=lambda t: jnp.zeros(2), substeps=1000))
    q, v = run(p, initial)
    np.testing.assert_allclose(q, jnp.cos(jnp.pi*times[:, None])*initial, atol=2e-8)
    np.testing.assert_allclose(v, -jnp.pi*jnp.sin(jnp.pi*times[:, None])*initial, atol=2e-7)
    energy = b.alfven_energy(p, q, v)
    np.testing.assert_allclose(energy, energy[0], rtol=1e-9)


def test_driven_cavity_matches_independent_images():
    x, spatial = grid(65)
    p = b.plan_alfven(spatial)
    drive = lambda t: jnp.where((t > 0)&(t < .5), jnp.sin(2*jnp.pi*t)**8, 0.)
    boundary = lambda t: jnp.array([drive(t), 0.])
    times = jnp.linspace(0., 2., 81)
    q, v = b.integrate_alfven(p, 0*x, 0*x, times, boundary=boundary, substeps=50)
    exact = sum(drive(times[:, None]-2*m-x)-drive(times[:, None]-2*m-2+x) for m in range(3))
    np.testing.assert_allclose(q, exact, atol=2e-5)
    np.testing.assert_allclose(q[:, 0], drive(times), atol=0)
    np.testing.assert_allclose(q[:, -1], 0., atol=0)
    np.testing.assert_allclose(v[:, 0], jax.vmap(jax.grad(drive))(times), atol=1e-13)


def test_validation_and_single_time():
    x, spatial = grid()
    for density in (0., -1., jnp.nan, lambda z: -jnp.ones_like(z)):
        with pytest.raises(ValueError, match='density'):
            b.plan_alfven(spatial, density=density)
    for kwargs in ({'magnetic_field':0.}, {'permeability':-1.}):
        with pytest.raises(ValueError):
            b.plan_alfven(spatial, **kwargs)
    p = b.plan_alfven(spatial)
    bc = lambda t: jnp.zeros(2)
    q, v = b.integrate_alfven(p, 0*x, 0*x, jnp.array([0.]), boundary=bc)
    np.testing.assert_array_equal(q, jnp.zeros((1, x.size)))
    np.testing.assert_array_equal(v, q)
    with pytest.raises(ValueError, match='substeps'):
        b.integrate_alfven(p, 0*x, 0*x, jnp.array([0.]), boundary=bc, substeps=0)
    with pytest.raises(ValueError, match='two real'):
        b.integrate_alfven(p, 0*x, 0*x, jnp.array([0.]), boundary=lambda t: jnp.zeros(3))


def test_off_grid_derivative_preserves_original_interpolant():
    x, spatial = grid(65)
    points = jnp.linspace(.013, .987, 87)
    values = jnp.exp(.3*x)+jnp.sin(2*x)
    result = jax.jit(lambda f: b.interpolate(spatial, f, points, derivative=1))(values)
    np.testing.assert_allclose(result, .3*jnp.exp(.3*points)+2*jnp.cos(2*points), atol=2e-8)
    np.testing.assert_allclose(b.interpolate(spatial, values, x, derivative=1),
                               b.differentiate(spatial, values), atol=2e-11)
    for order in (-1, True, 1.5, 20):
        with pytest.raises(ValueError, match='derivative'):
            b.interpolate(spatial, values, points, derivative=order)
