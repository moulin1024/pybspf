"""Independent geometry, compatibility, energy, residency and NS accuracy checks."""
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as la
from scipy.interpolate import BSpline

from bspf_jax.mapped_navier_stokes import (
    MappedNavierStokesPlan, _advance, _convection, _spline_jets,
)
from bspf_jax.mapped_poisson import _axis

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples' / 'pde'))
from mapped_spline_flow import manufactured, independent_error


@pytest.fixture(scope='module')
def gpu():
    try:
        return jax.devices('gpu')[0]
    except RuntimeError:
        pytest.skip('GPU device unavailable')


@pytest.fixture(scope='module')
def plan(gpu):
    return MappedNavierStokesPlan(elements=(3, 3), device=gpu, viscosity=.05, dt=.002, boundary="nitsche")


@pytest.mark.parametrize('degree', [2, 3, 4])
def test_second_derivatives_against_scipy(gpu, degree):
    knots, _, _ = _axis(4, degree, degree+3)
    points = np.unique(np.r_[np.linspace(0, 1, 61), np.linspace(0, 1, 5)])
    reference = BSpline(knots, np.eye(len(knots)-degree-1), degree)
    k, x = jax.device_put((knots, points), gpu)
    with jax.transfer_guard('disallow'):
        jets = _spline_jets(k, x, degree=degree)
        jax.block_until_ready(jets)
    for derivative, result in enumerate(jets):
        np.testing.assert_allclose(np.asarray(result), reference(points, nu=derivative), atol=5e-13, rtol=5e-13)


def test_piola_identity_physical_derivative_and_circulation(plan):
    p = plan
    rng = np.random.default_rng(61)
    state = jax.device_put(rng.normal(size=p.dofs)*.01, p.device)
    r, t = np.array((.24, .63)), np.array((.18, .79))
    values = jax.device_get(p.evaluate(state, r, t))
    op = p._operators(*jax.device_put((r, t), p.device))
    inv, jac, det = jax.device_get((op['inverse'], op['jacobian'], op['determinant']))
    # Piola image of the reference curl, independently using SciPy splines.
    kr, kt = jax.device_get(p.base.knots)
    br = BSpline(kr, np.eye(len(kr)-p.degree-1), p.degree)
    bt = BSpline(kt, np.eye(len(kt)-p.degree-1), p.degree)
    ids = np.asarray(p.ids)
    coefficients = np.asarray(state*p.scale)
    local = np.where(ids >= 0, coefficients[np.maximum(ids, 0)], 0)
    ref_r, ref_t = br(r, nu=1) @ local @ bt(t).T, br(r) @ local @ bt(t, nu=1).T
    reference = np.stack((ref_t, -ref_r), axis=-1).reshape(4, -1, 2)
    piola = np.einsum('pqij,pqj->pqi', jac, reference)/det[..., None]
    np.testing.assert_allclose(values['velocity'].reshape(4, -1, 2), piola, atol=3e-15)
    h = 1e-6
    differences = []
    for dr, dt in ((h, 0), (0, h)):
        plus = np.asarray(p.evaluate(state, r+dr, t+dt)['velocity'])
        minus = np.asarray(p.evaluate(state, r-dr, t-dt)['velocity'])
        differences.append((plus-minus)/(2*h))
    numerical = np.einsum('pqia,pqaj->pqij', np.stack(differences, axis=-1).reshape(4, -1, 2, 2), inv)
    np.testing.assert_allclose(values['gradient'].reshape(4, -1, 2, 2), numerical, atol=3e-9, rtol=3e-8)
    # The hole constant must be free: omitting it loses a circulation mode.
    mode = jnp.zeros_like(state).at[-1].set(1/p.scale[-1])
    wall = p.evaluate(mode, np.array((0., 1.)), np.linspace(0, 1, 19))
    np.testing.assert_allclose(np.asarray(wall['stream'])[:, 0], 1, atol=2e-15)
    np.testing.assert_allclose(np.asarray(wall['stream'])[:, 1], 0, atol=2e-15)
    assert np.max(np.linalg.norm(np.asarray(wall['velocity'])[:, 0], axis=-1)) > 1


def test_normal_continuity_and_closed_domain_energy(plan):
    p = plan
    state = jax.device_put(np.random.default_rng(71).normal(size=p.dofs)*.003, p.device)
    diagnostic = jax.device_get(p.diagnostics(state))
    assert diagnostic['divergence_linf'] < 2e-12
    assert diagnostic['normal_jump_linf'] < 2e-14
    assert diagnostic['boundary_normal_error'] < 2e-14
    assert abs(diagnostic['net_boundary_flux']) < 2e-14
    assert abs(diagnostic['convection_power']) < 2e-14
    assert diagnostic['tangent_jump_l2'] > 1e-4  # Exercise actual nonconforming traces.
    m, k = jax.device_get((p.data['mass'], p.data['stiffness']))
    np.testing.assert_allclose(k, k.T, atol=2e-12)
    assert la.eigvalsh(k, m).min() > 0
    # Viscosity dissipates energy; the central convection operator does no work.
    rhs = -np.asarray(_convection(p.data, state))-p.viscosity*k @ np.asarray(state)
    derivative = la.solve(m, rhs, assume_a='pos')
    power = np.asarray(state) @ m @ derivative
    np.testing.assert_allclose(power, -p.viscosity*diagnostic['viscous_quadratic'], atol=2e-13, rtol=2e-12)
    after = p.advance(state, steps=20)
    assert float(p.diagnostics(after)['kinetic_energy']) < diagnostic['kinetic_energy']


def test_gpu_time_loop_has_no_implicit_transfers(plan):
    p = plan
    state = jax.device_put(np.random.default_rng(17).normal(size=p.dofs)*1e-4, p.device)
    time = jax.device_put(0., p.device)
    _, _, _, forcing = manufactured(viscosity=p.viscosity)
    with jax.transfer_guard('disallow'):
        after = _advance(p.data, state, time, p.dt_device, steps=3, forcing=forcing)
        jax.block_until_ready(after)
    assert all(x.devices() == {p.device} for x in jax.tree.leaves(p.data))
    assert after.devices() == {p.device}
    assert np.all(np.isfinite(np.asarray(after)))


@pytest.mark.parametrize("boundary", ["strong", "nitsche"])
def test_continuous_ns_manufactured_convergence(gpu, boundary):
    errors = []
    for n in (4, 8):
        p = MappedNavierStokesPlan(elements=(n, n), device=gpu, viscosity=.02, dt=.002, boundary=boundary)
        _, velocity, exact, forcing = manufactured(viscosity=p.viscosity)
        state = p.advance(p.project(velocity), steps=50, forcing=forcing)
        error = independent_error(p, state, exact, .1)
        assert np.isfinite(error['velocity_l2'])
        errors.append(error['velocity_l2'])
    assert errors[0]/errors[1] > 4


def test_second_order_time_convergence(plan):
    p = plan
    # Low generalized modes avoid an unresolved impulsive boundary layer.
    # This CPU eigensystem chooses test data only; every evolution runs on GPU.
    mass, stiffness = jax.device_get((p.data['mass'], p.data['stiffness']))
    _, vectors = la.eigh(stiffness, mass, subset_by_index=(0, 2))
    initial = jax.device_put(.01*vectors[:, 0]+.007*vectors[:, 2], p.device)
    time = jax.device_put(0., p.device)
    states = []
    for dt in (.02, .01, .005, .00125):
        data = dict(p.data)
        dt_device = jax.device_put(dt, p.device)
        data['factor'] = jnp.linalg.cholesky(data['mass']+dt_device/2*data['nu']*data['stiffness'])
        state = _advance(data, initial, time, dt_device, steps=round(.2/dt))
        states.append(np.asarray(state))
    errors = [np.sqrt((a-states[-1]) @ mass @ (a-states[-1])) for a in states[:-1]]
    ratios = np.array(errors[:-1])/errors[1:]
    print('TEMPORAL_CONVERGENCE', {'errors': errors, 'ratios': ratios.tolist()}, flush=True)
    assert np.all(ratios > 3.5)
    assert np.all(ratios < 5.)


def test_nonzero_lift_and_pressure_gradient_balance(gpu):
    # u=(x,-y), p=-(x*x+y*y)/2 is a steady exact NS solution with
    # nonzero boundary data: convection is a pure pressure gradient, Delta u=0.
    # Overintegrate the curved geometry: this exact-balance test targets roundoff.
    p = MappedNavierStokesPlan(elements=(3, 3), device=gpu, viscosity=.05, dt=.002,
                              quadrature_order=15, lift=lambda x: x[0]*x[1])
    state = jax.device_put(np.zeros(p.dofs), gpu)
    assert float(jnp.linalg.norm(p.data['lift_load'])) < 2e-9
    assert float(jnp.linalg.norm(_convection(p.data, state))) < 2e-9
    after = p.advance(state, steps=50)
    error = jnp.sqrt(after @ p.data['mass'] @ after)
    assert float(error) < 2e-9
    assert float(p.diagnostics(after)['boundary_normal_error']) < 2e-14


def test_strong_boundary_constraints_preserve_circulation(gpu):
    p = MappedNavierStokesPlan(elements=(3, 3), device=gpu,
                              lift=lambda x: x[1], boundary='strong')
    state = jax.device_put(np.random.default_rng(51).normal(size=p.dofs)*.1, gpu)
    wall = p.evaluate(state, np.array((0., 1.)), np.linspace(0, 1, 39))
    target = np.broadcast_to(np.array((1., 0.)), wall['velocity'].shape)
    np.testing.assert_allclose(np.asarray(wall['velocity']), target, atol=2e-14)
    mode = jnp.zeros_like(state).at[-1].set(1/p.scale[-1])
    result = p.evaluate(mode, np.linspace(0, 1, 7), np.linspace(0, 1, 11))
    correction = np.asarray(result['stream'])-np.asarray(result['points'])[..., 1]
    np.testing.assert_allclose(correction[:, 0], 1, atol=2e-15)
    np.testing.assert_allclose(correction[:, -1], 0, atol=2e-15)
    assert np.max(np.linalg.norm(np.asarray(result['velocity'])-np.array((1., 0.)), axis=-1)) > 1
    stats = jax.device_get(p.diagnostics(state))
    assert stats['normal_jump_linf'] < 2e-13
    assert stats['boundary_velocity_error'] < 2e-14
