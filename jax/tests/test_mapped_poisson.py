"""Independent checks of the mapped scalar foundation for compatible splines."""
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import BSpline

from bspf_jax.mapped_poisson import (
    MappedPoissonPlan, _axis, _load, _pcg, apply_stiffness,
    patch_geometry, spline_values,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples' / 'pde'))
from mapped_spline_poisson import independent_errors, manufactured


@pytest.fixture(scope='module')
def gpu():
    try:
        return jax.devices('gpu')[0]
    except RuntimeError:
        pytest.skip('GPU device unavailable')


@pytest.mark.parametrize('degree', [1, 2, 3, 4])
def test_basis_matches_scipy_including_endpoints(gpu, degree):
    knots, _, _ = _axis(5, degree, degree+2)
    points = np.unique(np.r_[np.linspace(0, 1, 83), np.linspace(0, 1, 6)])
    reference = BSpline(knots, np.eye(len(knots)-degree-1), degree)
    k, x = jax.device_put((knots, points), gpu)
    with jax.transfer_guard('disallow'):
        values, derivatives = spline_values(k, x, degree=degree)
        jax.block_until_ready((values, derivatives))
    assert values.devices() == derivatives.devices() == {gpu}
    values, derivatives = jax.device_get((values, derivatives))
    np.testing.assert_allclose(values, reference(points), atol=3e-15)
    np.testing.assert_allclose(derivatives, reference(points, nu=1), atol=5e-14)
    np.testing.assert_allclose(values.sum(axis=1), 1, atol=2e-15)
    np.testing.assert_allclose(derivatives.sum(axis=1), 0, atol=5e-14)


def test_exact_geometry_and_analytic_jacobian(gpu):
    p = MappedPoissonPlan(elements=(3, 4), degree=2, device=gpu)
    r, t = jax.device_put((np.linspace(0, 1, 9), np.linspace(0, 1, 13)), gpu)
    points, jac, det, inverse = jax.device_get(patch_geometry(*p.geometry, r, t))
    assert det.min() > 0
    np.testing.assert_allclose(p.hole.level(points[:, 0]), 1, atol=4e-15)
    vertices = np.asarray(p.geometry[0])
    edges = np.roll(vertices, -1, axis=0)-vertices
    np.testing.assert_allclose(points[:, -1], vertices[:, None]+np.asarray(t)[None, :, None]*edges[:, None], atol=1e-15)
    np.testing.assert_allclose(points[:, :, -1], np.roll(points[:, :, 0], -1, axis=0), atol=1e-15)
    np.testing.assert_allclose(jac @ inverse, np.broadcast_to(np.eye(2), jac.shape), atol=3e-15)
    # Finite differences of physical positions do not use the analytic Jacobian.
    h = 1e-6
    for axis in range(2):
        delta_r, delta_t = (h, 0) if axis == 0 else (0, h)
        plus = np.asarray(patch_geometry(*p.geometry, r+delta_r, t+delta_t)[0])
        minus = np.asarray(patch_geometry(*p.geometry, r-delta_r, t-delta_t)[0])
        np.testing.assert_allclose((plus-minus)/(2*h), jac[..., axis], rtol=2e-8, atol=6e-10)


def test_resident_operator_and_cg_match_independent_dense_reference(gpu):
    p = MappedPoissonPlan(elements=(2, 3), degree=2, device=gpu)
    k_r, r, _ = _axis(2, 2, 4)
    k_t, t, _ = _axis(3, 2, 4)
    br = BSpline(k_r, np.eye(len(k_r)-3), 2)
    bt = BSpline(k_t, np.eye(len(k_t)-3), 2)
    indices, inverse, measure = jax.device_get((p.data['indices'], p.data['inverse'], p.data['measure']))
    gradients, values = [], []
    identity = np.eye(p.dofs).reshape(p.dofs, *p.shape)
    for coefficients in identity:
        local = np.moveaxis(np.pad(coefficients, ((1, 1), (0, 0)))[:, indices], 1, 0)
        ref_gradient = np.stack((br(r, nu=1) @ local @ bt(t).T,
                                 br(r) @ local @ bt(t, nu=1).T), axis=-1)
        gradients.append(np.einsum('...ji,...j->...i', inverse, ref_gradient))
        values.append(br(r) @ local @ bt(t).T)
    gradients, values = np.array(gradients), np.array(values)
    reference = np.einsum('iprtk,jprtk,prt->ij', gradients, gradients, measure)
    rng = np.random.default_rng(43)
    forcing = rng.normal(size=measure.shape)
    lift_gradient = rng.normal(size=(*measure.shape, 2))
    load = (np.einsum('iprt,prt,prt->i', values, forcing, measure)
            - np.einsum('iprtk,prtk,prt->i', gradients, lift_gradient, measure))
    basis, f, grad, rt, at, steps = jax.device_put((identity, forcing, lift_gradient, 1e-12, 1e-14, 1000), gpu)
    with jax.transfer_guard('disallow'):
        action = jax.vmap(apply_stiffness, in_axes=(None, 0))(p.data, basis)
        rhs = _load(p.data, f, grad)
        coeff, iterations, residual, threshold = _pcg(p.data, rhs, p.diagonal, rt, at, steps)
        jax.block_until_ready((action, rhs, coeff, residual))
    assert all(a.devices() == {gpu} for a in jax.tree.leaves((p.data, p.diagonal, action, rhs, coeff)))
    actual = np.asarray(action).reshape(p.dofs, p.dofs).T
    np.testing.assert_allclose(actual, reference, atol=2e-14, rtol=2e-13)
    np.testing.assert_allclose(actual, actual.T, atol=2e-14)
    assert np.linalg.eigvalsh(reference).min() > 0
    np.testing.assert_allclose(np.asarray(p.diagonal).ravel(), np.diag(reference), atol=2e-14)
    np.testing.assert_allclose(np.asarray(rhs).ravel(), load, atol=3e-14)
    np.testing.assert_allclose(np.asarray(coeff).ravel(), np.linalg.solve(reference, load), atol=5e-12, rtol=5e-11)
    assert float(residual) < 2*float(threshold)


def test_manufactured_convergence_with_nonzero_boundary_data(gpu):
    errors = []
    for n in (4, 8, 16):
        p = MappedPoissonPlan(elements=(n, n), device=gpu)
        exact, gradient, forcing, lift = manufactured(p)
        # Check the hand-derived source independently against automatic differentiation.
        z = jax.device_put(np.array((1.1, .6)), gpu)
        np.testing.assert_allclose(gradient(z), jax.grad(exact)(z), atol=1e-14)
        np.testing.assert_allclose(forcing(z), -jnp.trace(jax.hessian(exact)(z)), atol=1e-14)
        solution = p.solve(forcing, lift=lift)
        assert solution.coefficients.devices() == {gpu}
        assert solution.residual_norm / solution.rhs_norm < 2e-11
        e = independent_errors(p, solution)
        assert e['boundary_max_error'] < 2e-14
        assert e['seam_max_jump'] < 2e-14
        area = 8-np.pi*np.prod(p.hole.axes)
        assert abs(e['independent_area']-area) < 1e-10
        errors.append(e)
    for coarse, fine in zip(errors, errors[1:]):
        assert coarse['l2']/fine['l2'] > 13
        assert coarse['h1_seminorm']/fine['h1_seminorm'] > 7


def test_zero_load_and_unconverged_solve(gpu):
    p = MappedPoissonPlan(elements=(3, 3), degree=2, device=gpu)
    solution = p.solve(lambda z: 0.*z[0])
    assert solution.iterations == 0
    assert solution.residual_norm == 0
    assert np.count_nonzero(np.asarray(solution.coefficients)) == 0
    _, _, forcing, lift = manufactured(p)
    with pytest.raises(RuntimeError, match='did not converge'):
        p.solve(forcing, lift=lift, maxiter=1)
