"""Space and assembly checks for the BSPF-inspired triangle experiment."""
import numpy as np
import pytest

from examples.pde.simplex_bspf_comparison import Plan, Space, raw_basis, cases
from examples.pde.simplex_poisson import square_mesh, solve as original_solve, bernstein, triangle_rule


@pytest.mark.parametrize('family', ['polynomial', 'fourier'])
def test_bubbles_have_zero_trace_and_gradients(family):
    space = Space(3, 6, family)
    xy = np.array([[.2, 0.], [0., .4], [.3, .7]])
    b, _ = raw_basis(space, xy)
    np.testing.assert_allclose(b[:, 9:], 0., atol=2e-14)
    xy = np.array([[.23, .31], [.61, .13]])
    _, grad = raw_basis(space, xy)
    for d in (0, 1):
        offset = np.eye(2)[d]*1e-6
        fd = (raw_basis(space, xy+offset)[0]-raw_basis(space, xy-offset)[0])/2e-6
        np.testing.assert_allclose(grad[:, :, d], fd, atol=2e-9, rtol=2e-8)


def test_standard_space_matches_previous_full_fem():
    points, cells = square_mesh(3)
    u, f, g = cases()['smooth']
    plan = Plan(points, cells, Space(4, 4))
    c, stats = plan.solve(f, u)
    reference = original_solve(points, plan.cells, 4, f, u, condensed=False)
    xy, _ = triangle_rule(12)
    actual = plan.evaluate_basis(xy)[0]@c.T
    expected = bernstein(reference['alpha'], xy)[0]@reference['coefficients'].T
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
    assert stats['residual'] < 1e-12


@pytest.mark.parametrize('family', ['polynomial', 'fourier'])
def test_complete_cubic_patch_and_permutation_invariance(family):
    points, cells = square_mesh(3)
    u = lambda x: 1+x[:, 0]**3+x[:, 0]*x[:, 1]**2
    f = lambda x: -8*x[:, 0]
    g = lambda x: np.column_stack((3*x[:, 0]**2+x[:, 1]**2, 2*x[:, 0]*x[:, 1]))
    plan = Plan(points, cells, Space(3, 6, family))
    c, _ = plan.solve(f, u)
    errors = plan.errors(c, u, g)
    assert errors['h1_seminorm'] < 2e-11
    permuted = Plan(points, cells[:, [1, 0, 2]], Space(3, 6, family))
    cp, _ = permuted.solve(f, u)
    np.testing.assert_allclose(cp, c, atol=1e-12, rtol=1e-12)
    assert plan.stats['reference_whitening_error'] < 1e-10
    assert plan.stats['lift_orthogonality'] < 1e-12


def test_independent_quadrature_and_direct_metric_assembly():
    points, cells = square_mesh(3)
    u, f, g = cases()['localized']
    plan = Plan(points, cells, Space(3, 7, 'fourier'), quadrature=24)
    high = Plan(points, cells, Space(3, 7, 'fourier'), quadrature=36)
    c, _ = plan.solve(f, u)
    ch, _ = high.solve(f, u)
    xy, _ = triangle_rule(32)
    np.testing.assert_allclose(plan.evaluate_basis(xy)[0]@c.T,
                               high.evaluate_basis(xy)[0]@ch.T, atol=2e-12, rtol=2e-10)
    local = plan.locals[0]
    grad = plan.grad@local['inv']
    direct = np.einsum('qid,qjd,q->ij', grad, grad, plan.w*local['det'])
    np.testing.assert_allclose(direct, local['k'], atol=2e-12, rtol=2e-12)


def test_residual_selection_single_mode_energy_gain_and_full_recovery():
    from examples.pde.simplex_skeleton_adaptive import Skeleton
    points, cells = square_mesh(3)
    plan = Plan(points, cells, Space(6, 6))
    skeleton = Skeleton(plan)
    u, f, g = cases()['localized']
    skeleton.load(f)
    selected = skeleton.uniform(3)
    c, factor = skeleton.solve(selected)
    candidates, scores = skeleton.gain(selected, c, factor)
    winner = int(np.argmax(scores))
    enriched, _ = skeleton.solve(np.r_[selected, candidates[winner]])
    correction = enriched-c
    actual = correction@(skeleton.matrix@correction)
    np.testing.assert_allclose(actual, scores[winner], rtol=2e-11, atol=1e-15)
    full, _ = skeleton.solve(skeleton.uniform(6))
    direct, _ = plan.solve(f, u)
    np.testing.assert_allclose(skeleton.recover(full), direct, atol=2e-12, rtol=2e-12)
    assert np.linalg.norm((skeleton.matrix@enriched-skeleton.rhs)[np.r_[selected, candidates[winner]]]) < 1e-12


def test_diagonal_proxy_is_bounded_by_relaxed_single_mode_gain():
    from examples.pde.simplex_skeleton_adaptive import Skeleton
    points, cells = square_mesh(3)
    skeleton = Skeleton(Plan(points, cells, Space(6, 6)))
    skeleton.load(cases()['oscillatory'][1])
    selected = skeleton.uniform(3)
    coefficients, factor = skeleton.solve(selected)
    c1, exact = skeleton.gain(selected, coefficients, factor)
    c2, proxy = skeleton.gain(selected, coefficients, factor, score='diagonal')
    np.testing.assert_array_equal(c1, c2)
    assert np.all(proxy <= exact*(1+1e-12)+1e-15)
    assert np.all(proxy >= 0)
