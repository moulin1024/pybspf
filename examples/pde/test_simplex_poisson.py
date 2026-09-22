"""Targeted tests for the standalone triangular Poisson research prototype."""
import numpy as np
import pytest

from examples.pde.simplex_poisson import (bernstein, indices, square_mesh, solve, error,
                             manufactured, forcing, gradient)


@pytest.mark.parametrize('p', [1, 3, 5])
def test_partition_and_zero_trace_bubbles(p):
    alpha = indices(p)
    xy = np.array([[0., 0.], [.3, 0.], [0., .7], [.2, .8], [.2, .3]])
    b, g = bernstein(alpha, xy)
    np.testing.assert_allclose(b.sum(axis=1), 1., atol=1e-14)
    np.testing.assert_allclose(g.sum(axis=1), 0., atol=1e-14)
    np.testing.assert_allclose(b[:4, np.all(alpha > 0, axis=1)], 0., atol=1e-14)


def test_nonzero_quadratic_boundary_patch_and_cell_orientation():
    points, cells = square_mesh(4)
    cells[::2] = cells[::2, ::-1]  # clockwise and counterclockwise mixed
    exact = lambda x: x[:, 0]**2 + x[:, 0]*x[:, 1] + 2*x[:, 1]**2
    grad = lambda x: np.column_stack((2*x[:, 0]+x[:, 1], x[:, 0]+4*x[:, 1]))
    f = lambda x: np.full(len(x), -6.)
    result = solve(points, cells, 3, f, exact)
    errors = error(points, cells, result, exact, grad)
    assert errors['l2'] < 1e-12
    assert errors['h1_seminorm'] < 1e-11


def test_condensation_and_trace_continuity():
    points, cells = square_mesh(4)
    result = solve(points, cells, 5, forcing, manufactured)
    full = solve(points, cells, 5, forcing, manufactured, condensed=False)
    np.testing.assert_allclose(result['coefficients'], full['coefficients'], atol=2e-12, rtol=2e-12)
    assert result['stats']['lift_orthogonality'] < 1e-13
    traces = {}
    t = np.linspace(0, 1, 17)
    for cell, coeff in zip(cells, result['coefficients']):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(cell[i]), int(cell[j]))))
            xy = (1-t[:, None])*points[edge[0]] + t[:, None]*points[edge[1]]
            jac = (points[cell[1:]]-points[cell[0]]).T
            ref = (xy-points[cell[0]])@np.linalg.inv(jac).T
            values = bernstein(result['alpha'], ref)[0]@coeff
            if edge in traces:
                np.testing.assert_allclose(values, traces[edge], atol=2e-12)
            else:
                traces[edge] = values
    assert result['stats']['skeleton_dofs'] < result['stats']['total_dofs']


def test_cubic_convergence():
    errors = []
    for n in (4, 8):
        points, cells = square_mesh(n)
        result = solve(points, cells, 3, forcing, manufactured)
        errors.append(error(points, cells, result, manufactured, gradient))
    assert errors[0]['l2']/errors[1]['l2'] > 10
    assert errors[0]['h1_seminorm']/errors[1]['h1_seminorm'] > 5
