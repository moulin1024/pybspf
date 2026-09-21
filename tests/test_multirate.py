"""Independent colored-tree order audit and exact-ODE tests for MRI-GARK4."""

from functools import lru_cache
from itertools import combinations_with_replacement

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from pybspf.multirate import GAMMA0
from pybspf.multirate import GAMMA1
from pybspf.multirate import mri_gark_erk45a_step

jax.config.update("jax_enable_x64", True)


def expanded_tableau(m):
    """Expand actual slow/fast derivative events into an additive RK tableau.

    Independent NumPy symbolic-linear bookkeeping; does not invoke the runtime
    stepper. A row describes the state at one slow OR fast derivative call.
    H=1. Entries refer to previously evaluated colored derivatives.
    """
    size = 5 * (1 + 4 * m)
    a, history = np.zeros((size, size)), np.zeros((5, size))
    colors, times = [], []
    value = np.zeros(size)
    cursor = 0

    def event(state, color, time):
        nonlocal cursor
        a[cursor] = state
        colors.append(color)
        times.append(time)
        derivative = np.eye(1, size, cursor)[0]
        cursor += 1
        return derivative

    h = 1 / (5 * m)
    for i in range(5):
        history[i] = event(value, 0, i / 5)
        for k in range(m):

            def stage(state, offset):
                theta = (k + offset) / m
                derivative = event(state, 1, (i + theta) / 5)
                return derivative + (5 * (GAMMA0[i] + theta * GAMMA1[i])) @ history

            a1 = stage(value, 0)
            a2 = stage(value + h / 2 * a1, 0.5)
            a3 = stage(value + h / 2 * a2, 0.5)
            a4 = stage(value + h * a3, 1)
            value = value + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
    return a, value, np.array(colors), np.array(times)


@lru_cache(None)
def tree_size(tree):
    return 1 + sum(tree_size(child) for child in tree[1])


@lru_cache(None)
def tree_factorial(tree):
    return tree_size(tree) * np.prod(
        [tree_factorial(child) for child in tree[1]], dtype=int
    )


@lru_cache(None)
def colored_trees(order):
    if order == 1:
        return ((0, ()), (1, ()))
    pool = sorted(tree for n in range(1, order) for tree in colored_trees(n))
    result = []
    for count in range(1, order):
        for children in combinations_with_replacement(pool, count):
            if sum(tree_size(child) for child in children) == order - 1:
                result.extend((color, children) for color in (0, 1))
    return tuple(result)


@pytest.mark.parametrize("m", [1, 2, 3, 7])
def test_all_72_colored_order_conditions_through_order_four(m):
    a, b, colors, times = expanded_tableau(m)
    assert np.count_nonzero(colors == 0) == 5
    assert np.count_nonzero(colors == 1) == 20 * m
    assert np.max(abs(np.triu(a))) == 0  # explicit, including cross-partition couplings
    for color in (0, 1):
        # Both partitions have the same physical stage times, including nonautonomous use.
        np.testing.assert_allclose(a[:, colors == color].sum(axis=1), times, atol=3e-14)

    @lru_cache(None)
    def stage_weight(tree):
        result = np.ones(len(b))
        for child in tree[1]:
            selected = colors == child[0]
            result *= a[:, selected] @ stage_weight(child)[selected]
        return result

    trees = [tree for order in range(1, 5) for tree in colored_trees(order)]
    assert len(trees) == 72
    for tree in trees:
        selected = colors == tree[0]
        weight = b[selected] @ stage_weight(tree)[selected]
        assert abs(weight - 1 / tree_factorial(tree)) < 3e-13, (m, tree, weight)


def solve(fast, slow, initial, *, h, end, m=2):
    step = jax.jit(
        lambda y, t: mri_gark_erk45a_step(y, t, h, fast, slow, inner_steps=m)
    )
    state = initial
    for i in range(round(end / h)):
        state = step(state, i * h)
    return np.asarray(state)


def test_noncommuting_bidirectional_linear_system_fourth_order():
    fast = jnp.array([[-8.0, 3.0], [0.0, 0.0]])
    slow = jnp.array([[0.0, 0.0], [-2.0, -0.4]])
    initial = jnp.array([1.0, 0.3])
    exact = expm(np.asarray(fast + slow)) @ initial
    errors = [
        np.linalg.norm(
            solve(lambda t, y: fast @ y, lambda t, y: slow @ y, initial, h=h, end=1.0)
            - exact
        )
        for h in (0.2, 0.1, 0.05, 0.025)
    ]
    orders = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all((orders > 3.6) & (orders < 4.5)), (errors, orders)


def test_nonlinear_nonautonomous_manufactured_solution_fourth_order():
    def slow(t, y):
        return jnp.array([0.2 * y[0] * y[1], -0.3 * y[0] ** 2])

    def fast(t, y):
        a, b = jnp.exp(-t), 2 + jnp.sin(t)
        return jnp.array(
            [
                -4 * (y[0] - a) - a - 0.2 * a * b,
                -5 * (y[1] - b) + jnp.cos(t) + 0.3 * a * a,
            ]
        )

    exact = np.array([np.exp(-1), 2 + np.sin(1)])
    errors = [
        np.linalg.norm(solve(fast, slow, jnp.array([1.0, 2.0]), h=h, end=1.0) - exact)
        for h in (0.2, 0.1, 0.05, 0.025)
    ]
    orders = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all((orders > 3.6) & (orders < 4.5)), (errors, orders)


def test_linear_invariant_and_pytree_budget():
    def fast(t, y):
        a, b, reservoir = y
        exchange = 7 * b * b
        return (-a * 0, -exchange, exchange)

    def slow(t, y):
        a, b, reservoir = y
        exchange = a * b
        return (-exchange, exchange, reservoir * 0)

    initial = tuple(map(jnp.asarray, (0.4, 0.6, 0.0)))
    step = jax.jit(
        lambda y: mri_gark_erk45a_step(y, 0.0, 0.1, fast, slow, inner_steps=3)
    )
    state = initial
    for _ in range(20):
        state = step(state)
    np.testing.assert_allclose(sum(state), 1.0, atol=3e-14, rtol=0)


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_invalid_microstep_count(steps):
    with pytest.raises(ValueError):
        mri_gark_erk45a_step(
            jnp.array([1.0]),
            0.0,
            0.1,
            lambda t, y: -y,
            lambda t, y: y,
            inner_steps=steps,
        )
