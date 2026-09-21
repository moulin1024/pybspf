import jax
import jax.numpy as jnp
import numpy as np
from pybspf.time_integration import rk4_stages

def test_rk4_numpy_array_and_jax_coupled_fields():
    # Existing NumPy consumers retain NumPy output and all four diagnostics.
    original = np.array([1., 2.])
    value, stages = rk4_stages(original, .1, lambda x: (-x, x.copy()))
    assert isinstance(value, np.ndarray) and len(stages) == 4
    np.testing.assert_allclose(value, original*np.exp(-.1), rtol=1e-7)
    np.testing.assert_array_equal(stages[0], original)
    # Unequal field shapes, as in streamfunction and scalar slope coefficients.
    initial = (jnp.ones((2, 3)), jnp.ones((3, 4)))
    def step(state, dt):
        def rhs(y):
            return (-y[0], -2*y[1]), (jnp.sum(y[0]), jnp.sum(y[1]))
        return rk4_stages(state, dt, rhs)
    value, stages = jax.jit(step)(initial, .01)
    assert len(stages) == 4
    for result, expected in zip(value, (np.exp(-.01), np.exp(-.02))):
        np.testing.assert_allclose(result, expected, rtol=3e-11)
