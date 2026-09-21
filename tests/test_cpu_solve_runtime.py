"""Check concurrent solves with the documented CPU BLAS configuration.

Use a fresh interpreter: thread controls must precede library initialization.
The well-conditioned systems isolate runtime correctness from BSPF conditioning.
"""
import os
import subprocess
import sys


def test_concurrent_lu_with_serial_blas():
    source = '''
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jl
from scipy.linalg import lu_factor, lu_solve

rng = np.random.default_rng(321)
matrices = tuple(rng.normal(size=(34, 34)) + 40*np.eye(34) for _ in range(3))
solutions = tuple(rng.normal(size=(34, k)) for k in (6027, 4851, 4059))
rhs = tuple(a @ x for a, x in zip(matrices, solutions))
factors = tuple(jl.lu_factor(jnp.asarray(a)) for a in matrices)
jax_rhs = tuple(jnp.asarray(r) for r in rhs)
solve_all = jax.jit(lambda fs, bs: tuple(jl.lu_solve(f, b) for f, b in zip(fs, bs)))
for _ in range(30):
    for actual, expected in zip(solve_all(factors, jax_rhs), solutions):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
scipy_factors = tuple(lu_factor(a) for a in matrices)
with ThreadPoolExecutor(max_workers=3) as pool:
    for _ in range(15):
        results = pool.map(lambda args: lu_solve(*args), zip(scipy_factors, rhs))
        for actual, expected in zip(results, solutions):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
'''
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "JAX_PLATFORMS": "cpu"}
    result = subprocess.run([sys.executable, "-c", source], env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
