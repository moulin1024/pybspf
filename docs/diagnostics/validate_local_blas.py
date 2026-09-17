"""Validate concurrent JAX/SciPy solves via run_with_local_blas.py (macOS)."""

import ctypes
import json
import os
from pathlib import Path
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jl
from scipy.linalg import lu_factor, lu_solve
from concurrent.futures import ThreadPoolExecutor

rng = np.random.default_rng(321)
matrices = [rng.normal(size=(34, 34)) + 40 * np.eye(34) for _ in range(3)]
solutions = [rng.normal(size=(34, k)) for k in (6027, 4851, 4059)]
rhs = [a @ x for a, x in zip(matrices, solutions)]
factors = tuple(jl.lu_factor(jnp.asarray(a)) for a in matrices)
bs = tuple(jnp.asarray(b) for b in rhs)
solve = jax.jit(lambda fs, rs: tuple(jl.lu_solve(f, r) for f, r in zip(fs, rs)))
errors = []
for _ in range(100):
    errors.append(
        max(
            float(np.max(np.abs(np.asarray(v) - x)))
            for v, x in zip(solve(factors, bs), solutions)
        )
    )
sf = [lu_factor(a) for a in matrices]
se = []
with ThreadPoolExecutor(max_workers=3) as pool:
    for _ in range(100):
        out = list(pool.map(lambda i: lu_solve(sf[i], rhs[i]), range(3)))
        se.append(max(float(np.max(np.abs(v - x))) for v, x in zip(out, solutions)))
lib = ctypes.CDLL(os.environ["BSPF_BLAS_LIBRARY"])
lib.openblas_get_config.restype = ctypes.c_char_p
rt = ctypes.CDLL(None)
rt._dyld_image_count.restype = ctypes.c_uint32
rt._dyld_get_image_name.argtypes = [ctypes.c_uint32]
rt._dyld_get_image_name.restype = ctypes.c_char_p
paths = [rt._dyld_get_image_name(i).decode() for i in range(rt._dyld_image_count())]
paths = [
    p
    for p in paths
    if any(s in p.lower() for s in ("blas", "lapack", "libomp"))
    and p.endswith(".dylib")
]
expected_library = Path(os.environ["BSPF_BLAS_LIBRARY"]).resolve()
loaded_openblas = {Path(p).resolve() for p in paths if "openblas" in p.lower()}
assert loaded_openblas == {expected_library}, paths
report = {
    "omp_threads": os.environ["OMP_NUM_THREADS"],
    "config": lib.openblas_get_config().decode(),
    "loaded_libraries": paths,
    "jax_bad_runs": sum(not np.isfinite(e) or e > 1e-10 for e in errors),
    "jax_max_error": max(errors),
    "scipy_bad_runs": sum(not np.isfinite(e) or e > 1e-10 for e in se),
    "scipy_max_error": max(se),
    "repetitions": 100,
}
print(json.dumps(report, indent=2), flush=True)
assert report["jax_bad_runs"] == report["scipy_bad_runs"] == 0
Path(
    "build/openblas-atomic/validation_" + os.environ["OMP_NUM_THREADS"] + ".json"
).write_text(json.dumps(report, indent=2))
