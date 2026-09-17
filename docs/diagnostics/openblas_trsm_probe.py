"""Manual LP64 OpenBLAS concurrency probe; run in a fresh process.

Usage: OMP_NUM_THREADS=4 python openblas_trsm_probe.py /path/to/libopenblas
Repeat with OMP_NUM_THREADS=1. Requires NumPy, not JAX or SciPy.
The supplied library must export the LP64 Fortran symbol dtrsm_.
"""
import os,json,threading,ctypes,sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
rng=np.random.default_rng(321)
factors=[np.asfortranarray(np.tril(rng.normal(size=(34,34)))+40*np.eye(34)) for _ in range(3)]
expected=[rng.normal(size=(34,k)) for k in (6027,4851,4059)]
rhs=[np.asfortranarray(a@x) for a,x in zip(factors,expected)]
saved=[a.copy() for a in factors+rhs]
lock=threading.Lock()
lib=ctypes.CDLL(sys.argv[1])
trsm=lib.dtrsm_
trsm.restype=None
trsm.argtypes = ([ctypes.c_char_p] * 4 + [ctypes.POINTER(ctypes.c_int)] * 2
                + [ctypes.POINTER(ctypes.c_double), ctypes.c_void_p,
                   ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
                   ctypes.POINTER(ctypes.c_int)])
def solve(i):
    b=rhs[i].copy(order='F')
    m,n=map(ctypes.c_int,b.shape)
    alpha=ctypes.c_double(1.)
    trsm(b'L',b'L',b'N',b'N',ctypes.byref(m),ctypes.byref(n),ctypes.byref(alpha),ctypes.c_void_p(factors[i].ctypes.data),ctypes.byref(m),ctypes.c_void_p(b.ctypes.data),ctypes.byref(m))
    return b
def locked(i):
    with lock: return solve(i)
report={}
with ThreadPoolExecutor(max_workers=3) as pool:
    for name,run in [('sequential',lambda:list(map(solve,range(3)))),('concurrent',lambda:list(pool.map(solve,range(3)))),('locked',lambda:list(pool.map(locked,range(3))))]:
        errors=[]
        for _ in range(50):
            result=run()
            errors.append(max(float(np.max(np.abs(a-b))) for a,b in zip(result,expected)))
        report[name]={'bad_runs':sum(not np.isfinite(e) or e>1e-10 for e in errors),'max_abs_error':max(errors),'repetitions':50}
report['inputs_unchanged']=all(np.array_equal(a,b) for a,b in zip(factors+rhs,saved))
report['omp_threads']=os.environ.get('OMP_NUM_THREADS')
print(json.dumps(report,indent=2))
