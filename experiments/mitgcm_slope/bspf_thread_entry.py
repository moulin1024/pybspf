"""Record CPU-client thread initialization, then run the unchanged BSPF CLI."""
from pathlib import Path
import json
import os
import runpy
import psutil
import jax

process=psutil.Process()
before=process.num_threads()
devices=jax.devices()
print('THREAD_RUNTIME '+json.dumps(dict(
    pjrt_nproc=os.environ.get('PJRT_NPROC'),
    os_threads_before_client=before,os_threads_after_client=process.num_threads(),
    devices=[str(d) for d in devices],
    note='PJRT_NPROC sizes XLA compute/dispatch pools; total OS threads also include support and BLAS threads.')),flush=True)
runpy.run_path(str(Path(__file__).resolve().parents[2]/'examples/pde/isw_slope/run_slope.py'),run_name='__main__')
