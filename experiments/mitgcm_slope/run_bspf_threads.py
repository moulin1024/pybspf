"""Run the existing BSPF entrypoint with a configured CPU worker pool."""
from pathlib import Path
import argparse
import json
import os
import sys
from process_timing import run_logged

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tfinal',type=float,default=50.)
    p.add_argument('--dt',type=float,default=2.)
    a=p.parse_args()
    if a.threads<1: p.error('threads must be positive')
    out=a.out.resolve()
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    # The numerical driver's nonempty-output check requires logs beside its
    # output directory, rather than pre-creating files inside it.
    env=dict(os.environ,PJRT_NPROC=str(a.threads),JAX_PLATFORMS='cpu',
        OPENBLAS_NUM_THREADS=str(a.threads),OMP_NUM_THREADS=str(a.threads),
        MKL_NUM_THREADS=str(a.threads),VECLIB_MAXIMUM_THREADS=str(a.threads),
        XLA_FLAGS='--xla_cpu_multi_thread_eigen=true')
    command=[sys.executable,str(Path(__file__).with_name('bspf_thread_entry.py')),
        '--case','R2_321x161','--dt',str(a.dt),'--tfinal',str(a.tfinal),'--save','50','--out',str(out)]
    timing=run_logged(command,cwd=ROOT,env=env,log=out.with_suffix('.log'))
    summary=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else {}
    runtime=[json.loads(line.removeprefix('THREAD_RUNTIME ')) for line in out.with_suffix('.log').read_text().splitlines() if line.startswith('THREAD_RUNTIME ')]
    timing['cpu_runtime_initialization']=runtime
    timing.update(threads_requested=a.threads,command=command,
        thread_configuration={k:env[k] for k in ('PJRT_NPROC','JAX_PLATFORMS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','XLA_FLAGS')},
        successful=timing['returncode']==0 and summary.get('completed',False),summary=summary)
    out.mkdir(exist_ok=True)
    (out/'wall_time_threads.json').write_text(json.dumps(timing,indent=2))
    print(json.dumps(timing,indent=2))
    if not timing['successful']:raise SystemExit('BSPF run failed')


if __name__=='__main__':main()
