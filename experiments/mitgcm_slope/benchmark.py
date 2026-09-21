"""Time an already-built MITgcm run and validate its final binary fields."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import numpy as np
from process_timing import run_logged


def benchmark(executable, run_dir, source):
    executable, run_dir, source = (p.resolve() for p in (executable, run_dir, source))
    if (run_dir/'console.log').exists():
        raise FileExistsError('Use a fresh prepared run directory; refusing to overwrite an earlier run.')
    case = json.loads((run_dir/'case.json').read_text())
    steps = int(round(case['final_time_s']/case['dt_s']))
    threads = case.get('threads',1)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE',
               OMP_MAX_ACTIVE_LEVELS='1', OMP_STACKSIZE='400M',
               OPENBLAS_NUM_THREADS='1')
    metrics=run_logged([str(executable)],cwd=run_dir,env=env,log=run_dir/'console.log')
    timing = dict(**metrics,threads_requested=threads,
        thread_configuration={k:env[k] for k in ('OMP_NUM_THREADS','OMP_DYNAMIC','OMP_MAX_ACTIVE_LEVELS','OMP_STACKSIZE')},
        source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip(),
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        timing_scope=f'{threads}-thread executable startup + initialization + {steps} steps + output, excluding compilation',
        case=case)
    stdout = (run_dir/'STDOUT.0000').read_text() if (run_dir/'STDOUT.0000').exists() else (run_dir/'console.log').read_text()
    stderr = (run_dir/'STDERR.0000').read_text() if (run_dir/'STDERR.0000').exists() else ''
    thread_match=re.search(r'nThreads\s*=\s*(\d+)',stdout)
    timing['threads_reported_by_model']=int(thread_match[1]) if thread_match else None
    timing['thread_configuration_verified']=timing['threads_reported_by_model']==threads
    fields = {}
    for var in ('U','V','W','T'):
        files = list(run_dir.glob(f'{var}.{steps:010d}*.data'))
        stats = []
        for f in files:
            data = np.fromfile(f, dtype='>f8')
            stats.append(dict(file=f.name, values=data.size, finite=bool(np.isfinite(data).all()),
                              min=float(data.min()) if np.isfinite(data).all() else None,
                              max=float(data.max()) if np.isfinite(data).all() else None))
        fields[var] = stats
    timing['final_fields'] = fields
    residuals = {str(dim): [float(v) for v in re.findall(
        rf'cg{dim}d_last_res\s*=\s*([0-9.Ee+\-]+)', stdout)] for dim in (2, 3)}
    iterations = {str(dim): [int(v) for v in re.findall(
        (r'cg2d_iters\(min,last\)\s*=\s*-?\d+\s+(\d+)' if dim == 2 else
         r'cg3d_iters \(last\)\s*=\s*(\d+)'), stdout)] for dim in (2, 3)}
    timing['pressure_residuals'] = residuals
    timing['pressure_iterations'] = iterations
    timing['pressure_converged'] = all(len(residuals[str(dim)]) == steps and
        all(np.isfinite(v) and v <= 1.e-11 for v in residuals[str(dim)]) for dim in (2,3))
    timing['successful'] = (timing['thread_configuration_verified'] and timing['pressure_converged'] and timing['returncode'] == 0 and 'Execution ended Normally' in stdout
                            and all(fields[k] and all(v['finite'] for v in fields[k]) for k in fields))
    timing['stderr_tail'] = stderr[-2000:]
    timing['pressure_summary_lines'] = [line for line in stdout.splitlines()
        if re.search(r'(cg[23]d|CG[23]D)', line)][-20:]
    (run_dir/'wall_time.json').write_text(json.dumps(timing, indent=2, allow_nan=False))
    print(json.dumps(timing, indent=2, allow_nan=False))
    if not timing['successful']:
        raise SystemExit('MITgcm failed validation; timing is not an accepted benchmark.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--executable', required=True, type=Path)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--source', required=True, type=Path)
    args = p.parse_args()
    benchmark(args.executable, args.run, args.source)
