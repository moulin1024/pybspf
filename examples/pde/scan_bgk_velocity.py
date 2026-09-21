"""Run the fixed Nx=33 velocity sweep, retaining restartable histories and logs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='build/bgk_velocity_scan')
    ap.add_argument('--end',type=float,default=1200.)
    ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if args.jobs<1:raise ValueError('jobs must be positive')
    root=Path(__file__).resolve().parents[2]
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')

    def run(pair):
        nv,nmu=pair;name=f'v{nv}_m{nmu}_dt01';directory=out/name
        if (directory/'history.npz').exists() and not args.resume:
            raise FileExistsError(f'{directory} already contains results; use --resume')
        command=[sys.executable,str(root/'examples/pde/bgk_itg_saturation.py'),
            '--nx','33','--nv',str(nv),'--nmu',str(nmu),'--dt','0.1',
            '--radial-kernel','tensor','--end',str(args.end),'--average-start','400',
            '--out',str(directory)]
        if args.resume and (directory/'history.npz').exists():command.append('--resume')
        with (out/f'{name}.log').open('a' if args.resume else 'w') as log:
            subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f'{name}: complete',flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run,[(24,16),(16,12),(12,8),(8,6)]))


if __name__=='__main__':main()
