"""Run isolated mapped ripple controls in separate GPU processes."""
from pathlib import Path
import subprocess
import sys

root=Path('build/mapped_ripple')
root.mkdir(parents=True,exist_ok=True)
cases={
    'lift_c4':['--lift','c4'],
    'radial24':['--elements','24','16'],
    'tangent24':['--elements','16','24'],
    'degree4':['--degree','4'],
}
for name,args in cases.items():
    print('START',name,flush=True)
    with (root/(name+'.log')).open('w') as log:
        subprocess.run([sys.executable,str(Path(__file__).with_name('source_audit.py')),
                        *args,'--evolve','2','--out',str(root/name)],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',name,flush=True)
