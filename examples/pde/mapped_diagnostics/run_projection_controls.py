"""Projection localization and harmonicity checks after run_controls.py."""
from pathlib import Path
import subprocess
import sys

folder=Path(__file__).resolve().parent
root=Path('build/mapped_ripple')
commands=[
    ('projection_strong','projection_audit.py',['--out',str(root/'projection_strong')]),
    ('projection_nitsche','projection_audit.py',['--boundary','nitsche','--out',str(root/'projection_nitsche')]),
    ('degree4_c4','source_audit.py',['--degree','4','--lift','c4','--evolve','2','--out',str(root/'degree4_c4')]),
    ('mean_value','mean_value_audit.py',[]),
]
for name,script,args in commands:
    print('START',name,flush=True)
    with (root/(name+'.log')).open('w') as log:
        subprocess.run([sys.executable,str(folder/script),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',name,flush=True)
