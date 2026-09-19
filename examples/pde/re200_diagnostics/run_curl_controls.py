"""Reproduce the rejected curl-test formulation controls sequentially on GPU."""
from pathlib import Path
import subprocess
import sys

root=Path('build/immersed_flow/re200_ripple_study/curl_residual')
root.mkdir(parents=True,exist_ok=True)
script=Path(__file__).with_name('curl_residual_control.py')
cases=[
    ('zero',['--strength','0']),
    ('strength1',[]),
    ('unweighted_dt0025',['--dt','.0025','--time','1']),
    ('boundary_compatible',['--boundary-compatible']),
    ('boundary_cell',['--boundary-compatible','--strength','9.869604401089358','--time','20']),
    ('boundary_cell_dt0025',['--boundary-compatible','--strength','9.869604401089358','--dt','.0025','--time','3']),
    ('boundary_cell_dt0025_q4',['--boundary-compatible','--strength','9.869604401089358','--dt','.0025','--quadrature','4','--time','3']),
]
for name,args in cases:
    print('START',name,flush=True)
    with (root/(name+'.log')).open('w') as log:
        subprocess.run([sys.executable,str(script),*args,'--out',str(root/name)],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',name,flush=True)
