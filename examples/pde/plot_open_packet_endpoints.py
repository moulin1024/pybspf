"""Plot saved endpoint diagnostics; no simulation or fitted convergence data."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=Path('build/open_slab_packet')
    rows=json.loads((out/'endpoint_comparison.json').read_text())
    scalar=json.loads((out/'same_nodes_derivative.json').read_text())
    mild=json.loads((out/'mild_knots.json').read_text())
    uniform=json.loads((out/'uniform_knots.json').read_text())
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    colors=['#2463a5','#db6d20']; x=np.arange(4)
    labels=['FD\n9 points','Cheb12\n16 points',
            'Cheb12\n24 points','Cheb12\n32 points']
    for n,c,offset in [(129,colors[0],-.18),(257,colors[1],.18)]:
        data=[r for r in rows if r['n']==n]
        vals=[r['packet_error'] for r in data]
        axs[0,0].bar(x+offset,vals,width=.34,color=c,label=f'N = {n}',bottom=0)
        for xx,v in zip(x+offset,vals):
            axs[0,0].annotate(f'{v:.2e}',(xx,v),xytext=(0,5 if n==129 else 17),textcoords='offset points',ha='center',fontsize=8)
        axs[1,0].semilogy(x,[r['map_norm'] for r in data],'o-',color=c,label=f'N = {n}')
    axs[0,0].set(yscale='log',ylim=(3e-11,5e-6),ylabel='Relative distribution L2 error',title='A  Open GK packet: endpoint-window comparison')
    axs[1,0].set(ylabel='Coefficient-map spectral norm',title='C  Coefficient amplification vs. endpoint window')
    for a in [axs[0,0],axs[1,0]]:
        a.set_xticks(x,labels,fontsize=9);a.legend();a.grid(axis='y',which='major',alpha=.2);a.set_axisbelow(True)
    a=axs[0,1]
    for data,label,c in [(uniform,'Uniform knots + FD9','#81909c'),(mild,'Mild knots + FD9',colors[0])]:
        a.loglog([r['n']-1 for r in data],[r['relative_g_error'] for r in data],'o-',color=c,label=label)
    data=[r for r in rows if r['points']==24]
    a.loglog([r['n']-1 for r in data],[r['packet_error'] for r in data],'D',ms=8,color=colors[1],label='Mild knots + Cheb12 / 24 (two runs)')
    a.set(xlabel='N - 1',ylabel='Relative distribution L2 error',title='B  New results against the full grid sweep')
    a.legend(fontsize=9);a.grid(which='both',alpha=.2)
    a=axs[1,1]
    values=[scalar['direct_max'],scalar['precomputed_map_max']]
    a.bar([0,1],values,width=.5,color=[colors[0],colors[1]])
    a.set(yscale='log',ylim=(1e-14,1e-8),ylabel='Maximum absolute first-derivative error',title='D  Same nodes, function and knots: N = 257')
    a.set_xticks([0,1],['Standard BSPF\nsolve for each input','Precomputed coefficient map\nthen the same derivative formula'])
    for i,v in enumerate(values):a.text(i,v*1.5,f'{v:.3e}',ha='center')
    a.text(.5,.91,f'Map path error / standard error = {values[1]/values[0]:,.0f}',transform=a.transAxes,ha='center',fontsize=10)
    a.grid(axis='y',which='major',alpha=.2);a.set_axisbelow(True)
    fig.suptitle('BSPF endpoint diagnostics: accuracy, amplification and the fine-grid plateau',fontsize=16)
    fig.savefig(out/'endpoint_diagnostics.png',dpi=190)
    fig.savefig(out/'endpoint_diagnostics.pdf')
    plt.close(fig)
    print(out/'endpoint_diagnostics.png')

if __name__=='__main__':main()
