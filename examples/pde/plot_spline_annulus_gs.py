"""Render accepted GS flux samples with prescribed boundary traces; no PDE mesh."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from spline_annulus_convergence import geometry
from bspf_models.elliptic.spline_annulus import SplineAnnulus


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    report=json.loads((args.directory/'results.json').read_text())
    if not report['passed']: raise ValueError('refusing to label an unaccepted run as a solution')
    data=np.load(args.directory/'fields.npz')
    domain,_=geometry()
    if report['degree']==5:
        curves=[]
        for b in domain.boundaries:
            n=len(b.knots)-1;c=b.curve.c[:n]
            curves.append(BSpline(np.arange(-5,n+6),np.vstack((c,c[:5])),5,extrapolate='periodic'))
        domain=SplineAnnulus(*curves)
    points=[data['points']];values=[data['nonlinear'].real];curves=[]
    for b,value in zip(domain.boundaries,(0.,.02)):
        c=b.curve(np.linspace(b.a,b.b,500,endpoint=False))+[3.,0.]
        curves.append(c);points.append(c);values.append(np.full(len(c),value))
    points=np.vstack(points);values=np.concatenate(values)
    mesh=tri.Triangulation(points[:,0],points[:,1])
    vertices=points[mesh.triangles]-[3.,0.]
    mask=~domain.contains(vertices.mean(axis=1))
    mesh.set_mask(mask)
    fig,ax=plt.subplots(figsize=(7.4,6.2),layout='constrained')
    im=ax.tripcolor(mesh,values,shading='gouraud',cmap='viridis',vmin=0,vmax=.02)
    ax.tricontour(mesh,values,levels=np.linspace(.002,.018,9),colors='white',linewidths=.65,alpha=.8)
    from matplotlib.patches import Polygon
    ax.add_patch(Polygon(curves[1], closed=True, facecolor='white', edgecolor='none', zorder=3))
    for c in curves: ax.plot(*np.vstack((c,c[0])).T,'k',lw=1)
    ax.set(aspect='equal',xlabel='R',ylabel='Z',
           title=f"Nonlinear Grad-Shafranov flux | degree-{report['degree']} B-splines\nOuter flux = 0; inner flux = 0.02")
    fig.colorbar(im,ax=ax,label='poloidal flux psi')
    fig.get_layout_engine().set(rect=(0,.05,1,.95))
    fig.text(.5,.015,'Interpolated computed interior samples + prescribed boundary traces; no fitted PDE mesh',ha='center',fontsize=8)
    fig.savefig(args.directory/'nonlinear_flux.png',dpi=180)
    fig.savefig(args.directory/'nonlinear_flux.pdf')

if __name__=='__main__':main()
