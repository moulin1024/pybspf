"""Reynolds-independent background rule resolving rational corner functions.

Only fixed geometric end caps use a cubic coordinate map. Bulk orders follow
the background degree, not the viscous-layer thickness. Curved ellipse patches
retain the exact channel geometry.
"""
import numpy as np
from scipy.special import roots_legendre


def axis_rule(lo,hi,density,*,left=False,right=False,cap=.15,order=20):
    width=min(cap,(hi-lo)/4)
    cuts=[lo]+([lo+width] if left else [])+([hi-width] if right else [])+[hi]
    points=[];weights=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        graded_left=left and a==lo
        graded_right=right and b==hi
        n=order if graded_left or graded_right else max(12,int(np.ceil(density*(b-a))))
        z,w=roots_legendre(n);z=(z+1)/2;w=w/2
        if graded_left:w=w*3*z*z;z=z**3
        elif graded_right:w=w*3*(1-z)**2;z=1-(1-z)**3
        points.append(a+(b-a)*z);weights.append((b-a)*w)
    return np.concatenate(points),np.concatenate(weights)


def corner_channel_quadrature(bounds,hole,nx,ny,factor=2.5,x_breaks=(),*,corner_order=20):
    if hole is None:raise ValueError('This background rule requires an ellipse')
    left,right,h=bounds;cx,cy=hole.center;a,b=hole.axes
    y,wy=axis_rule(-h,h,factor*ny/(2*h),left=True,right=True,order=corner_order)
    points=[];weights=[]
    for lo,hi in ((left,cx-a),(cx+a,right)):
        cuts=[lo]+sorted(z for z in x_breaks if lo<z<hi)+[hi]
        for aa,bb in zip(cuts[:-1],cuts[1:]):
            x,wx=axis_rule(aa,bb,factor*nx/(right-left),left=aa==left,right=bb==right,
                           order=corner_order)
            xx,yy=np.meshgrid(x,y,indexing='ij')
            points.append(np.column_stack((xx.ravel(),yy.ravel())))
            weights.append(np.outer(wx,wy).ravel())
    nt=max(20,int(np.ceil(factor*(nx*2*a/(right-left)+ny*b/h)))+8)
    z,w=roots_legendre(nt);theta=(z+1)*np.pi/2;wt=w*np.pi/2
    x=cx+a*np.cos(theta)
    for sign in (-1,1):
        wall=sign*h;edge=cy+sign*b*np.sin(theta)
        ns=max(12,int(np.ceil(factor*ny*(h-sign*cy)/(2*h)))+4)
        z,w=roots_legendre(ns);s=(z+1)/2;ws=w/2
        yy=edge[:,None]*(1-s)+wall*s
        xx=np.broadcast_to(x[:,None],yy.shape)
        jac=a*np.sin(theta)*abs(wall-edge)
        points.append(np.column_stack((xx.ravel(),yy.ravel())))
        weights.append((wt[:,None]*ws*jac[:,None]).ravel())
    return np.vstack(points),np.concatenate(weights)
