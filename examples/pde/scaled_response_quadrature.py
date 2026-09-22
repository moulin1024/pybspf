"""Fixed-budget positive quadrature with thickness-scaled boundary coordinates.

The ellipse-to-rectangle ray map covers the fluid exactly. Each ray has an
inner layer, a bulk interval and an outer layer. Their orders do not depend on
Reynolds number; only the layer widths and physical nodes change. This is a
research integration rule, not a modification of the response functions.
"""
from functools import lru_cache
import numpy as np
from scipy.special import roots_legendre


@lru_cache(None)
def unit_gauss(order):
    x,w=roots_legendre(order)
    return (x+1)/2,w/2


@lru_cache(None)
def layer_gauss(order,outer=False):
    """Split a fixed node budget across three dimensionless layer scales."""
    cuts=(0.,1/16,1/4,1.)
    counts=(order//3,order//3,order-2*(order//3))
    nodes=[];weights=[]
    for lo,hi,n in zip(cuts[:-1],cuts[1:],counts):
        z,w=unit_gauss(n);nodes.append(lo+(hi-lo)*z);weights.append((hi-lo)*w)
    z,w=np.concatenate(nodes),np.concatenate(weights)
    return (1-z[::-1],w[::-1]) if outer else (z,w)


def boundary_rays(bounds,hole,nx,ny,factor=2.5,x_breaks=(),*,corner_width=0.,minimum_tangent_order=20):
    """Outer boundary nodes, positive cross products, and parameter weights."""
    left,right,h=bounds
    c=np.asarray(hole.center);a,b=hole.axes
    vertices=np.array([[left,-h],[right,-h],[right,h],[left,h]])
    rays=[];weights=[];crosses=[];normal_distances=[]
    for i in range(4):
        first,last=vertices[i],vertices[(i+1)%4];v=last-first
        axis=int(np.argmax(abs(v)));length=abs(v[axis])
        # Geometry-dependent tangent panels resolve changes in ellipse angle.
        # They do not change with viscosity or layer thickness.
        width=(a,b)[axis]
        positions=[c[axis]+k*width for k in (-2,-1,0,1,2)]
        if axis==0:positions.extend(x_breaks)
        cuts=sorted(set([0.,1.]+[(z-first[axis])/v[axis] for z in positions
                                if 0<(z-first[axis])/v[axis]<1]))
        cap=min(corner_width/length,.25)
        if cap>0:cuts=sorted(set(cuts+[cap,1-cap]))
        tangent_count=(nx,ny)[axis]
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            n=max(minimum_tangent_order,int(np.ceil(factor*tangent_count*(hi-lo))))
            s,ws=unit_gauss(n)
            if cap>0 and lo==0:
                ws=ws*3*s*s;s=s**3
            elif cap>0 and hi==1:
                ws=ws*3*(1-s)**2;s=1-(1-s)**3
            s=lo+(hi-lo)*s;ws=ws*(hi-lo)
            ray=first+s[:,None]*v-c
            cross=ray[:,0]*v[1]-ray[:,1]*v[0]
            if not np.all(cross>0):raise ValueError('Ellipse center must be strictly inside rectangle')
            rays.append(ray);weights.append(ws);crosses.append(cross)
            normal_distances.append(np.full(n,cross[0]/length))
    return tuple(np.concatenate(a) for a in (rays,weights,crosses,normal_distances))


def scaled_channel_quadrature(bounds,hole,nx,ny,factor=2.5,x_breaks=(),*,
                              layer_lengths,layer_order=48,bulk_order=32,extent=32.,
                              resolve_corners=False,minimum_tangent_order=20):
    """Integrate in scaled near-wall coordinates without adding points as Re grows.

    layer_lengths are the thin family only; broad exponentials remain resolved
    by the bulk rule. All terms of a weak form should use this same positive
    rule unless a separate, consistent block integration is implemented.
    """
    lengths=np.asarray(layer_lengths,dtype=float)
    if lengths.ndim!=1 or not len(lengths) or not np.all(np.isfinite(lengths)&(lengths>0)):
        raise ValueError('Require positive finite thin-layer lengths')
    if hole is None:raise ValueError('The ray rule requires an elliptic hole')
    if not isinstance(layer_order,int) or layer_order<6 or not isinstance(bulk_order,int) or bulk_order<2:
        raise ValueError('Require layer_order >= 6 and bulk_order >= 2')
    if not np.isfinite(extent) or extent<=0:raise ValueError('extent must be positive')
    rays,tw,cross,normal=boundary_rays(bounds,hole,nx,ny,factor,x_breaks,
                                     corner_width=.15 if resolve_corners else 0.,
                                     minimum_tangent_order=minimum_tangent_order)
    axes=np.asarray(hole.axes);rmax=np.linalg.norm(rays/axes,axis=1);span=rmax-1
    if np.any(span<=0):raise ValueError('Ellipse must be strictly inside rectangle')
    scale=float(np.max(lengths));minor=float(np.min(axes))
    inner=np.minimum(.25,extent*scale/(minor*span))
    outer=np.minimum(.25,extent*scale/(normal*span/rmax))
    nodes=[];weights=[]
    outer_rule=layer_gauss(layer_order,True)
    if resolve_corners:
        cuts=(0.,1/16,1/4,1.)
        counts=(layer_order//3,layer_order//3,layer_order-2*(layer_order//3))
        zz=[];ww=[]
        for low,high,count in zip(cuts[:-1],cuts[1:],counts):
            z,w=unit_gauss(count)
            if low==0:w=w*3*z*z;z=z**3
            zz.append(low+(high-low)*z);ww.append((high-low)*w)
        outer_rule=1-np.concatenate(zz)[::-1],np.concatenate(ww)[::-1]
    for lo,hi,rule in ((np.zeros_like(inner),inner,layer_gauss(layer_order)),
                       (inner,1-outer,unit_gauss(bulk_order)),
                       (1-outer,np.ones_like(outer),outer_rule)):
        z,w=rule
        t=lo[:,None]+(hi-lo)[:,None]*z
        r=1+span[:,None]*t
        pts=np.asarray(hole.center)+r[:,:,None]/rmax[:,None,None]*rays[:,None,:]
        jac=cross[:,None]*r*span[:,None]/rmax[:,None]**2
        ww=tw[:,None]*w*(hi-lo)[:,None]*jac
        nodes.append(pts.reshape(-1,2));weights.append(ww.ravel())
    return np.vstack(nodes),np.concatenate(weights)
