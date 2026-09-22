"""Low-node exponential-weight assembly of the full thin response energy block.

Includes same-boundary blocks, inlet corners, obstacle/wall cross terms and
opposite-wall terms. The full NS adapter in weighted_response_space.py uses these pair blocks.
See WEIGHTED_RESPONSE_INTEGRATION.md.
"""
from time import perf_counter
import numpy as np
from scipy.special import roots_legendre
from bspf_models.elliptic.immersed_poisson import EllipticHole
from bspf_models.fluids.immersed_flow import channel_quadrature
from exponential_weight_quadrature import exponential_gauss
from short_response_space import ResponseModes


def polar_rays(bounds,hole,count):
    theta=2*np.pi*(np.arange(count)+.371)/count
    direction=np.column_stack((hole.axes[0]*np.cos(theta),hole.axes[1]*np.sin(theta)))
    left,right,h=bounds;cx,cy=hole.center
    rx=np.where(direction[:,0]>0,(right-cx)/direction[:,0],(left-cx)/direction[:,0])
    ry=np.where(direction[:,1]>0,(h-cy)/direction[:,1],(-h-cy)/direction[:,1])
    return direction,min(hole.axes)*(np.minimum(rx,ry)-1)


def weighted_hole_gram(re,order,angles=64,*,kind="energy"):
    hole=EllipticHole();bounds=(-1,5,1);minor=min(hole.axes)
    ell=np.sqrt(.01*(2/3)*.46/re);lengths=ell*np.array([.5,1,2,4])
    modes=[ResponseModes(bounds,hole,[length],16,0) for length in lengths]
    direction,H=polar_rays(bounds,hole,angles)
    matrix=np.zeros((132,132));expensive_points=0;basis_values=0
    start=perf_counter()
    for i in range(4):
        for j in range(i,4):
            effective=1/(1/lengths[i]+1/lengths[j])
            nodes,weights=zip(*(exponential_gauss(effective,h,order) for h in H))
            d=np.array(nodes);w=np.array(weights)
            r=1+d/minor
            p=(hole.center+r[:,:,None]*direction[:,None,:]).reshape(-1,2)
            # Exponential is already in the measure. Restore only the smooth
            # operator factors; exponents at these weighted nodes remain modest.
            def operators(k):
                return modes[k].operators(p,strip_exponential=True,family='hole')[1:]
            A=operators(i);B=A if i==j else operators(j)
            jac=np.prod(hole.axes)*r/minor*(2*np.pi/angles)
            ww=(w*jac).ravel()
            block=energy(A,B,ww,kind)
            si=slice(33*i,33*(i+1));sj=slice(33*j,33*(j+1))
            matrix[si,sj]=block;matrix[sj,si]=block.T
            expensive_points+=len(p)
            basis_values+=len(p)*33*(1 if i==j else 2)
    return matrix,dict(radial_nodes=order,angles=angles,points_per_block=order*angles,
                       block_node_visits=expensive_points,basis_value_visits=basis_values,
                       seconds=perf_counter()-start)


def wall_nodes(bounds,hole,family,length,order,tangent_order=48):
    left,right,h=bounds;cx,cy=hole.center;a,b=hole.axes
    z,w=roots_legendre(tangent_order)
    if family=='inlet':tangent=h*z;tw=h*w;depth=right-left
    else:tangent=(left+right)/2+(right-left)/2*z;tw=(right-left)/2*w;depth=2*h
    points=[];weights=[]
    for s,ws in zip(tangent,tw):
        intervals=[(0.,depth)]
        projected=(s-cy)/b if family=='inlet' else (s-cx)/a
        if abs(projected)<1:
            radius=(a if family=='inlet' else b)*np.sqrt(1-projected**2)
            if family=='inlet':near,far=cx-radius-left,cx+radius-left
            elif family=='top':near,far=h-cy-radius,h-cy+radius
            else:near,far=h+cy-radius,h+cy+radius
            intervals=[(0.,near),(far,depth)]
        for lo,hi in intervals:
            prefactor=np.exp(-lo/length)
            if prefactor==0:continue  # Below representable measure, not a chosen cutoff.
            d,wd=exponential_gauss(length,hi-lo,order);d=d+lo
            if family=='top':p=np.column_stack((np.full(order,s),h-d))
            elif family=='bottom':p=np.column_stack((np.full(order,s),d-h))
            else:p=np.column_stack((left+d,np.full(order,s)))
            points.append(p);weights.append(ws*prefactor*wd)
    return np.vstack(points),np.concatenate(weights)


def weighted_wall_gram(re,family,order,tangent_order=48,*,kind="energy"):
    hole=EllipticHole();bounds=(-1,5,1)
    ell=np.sqrt(.01*(2/3)*.46/re);lengths=ell*np.array([.5,1,2,4])
    modes=[ResponseModes(bounds,hole,[length],0,24) for length in lengths]
    matrix=np.zeros((100,100));visits=0;values=0;start=perf_counter()
    for i in range(4):
        for j in range(i,4):
            effective=1/(1/lengths[i]+1/lengths[j])
            p,w=wall_nodes(bounds,hole,family,effective,order,tangent_order)
            A=modes[i].operators(p,strip_exponential=True,family=family)[1:]
            B=A if i==j else modes[j].operators(p,strip_exponential=True,family=family)[1:]
            block=energy(A,B,w,kind)
            si=slice(25*i,25*(i+1));sj=slice(25*j,25*(j+1))
            matrix[si,sj]=block;matrix[sj,si]=block.T
            visits+=len(w);values+=len(w)*25*(1 if i==j else 2)
    return matrix,dict(reynolds=re,family=family,radial_nodes=order,tangent_nodes=tangent_order,
                       block_node_visits=visits,basis_value_visits=values,seconds=perf_counter()-start)


def energy(A,B,w,kind="energy"):
    factors={"energy":(1,1,2,1,1),"mass":(1,1,0,0,0),"stiffness":(0,0,2,1,1)}[kind]
    return sum(c*(a.T@(w[:,None]*b)) for a,b,c in zip(A,B,factors) if c)


def complete_thin_gram(re,order=8,*,kind="energy"):
    hole=EllipticHole();bounds=(-1,5,1);minor=min(hole.axes)
    ell=np.sqrt(.01*(2/3)*.46/re);lengths=ell*np.array([.5,1,2,4])
    modes=[ResponseModes(bounds,hole,[s],16,24) for s in lengths]
    families=[('hole',0,33),('top',33,25),('bottom',58,25),('inlet',83,25)]
    ids={name:np.concatenate([np.arange(n)+offset+108*k for k in range(4)]) for name,offset,n in families}
    G=np.zeros((432,432));visits=0;values=0;start=perf_counter()
    for family,offset,n in families:
        block,rec=weighted_hole_gram(re,order,kind=kind) if family=='hole' else weighted_wall_gram(re,family,order,kind=kind)
        G[np.ix_(ids[family],ids[family])]=block
        visits+=rec['block_node_visits'];values+=rec['basis_value_visits']
    # Corner interactions have two known exponential directions, so both
    # coordinates use weighted rules. The far-away hole correction is retained.
    theta=2*np.pi*(np.arange(32)+.37)/32;z,wz=roots_legendre(12)
    radius=(z+1)/2;wr=wz/2
    hp=(hole.center+radius[:,None,None]*np.column_stack((hole.axes[0]*np.cos(theta),hole.axes[1]*np.sin(theta)))[None,:,:]).reshape(-1,2)
    hw=np.broadcast_to(np.prod(hole.axes)*radius[:,None]*wr[:,None]*(2*np.pi/32),(12,32)).ravel()
    for family,offset in [('top',33),('bottom',58)]:
        for i,si in enumerate(lengths):
            for j,sj in enumerate(lengths):
                dx,wx=exponential_gauss(sj,6.,order)
                dy,wy=exponential_gauss(si,2.,order)
                xx,yy=np.meshgrid(-1+dx,1-dy if family=='top' else -1+dy,indexing='ij')
                p=np.column_stack((xx.ravel(),yy.ravel()));weights=np.outer(wx,wy).ravel()
                A=modes[i].operators(p,strip_exponential=True,family=family)[1:]
                B=modes[j].operators(p,strip_exponential=True,family='inlet')[1:]
                block=energy(A,B,weights,kind)
                ha=modes[i].operators(hp,family=family)[1:]
                hb=modes[j].operators(hp,family='inlet')[1:]
                block-=energy(ha,hb,hw,kind)
                ii=np.arange(25)+offset+108*i;jj=np.arange(25)+83+108*j
                G[np.ix_(ii,jj)]=block;G[np.ix_(jj,ii)]=block.T
                visits+=len(p)+len(hp);values+=(len(p)+len(hp))*50
    # Obstacle/wall exponent sums are affine in the ellipse radial coordinate.
    # Reverse the normal coordinate when the combined decay changes sign.
    direction,H=polar_rays(bounds,hole,32)
    for family,offset,D,alpha in [('top',33,1-hole.center[1],direction[:,1]),
                                  ('bottom',58,1+hole.center[1],-direction[:,1]),
                                  ('inlet',83,hole.center[0]+1,-direction[:,0])]:
        for i,si in enumerate(lengths):
            for j,sj in enumerate(lengths):
                pts=[];weights=[]
                for ray,upper,aa in zip(direction,H,alpha):
                    first=(D-aa)/sj
                    last=upper/si+(D-aa*(1+upper/minor))/sj
                    kappa=(last-first)/upper
                    pref=np.exp(-min(first,last))
                    if pref==0:continue
                    if abs(kappa)*upper<1e-10:
                        z,w=roots_legendre(order);d=upper*(z+1)/2;ww=upper*w/2
                    else:
                        d,ww=exponential_gauss(1/abs(kappa),upper,order)
                        if kappa<0:d=upper-d
                    r=1+d/minor
                    pts.append(hole.center+r[:,None]*ray)
                    weights.append(pref*ww*np.prod(hole.axes)*r/minor*(2*np.pi/32))
                if pts:
                    p=np.vstack(pts);ww=np.concatenate(weights)
                    A=modes[i].operators(p,strip_exponential=True,family='hole')[1:]
                    B=modes[j].operators(p,strip_exponential=True,family=family)[1:]
                    block=energy(A,B,ww,kind)
                    ii=np.arange(33)+108*i;jj=np.arange(25)+offset+108*j
                    G[np.ix_(ii,jj)]=block;G[np.ix_(jj,ii)]=block.T
                    visits+=len(p);values+=len(p)*58
    # Opposite-wall products are exponentially small, but are included using
    # a fixed exact-geometry rule rather than dropped as zero.
    p,w=channel_quadrature(bounds,hole,9,7,1.)
    both=ResponseModes(bounds,hole,lengths,0,24)
    top=both.operators(p,family='top')[1:];bottom=both.operators(p,family='bottom')[1:]
    block=energy(top,bottom,w,kind)
    G[np.ix_(ids['top'],ids['bottom'])]=block
    G[np.ix_(ids['bottom'],ids['top'])]=block.T
    visits+=len(p);values+=len(p)*200
    raw_asymmetry=float(np.max(abs(G-G.T)))
    G=(G+G.T)/2
    return G,dict(reynolds=re,radial_nodes=order,block_node_visits=visits,
                  basis_value_visits=values,seconds=perf_counter()-start,
                  raw_symmetry_relative=raw_asymmetry/max(float(np.max(abs(G))),1e-300),
                  previous_production_basis_value_visits=91136*432)
