"""Experimental geometric short-response enrichment; no force snapshots or filters.

Host-only research adapter around an existing rational ImmersedFlowPlan. Every
added streamfunction vanishes to second order on every boundary (including the
outlet). Existing outlet freedom remains in the original space. Analytic jets
supply all velocity derivatives. Neither the lift nor time integrator changes.
"""
import numpy as np
import scipy.linalg as la
from bspf_models.fluids.immersed_flow import ImmersedFlowPlan


class Jet:
    """Value, x, y, xx, xy, yy derivatives with NumPy broadcasting."""
    def __init__(self, value, dx=0, dy=0, dxx=0, dxy=0, dyy=0):
        self.a = (value, dx, dy, dxx, dxy, dyy)
    def __add__(self, other):
        other = other if isinstance(other, Jet) else Jet(other)
        return Jet(*(a+b for a,b in zip(self.a, other.a)))
    __radd__ = __add__
    def __neg__(self): return Jet(*(-a for a in self.a))
    def __sub__(self, other): return self + -other
    def __rsub__(self, other): return -self + other
    def __mul__(self, other):
        other = other if isinstance(other, Jet) else Jet(other)
        a,x,y,xx,xy,yy=self.a; b,u,v,uu,uv,vv=other.a
        return Jet(a*b,x*b+a*u,y*b+a*v,xx*b+2*x*u+a*uu,
                   xy*b+x*v+y*u+a*uv,yy*b+2*y*v+a*vv)
    __rmul__=__mul__
    def compose(self, value, first, second):
        _,x,y,xx,xy,yy=self.a
        return Jet(value,first*x,first*y,first*xx+second*x*x,
                   first*xy+second*x*y,first*yy+second*y*y)
    def __pow__(self, p):
        a=self.a[0]
        return self.compose(a**p,p*a**(p-1),p*(p-1)*a**(p-2))
    def exp(self):
        v=np.exp(self.a[0]);return self.compose(v,v,v)


def chebyshev(z, degree):
    a,b=Jet(np.ones_like(z.a[0])),z
    yield a
    if degree: yield b
    for k in range(2,degree+1):
        a,b=b,2*z*b-a
        yield b


def matched_primitive(distance, length):
    """R(n)=1-(1+n/length)exp(-n/length), with stable wall values and jets."""
    n=distance.a[0]; t=n/length; e=np.exp(-t)
    value=-np.expm1(-t)-t*e
    small=np.abs(t)<1e-3
    series=t*t*(.5+t*(-1/3+t*(1/8+t*(-1/30+t*(1/144+t*(-1/840+t/5760))))))
    value=np.where(small,series,value)
    return distance.compose(value,t*e/length,(1-t)*e/length**2)


class ResponseModes:
    def __init__(self, bounds, hole, lengths, angular_order=16, wall_order=24,
                 *, pairs=()):
        self.bounds,self.hole=bounds,hole
        self.lengths=tuple(lengths)
        pairs=tuple(pairs)
        if (not self.lengths and not pairs) or not all(np.isfinite(d) and d>0 for d in self.lengths):
            raise ValueError('Require positive finite layer lengths')
        if len(set(self.lengths)) != len(self.lengths): raise ValueError('Duplicate layer lengths')
        if not all(isinstance(k,int) and k>=0 for k in (angular_order,wall_order)):
            raise ValueError('Require nonnegative integer tangential orders')
        self.angular_order,self.wall_order=angular_order,wall_order
        self.pairs=tuple((float(ell),float(outer)) for ell,outer in pairs)
        if not all(np.isfinite(ell) and np.isfinite(outer) and 0<ell<outer
                   for ell,outer in self.pairs):
            raise ValueError('Each pair requires finite 0 < thin < outer')
        if len(set(self.pairs)) != len(self.pairs): raise ValueError('Duplicate pairs')

    def operators(self, points, *, strip_exponential=False, family=None):
        """Physical operators, or their individually factored exponential amplitudes.

        With strip_exponential=True each local-mode column is multiplied by
        exp(distance/length), analytically, including its derivative columns.
        These amplitudes are for weighted integration, not field evaluation.
        """
        if strip_exponential and self.pairs:
            raise ValueError('Paired modes contain two exponential weights; split them first')
        if family not in (None,'hole','top','bottom','inlet'):
            raise ValueError('Unknown boundary family')
        if family is not None and self.pairs:
            raise ValueError('Family selection currently requires unpaired modes')
        points=np.asarray(points)
        x,y=Jet(points[:,0],1),Jet(points[:,1],0,1)
        left,right,h=self.bounds;(cx,cy),(a,b)=self.hole.center,self.hole.axes
        X,Y=(x-cx)*(1/a),(y-cy)*(1/b)
        r=(X*X+Y*Y)**.5
        d=(r-1)*min(a,b)
        # Exact double zeros at inlet, outlet and horizontal walls.
        rectangle=((x-left)*(right-x)*(1-(y*(1/h))**2))**2
        rectangle=rectangle*(1/((cx-left)*(right-cx)*(1-(cy/h)**2))**2)
        f=(X*X+Y*Y-1)**2
        c,s=X*r**-1,Y*r**-1
        harmonics=[Jet(np.ones(len(points)))];ck,sk=c,s
        for k in range(1,self.angular_order+1):
            harmonics.extend((ck,sk));ck,sk=ck*c-sk*s,sk*c+ck*s
        result=[]
        def decay(distance,length):
            if strip_exponential:
                return distance.compose(np.ones(len(points)),-1/length,1/length**2)
            return ((-1/length)*distance).exp()
        for ell in self.lengths:
            if family in (None,'hole'):
                radial=d*d*decay(d,ell)*rectangle
                result.extend(radial*q for q in harmonics)
            for name,distance,tangent in (('top',h-y,(2*x-left-right)*(1/(right-left))),
                                          ('bottom',y+h,(2*x-left-right)*(1/(right-left))),
                                          ('inlet',x-left,y*(1/h))):
                if family is not None and family!=name:continue
                # rectangle contains distance^2 already; polynomial f gives the
                # exact hole constraint without a fitted boundary correction.
                radial=rectangle*f*decay(distance,ell)
                result.extend(radial*q for q in chebyshev(tangent,self.wall_order))
        # Pair a thin velocity correction with broad compensating flow. Each
        # R has a double zero at its own wall; remove that wall's factor from
        # the envelope so the intended near-wall shape is not multiplied by n^2.
        norm=((cx-left)*(right-cx)*(1-(cy/h)**2))**2
        wall_envelopes=(
            ((x-left)*(right-x)*(h+y)*(1/h**2))**2*(1/norm),
            ((x-left)*(right-x)*(h-y)*(1/h**2))**2*(1/norm),
            ((right-x)*(1-(y*(1/h))**2))**2*(1/norm))
        walls=((h-y,(2*x-left-right)*(1/(right-left))),
               (y+h,(2*x-left-right)*(1/(right-left))),
               (x-left,y*(1/h)))
        for ell,outer in self.pairs:
            radial=(matched_primitive(d,ell)-matched_primitive(d,outer))*rectangle
            result.extend(radial*q for q in harmonics)
            for (distance,tangent),envelope in zip(walls,wall_envelopes):
                radial=(matched_primitive(distance,ell)-matched_primitive(distance,outer))*envelope*f
                result.extend(radial*q for q in chebyshev(tangent,self.wall_order))
        arrays=[np.column_stack([np.broadcast_to(j.a[k],len(points)) for j in result])
                for k in range(6)]
        p,x,y,xx,xy,yy=arrays
        return p,y,-x,xy,yy,-xx


class EnrichedFlowPlan:
    """Direct sum with energy-orthogonalized response modes, retaining old space."""
    _gram=ImmersedFlowPlan._gram
    explicit=ImmersedFlowPlan.explicit
    rhs=ImmersedFlowPlan.rhs
    force_load=ImmersedFlowPlan.force_load
    diagnostics=ImmersedFlowPlan.diagnostics
    def __getattr__(self,name):return getattr(self.base,name)

    def __init__(self,base,modes,rcond=1e-10):
        if not base.rational_wall or base.buffer_strength != 0:
            raise ValueError('Research adapter currently requires rational wall and no sponge')
        if not np.isfinite(rcond) or not 0<rcond<1:
            raise ValueError('rcond must be between zero and one')
        self.base,self.modes=base,modes
        extra=modes.operators(base.points)[1:]
        M,K=self._gram(extra)
        self.scaling=1/np.sqrt(np.diag(M+K))
        extra=tuple(o*self.scaling for o in extra)
        B=base.operators_fluid; w=base.weights
        def cross(a,b):return a.T@(w[:,None]*b)
        C=sum(cross(a,b)*v for a,b,v in zip(B,extra,(1,1,2,1,1)))
        self.remove=la.solve(base.mass+base.stiffness,C,assume_a='pos')
        residual=tuple(n-b@self.remove for b,n in zip(B,extra))
        del extra,M,K,C
        mr,kr=self._gram(residual)
        ev,vec=la.eigh(mr+kr);keep=ev>rcond*ev[-1]
        self.rotation=vec[:,keep]/np.sqrt(ev[keep])
        self.remove=self.remove@self.rotation
        self.scaling_rotation=self.scaling[:,None]*self.rotation
        self.added_dofs=int(keep.sum());self.dofs=base.dofs+self.added_dofs
        self.operators_fluid=tuple(np.column_stack((b,r@self.rotation)) for b,r in zip(B,residual))
        del residual,B
        self.mass,self.stiffness=self._gram(self.operators_fluid)
        self.mass=(self.mass+self.mass.T)/2
        self.stiffness=(self.stiffness+self.stiffness.T)/2
        self.mass_factor=la.cho_factor(self.mass)
        self.linear=base.nu*self.stiffness
        self.linear_lift=np.zeros(self.dofs)
        self.stokes_state=np.zeros(self.dofs)
        self.out_ops=tuple(np.column_stack((b,-b@self.remove)) for b in base.out_ops)
        self.info=dict(lengths=list(modes.lengths),angular_order=modes.angular_order,
                      wall_order=modes.wall_order,pairs=[list(p) for p in modes.pairs],raw_modes=len(ev),added_dofs=self.added_dofs,
                      discarded=int((~keep).sum()),min_kept_energy_eigenvalue=float(ev[keep][0]),
                      max_energy_eigenvalue=float(ev[-1]),rcond=rcond)

    def stepper(self,dt,*,device=None):
        if device is not None: raise ValueError('Experimental enrichment is host-only')
        return ImmersedFlowPlan.stepper(self,dt)

    def split(self,state):
        state=np.asarray(state)
        return state[:self.base.dofs]-self.remove@state[self.base.dofs:],self.scaling_rotation@state[self.base.dofs:]

    def evaluate(self,state,points):
        a,c=self.split(state)
        return tuple(f+o@c for f,o in zip(self.base.evaluate(a,points),self.modes.operators(points)))

    def grid_many(self,states,x,y,*,batch_size=64):
        states=np.asarray(states);ab=[self.split(s) for s in states]
        xx,yy=np.meshgrid(x,y);points=np.column_stack((xx.ravel(),yy.ravel()))
        mask=self.hole.level(points)>=1-1e-13
        ids=np.flatnonzero(mask)
        coefficients=np.column_stack([c for a,c in ab])
        corrections=[np.empty((len(ids),len(states))) for _ in range(4)]
        for start in range(0,len(ids),2048):
            stop=min(start+2048,len(ids))
            ops=self.modes.operators(points[ids[start:stop]])
            for values,o in zip(corrections,(ops[0],ops[1],ops[2],ops[5]-ops[4])):
                values[start:stop]=o@coefficients
        for k,g in enumerate(self.base.grid_many(np.array([a for a,c in ab]),x,y,batch_size=batch_size)):
            for name,addition in zip(('psi','u','v','vorticity'),corrections):
                g[name].flat[ids]+=addition[:,k]
            yield g

    def grid(self,state,x,y):return next(self.grid_many(np.asarray([state]),x,y))
