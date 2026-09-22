"""Experimental block-weighted assembly and nonlinear action for channel NS.

Only the smooth background uses the bulk rule. Every thin-family integral uses
an exponential measure in its own normal coordinate. No background-by-background
matrix is formed on the layer nodes, and no Q-by-N matrix is postmultiplied by
the full background rotation there. Pairwise exponential rules supply thin M/K.
"""
from time import perf_counter
from functools import cached_property
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
import scipy.linalg as la
from scipy.special import roots_legendre
from bspf_models.elliptic.immersed_poisson import EllipticHole
from bspf_models.fluids.immersed_flow import channel_quadrature
from exponential_weight_quadrature import exponential_gauss
from short_response_space import ResponseModes, EnrichedFlowPlan
from weighted_response_assembly import complete_thin_gram, energy


FAMILIES = (('hole', 0, 33), ('top', 33, 25),
            ('bottom', 58, 25), ('inlet', 83, 25))


def refine_rational_background_stiffness(base,factor):
    """One-off linear integration refinement; keep the nonlinear volume rule.

    Assemble in the original scaled coordinates, then transform the small Gram
    matrix. No Q-by-N rotation is needed on the additional linear-only points.
    This helper is restricted to the zero-sponge rational lift used here.
    """
    if not base.rational_wall or base.buffer_strength!=0:
        raise ValueError('Linear-only refinement requires the zero-sponge rational plan')
    if 'stokes_state' in base.__dict__:
        raise ValueError('Refine before requesting a Stokes state')
    if not np.isfinite(factor) or factor<=base.quadrature_factor:
        raise ValueError('Require a larger finite linear quadrature factor')
    start=perf_counter()
    points,weights=channel_quadrature(base.bounds,base.hole,base.nx,base.ny,
                                      factor,(base.buffer_start,))
    gram=np.zeros((base.ndofs,base.ndofs))
    for lo in range(0,len(weights),4096):
        hi=min(lo+4096,len(weights));w=weights[lo:hi]
        for op,multiplier in zip(base.operators(points[lo:hi])[3:],(2,1,1)):
            scaled=op*base.scale
            gram+=multiplier*(scaled.T@(w[:,None]*scaled))
    normalized=base.transform/base.scale[:,None]
    stiffness=normalized.T@gram@normalized
    stiffness=(stiffness+stiffness.T)/2
    change=float(np.linalg.norm(stiffness-base.stiffness)/np.linalg.norm(stiffness))
    base.stiffness=stiffness
    base.linear=base.nu*stiffness
    seconds=perf_counter()-start
    base.setup_seconds+=seconds
    return dict(quadrature_factor=factor,points=len(weights),seconds=seconds,
                relative_matrix_change=change,nonlinear_points=len(base.weights))


@contextmanager
def basis_evaluation_pool(base, workers=4):
    """Reuse the existing MPFR chunk evaluator; preserve its numerical precision."""
    previous=base._basis_executor
    if previous is not None or workers==1:
        yield
        return
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
        base._basis_executor=pool
        try:
            yield
        finally:
            base._basis_executor=previous


def normal_rule(length, upper, order):
    """Composite weighted Gaussian rule, fixed dimensionless panel boundaries.

    The shortest scale is length/8. Splitting the known exponential variation
    also resolves its quadratic products without increasing order with Re.
    The final interval extends to the physical boundary; no tail is truncated.
    """
    cuts = [0.] + [length*z for z in (.25, 1., 4., 16.) if length*z < upper] + [upper]
    nodes, weights = [], []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        z, w = exponential_gauss(length, hi-lo, order)
        nodes.append(lo+z)
        weights.append(np.exp(-lo/length)*w)
    return np.concatenate(nodes), np.concatenate(weights)


def tangent_rule(lo, hi, count, layer, order, factor=1.5):
    # Both end corners are resolved in scaled coordinates as well. Bulk tangent
    # resolution is set by the background degree, independently of Reynolds.
    width = min((hi-lo)/4, 16*layer)
    cuts = sorted(set([lo, hi, lo+width, hi-width,
                       *np.linspace(lo, hi, 5)[1:-1]]))
    nodes, weights = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if a == lo or b == hi:
            cc = [a, a+(b-a)/16, a+(b-a)/4, b] if a == lo else [a, b-(b-a)/4, b-(b-a)/16, b]
            panels = zip(cc[:-1], cc[1:])
        else:
            panels = [(a, b)]
        for c, d in panels:
            n = max(order, int(np.ceil(factor*count*(d-c)/(hi-lo))))
            z, w = roots_legendre(n)
            nodes.append((c+d)/2+(d-c)*z/2)
            weights.append((d-c)*w/2)
    return np.concatenate(nodes), np.concatenate(weights)


def family_rule(bounds, hole, family, length, order=12, nx=73, ny=33, angles=128, tangent_factor=1.5):
    """Points, exp(-distance/length) weights, and dimensionless distance."""
    left, right, h = bounds
    cx, cy = hole.center
    a, b = hole.axes
    points, weights, distances = [], [], []
    if family == 'hole':
        # Split at rectangle-corner ray directions, where the radial extent has
        # a derivative jump. Tangential panels otherwise depend only on geometry.
        corners = np.array([[left,-h],[right,-h],[right,h],[left,h]])
        cuts = sorted(set([0., 2*np.pi, *np.mod(np.arctan2((corners[:,1]-cy)/b,
                                                         (corners[:,0]-cx)/a),2*np.pi)]))
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            z, wz = roots_legendre(max(12, int(np.ceil(angles*(hi-lo)/(2*np.pi)))))
            for theta, wt in zip((lo+hi)/2+(hi-lo)*z/2, (hi-lo)*wz/2):
                ray = np.array([a*np.cos(theta),b*np.sin(theta)])
                rx = ((right-cx) if ray[0]>0 else (left-cx))/ray[0]
                ry = ((h-cy) if ray[1]>0 else (-h-cy))/ray[1]
                upper = min(a,b)*(min(rx,ry)-1)
                d, w = normal_rule(length, upper, order)
                r = 1+d/min(a,b)
                points.append(hole.center+r[:,None]*ray)
                weights.append(w*wt*a*b*r/min(a,b))
                distances.append(d/length)
    else:
        lo, hi, count = (-h,h,ny) if family=='inlet' else (left,right,nx)
        tangent, tw = tangent_rule(lo,hi,count,length,order,tangent_factor)
        depth = right-left if family=='inlet' else 2*h
        for s, wt in zip(tangent,tw):
            projected = (s-cy)/b if family=='inlet' else (s-cx)/a
            intervals = [(0.,depth)]
            if abs(projected)<1:
                radius=(a if family=='inlet' else b)*np.sqrt(1-projected**2)
                center_distance={'inlet':cx-left,'top':h-cy,'bottom':h+cy}[family]
                intervals=[(0.,center_distance-radius),(center_distance+radius,depth)]
            for low, high in intervals:
                prefactor=np.exp(-low/length)
                if prefactor==0: continue  # Measure below floating-point range.
                d,w=normal_rule(length,high-low,order)
                d=d+low
                if family=='inlet': p=np.column_stack((left+d,np.full(len(d),s)))
                elif family=='top': p=np.column_stack((np.full(len(d),s),h-d))
                else: p=np.column_stack((np.full(len(d),s),d-h))
                points.append(p);weights.append(w*wt*prefactor);distances.append(d/length)
    return np.vstack(points),np.concatenate(weights),np.concatenate(distances)


def convection(a,b):
    """(a.velocity dot grad) b.velocity, with b.v_y=-b.u_x."""
    return a[0]*b[2]+a[1]*b[3], a[0]*b[4]-a[1]*b[2]


class WeightedResponsePlan(EnrichedFlowPlan):
    def __init__(self, base, *, order=12, angles=128, rcond=1e-10, broad_lengths=(.1,.2), tangent_factor=1.5, progress=print):
        start=perf_counter()
        if base.bounds != (-1.,5.,1.) or base.hole != EllipticHole():
            raise ValueError('Experimental pair assembler requires default channel geometry')
        if not (base.rational_wall or base.factored_wall) or base.buffer_strength != 0:
            raise ValueError('Requires rational or geometric factor wall and zero sponge')
        self.base=base
        lengths=np.sqrt(.01*base.nu)*np.array([.5,1,2,4])
        self.thin_modes=ResponseModes(base.bounds,base.hole,lengths)
        broad_lengths=tuple(broad_lengths)
        self.background_modes=(ResponseModes(base.bounds,base.hole,broad_lengths) if broad_lengths else None)
        self.modes=ResponseModes(base.bounds,base.hole,[*lengths,*broad_lengths])
        broad_count=108*len(broad_lengths)
        def broad_operators(points):
            return (self.background_modes.operators(points)[1:] if self.background_modes is not None
                    else tuple(np.empty((len(points),0)) for _ in range(5)))
        n=base.dofs
        # Optional broad columns stay raw until joint energy truncation.
        broad=broad_operators(base.points)
        self.bulk=(tuple(np.column_stack((a,b)) for a,b in zip(base.operators_fluid,broad))
                   if broad_count else base.operators_fluid)
        def bulk_block(bb,kind):
            cross=energy(base.operators_fluid,broad,base.weights,kind)
            extra=energy(broad,broad,base.weights,kind)
            return np.block([[bb,cross],[cross.T,extra]])
        mb=bulk_block(base.mass,'mass');kb=bulk_block(base.stiffness,'stiffness')
        del broad
        re=(2/3)*.46/base.nu
        me,_=complete_thin_gram(re,order,kind='mass')
        ke,_=complete_thin_gram(re,order,kind='stiffness')
        cm=np.zeros((n+broad_count,432));ck=np.zeros_like(cm)
        self.local=[]
        lm=np.zeros(432+broad_count);lk=np.zeros(432+broad_count)
        for family,offset,count in FAMILIES:
            ids=np.concatenate([np.arange(count)+offset+108*k for k in range(4)])
            p,w,z=family_rule(base.bounds,base.hole,family,max(lengths),order,
                              base.nx,base.ny,angles,tangent_factor)
            raw,lift=base.operators(p,with_base=True)
            lift=tuple(o@base.lift_coefficients+b for o,b in zip(raw,lift))[1:]
            broad=broad_operators(p)
            raw=(tuple(np.column_stack((a,b)) for a,b in zip(raw[1:],broad))
                 if broad_count else raw[1:])
            del broad
            thin=self.thin_modes.operators(p)[1:]
            amplitude=self.thin_modes.operators(p,strip_exponential=True,family=family)[1:]
            decay=np.column_stack([np.exp(z*(1-max(lengths)/ell))[:,None]*np.ones((1,count))
                                   for ell in lengths])
            amplitude=tuple(a*decay for a in amplitude)
            for matrix,kind in ((cm,'mass'),(ck,'stiffness')):
                cross=energy(raw,amplitude,w,kind)
                matrix[:n,ids]=base.transform.T@cross[:base.ndofs]
                matrix[n:,ids]=cross[base.ndofs:]
            lm[ids]=sum(a.T@(w*b) for a,b in zip(amplitude[:2],lift[:2]))
            lk[ids]=sum(c*a.T@(w*b) for a,b,c in zip(amplitude[2:],lift[2:],(2,1,1)))
            self.local.append(dict(family=family,ids=ids,points=p,weights=w,
                                   raw=raw,thin=thin,amplitude=amplitude,lift=lift))
            progress(f'WEIGHTED {family}: {len(w)} nodes; elapsed {perf_counter()-start:.1f}s',flush=True)
        # Keep thin scales first, followed by any optional broad scales.
        def blocks(bb,be,ee):
            return np.column_stack((be[:n],bb[:n,n:])),np.block([[ee,be[n:].T],[be[n:],bb[n:,n:]]])
        cm,me=blocks(mb,cm,me);ck,ke=blocks(kb,ck,ke)
        self.cross_mass,self.extra_mass=cm,me
        self.cross_stiffness,self.extra_stiffness=ck,ke
        self.scaling=1/np.sqrt(np.diag(me+ke))
        c=(cm+ck)*self.scaling
        remove=la.solve(base.mass+base.stiffness,c,assume_a='pos')
        residual=(me+ke)*self.scaling[:,None]*self.scaling-c.T@remove
        ev,vec=la.eigh((residual+residual.T)/2)
        if ev[0] < -rcond*ev[-1]:
            raise ArithmeticError(f'Indefinite weighted Schur complement: {ev[0]:.3e}; max={ev[-1]:.3e}')
        keep=ev>rcond*ev[-1]
        rotation=vec[:,keep]/np.sqrt(ev[keep])
        self.rotation=rotation
        self.remove=remove@rotation
        self.scaling_rotation=self.scaling[:,None]*rotation
        self.added_dofs=int(keep.sum());self.dofs=n+self.added_dofs
        def assemble(bb,be,ee):
            be=be*self.scaling
            rr=ee*self.scaling[:,None]*self.scaling-be.T@remove-remove.T@be+remove.T@bb@remove
            cross=(be-bb@remove)@rotation
            matrix=np.block([[bb,cross],[cross.T,rotation.T@rr@rotation]])
            return (matrix+matrix.T)/2
        self.mass=assemble(base.mass,cm,me)
        self.stiffness=assemble(base.stiffness,ck,ke)
        self.mass_factor=la.cho_factor(self.mass)
        self.linear=base.nu*self.stiffness
        self.out_ops=tuple(np.column_stack((b,-b@self.remove)) for b in base.out_ops)
        blm=sum(o.T@(base.weights*l) for o,l in zip(self.bulk[:2],base.lift_fields[1:3]))
        blk=sum(c*o.T@(base.weights*l) for o,l,c in zip(self.bulk[2:],base.lift_fields[3:],(2,1,1)))
        lm[432:]=blm[n:];lk[432:]=blk[n:]
        self.lift_mass=np.r_[blm[:n],self.scaling_rotation.T@lm-self.remove.T@blm[:n]]
        self.lift_stiffness=np.r_[blk[:n],self.scaling_rotation.T@lk-self.remove.T@blk[:n]]
        self.linear_lift=(np.zeros(self.dofs) if base.rational_wall
                          else base.nu*self.lift_stiffness)
        self.info=dict(quadrature='exponential block',bulk_points=len(base.weights),
                       local_points={q['family']:len(q['weights']) for q in self.local},
                       normal_order=order,angles=angles,tangent_factor=tangent_factor,lengths=list(self.modes.lengths),
                       background_dofs=n,added_dofs=self.added_dofs,raw_modes=432+broad_count,
                       rcond=rcond,min_energy_eigenvalue=float(ev[0]),
                       min_kept_energy_eigenvalue=float(ev[keep][0]),
                       setup_seconds=perf_counter()-start)

    @cached_property
    def stokes_state(self):
        return la.solve(self.linear,-self.linear_lift,assume_a='pos')

    def compatible_state(self,**kwargs):
        return np.r_[self.base.compatible_state(**kwargs),np.zeros(self.added_dofs)]

    def refine_nonlinear_hole_rule(self,angles):
        """Refine only the hole's nonlinear tangent rule, keeping M/K and space.

        This is a diagnostic refinement. Acceptance still requires an independent
        nonlinear audit; it does not change rank selection or the initial state.
        """
        if not isinstance(angles,int) or angles<16:
            raise ValueError('Require an integer angular budget >= 16')
        start=perf_counter()
        base=self.base;lengths=self.thin_modes.lengths
        p,w,z=family_rule(base.bounds,base.hole,'hole',max(lengths),
                          self.info['normal_order'],base.nx,base.ny,angles)
        raw,lift=base.operators(p,with_base=True)
        lift=tuple(o@base.lift_coefficients+b for o,b in zip(raw,lift))[1:]
        broad=(self.background_modes.operators(p)[1:] if self.background_modes is not None
               else tuple(np.empty((len(p),0)) for _ in range(5)))
        raw=(tuple(np.column_stack((a,b)) for a,b in zip(raw[1:],broad))
             if self.background_modes is not None else raw[1:])
        thin=self.thin_modes.operators(p)[1:]
        amplitude=self.thin_modes.operators(p,strip_exponential=True,family='hole')[1:]
        decay=np.column_stack([np.exp(z*(1-max(lengths)/ell))[:,None]*np.ones((1,33))
                               for ell in lengths])
        amplitude=tuple(a*decay for a in amplitude)
        ids=np.concatenate([np.arange(33)+108*k for k in range(4)])
        self.local[0]=dict(family='hole',ids=ids,points=p,weights=w,raw=raw,
                           thin=thin,amplitude=amplitude,lift=lift)
        self.info['nonlinear_hole_angles']=angles
        self.info['local_points']['hole']=len(w)
        self.info['setup_seconds']+=perf_counter()-start

    def explicit(self,state,time=0.):
        ab,c=self.split(state)
        cb=np.r_[ab,c[432:]]
        raw_coeff=np.r_[self.base.transform@ab,c[432:]]
        background=tuple(o@cb+l for o,l in zip(self.bulk,self.base.lift_fields[1:]))
        force=convection(background,background)
        fb=-sum(o.T@(self.base.weights*f) for o,f in zip(self.bulk[:2],force))
        fe=np.zeros(432);raw_load=np.zeros(len(raw_coeff))
        for q in self.local:
            w=q['weights'];ids=q['ids']
            b=tuple(o@raw_coeff+l for o,l in zip(q['raw'],q['lift']))
            e=tuple(o@c[:432] for o in q['thin'])
            eg=tuple(o@c[ids] for o in q['amplitude'])
            total=tuple(x+y for x,y in zip(b,e))
            force=convection(total,total)
            fe[ids]=-sum(o.T@(w*f) for o,f in zip(q['amplitude'][:2],force))
            # Sum_g [N(B,E_g)+N(E_g,B)+N(E_g,E)] = N(B+E)-N(B).
            pieces=(convection(b,eg),convection(eg,b),convection(eg,e))
            correction=tuple(sum(p[k] for p in pieces) for k in (0,1))
            raw_load-=sum(o.T@(w*f) for o,f in zip(q['raw'][:2],correction))
        n=self.base.dofs
        fb[:n]+=self.base.transform.T@raw_load[:self.base.ndofs]
        fb[n:]+=raw_load[self.base.ndofs:]
        outu,outv=[o@ab+l for o,l in zip(self.base.out_ops,self.out_lift)]
        incoming=np.minimum(outu,0)*self.out_weights
        fb[:n]+=sum(o.T@(incoming*f) for o,f in zip(self.base.out_ops,(outu,outv)))
        extra=np.r_[fe,fb[n:]]
        return np.r_[fb[:n],self.scaling_rotation.T@extra-self.remove.T@fb[:n]]-self.linear_lift

    def diagnostics(self,state):
        w=self.base.weights;u,v,ux,uy,vx=self.base.lift_fields[1:]
        kinetic=float(w@(u*u+v*v)/2+state@self.mass@state/2+state@self.lift_mass)
        dissipation=float(self.nu*(w@(2*ux*ux+uy*uy+vx*vx)+state@self.stiffness@state+2*state@self.lift_stiffness))
        ab,c=self.split(state);raw_coeff=np.r_[self.base.transform@ab,c[432:]]
        speed=0.
        for q in self.local:
            u,v=[b@raw_coeff+l+e@c[:432] for b,l,e in zip(q['raw'][:2],q['lift'][:2],q['thin'][:2])]
            speed=max(speed,float(np.max(np.hypot(u,v))))
        outu=self.out_ops[0]@state+self.out_lift[0]
        derivative=self.rhs(state)
        return dict(kinetic_energy=kinetic,dissipation=dissipation,max_speed_on_layer_nodes=speed,
                    min_outlet_u=float(np.min(outu)),flux_in=4*self.peak*self.bounds[2]/3,
                    flux_out=float(self.out_weights@outu),sponge_perturbation_dissipation=0.,
                    acceleration_l2=float(np.sqrt(max(derivative@self.mass@derivative,0))))
