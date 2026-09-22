"""Axisymmetric free-boundary GS prototype with prescribed filament coils.

The computational spline boundary is in vacuum, NOT the plasma interface.
Its Dirichlet values are updated from the free-space ring-current Green function
and plasma volume quadrature. The plasma edge is the flux surface through a
prescribed limiter point. Single-axis, closed limited plasmas only: no separatrix,
coil-current optimisation, wall dynamics or R=0 treatment.

Green convention: -Delta* psi = mu0 R j_phi, psi -> 0 at infinity for rings.
All pair interactions are blocked, never stored as global dense matrices.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import ellipk, ellipe, hyp2f1
from scipy.ndimage import label
from scipy.optimize import brentq


def ring_flux(points, sources, currents, *, mu0=4e-7*np.pi, block_size=64, source_block=512):
    """Flux of circular filaments; currents include quadrature weights for j_phi.

    Singular filament targets are rejected, not softened. This is not a volume
    self-quadrature rule: plasma volume sources are used only at vacuum targets.
    """
    x, y, current = np.asarray(points, float), np.asarray(sources, float), np.asarray(currents, float)
    if (x.ndim != 2 or x.shape[1] != 2 or y.ndim != 2 or y.shape[1] != 2
            or current.shape != (len(y),) or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y)) or not np.all(np.isfinite(current))
            or np.any(x[:,0] <= 0) or np.any(y[:,0] <= 0)
            or not np.isfinite(mu0) or mu0 <= 0):
        raise ValueError('finite R>0 points, sources, currents and positive mu0 required')
    if any(not isinstance(n,(int,np.integer)) or n <= 0 for n in (block_size,source_block)):
        raise ValueError('block sizes must be positive integers')
    active = current != 0
    y, current = y[active], current[active]
    result = np.zeros(len(x))
    for a in range(0,len(x),block_size):
        t = x[a:a+block_size]
        for b in range(0,len(y),source_block):
            s = y[b:b+source_block]
            product = t[:,0,None]*s[None,:,0]
            m = 4*product/((t[:,0,None]+s[None,:,0])**2+(t[:,1,None]-s[None,:,1])**2)
            if np.any(m >= 1):
                raise ValueError('target on or too close to a singular filament')
            factor = np.empty_like(m)
            small = m < 1e-3
            # Stable analytic small-m expression avoids K/E cancellation.
            factor[small] = np.pi/16*m[small]**1.5*hyp2f1(1.5,1.5,3.,m[small])
            z = m[~small]
            factor[~small] = ((2-z)*ellipk(z)-2*ellipe(z))/np.sqrt(z)
            result[a:a+len(t)] += mu0/(2*np.pi)*(np.sqrt(product)*factor)@current[b:b+source_block]
    return result


@dataclass(frozen=True)
class FilamentCoils:
    positions: np.ndarray
    currents: np.ndarray
    mu0: float = 4e-7*np.pi

    def flux(self, points):
        return ring_flux(points, self.positions, self.currents, mu0=self.mu0)


def volume_rule(plan, n):
    """Tensor Gauss rule on the bounding box, filtered to the full section.

    Used for a compact plasma current separated from the computational boundary;
    this is not high-order cut-cell integration for currents touching that boundary.
    """
    if not isinstance(n,(int,np.integer)) or n < 8:
        raise ValueError('volume order must be an integer >=8')
    z,w = np.polynomial.legendre.leggauss(n)
    bounds = plan.domain.bounds
    axes = [bounds[i].mean()+np.ptp(bounds[i])/2*z for i in (0,1)]
    x,y = np.meshgrid(*axes,indexing='ij')
    local = np.column_stack((x.ravel(),y.ravel()))
    mask = plan.domain.contains(local)
    weights = np.outer(w,w).ravel()*np.prod(np.ptp(bounds,axis=1))/4
    return plan.physical(local[mask]), weights[mask], mask


class FreeBoundaryGSPlan:
    """Wrap a full-section SplineAnnulusGSPlan with free-space boundary updates.

    j_phi = amplitude * [beta R/R0 + (1-beta) R0/R] * positive(psi-psi_edge)^power.
    The amplitude enforces the prescribed positive total plasma current. This
    corresponds to pressure and FF' profiles with the same flux dependence.
    ``power`` controls current smoothness at the moving plasma interface.
    """
    def __init__(self, section_plan, coils, limiter, *, plasma_current,
                 beta=.5, power=6, volume_order=48):
        if len(section_plan.domain.boundaries) != 1:
            raise ValueError('free-boundary GS requires a full section without an inner hole')
        if section_plan.major_radius <= 0:
            raise ValueError('free-boundary profile normalization requires positive major_radius')
        self.plan, self.coils = section_plan, coils
        self.limiter = np.asarray(limiter,dtype=float)
        if (self.limiter.shape != (2,) or not np.all(np.isfinite(self.limiter))
                or not section_plan.domain.contains((self.limiter-[section_plan.major_radius,0])[None,:])[0]):
            raise ValueError('limiter point must be inside the computational section')
        if not np.isfinite(plasma_current) or plasma_current <= 0 or not 0 <= beta <= 1:
            raise ValueError('require positive plasma_current and beta in [0,1]')
        if not isinstance(power,(int,np.integer)) or power < 2:
            raise ValueError('require integer power>=2 for vanishing interface current')
        positions = np.asarray(coils.positions,float)
        # Validate coil arrays and exclude singularities from the PDE domain.
        coils.flux(self.limiter[None,:])
        if np.any(section_plan.domain.contains(positions-[section_plan.major_radius,0])):
            raise ValueError('coils must lie outside the computational section')
        self.current, self.beta, self.power = float(plasma_current), beta, power
        self.volume_order = volume_order
        self.quad, self.weights, self.quad_mask = volume_rule(section_plan,volume_order)
        self.grids = [section_plan.physical(g) for g in section_plan._grids]
        self.grids += [self.quad,self.limiter[None,:]]

    def _shape(self, points, psi, edge):
        r = points[:,0]/self.plan.major_radius
        return (self.beta*r+(1-self.beta)/r)*np.maximum(psi-edge,0.)**self.power

    def solve(self, initial_flux, *, tolerance=1e-5, pde_tolerance=1e-6,
              source_tolerance=1e-6, boundary_tolerance=1e-7,
              max_iterations=40, relaxation=.5, anderson_depth=4, callback=None):
        """Damped/Anderson outer Picard; inner GS, current and boundary closure gates.

        ``initial_flux`` must seed a positive-current region above the limiter
        flux. No prescribed plasma shape is passed to the solver.
        """
        if (not np.isfinite(tolerance) or tolerance <= 0 or not 0 < relaxation <= 1
                or not isinstance(max_iterations,int) or max_iterations < 1
                or not isinstance(anderson_depth,int) or not 0 <= anderson_depth <= 8):
            raise ValueError('invalid free-boundary iteration parameters')
        raw = [np.asarray(initial_flux(x)) for x in self.grids]
        if any(np.iscomplexobj(a) and np.max(abs(a.imag)) > 1e-10*max(1.,np.max(abs(a))) for a in raw):
            raise ValueError('initial flux must be real')
        states = [np.asarray(a.real,float) for a in raw]
        if any(a.shape != (len(x),) or not np.all(np.isfinite(a)) for a,x in zip(states,self.grids)):
            raise ValueError('invalid initial flux')
        history=[]
        mixing=[]
        cuts=np.cumsum([len(x) for x in self.grids])[:-1]
        b=self.plan.domain.boundaries[0]
        t=b.a+(np.arange(97)+.371)/97*(b.b-b.a)
        boundary_probes=self.plan.physical(b.curve(t))
        for iteration in range(max_iterations):
            edge=float(states[3][0])
            shape=[self._shape(x,u,edge) for x,u in zip(self.grids[:3],states[:3])]
            integral=float(self.weights@shape[2])
            if not np.isfinite(integral) or integral <= 0:
                raise ValueError('initial/current iterate has no positive plasma region')
            amplitude=self.current/integral
            currents=[amplitude*f for f in shape]
            def source(x):
                for grid,j in zip(self.grids[:2],currents[:2]):
                    if x.shape == grid.shape and np.array_equal(x,grid):
                        return self.coils.mu0*x[:,0]*j
                raise ValueError('unexpected inner source grid')
            def boundary_from(j):
                weighted=self.weights*j
                cache={}
                def flux(x):
                    key=np.asarray(x).tobytes()
                    if key not in cache:
                        cache[key]=self.coils.flux(x)+ring_flux(x,self.quad,weighted,mu0=self.coils.mu0)
                    return cache[key]
                return flux
            boundary=boundary_from(currents[2])
            def initial(x):
                for grid,u in zip(self.grids[:2],states[:2]):
                    if x.shape == grid.shape and np.array_equal(x,grid):
                        return u
                raise ValueError('unexpected initial flux grid')
            inner=self.plan.solve(source,(boundary,),initial_flux=initial,tolerance=pde_tolerance,
                                  source_tolerance=source_tolerance,boundary_tolerance=boundary_tolerance,max_iterations=12)
            updated=[inner.flux(x).real for x in self.grids]
            new_edge=float(updated[3][0])
            new_shape=[self._shape(x,u,new_edge) for x,u in zip(self.grids[:3],updated[:3])]
            new_integral=float(self.weights@new_shape[2])
            if not np.isfinite(new_integral) or new_integral <= 0:
                raise ValueError('plasma disappeared; revise coils, current or initial state')
            new_amplitude=self.current/new_integral
            new_current=[new_amplitude*f for f in new_shape]
            current_error=max(float(np.max(abs(a-b))/max(np.max(abs(b)),1e-300))
                              for a,b in zip(currents,new_current))
            # Recompute the FULL nonlinear GS residual against the NEW profiles.
            x=self.grids[1]; u=updated[1]/np.sqrt(x[:,0])
            represented=inner.transformed.particular.source_values(self.plan.source_plan.validation)
            rhs=self.coils.mu0*x[:,0]*new_current[1]/np.sqrt(x[:,0])
            residual=float(np.max(abs(represented+3*u/(4*x[:,0]**2)-rhs))/max(np.max(abs(rhs)),1e-300))
            new_boundary=boundary_from(new_current[2])
            scale=max(np.ptp(updated[2]),1e-300)
            boundary_change=float(np.max(abs(new_boundary(boundary_probes)-boundary(boundary_probes)))/scale)
            row=dict(iteration=iteration+1,current_relative_change=current_error,
                     nonlinear_pde_residual=residual,boundary_relative_change=boundary_change,
                     edge_flux=new_edge,axis_flux_sample=float(np.max(updated[2])),
                     inner_source_error=inner.history[-1]['source_relative_max'])
            history.append(row)
            if callback: callback(row)
            if max(current_error,residual,boundary_change) <= tolerance:
                # Reject multiple disjoint plasmas or plasma touching the outer box.
                active=np.zeros(self.volume_order**2,dtype=bool)
                active[self.quad_mask]=updated[2]>new_edge
                if label(active.reshape(self.volume_order,self.volume_order))[1] != 1:
                    raise ValueError('only a single connected limited plasma is supported')
                outer_flux=new_boundary(boundary_probes)
                if np.max(outer_flux) >= new_edge:
                    raise ValueError('plasma touches computational boundary or topology is unsupported')
                result=FreeBoundaryGSSolution(self,inner,new_edge,new_amplitude,history,new_current[2].copy())
                audit=result.current_audit()
                check_points,check_weights,_=volume_rule(self.plan,self.volume_order+11)
                check_current=result.current_density(check_points)
                high_boundary=self.coils.flux(boundary_probes)+ring_flux(
                    boundary_probes,check_points,check_weights*check_current,mu0=self.coils.mu0)
                audit['boundary_quadrature_relative_change']=float(np.max(abs(high_boundary-outer_flux))/scale)
                row['quadrature_audit']=audit
                if max(audit['relative_error'],audit['boundary_quadrature_relative_change']) > tolerance:
                    raise ValueError(f'free-boundary volume quadrature unresolved: {audit}; increase volume_order')
                return result
            flat=np.concatenate(states)
            residual_vector=np.concatenate(updated)-flat
            next_flat=flat+relaxation*residual_vector
            if anderson_depth:
                mixing.append((flat.copy(),residual_vector.copy()))
                mixing=mixing[-(anderson_depth+1):]
                if len(mixing)>1:
                    dx=np.column_stack([b[0]-a[0] for a,b in zip(mixing[:-1],mixing[1:])])
                    dr=np.column_stack([b[1]-a[1] for a,b in zip(mixing[:-1],mixing[1:])])
                    coefficients=np.linalg.lstsq(dr,residual_vector,rcond=1e-10)[0]
                    if np.linalg.norm(coefficients) <= 10:
                        candidate=next_flat-(dx+relaxation*dr)@coefficients
                        split=np.split(candidate,cuts)
                        if (np.all(np.isfinite(candidate)) and
                                np.max(split[2]) > split[3][0] and
                                np.linalg.norm(candidate-flat) <= 5*np.linalg.norm(residual_vector)):
                            next_flat=candidate
            states=np.split(next_flat,cuts)
        raise RuntimeError(f'free-boundary Picard failed after {max_iterations} iterations: {history[-1]}')


@dataclass
class FreeBoundaryGSSolution:
    plan: FreeBoundaryGSPlan
    equilibrium: object
    edge_flux: float
    amplitude: float
    history: list
    quadrature_current: np.ndarray

    def flux(self, points):
        return self.equilibrium.flux(points)

    def current_density(self, points):
        points=np.asarray(points)
        return self.amplitude*self.plan._shape(points,self.flux(points).real,self.edge_flux)

    def pressure(self, points):
        delta=np.maximum(self.flux(points).real-self.edge_flux,0.)
        return self.amplitude*self.plan.beta/self.plan.plan.major_radius*delta**(self.plan.power+1)/(self.plan.power+1)

    def exterior_flux(self, points):
        """Free-space coil + plasma flux outside the computational section."""
        points=np.asarray(points,float)
        if np.any(self.plan.plan.domain.contains(points-[self.plan.plan.major_radius,0])):
            raise ValueError('exterior_flux requires targets outside the computational section')
        return self.plan.coils.flux(points)+ring_flux(points,self.plan.quad,
            self.plan.weights*self.quadrature_current,mu0=self.plan.coils.mu0)

    def magnetic_axis(self, guess=None, **kwargs):
        return self.equilibrium.magnetic_axis(guess,**kwargs)

    def current_audit(self, order=None):
        points,weights,_=volume_rule(self.plan.plan,order or self.plan.volume_order+11)
        measured=float(weights@self.current_density(points))
        return dict(current=measured,relative_error=abs(measured-self.plan.current)/self.plan.current)

    def plasma_boundary(self, axis=None, *, count=64):
        """Trace first edge-flux crossing along rays; requires a star-shaped LCFS.

        This limited prototype does not classify X-points/separatrices. A missing
        crossing is rejected rather than replaced by a computational-wall point.
        """
        if not isinstance(count,int) or count < 8:
            raise ValueError('boundary count must be an integer >=8')
        axis=np.asarray(axis if axis is not None else self.magnetic_axis()['position'])
        if self.flux(axis[None,:]).real[0] <= self.edge_flux:
            raise ValueError('axis must lie above edge flux')
        domain=self.plan.plan.domain
        extent=np.linalg.norm(np.ptp(domain.bounds,axis=1))
        boundary=[]
        for theta in np.arange(count)*2*np.pi/count:
            direction=np.array([np.cos(theta),np.sin(theta)])
            distances=np.linspace(0,extent,65)
            probes=axis+distances[:,None]*direction
            inside=domain.contains(probes-[self.plan.plan.major_radius,0])
            end=np.flatnonzero(~inside)
            stop=int(end[0]) if len(end) else len(probes)
            f=self.flux(probes[:stop]).real-self.edge_flux
            crossings=np.flatnonzero(f <= 0)
            if not len(crossings):
                raise ValueError('no closed plasma boundary before the computational wall')
            k=int(crossings[0])
            if np.any(f[k:] > 1e-10*max(1.,abs(f[0]))):
                raise ValueError('plasma interface is not star-shaped about the supplied axis')
            radius=brentq(lambda r:float(self.flux((axis+r*direction)[None,:]).real[0]-self.edge_flux),
                          distances[k-1],distances[k],xtol=1e-10)
            boundary.append(axis+radius*direction)
        return np.asarray(boundary)
