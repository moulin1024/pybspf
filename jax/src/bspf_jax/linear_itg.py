"""Electrostatic linear ITG in a constant-curvature, shearless local model.

A finite Dirichlet radial interval uses standard BSPF trial functions and a
self-adjoint Galerkin Laplacian. One nonzonal (ky,k_parallel) Fourier harmonic
retains full velocity-dependent Bessel FLR. Gradients are local coefficients,
not a globally evolving Maxwellian. This is not a Cyclone/tokamak benchmark.
"""
from dataclasses import dataclass
from functools import partial
import numpy as np
from scipy.linalg import eigh
from scipy.special import roots_hermitenorm, roots_laguerre, j0
import jax
import jax.numpy as jnp
from .plans import plan_1d
from .fast_axis import sample_aligned_knots
from .galerkin import _quadrature_rule, _quadrature_trial


@partial(jax.tree_util.register_dataclass,
    data_fields=['x','points','weights','eigenvalues','values','derivative_values','samples','bspf'],meta_fields=[])
@dataclass(frozen=True)
class ITGRadial:
    x: object
    points: object
    weights: object
    eigenvalues: object
    values: object
    derivative_values: object
    samples: object
    bspf: object


def plan_itg_radial(n=65, *, length=12., degree=7, n_basis=16,
                    endpoint_blend=.5, quadrature_order=12):
    """Standard BSPF solves on smooth basis columns, never FastAxis.C @ f.

    All N-2 Dirichlet degrees of freedom are retained. Sine sample columns are
    a well-conditioned change of basis, not an analytic sine Laplacian: every
    column is decomposed by the ordinary BSPF KKT solve and differentiated by
    BSPF at quadrature points. Small dense radial mass/stiffness matrices are
    intentional for N<=129. No 5D phase-space matrix is assembled.
    """
    if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or not 17<=n<=129:
        raise ValueError('radial grid requires 17 <= n <= 129')
    if not np.isfinite(length) or length<=0:
        raise ValueError('length must be positive')
    x=jnp.linspace(0.,length,n)
    bp=plan_1d(x,degree=degree,n_basis=None,
        knots=sample_aligned_knots(x,degree=degree,n_basis=n_basis,endpoint_blend=endpoint_blend),
        boundary_points=degree+2)
    trial=jnp.sin(jnp.pi*x[:,None]/length*jnp.arange(1,n-1)[None,:])
    trial=trial.at[0].set(0).at[-1].set(0)
    weights,(q,g)=_quadrature_trial(bp,trial,(0,1),quadrature_order)
    points,_=_quadrature_rule(bp,quadrature_order)
    mass=np.asarray(q.T@(weights[:,None]*q))
    stiffness=np.asarray(g.T@(weights[:,None]*g))
    lam,u=eigh((stiffness+stiffness.T)/2,(mass+mass.T)/2)
    if not np.all(np.isfinite(lam)) or np.any(lam<=0):
        raise ValueError('Dirichlet radial Laplacian must be positive')
    samples=np.asarray(trial)@u
    orientation=np.sign(np.sum(samples*np.asarray(trial),axis=0))
    u*=np.where(orientation==0,1,orientation)
    return ITGRadial(x,points,weights,jnp.asarray(lam),q@jnp.asarray(u),g@jnp.asarray(u),trial@jnp.asarray(u),bp)


@partial(jax.tree_util.register_dataclass,
    data_fields=['radial','velocity','mu','sqrt_weights','energy','b','polarization',
                 'omega_motion','omega_density','omega_temperature','ky','rho'],meta_fields=[])
@dataclass(frozen=True)
class LinearITG:
    radial: ITGRadial
    velocity: object
    mu: object
    sqrt_weights: object
    energy: object
    b: object
    polarization: object
    omega_motion: object
    omega_density: object
    omega_temperature: object
    ky: object
    rho: object


def plan_linear_itg(radial=None, *, n_x=65,n_v=48,n_mu=32,
                    ky=.3,k_parallel=.1,curvature=.2,a_n=0.,a_t=4.,rho=1.,tau=1.):
    """Set local drive: omega_*^T=rho*ky*(a_n+a_t*(v^2/2+mu-3/2)).

    v has unit Maxwell variance, mu has exp(-mu) measure. Positive curvature
    and a_t define the bad-curvature convention. ky=0 is excluded: zonal
    electron response belongs to the subsequent nonlinear model.
    """
    if not all(np.isfinite(z) for z in [ky,k_parallel,curvature,a_n,a_t,rho,tau]) or ky==0 or rho<0 or tau<=0:
        raise ValueError('finite parameters, nonzero ky, rho>=0 and tau>0 required')
    if any(isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<2 for n in [n_v,n_mu]):
        raise ValueError('velocity quadrature orders must be integers >=2')
    if radial is None: radial=plan_itg_radial(n_x)
    v,wv=roots_hermitenorm(n_v);wv/=np.sqrt(2*np.pi)
    mu,wm=roots_laguerre(n_mu)
    vv,mm=np.meshgrid(v,mu,indexing='ij')
    vv=vv.ravel();mm=mm.ravel();s=np.sqrt((wv[:,None]*wm).ravel())
    energy=vv*vv/2+mm
    gyro=j0(rho*np.sqrt(2*(np.asarray(radial.eigenvalues)[:,None]+ky*ky)*mm[None,:]))
    b=gyro*s[None,:]
    d=tau+1-np.sum(b*b,axis=1)
    if np.any(d<=0):raise ValueError('polarization must be positive')
    return LinearITG(radial,*map(jnp.asarray,[vv,mm,s,energy,b,d,
        k_parallel*vv+rho*ky*curvature*(vv*vv+mm),np.full_like(vv,rho*ky*a_n),
        rho*ky*a_t*(energy-1.5),ky,rho]))


def linear_itg_fields(p,x):
    """Modal phi and sqrt(w)*gyroaverage(phi); x=sqrt(w)*g, shape (N-2,Nv*Nmu)."""
    phi=jnp.sum(p.b*x,axis=1)/p.polarization
    return phi,p.b*phi[:,None]


def linear_itg_rhs(p,x):
    _,psi=linear_itg_fields(p,x)
    return -1j*p.omega_motion*(x+psi)+1j*(p.omega_density+p.omega_temperature)*psi


def linear_itg_diagnostics(p,x):
    """Entropy, field energy, W, density power, temperature power, Gamma, Q.

    Complex-amplitude norm W=1/2*(||x||^2+phi*D*phi). A real-harmonic average
    adds a common factor 1/2. Radial flux convention v_Ex=+i*rho*ky*gyro(phi).
    Q uses (energy-3/2), the temperature-gradient-conjugate heat flux.
    """
    phi,psi=linear_itg_fields(p,x)
    entropy=.5*jnp.sum(jnp.abs(x)**2)
    field=.5*jnp.sum(p.polarization*jnp.abs(phi)**2)
    h=x+psi
    density_power=jnp.real(jnp.vdot(h,1j*p.omega_density*psi))
    temperature_power=jnp.real(jnp.vdot(h,1j*p.omega_temperature*psi))
    flux=jnp.real(jnp.vdot(x,1j*p.rho*p.ky*psi))
    heat=jnp.real(jnp.vdot(x,1j*p.rho*p.ky*(p.energy-1.5)*psi))
    return jnp.stack((entropy,field,entropy+field,density_power,temperature_power,flux,heat))


def linear_itg_initial(p, *, radial_mode=1, amplitude=1e-7, omega=None):
    """A density seed, or a velocity eigenmode seed with supplied omega.

    omega is never used by the RHS. A reference-root seed is for validation;
    generic density seeds demonstrate unforced selection of the growing mode.
    """
    if not 1<=radial_mode<=p.radial.eigenvalues.size:
        raise ValueError('radial_mode out of range')
    j=radial_mode-1
    if omega is None:
        row=amplitude*p.b[j]
    else:
        row=amplitude*p.b[j]*(p.omega_motion-p.omega_density-p.omega_temperature)/(omega-p.omega_motion)
    return jnp.zeros_like(p.b,dtype=jnp.complex128).at[j].set(row)


@partial(jax.jit,static_argnames=['steps','save_every'])
def integrate_linear_itg(p,x,dt,*,steps,save_every=10):
    if steps<1 or save_every<1 or steps%save_every:
        raise ValueError('steps must be positive and divisible by save_every')
    def rhs(u):
        return linear_itg_rhs(p,u),linear_itg_diagnostics(p,u)[3:5]
    def block(state,_):
        def step(_,state):
            u,budget=state
            a,pa=rhs(u);b,pb=rhs(u+dt*a/2);c,pc=rhs(u+dt*b/2);d,pd=rhs(u+dt*c)
            return u+dt*(a+2*b+2*c+d)/6,budget+dt*(pa+2*pb+2*pc+pd)/6
        state=jax.lax.fori_loop(0,save_every,step,state)
        return state,state
    zero=jnp.zeros(2,dtype=jnp.float64)
    _,(hist,budgets)=jax.lax.scan(block,(x,zero),None,length=steps//save_every)
    return jnp.concatenate((x[None],hist)),jnp.arange(steps//save_every+1)*dt*save_every,jnp.concatenate((zero[None],budgets))
