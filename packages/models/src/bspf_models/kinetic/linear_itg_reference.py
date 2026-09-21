"""Independent NumPy/SciPy local ITG dispersion and velocity-matrix checks.

The radial reference is the analytic Dirichlet kx=m*pi/L. This module does
not import the production radial discretization or RHS. Upper-half-plane
roots describe growing modes; discrete real poles are not Landau damping.
"""
import numpy as np
from scipy.special import roots_hermitenorm,roots_laguerre,j0
from scipy.optimize import root


def velocity_data(*,kx=np.pi/12,ky=.3,k_parallel=.1,curvature=.2,
                  a_n=0.,a_t=4.,rho=1.,tau=1.,n_v=80,n_mu=64):
    v,wv=roots_hermitenorm(n_v);wv/=np.sqrt(2*np.pi)
    mu,wm=roots_laguerre(n_mu)
    v,mu=np.meshgrid(v,mu,indexing='ij');w=(wv[:,None]*wm).ravel()
    v=v.ravel();mu=mu.ravel()
    gyro=j0(rho*np.sqrt(2*(kx*kx+ky*ky)*mu))
    motion=k_parallel*v+rho*ky*curvature*(v*v+mu)
    star=rho*ky*(a_n+a_t*(v*v/2+mu-1.5))
    return w,gyro,motion,star,tau


def dispersion(omega,**parameters):
    """h-form quasineutrality: 1+tau = <J0^2 (omega-omega*)/(omega-Omega)>."""
    w,j,motion,star,tau=velocity_data(**parameters)
    return 1+tau-np.sum(w*j*j*(omega-star)/(omega-motion))


def growing_root(*,guess=.32+.16j,**parameters):
    w,j,motion,star,tau=velocity_data(**parameters)
    def residual(z):return 1+tau-np.sum(w*j*j*(z-star)/(z-motion))
    def fun(y):
        r=residual(y[0]+1j*y[1]);return [r.real,r.imag]
    sol=root(fun,[complex(guess).real,complex(guess).imag],tol=1e-11)
    omega=complex(*sol.x)
    if not sol.success or abs(residual(omega))>1e-9 or omega.imag<=1e-7:
        raise ValueError('no resolved growing root from this seed; this does not prove stability')
    return omega


def velocity_generator(**parameters):
    """Small independent dense velocity matrix for tests only, not time stepping."""
    w,j,motion,star,tau=velocity_data(**parameters)
    b=np.sqrt(w)*j;d=1+tau-np.sum(b*b)
    return -1j*(np.diag(motion)+np.outer((motion-star)*b,b)/d)


def continuum_dispersion(omega, *, kx=np.pi/12,ky=.3,k_parallel=.1,
                         curvature=.2,a_n=0.,a_t=4.,rho=1.,tau=1.):
    """Upper-half-plane continuous velocity integral, independent of velocity grids.

    Integrate the Maxwellian parallel velocity analytically using the plasma
    dispersion function; adaptive quadrature handles mu. Positive rho*ky*curvature
    and Im(omega)>0 are required. This is not a damped-mode Landau continuation.
    """
    from scipy.special import wofz
    from scipy.integrate import quad
    c=rho*ky*curvature
    if c<=0 or omega.imag<=0:
        raise ValueError('continuous reference requires rho*ky*curvature>0 and Im(omega)>0')
    quotient=rho*ky*a_t/(2*c)
    def integrand(mu):
        disc=np.sqrt(k_parallel*k_parallel+4*c*(omega-c*mu))
        plus=(-k_parallel+disc)/(2*c)
        minus=(-k_parallel-disc)/(2*c)
        # One pole is above and one below the real-v integration contour.
        upper=1j*np.sqrt(np.pi/2)*wofz(plus/np.sqrt(2))
        lower=-1j*np.sqrt(np.pi/2)*wofz(-minus/np.sqrt(2))
        r0=omega-rho*ky*(a_n+a_t*(mu-1.5))-quotient*(omega-c*mu)
        r1=quotient*k_parallel
        average=quotient-((r0+r1*plus)*upper-(r0+r1*minus)*lower)/disc
        return np.exp(-mu)*j0(rho*np.sqrt(2*(kx*kx+ky*ky)*mu))**2*average
    turning=(omega.real+k_parallel*k_parallel/(4*c))/c
    width=omega.imag/c
    points=sorted(set(t for t in [turning+k*width for k in [-20,-1,0,1,20]] if 0<t<60)) or None
    re=quad(lambda m:integrand(m).real,0,60,points=points,epsabs=2e-11,epsrel=2e-11,limit=250)[0]
    im=quad(lambda m:integrand(m).imag,0,60,points=points,epsabs=2e-11,epsrel=2e-11,limit=250)[0]
    return 1+tau-complex(re,im)


def continuum_growing_root(*,guess=.32+.16j,**parameters):
    """Positive growth parameterization avoids crossing the Landau contour."""
    def fun(y):
        if not -25<y[1]<5:return [1e3+abs(y[1]),1e3+abs(y[0])]
        omega=y[0]+1j*np.exp(y[1])
        r=continuum_dispersion(omega,**parameters)
        return [r.real,r.imag]
    if complex(guess).imag<=0:raise ValueError('guess must have positive growth')
    sol=root(fun,[complex(guess).real,np.log(complex(guess).imag)],tol=1e-9)
    omega=sol.x[0]+1j*np.exp(sol.x[1])
    if not sol.success or omega.imag<1e-7 or np.linalg.norm(fun(sol.x))>1e-8:
        raise ValueError('no continuous growing root resolved; absence is not a stability proof')
    return omega
