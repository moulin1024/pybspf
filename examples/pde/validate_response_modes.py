"""Analytic-jet and essential-boundary checks for the research enrichment."""
import numpy as np
from bspf_models.elliptic.immersed_poisson import EllipticHole
from short_response_space import ResponseModes


def check_modes(m):
    hole=m.hole
    theta=np.linspace(0,2*np.pi,117,endpoint=False)
    boundary=np.column_stack((hole.center[0]+hole.axes[0]*np.cos(theta),hole.center[1]+hole.axes[1]*np.sin(theta)))
    bx=np.linspace(-1,5,123);by=np.linspace(-1,1,113)
    boundary=np.vstack((boundary,*(np.column_stack((bx,np.full_like(bx,s))) for s in [-1,1]),
                        *(np.column_stack((np.full_like(by,s),by)) for s in [-1,5])))
    ops=m.operators(boundary)
    assert np.max(np.abs(ops[0]))<1e-20
    assert max(np.max(np.abs(a)) for a in ops[1:3])<1e-12
    # Points resolve each boundary family; finite differences independently
    # check first and second derivatives, including curvature and cross terms.
    pts=np.vstack((np.column_stack((hole.center[0]+(hole.axes[0]+.012)*np.cos(theta),
                                   hole.center[1]+(hole.axes[1]+.012)*np.sin(theta))),
                   [[-.99,.3],[.7,.99],[1.1,-.99],[-.8,.95]]))
    ops=m.operators(pts);h=1e-6
    for axis in [0,1]:
        shift=np.eye(2)[axis]*h
        plus=m.operators(pts+shift);minus=m.operators(pts-shift)
        pairs=[(0,-ops[2]),(1,ops[3]),(2,ops[5])] if axis==0 else [(0,ops[1]),(1,ops[4]),(2,-ops[3])]
        for k,target in pairs:
            derivative=(plus[k]-minus[k])/(2*h)
            scale=np.maximum(np.max(abs(target),axis=0),1e-10)
            err=np.max(abs(derivative-target)/scale)
            assert err<2e-6,(axis,k,err)
    print('PASS: analytic values, gradients, Hessians and double-zero boundary constraints')


def main():
    from scipy.integrate import quad
    from short_response_space import Jet, matched_primitive
    for lengths,pairs in [((.01,.025),()), ((),((.01,.2),(.025,.2))),
                           ((.01,.025),((.01,.2),(.025,.2)))]:
        check_modes(ResponseModes((-1,5,1),EllipticHole(),lengths,4,6,pairs=pairs))
    # Integral displacement in the thin part, compensation in the outer part.
    ell,outer=.01,.2
    R=lambda n,h: matched_primitive(Jet(np.array([n]),1),h)
    for n in [0.,1e-15,1e-8,.003,.02,.1,1.]:
        p=R(n,ell)-R(n,outer)
        expected=quad(lambda q:n*n*np.exp(-n/q)/q**3,ell,outer,epsabs=1e-14)[0]
        np.testing.assert_allclose(p.a[0],expected,rtol=1e-10,atol=1e-14)
    velocity=lambda n: float((R(n,ell)-R(n,outer)).a[1][0])
    np.testing.assert_allclose(quad(velocity,0,np.inf,epsabs=1e-12)[0],0,atol=1e-12)
    assert quad(velocity,0,5*ell)[0]>.9
    assert quad(velocity,5*ell,np.inf)[0]<-.9
    print('PASS: paired modes have thin displacement and broad compensation with zero net flux')


if __name__=='__main__':main()
