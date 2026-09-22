"""Analytic checks for the small exponential-weight rules and factored jets."""
from math import gamma
import numpy as np
from scipy.special import gammainc
from bspf_models.elliptic.immersed_poisson import EllipticHole
from exponential_weight_quadrature import truncated_exponential_gauss
from short_response_space import ResponseModes


def main():
    worst=0.
    for order in (8,12,16):
        for upper in (.01,.1,1.,4.,10.,40.,100.,1e4):
            z,w=truncated_exponential_gauss(upper,order)
            assert np.all((z>0)&(z<upper)) and np.all(w>0)
            for k in range(2*order):
                exact=gamma(k+1)*gammainc(k+1,upper)
                error=abs(w@z**k-exact)/exact
                worst=max(worst,error)
                assert error<1e-11,(order,upper,k,error)
    hole=EllipticHole();lengths=(.01,.025)
    modes=ResponseModes((-1,5,1),hole,lengths,4,6)
    p=np.random.default_rng(492).uniform([-1,-1],[5,1],(200,2))
    p=p[hole.level(p)>1];full=modes.operators(p)
    distances=(min(hole.axes)*(np.sqrt(hole.level(p))-1),1-p[:,1],p[:,1]+1,p[:,0]+1)
    for family,offset,count,d in zip(('hole','top','bottom','inlet'),(0,9,16,23),(9,7,7,7),distances):
        ids=np.concatenate([np.arange(count)+offset+30*k for k in range(2)])
        selected=modes.operators(p,family=family)
        stripped=modes.operators(p,family=family,strip_exponential=True)
        decay=np.exp(-d[:,None]/np.repeat(lengths,count)[None,:])
        for physical,subset,amplitude in zip(full,selected,stripped):
            np.testing.assert_array_equal(physical[:,ids],subset)
            np.testing.assert_allclose(amplitude*decay,subset,rtol=2e-12,atol=1e-12)
    print(f'PASS: positive truncated exponential Gaussian rules; maximum moment relative error {worst:.3e}')
    print('PASS: selected family and analytically factored values, velocities and gradients reproduce original operators')


if __name__=='__main__':main()
