"""Independent quadrature refinement of the new analytic response modes."""
import argparse
import json
import numpy as np
from bspf_models.fluids.immersed_flow import channel_quadrature
from bspf_models.elliptic.immersed_poisson import EllipticHole
from short_response_space import ResponseModes


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family',choices=['local','hybrid','broad'],default='local')
    ap.add_argument('--reynolds',type=float,default=20.)
    ap.add_argument('--factors',type=float,nargs='+',default=[4,6,8])
    args=ap.parse_args()
    if not np.isfinite(args.reynolds) or args.reynolds<=0:
        ap.error('reynolds must be finite and positive')
    if len(args.factors)<2 or not all(np.isfinite(f) and f>=1 for f in args.factors) or any(a>=b for a,b in zip(args.factors,args.factors[1:])):
        ap.error('factors must be an increasing sequence of at least two values >= 1')
    ell=np.sqrt(.01*(2/3)*.46/args.reynolds)
    lengths=ell*np.array([.5,1,2,4]);pairs=()
    if args.family=='hybrid':pairs=((ell,.2),(2*ell,.2))
    if args.family=='broad':lengths=np.r_[lengths,.1,.2]
    h=EllipticHole();m=ResponseModes((-1,5,1),h,lengths,pairs=pairs)
    grams=[]
    for factor in args.factors:
        p,w=channel_quadrature((-1,5,1),h,73,33,factor,(3.,))
        G=None
        for ids in np.array_split(np.arange(len(p)),24):
            ops=m.operators(p[ids])[1:]
            g=sum(v*(o.T@(w[ids,None]*o)) for o,v in zip(ops,(1,1,2,1,1)))
            G=g if G is None else G+g
        grams.append(G)
        print('QUAD',factor,len(p),flush=True)
    s=1/np.sqrt(np.diag(grams[-1]))
    for k,G in zip(args.factors[:-1],grams[:-1]):
        err=(G-grams[-1])*s[:,None]*s
        print(json.dumps(dict(factor=k,max_diagonal_relative=float(np.max(abs(np.diag(err)))),
             scaled_gram_relative_frobenius=float(np.linalg.norm(err)/np.linalg.norm(grams[-1]*s[:,None]*s)))),flush=True)


if __name__=='__main__':main()
