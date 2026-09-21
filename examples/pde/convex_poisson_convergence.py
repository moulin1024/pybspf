"""17/33/65/129 convex-domain convergence using one matrix-free backend."""

import argparse,json,time
from pathlib import Path

import jax
import numpy as np
import scipy.linalg as la

from bspf_models.elliptic.convex_poisson_iterative import IterativeConvexPlan
from bspf_models.elliptic.convex_poisson import trace_transform
from bspf_models.elliptic.embedded_poisson import benchmark_domains
from bspf_models.elliptic.random_wave_mms import RandomWaveMMS
from bspf_models.elliptic.smooth_extension import factors
from bspf_models.elliptic.smooth_extension import evaluate_factors
from embedded_poisson_approximation import interior

jax.config.update("jax_enable_x64",True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes",type=int,nargs="+",default=[17,33,65,129])
    parser.add_argument("--maxiter",type=int,default=6000)
    parser.add_argument("--tolerance",type=float,default=1e-12)
    parser.add_argument("--sample-factor",type=int,default=3)
    parser.add_argument("--bands",type=int,nargs="+",default=[4,12])
    parser.add_argument("--out",type=Path,default=Path("build/convex_poisson_convergence"))
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    domain=benchmark_domains()[0]
    check=interior(domain,257,263,.613)
    for n in args.nodes:
        start=time.monotonic();print("BUILD",n,flush=True)
        plan=IterativeConvexPlan(domain,nodes=n,sample_factor=args.sample_factor)
        print("BUILT",n,time.monotonic()-start,"shape",plan.operator.shape,flush=True)
        # Exact tensor factors on independent Cartesian abscissae are cheap.
        basis=factors(plan.line,check)
        bp,_=plan.arc.sample(2*plan.boundary_count,offset=.371)
        bb=factors(plan.line,bp)
        results=[]
        for band in args.bands:
            m=RandomWaveMMS.create(kmax=band*np.pi)
            print("SOLVE",n,band,flush=True)
            c,info=plan.solve(lambda p:m.evaluate(p)[2],lambda p:m.evaluate(p)[0],
                              tolerance=args.tolerance,maxiter=args.maxiter,progress=True)
            value,grad,lap=evaluate_factors(basis,c);exact,eg,ef=m.evaluate(check)
            edge=evaluate_factors(bb,c)[0]-m.evaluate(bp)[0]
            row=dict(nodes=n,band=band,sample_factor=args.sample_factor,
                     value=float(la.norm(value-exact)/la.norm(exact)),
                     gradient=float(la.norm(grad-eg)/la.norm(eg)),
                     laplacian=float(la.norm(lap+ef)/la.norm(ef)),
                     boundary_h32=float(la.norm(trace_transform(edge,plan.arc.length))),
                     boundary_linf=float(abs(edge).max()),elapsed=time.monotonic()-start,**info)
            results.append(row);print("RESULT",json.dumps(row),flush=True)
            np.savez(args.out/f"n{n}_{band}pi.npz",coefficient=c,points=check,value=value,exact=exact)
            (args.out/f"n{n}.json").write_text(json.dumps(results,indent=2)+"\n")


if __name__=="__main__":main()
