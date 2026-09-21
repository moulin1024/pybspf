"""Measure complete transport+projected multiplication axis applications.

Dense matrices are constructed ONLY as timing references. Both paths use the
same quadrature and trial functions; correctness is tested independently
in test_fast_axis.py against direct Fourier sums. Warm JIT calls block before timing ends.
"""

import pybspf.plans as bspf_plans
import argparse, json, time, gc
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from pybspf.fast_axis import plan_fast_axis
from pybspf.fast_axis import axis_values
from pybspf.fast_axis import axis_transport
from pybspf.fast_axis import plan_axis_multiplier
from pybspf.fast_axis import axis_apply_multiplier
from pybspf.fast_axis import boundary_clustered_knots


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes',type=int,nargs='+',default=[64,256,1024,2048])
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--out',type=Path,default=Path('build/fast_mirror/axis_benchmark.json'))
    args=parser.parse_args();reports=[]
    for n in args.sizes:
        t0=time.perf_counter()
        x=jnp.linspace(-4.,4.,n)
        p=bspf_plans.plan_1d(x,degree=7,knots=boundary_clustered_knots(x),boundary_points=9)
        a=plan_fast_axis(p,quadrature_order=12)
        op=plan_axis_multiplier(a,a.points)
        op.symbol.block_until_ready(); setup=time.perf_counter()-t0
        # Verification/timing reference, deliberately explicit and not called
        # from any matrix-free factory or runtime function.
        # Probe in narrow blocks so reference construction does not create
        # FFT work arrays with both a full axis and a full identity batch.
        block=32
        transport_host=np.empty((n,n)); multiplication_host=np.empty((n,n))
        get_t=jax.jit(lambda f:axis_transport(a,f))
        get_m=jax.jit(lambda f:axis_apply_multiplier(a,op,f))
        for start in range(0,n,block):
            width=min(block,n-start)
            probe=np.zeros((n,block))
            probe[np.arange(start,start+width),np.arange(width)]=1
            transport_host[:,start:start+width]=np.asarray(get_t(jnp.asarray(probe)))[:,:width]
            multiplication_host[:,start:start+width]=np.asarray(get_m(jnp.asarray(probe)))[:,:width]
        transport=jnp.asarray(transport_host);multiplication=jnp.asarray(multiplication_host)
        del transport_host,multiplication_host,get_t,get_m,probe
        f=jnp.asarray(np.random.default_rng(0).normal(size=(n,args.batch)))
        fast=jax.jit(lambda a,op,f:axis_apply_multiplier(a,op,axis_transport(a,f)))
        dense=jax.jit(lambda t,m,f:m@(t@f))
        actual=fast(a,op,f);reference=dense(transport,multiplication,f)
        actual.block_until_ready();reference.block_until_ready()
        error=float(jnp.max(jnp.abs(actual-reference))/jnp.max(jnp.abs(reference)))
        def bench(fn):
            samples=[]
            for _ in range(7):
                start=time.perf_counter()
                for _ in range(5): fn().block_until_ready()
                samples.append((time.perf_counter()-start)/5)
            return float(np.median(samples))
        fast_s=bench(lambda:fast(a,op,f));dense_s=bench(lambda:dense(transport,multiplication,f))
        record=dict(n=n,batch=args.batch,core=a.mass_factor.shape[0],setup_seconds=setup,
            fast_seconds=fast_s,dense_seconds=dense_s,speedup=dense_s/fast_s,
            fast_plan_bytes=sum(x.size*x.dtype.itemsize for x in jax.tree_util.tree_leaves((a,op))),
            dense_runtime_bytes=int(transport.nbytes+multiplication.nbytes),relative_difference=error)
        reports.append(record);print(json.dumps(record),flush=True)
        assert error<1e-7
        del a,op,p,transport,multiplication,actual,reference,fast,dense,f
        jax.clear_caches();gc.collect()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':main()
