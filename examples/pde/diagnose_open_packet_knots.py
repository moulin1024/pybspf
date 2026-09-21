"""Knot grading: coefficient amplification, mass-core conditioning and interpolation.

No time integration; the output directory is created automatically.
"""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp,numpy as np,json
from dataclasses import replace
from bspf_models.kinetic.open_slab_packet import *
from pybspf.fast_axis import axis_values
from pybspf.fast_axis import plan_fast_axis
from pybspf.fast_axis import sample_aligned_knots
from pybspf.plans import plan_1d

def main():
    from pathlib import Path
    Path("build/open_slab_packet").mkdir(parents=True,exist_ok=True)
    rows=[]
    for n in [129,193,257]:
        p=plan_open_slab_packet(n,n_v=192)
        for alpha in [0.,.25,.5,.75,1.]:
            s=np.linspace(0,1,10);u=(1-alpha)*s+alpha*(1-np.cos(np.pi*s))/2
            idx=np.rint((n-1)*u).astype(int);z=np.asarray(p.axis.x)
            knots=sample_aligned_knots(z,degree=7,n_basis=16,endpoint_blend=alpha)
            a=plan_fast_axis(plan_1d(z,degree=7,knots=jnp.asarray(knots),boundary_points=9),quadrature_order=12)
            pp=replace(p,axis=a)
            ref=packet_reference(pp,a.points,1.2);interp=axis_values(a,packet_reference(pp,a.x,1.2));ref0=packet_reference(pp,a.points,0.)
            norm=lambda v: jnp.sqrt(jnp.sum(a.weights[:,None,None]*v*v*pp.gamma[None,None,:]))
            r=dict(n=n,alpha=alpha,indices=idx.tolist(),map_norm=float(np.linalg.norm(np.asarray(a.coefficient_map),2)),mass_core_condition=float(np.linalg.cond(np.asarray(a.mass_factor))**2),constant_error=float(jnp.max(jnp.abs(axis_values(a,jnp.ones(n))-1))),interpolation=float(norm(interp-ref)/norm(ref0)))
            rows.append(r);print(json.dumps(r),flush=True)
            jax.clear_caches()
    open('build/open_slab_packet/knot_probe.json','w').write(json.dumps(rows,indent=2))


if __name__ == "__main__":
    main()
