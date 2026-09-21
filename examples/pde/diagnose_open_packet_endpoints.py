"""Isolate endpoint estimation at fixed BSPF knots: scalar differentiation and GK evolution."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp,numpy as np,json
from dataclasses import replace
from pybspf.plans import plan_1d
from pybspf.fast_axis import plan_fast_axis
from pybspf.fast_axis import sample_aligned_knots
from pybspf.fast_axis import axis_values
from pybspf.operators import differentiate
from bspf_models.kinetic.open_slab_packet import plan_open_slab_packet
from bspf_models.kinetic.open_slab_packet import integrate_open_packet
from bspf_models.kinetic.open_slab_packet import packet_initial
from bspf_models.kinetic.open_slab_packet import packet_reference

def main():
    from pathlib import Path
    Path("build/open_slab_packet").mkdir(parents=True,exist_ok=True)
    rows=[]
    for n in [129,257]:
        p=plan_open_slab_packet(n,n_v=192,endpoint_blend=.5)
        for method,points,modes in [('finite_difference',9,None),('chebyshev',16,12),('chebyshev',24,12),('chebyshev',32,12)]:
            z=p.axis.x
            zp=plan_1d(z,degree=7,knots=sample_aligned_knots(z,degree=7,n_basis=16,endpoint_blend=.5),boundary_points=points,endpoint_method=method,chebyshev_modes=modes)
            a=plan_fast_axis(zp,quadrature_order=12); pp=replace(p,axis=a)
            # Elementary, nonperiodic analytic function and exact endpoint jets.
            f=jnp.exp(z/3)+jnp.sin(1.3*z)
            exact=jnp.exp(z/3)/3+1.3*jnp.cos(1.3*z)
            jets=jnp.stack([jnp.stack([jnp.exp(t/3)/3**k+1.3**k*jnp.sin(1.3*t+k*jnp.pi/2) for k in range(6)]) for t in [z[0],z[-1]]])
            err=float(jnp.max(jnp.abs(differentiate(zp,f)-exact)))
            err_exact=float(jnp.max(jnp.abs(differentiate(zp,f,boundary=jets)-exact)))
            h,t,b=integrate_open_packet(pp,packet_initial(pp),.0005,steps=2400,save_every=2400)
            ref=packet_reference(pp,a.points,1.2);ref0=packet_reference(pp,a.points,0.)
            norm=lambda u:jnp.sqrt(jnp.sum(a.weights[:,None,None]*u*u*pp.gamma[None,None,:]))
            r=dict(n=n,method=method,points=points,modes=modes,map_norm=float(np.linalg.norm(np.asarray(a.coefficient_map),2)),simple_derivative_max=err,exact_jets_derivative_max=err_exact,packet_error=float(norm(axis_values(a,h[-1])-ref)/norm(ref0)))
            rows.append(r);print(json.dumps(r),flush=True)
            open('build/open_slab_packet/endpoint_comparison.json','w').write(json.dumps(rows,indent=2))
            jax.clear_caches()

if __name__ == "__main__":
    main()
