"""Compare direct BSPF solves and a precomputed coefficient map at identical nodes."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp,json
from bspf_jax.plans import plan_1d
from bspf_jax.fast_axis import plan_fast_axis,sample_aligned_knots
from bspf_jax.operators import differentiate,decompose
from pathlib import Path
Path("build/open_slab_packet").mkdir(parents=True,exist_ok=True)
z=jnp.linspace(-3.,3.,257)
p=plan_1d(z,degree=7,knots=sample_aligned_knots(z,degree=7,n_basis=16,endpoint_blend=.5),boundary_points=9)
a=plan_fast_axis(p,quadrature_order=12)
f=jnp.exp(z/3)+jnp.sin(1.3*z);exact=jnp.exp(z/3)/3+1.3*jnp.cos(1.3*z)
c=a.coefficient_map@f
fast=p.basis[1]@c+jnp.fft.ifft(1j*p.omega*jnp.fft.fft(f-p.basis[0]@c)).real
r=dict(n=257,direct_max=float(jnp.max(jnp.abs(differentiate(p,f)-exact))),precomputed_map_max=float(jnp.max(jnp.abs(fast-exact))),coefficient_difference=float(jnp.max(jnp.abs(c-decompose(p,f).coefficients))))
print(r);open('build/open_slab_packet/same_nodes_derivative.json','w').write(json.dumps(r,indent=2))
