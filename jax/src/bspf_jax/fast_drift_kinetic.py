"""Matrix-free axis implementation of the static 1z2v mirror model."""
from dataclasses import dataclass
from functools import partial
from numbers import Integral
import jax
import jax.numpy as jnp
import numpy as np
from .fast_axis import (plan_fast_axis, axis_values, axis_adjoint, axis_project,
                        axis_transport, axis_multiply, axis_lift, plan_axis_multiplier, axis_apply_multiplier)


@partial(jax.tree_util.register_dataclass,
         data_fields=['z_axis','v_axis','mu','mu_weights','field','field_ends',
                      'acceleration','velocity_operator','force_operator','number_z','number_v','parallel_v','magnetic_z'],
         meta_fields=['mass'])
@dataclass(frozen=True)
class FastDriftKineticPlan:
    z_axis: object
    v_axis: object
    mu: jax.Array
    mu_weights: jax.Array
    field: jax.Array
    field_ends: jax.Array
    acceleration: jax.Array
    velocity_operator: object
    force_operator: object
    number_z: jax.Array
    number_v: jax.Array
    parallel_v: jax.Array
    magnetic_z: jax.Array
    mass: float

    @property
    def transport(self): return self
    @property
    def z(self): return self.z_axis.x
    @property
    def v(self): return self.v_axis.x
    @property
    def z_points(self): return self.z_axis.points
    @property
    def v_points(self): return self.v_axis.points
    @property
    def z_weights(self): return self.z_axis.weights
    @property
    def v_weights(self): return self.v_axis.weights


def plan_fast_drift_kinetic(z_plan,v_plan,*,magnetic_field,magnetic_gradient,
                            mu_max,n_mu=8,mass=1.,quadrature_order=8):
    for name,value in [('mass',mass),('mu_max',mu_max)]:
        a=np.asarray(value)
        if a.ndim or np.iscomplexobj(a) or not np.isfinite(a) or a<=0:
            raise ValueError(f'{name} must be a finite positive scalar')
    if isinstance(n_mu,bool) or not isinstance(n_mu,Integral) or n_mu<1:
        raise ValueError('n_mu must be a positive integer')
    z=plan_fast_axis(z_plan,quadrature_order=quadrature_order)
    v=plan_fast_axis(v_plan,quadrature_order=quadrature_order)
    def evaluate(fn,points,name,positive=False):
        a=np.asarray(fn(points))
        if np.iscomplexobj(a) or not np.isfinite(a).all() or (positive and np.any(a<=0)):
            raise ValueError(f'{name} must be real, finite'+(' and positive' if positive else ''))
        return jnp.broadcast_to(jnp.asarray(a,dtype=z.x.dtype),points.shape)
    field=evaluate(magnetic_field,z.points,'magnetic_field',True)
    bn=evaluate(magnetic_field,z.x,'magnetic_field',True)
    acceleration=-evaluate(magnetic_gradient,z.points,'magnetic_gradient')/mass
    nodes,weights=np.polynomial.legendre.leggauss(n_mu)
    mu=jnp.asarray((nodes+1)*float(mu_max)/2)
    wm=jnp.asarray(weights*float(mu_max)/2)
    return FastDriftKineticPlan(z,v,mu,wm,field,bn[jnp.array([0,-1])],acceleration,
        plan_axis_multiplier(v,v.points),plan_axis_multiplier(z,acceleration),
        axis_adjoint(z,z.weights),axis_adjoint(v,v.weights),
        axis_adjoint(v,v.weights*(mass*v.points**2/2)),axis_adjoint(z,z.weights*field),float(mass))


def fast_boundary_fluxes(p,t,f,inflow,*,logarithmic=False):
    mu=p.mu[None,:]
    value=jnp.exp if logarithmic else lambda a:a
    def data(z,v,shape):
        a=jnp.asarray(inflow(t,z,v,mu))
        if jnp.iscomplexobj(a): raise ValueError('inflow must be real')
        return value(jnp.broadcast_to(a,shape))
    v=p.v_points[:,None]
    zl=jnp.maximum(v,0)*data(p.z[0],v,(v.size,mu.size))+jnp.minimum(v,0)*value(axis_values(p.v_axis,f[0]))
    zr=jnp.minimum(v,0)*data(p.z[-1],v,(v.size,mu.size))+jnp.maximum(v,0)*value(axis_values(p.v_axis,f[-1]))
    a=p.acceleration[:,None]*mu
    z=p.z_points[:,None]
    vl=jnp.maximum(a,0)*data(z,p.v[0],a.shape)+jnp.minimum(a,0)*value(axis_values(p.z_axis,f[:,0]))
    vr=jnp.minimum(a,0)*data(z,p.v[-1],a.shape)+jnp.maximum(a,0)*value(axis_values(p.z_axis,f[:,-1]))
    return zl,zr,vl,vr


def fast_drift_rhs(p,t,f,*,inflow):
    from .drift_kinetic import _boundary_rates
    velocity=jnp.moveaxis(axis_apply_multiplier(p.v_axis,p.velocity_operator,jnp.moveaxis(f,1,0)),0,1)
    result=axis_transport(p.z_axis,velocity)
    dv=jnp.moveaxis(axis_transport(p.v_axis,jnp.moveaxis(f,1,0)),0,1)
    result+=axis_apply_multiplier(p.z_axis,p.force_operator,dv)*p.mu[None,None,:]
    flux=fast_boundary_fluxes(p,t,f,inflow)
    zl,zr,vl,vr=flux
    result+=axis_lift(p.z_axis,axis_project(p.v_axis,zl),axis_project(p.v_axis,zr))
    result+=jnp.moveaxis(axis_lift(p.v_axis,axis_project(p.z_axis,vl),axis_project(p.z_axis,vr)),0,1)
    return result,_boundary_rates(p,flux)


def fast_moments(p,f):
    n=jnp.einsum('...ijm,i,j,m->...',f,p.number_z,p.number_v,p.mu_weights)
    kp=jnp.einsum('...ijm,i,j,m->...',f,p.number_z,p.parallel_v,p.mu_weights)
    km=jnp.einsum('...ijm,i,j,m->...',f,p.magnetic_z,p.number_v,p.mu_weights*p.mu)
    return jnp.stack((n,kp,km,kp+km),axis=-1)


def fast_log_diagnostics(p,g):
    shape=(p.z.size,p.v.size,p.mu.size)
    if g.shape[-3:]!=shape or jnp.iscomplexobj(g):
        raise ValueError(f'log_distribution must be real with trailing shape {shape}')
    leading=g.shape[:-3]
    weights=p.z_weights[:,None,None]*p.v_weights[None,:,None]*p.mu_weights[None,None,:]
    kp=p.mass*p.v_points[None,:,None]**2/2
    km=p.field[:,None,None]*p.mu[None,None,:]
    def one(state):
        z_values=axis_values(p.z_axis,state)
        values=jnp.exp(jnp.moveaxis(axis_values(p.v_axis,jnp.moveaxis(z_values,1,0)),0,1))
        wf=weights*values
        n,parallel,perpendicular=jnp.sum(wf),jnp.sum(wf*kp),jnp.sum(wf*km)
        return jnp.stack((n,parallel,perpendicular,parallel+perpendicular)),jnp.exp(jnp.min(state)),jnp.min(values)
    m,n,q=jax.lax.map(one,g.reshape((-1,)+shape))
    return m.reshape(leading+(4,)),n.reshape(leading),q.reshape(leading)
