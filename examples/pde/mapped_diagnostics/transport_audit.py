"""Measure centered spline transport dispersion and its mapped-space analogue.

The periodic test uses quadratic velocity splines, the uniform 1D counterpart
of the derivative of the production cubic streamfunction. The mapped test
projects the exact derivative of a compactly supported divergence-free mode.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np

from bspf_jax.mapped_navier_stokes import MappedNavierStokesPlan,_spline_jets,_load


def main():
    jax.config.update('jax_enable_x64',True);device=jax.devices('gpu')[0]
    out=Path('build/mapped_ripple/transport');out.mkdir(parents=True,exist_ok=True)
    n,degree=64,2
    gauss,weights=np.polynomial.legendre.leggauss(4)
    x=((np.arange(n)[:,None]+(gauss+1)/2)/n).ravel()
    w=np.tile(weights/(2*n),n)
    knots=np.arange(-degree,n+degree+1,dtype=float)/n
    k,x,w,ids=jax.device_put((knots,x,w,np.arange(n+degree)%n),device)
    b,d,_=_spline_jets(k,x,degree=degree)
    periodic=jax.nn.one_hot(ids,n,dtype=jnp.float64)
    b,d=b @ periodic,d @ periodic
    mass=b.T @ (w[:,None]*b)
    advection=b.T @ (w[:,None]*d)
    diffusion=d.T @ (w[:,None]*d)
    theta=jax.device_put(np.linspace(0,np.pi,2049),device)
    offsets=jnp.arange(n);offsets=jnp.where(offsets<=n//2,offsets,offsets-n)
    phase=jnp.exp(1j*theta[:,None]*offsets[None,:])
    mass_symbol=jnp.real(phase @ mass[0])
    frequency=jnp.imag(phase @ advection[0])/mass_symbol/n
    damping=jnp.real(phase @ diffusion[0])/mass_symbol/n**2
    group=jnp.gradient(frequency,theta)
    rows=[]
    for fraction in (.25,.5,.75,.9,1.):
        i=round(fraction*(len(theta)-1))
        rows.append(dict(fraction_of_nyquist=fraction,dimensionless_frequency=float(frequency[i]),
                         phase_speed_over_U=float(frequency[i]/theta[i]),group_speed_over_U=float(group[i]),
                         dimensionless_diffusion=float(damping[i])))
    theta_h,freq_h,group_h,damping_h=jax.device_get((theta,frequency,group,damping))
    threshold=float(theta_h[np.flatnonzero(group_h<0)[0]]/np.pi)
    print('PERIODIC',json.dumps(dict(modes=rows,group_reversal_fraction_of_nyquist=threshold)),flush=True)
    p=MappedNavierStokesPlan(elements=(16,16),device=device,viscosity=(2/3)*.46/200)
    nr,nt=19,19
    ir=np.arange(nr);jt=np.arange(nt)
    radial_window=np.zeros(nr);radial_window[4:15]=np.hanning(11)
    tangent_window=np.zeros(nt);tangent_window[4:13]=np.hanning(9)
    tests=[]
    for fraction in (.125,.25,.5,.75,.9,1.):
        coeff=np.zeros(p.dofs)
        for i in range(2,nr-2):
            for j in range(1,nt-1):
                index=(i-2)*(4*(nt-1))+3*(nt-1)+j
                coeff[index]=radial_window[i]*np.cos(np.pi*fraction*(i-9))*tangent_window[j]
        state=jax.device_put(coeff,device)/p.scale
        norm=jnp.sqrt(state @ p.data['mass'] @ state)
        state/=norm
        gradient=p.data['g'] @ state
        # Constant unit advection in physical x. This force is divergence-free
        # and vanishes near every boundary and interface, so its continuous
        # Leray projection is itself. No lift or nonlinear aliasing is involved.
        force=gradient[:,:,0]
        rhs=_load(p.data['v'],force,p.data['w'])
        projected=jl.cho_solve((p.data['mass_factor'],True),rhs)
        exact_norm=jnp.sqrt(jnp.sum(p.data['w']*jnp.sum(force*force,axis=-1)))
        actual_norm=jnp.sqrt(projected @ p.data['mass'] @ projected)
        error=(p.data['v'] @ projected)-force
        error_norm=jnp.sqrt(jnp.sum(p.data['w']*jnp.sum(error*error,axis=-1)))
        trace=(p.data['bv'] @ state)
        record=dict(fraction_of_radial_coefficient_nyquist=fraction,physical_force_l2=float(exact_norm),
                    projected_force_l2=float(actual_norm),projection_norm_ratio=float(actual_norm/exact_norm),
                    relative_projection_error=float(error_norm/exact_norm),boundary_velocity_max=float(jnp.max(jnp.abs(trace))))
        tests.append(record);print('MAPPED',json.dumps(record),flush=True)
    data=dict(periodic_velocity_degree=degree,periodic_modes=rows,
              group_reversal_fraction_of_nyquist=threshold,mapped_modes=tests)
    (out/'report.json').write_text(json.dumps(data,indent=2)+'\n')
    np.savez_compressed(out/'dispersion.npz',theta=theta_h,frequency=freq_h,group=group_h,damping=damping_h)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].plot(theta_h/np.pi,freq_h,label='Galerkin spline transport')
    axes[0].plot(theta_h/np.pi,theta_h,'--',label='Exact advection')
    axes[0].set(xlabel='Wavenumber / Nyquist',ylabel='Frequency × h / U');axes[0].legend()
    axes[1].plot(theta_h/np.pi,group_h);axes[1].axhline(0,color='black',lw=.5)
    axes[1].set(xlabel='Wavenumber / Nyquist',ylabel='Group velocity / U')
    fig.savefig(out/'dispersion.png',dpi=150)


if __name__=='__main__':main()
