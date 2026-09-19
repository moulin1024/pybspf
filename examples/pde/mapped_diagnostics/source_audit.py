"""Locate mapped Re=200 ripples in raw fields and frozen nonlinear projection."""
import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fields import evaluate, local_coefficients
from bspf_jax.mapped_navier_stokes import (
    MappedNavierStokesPlan, channel_streamfunction, _convection, _load,
)
from bspf_jax.immersed_poisson import EllipticHole


def rms(a): return float(np.sqrt(np.mean(np.asarray(a)**2)))


def metrics(w):
    return dict(rms=rms(w), max=float(np.max(np.abs(w))),
                d2x=rms(np.diff(w, n=2, axis=0)), d2y=rms(np.diff(w, n=2, axis=1)),
                d4x=rms(np.diff(w, n=4, axis=0)), d4y=rms(np.diff(w, n=4, axis=1)))


@jax.jit
def convection_parts(d, state):
    u = d['v'] @ state+d['lift_v']
    g = d['g'] @ state+d['lift_g']
    adv = jnp.einsum('qij,qj->qi', g, u)
    volume = .5*(_load(d['v'], adv, d['w'])-_load(d['g'], u[:, :, None]*u[:, None, :], d['w']))
    im, ip = d['im'] @ state+d['lift_i'], d['ip'] @ state+d['lift_i']
    normal = .5*jnp.sum((im+ip)*d['inormal'], axis=-1)
    interface = .5*(_load(d['im'], ip, d['iw']*normal)-_load(d['ip'], im, d['iw']*normal))
    b = d['bv'] @ state+d['lift_b']
    boundary = .5*_load(d['bv'], b, d['bw']*jnp.sum(b*d['bnormal'], axis=-1))
    strong_bulk = _load(d['v'], adv, d['w'])
    strong_seam = -.5*_load(d['im']+d['ip'], im-ip, d['iw']*normal)
    return dict(skew_bulk=volume, skew_seam=interface, boundary=boundary,
                strong_bulk=strong_bulk, strong_seam=strong_seam)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--elements',nargs=2,type=int,default=(16,16))
    ap.add_argument('--degree',type=int,default=3)
    ap.add_argument('--quadrature',type=int)
    ap.add_argument('--out',type=Path,default=Path('build/mapped_ripple/baseline'))
    ap.add_argument('--history',type=Path)
    ap.add_argument('--evolve',type=float,default=0.)
    ap.add_argument('--lift',choices=('quintic','c4'),default='quintic')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    jax.config.update('jax_enable_x64',True)
    device=jax.devices('gpu')[0]
    bounds=(-1.,3.,-1.,1.);hole=EllipticHole();nu=(2/3)*.46/200
    lift=channel_streamfunction(bounds,hole)
    if args.lift == 'c4':
        cx,cy=hole.center;a,b=hole.axes
        radius=.9*min((cx+1)/a,(3-cx)/a,(cy+1)/b,(1-cy)/b)
        constant=cy-cy**3/3
        def lift(z):
            q=jnp.sqrt(((z[0]-cx)/a)**2+((z[1]-cy)/b)**2)
            s=jnp.clip((q-1)/(radius-1),0.,1.)
            cutoff=s**5*(126+s*(-420+s*(540+s*(-315+70*s))))
            return constant+cutoff*(z[1]-z[1]**3/3-constant)
    p=MappedNavierStokesPlan(elements=tuple(args.elements),degree=args.degree,viscosity=nu,dt=.005,
                            quadrature_order=args.quadrature,lift=lift,device=device)
    print('SETUP',p.dofs,p.setup_seconds,flush=True)
    x=np.linspace(-.85,-.45,97);y=np.linspace(-.35,.30,161)
    xy=np.stack(np.meshgrid(x,y,indexing='ij'),axis=-1)
    points=jax.device_put(xy.reshape(-1,2),device)
    shape=xy.shape[:-1]
    state=p.stokes_state
    # Cross-check the independent physical-coordinate evaluator at patch interior points.
    check=p.evaluate(state,np.array((.19,.42,.73)),np.array((.17,.43,.78)))
    independent=evaluate(check['points'].reshape(-1,2),local_coefficients(p,state),p.base.geometry,p.base.knots,
                         degree=p.degree,lift=lift)
    parity={k:float(jnp.max(jnp.abs(independent[k].reshape(check[k].shape)-check[k])))
            for k in ('velocity','gradient','vorticity')}
    print('PARITY',parity,flush=True)
    parts=convection_parts(p.data,state)
    full=_convection(p.data,state)
    linear=nu*(p.data['lift_load']-p.data['stiffness'] @ state)
    acceleration=jl.cho_solve((p.data['mass_factor'],True),-full)
    scalars=dict(linear_relative_to_convection=float(jnp.linalg.norm(linear)/jnp.linalg.norm(full)),
                 mass_relative_residual=float(jnp.linalg.norm(p.data['mass'] @ acceleration+full)/jnp.linalg.norm(full)),
                 weak_vs_strong_relative=float(jnp.linalg.norm(full-parts['strong_bulk']-parts['strong_seam'])/jnp.linalg.norm(full)),
                 parity=parity)
    initial=evaluate(points,local_coefficients(p,state),p.base.geometry,p.base.knots,degree=p.degree,lift=lift,order=3)
    fields={k:np.asarray(v).reshape(*shape,*v.shape[1:]) for k,v in initial.items()}
    fields['physical_convective_curl']=-np.sum(fields['velocity']*fields['vorticity_gradient'],axis=-1)
    fields['projected_convective_curl']=np.asarray(evaluate(points,local_coefficients(p,acceleration),p.base.geometry,
            p.base.knots,degree=p.degree)['vorticity']).reshape(shape)
    fields['projection_curl_error']=fields['projected_convective_curl']-fields['physical_convective_curl']
    zero=jnp.zeros_like(local_coefficients(p,state))
    fields['lift_vorticity']=np.asarray(evaluate(points,zero,p.base.geometry,p.base.knots,degree=p.degree,lift=lift)['vorticity']).reshape(shape)
    for key in ('strong_bulk','strong_seam'):
        derivative=jl.cho_solve((p.data['mass_factor'],True),-parts[key])
        fields['projected_'+key]=np.asarray(evaluate(points,local_coefficients(p,derivative),p.base.geometry,p.base.knots,
                                                    degree=p.degree)['vorticity']).reshape(shape)
    first=p.advance(state,steps=1)
    fields['first_step_vorticity']=np.asarray(evaluate(points,local_coefficients(p,first),p.base.geometry,p.base.knots,
                                                    degree=p.degree,lift=lift)['vorticity']).reshape(shape)
    line_r=np.linspace(.35,.95,2049)
    line_d=np.array((-1.,.3))-np.array(hole.center)
    line_rho=1/np.sqrt(np.sum((line_d/np.array(hole.axes))**2))
    line_points=np.array(hole.center)+(line_rho+line_r[:,None]*(1-line_rho))*line_d
    line_gpu=jax.device_put(line_points,device)
    def line_values(a):
        return np.asarray(evaluate(line_gpu,local_coefficients(p,a),p.base.geometry,p.base.knots,
                                   degree=p.degree,lift=lift)['vorticity'])
    fields['line_r']=line_r;fields['line_points']=line_points;fields['line_omega_initial']=line_values(state)
    if args.evolve > 0:
        count=round(args.evolve/p.dt)
        evolved=p.advance(state,steps=count)
        evolved_field=evaluate(points,local_coefficients(p,evolved),p.base.geometry,p.base.knots,degree=p.degree,lift=lift)
        fields[f'omega_t{args.evolve:g}']=np.asarray(evolved_field['vorticity']).reshape(shape)
        fields['line_omega_evolved']=line_values(evolved)
        scalars['evolved_time']=count*p.dt
        scalars['evolved_diagnostics']={k:np.asarray(v).tolist() for k,v in jax.device_get(p.diagnostics(evolved)).items()}
    stats={k:metrics(v) for k,v in fields.items() if v.ndim==2 and v.shape==shape}
    history=[]
    if args.history:
        saved=np.load(args.history)
        for time in (0.,.05,.1,.25,.5,1.,2.,5.,10.,20.):
            idx=int(np.argmin(np.abs(saved['times']-time)))
            a=jax.device_put(saved['coefficients'][idx],device)
            v=evaluate(points,local_coefficients(p,a),p.base.geometry,p.base.knots,degree=p.degree,lift=lift)
            omega=np.asarray(v['vorticity']).reshape(shape)
            velocity=np.asarray(v['velocity']).reshape(*shape,2)
            fields[f'omega_t{time:g}']=omega
            if time in (2.,20.):fields[f'line_omega_t{time:g}']=line_values(a)
            history.append(dict(time=float(saved['times'][idx]),vorticity=metrics(omega),
                                u=metrics(velocity[...,0]),v=metrics(velocity[...,1])))
    payload=dict(elements=p.elements,degree=p.degree,quadrature=p.quadrature_order,dofs=p.dofs,lift=args.lift,
                 probe_bounds=[float(x[0]),float(x[-1]),float(y[0]),float(y[-1])],scalars=scalars,metrics=stats,history=history)
    (args.out/'report.json').write_text(json.dumps(payload,indent=2)+'\n')
    np.savez_compressed(args.out/'fields.npz',points=xy,**fields)
    np.savez_compressed(args.out/'stokes.npz',coefficients=np.asarray(state),scale=np.asarray(p.scale),
                        local=np.asarray(local_coefficients(p,state)))
    print('RESULT',json.dumps(payload),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    names=['vorticity','lift_vorticity','projected_convective_curl','physical_convective_curl','projection_curl_error',
           'omega_t20' if args.history else (f'omega_t{args.evolve:g}' if args.evolve else 'first_step_vorticity')]
    for ax,key in zip(axes.flat,names):
        f=fields[key];limit=np.max(np.abs(f))
        im=ax.pcolormesh(x,y,f.T,shading='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.set(title=key,xlabel='x',ylabel='y');fig.colorbar(im,ax=ax)
    fig.savefig(args.out/'source.png',dpi=140)


if __name__=='__main__':main()
