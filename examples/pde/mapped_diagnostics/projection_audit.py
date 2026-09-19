"""Test nonlocal curl leakage of the resident divergence-free mass projection."""
import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from fields import evaluate,local_coefficients
from source_audit import metrics,rms
from bspf_jax.mapped_navier_stokes import (MappedNavierStokesPlan,channel_streamfunction,
                                         _convection,_load,_geometry_jets)
from bspf_jax.mapped_poisson import _axis
from bspf_jax.immersed_poisson import EllipticHole


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--boundary',choices=('strong','nitsche'),default='strong')
    ap.add_argument('--out',type=Path,default=Path('build/mapped_ripple/projection'))
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    jax.config.update('jax_enable_x64',True);device=jax.devices('gpu')[0]
    bounds=(-1.,3.,-1.,1.);hole=EllipticHole();lift=channel_streamfunction(bounds,hole)
    p=MappedNavierStokesPlan(elements=(16,16),viscosity=(2/3)*.46/200,dt=.005,lift=lift,
                            boundary=args.boundary,device=device)
    d=p.data
    x=np.linspace(-.85,-.45,97);y=np.linspace(-.35,.30,161)
    xy=np.stack(np.meshgrid(x,y,indexing='ij'),axis=-1)
    probe=jax.device_put(xy.reshape(-1,2),device);shape=xy.shape[:-1]
    def project(rhs):return jl.cho_solve((d['mass_factor'],True),rhs)
    def curl(a):return np.asarray(evaluate(probe,local_coefficients(p,a),p.base.geometry,p.base.knots,
                                           degree=p.degree)['vorticity']).reshape(shape)
    points=d['points']
    # This smooth compact force lies entirely in the top patch, outside the probe.
    bx=jnp.maximum(1-((points[:,0]-.2)/.2)**2,0.)**4
    by=jnp.maximum(1-((points[:,1]-.45)/.1)**2,0.)**4
    force=jnp.stack((jnp.zeros_like(bx),bx*by),axis=-1)
    fnorm=jnp.sqrt(jnp.sum(d['w']*jnp.sum(force*force,axis=-1)))
    force/=fnorm
    rhs=_load(d['v'],force,d['w']);a=project(rhs)
    fields=dict(points=xy,compact_force_projected_curl=curl(a))
    compact=dict(force_l2=float(jnp.sqrt(jnp.sum(d['w']*jnp.sum(force**2,axis=-1)))),
                 force_support=[0.,.4,.35,.55],probe_force_and_curl_exactly_zero=True,
                 projected_curl=metrics(fields['compact_force_projected_curl']),
                 projected_velocity_l2=float(jnp.sqrt(a @ d['mass'] @ a)),
                 mass_relative_residual=float(jnp.linalg.norm(d['mass'] @ a-rhs)/jnp.linalg.norm(rhs)))
    print('COMPACT',json.dumps(compact),flush=True)
    def mask(z):
        q=jnp.sqrt(jnp.sum(((z-jnp.array(hole.center))/jnp.array(hole.axes))**2,axis=-1))
        s=jnp.clip(2-q,0,1)
        return s**3*(10+s*(-15+6*s))
    chi=mask(points)
    r=jax.device_put(_axis(16,3,p.quadrature_order)[1],device)
    ends=jax.device_put(np.array((0.,1.)),device)
    ipoints=_geometry_jets(p.base.geometry,r,ends)[0].reshape(4,len(r),2,2)[:,:,1].reshape(-1,2)
    ichi=mask(ipoints)
    records=[]
    states=[('stokes',p.stokes_state)]
    if args.boundary=='strong':
        saved=np.load('build/mapped_spline_flow_re200_strong/solution.npz')
        states.append(('t20',jax.device_put(saved['coefficients'],device)))
    for name,state in states:
        u=d['v'] @ state+d['lift_v'];g=d['g'] @ state+d['lift_g']
        adv=jnp.einsum('qij,qj->qi',g,u)
        im=d['im'] @ state+d['lift_i'];ip=d['ip'] @ state+d['lift_i']
        un=.5*jnp.sum((im+ip)*d['inormal'],axis=-1)
        total=-_convection(d,state)
        near_bulk=-_load(d['v'],adv,d['w']*chi)
        near_seam=.5*_load(d['im']+d['ip'],im-ip,d['iw']*un*ichi)
        near=near_bulk+near_seam
        rest=total-near
        responses={key:curl(project(value)) for key,value in
                   dict(total=total,near_body=near,near_seam=near_seam,remainder=rest).items()}
        record=dict(state=name,probe_mask_max=float(jnp.max(mask(probe))),
                    metrics={k:metrics(v) for k,v in responses.items()},correlations={})
        for axis in (0,1):
            full=np.diff(responses['total'],n=4,axis=axis).ravel()
            small=np.diff(responses['near_body'],n=4,axis=axis).ravel()
            record['correlations']['d4'+('x' if axis==0 else 'y')]=dict(
                near_to_total_rms=rms(small)/rms(full),correlation=float(np.corrcoef(full,small)[0,1]))
        fields.update({name+'_'+key:value for key,value in responses.items()})
        records.append(record);print('ACTUAL',json.dumps(record),flush=True)
    (args.out/'report.json').write_text(json.dumps(dict(boundary=p.boundary,compact=compact,actual=records),indent=2)+'\n')
    np.savez_compressed(args.out/'fields.npz',**fields)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,key in zip(axes,('compact_force_projected_curl','stokes_near_body','stokes_total')):
        f=fields[key];a=np.max(np.abs(f))
        im=ax.pcolormesh(x,y,f.T,cmap='RdBu_r',vmin=-a,vmax=a,shading='auto')
        ax.set(title=key,xlabel='x',ylabel='y');fig.colorbar(im,ax=ax)
    fig.savefig(args.out/'leakage.png',dpi=140)


if __name__=='__main__':main()
