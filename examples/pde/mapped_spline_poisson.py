"""GPU Poisson convergence on four exact rectangle-minus-ellipse spline patches."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from bspf_jax.mapped_poisson import MappedPoissonPlan


def manufactured(plan):
    """Nonzero boundary data and independently differentiated polynomial forcing."""
    left,right,bottom,top = plan.bounds
    cx,cy = plan.hole.center
    a,b = plan.hole.axes
    scale = (right-left)**2*(top-bottom)**2
    def lift(z): return .3+.2*z[0]-.1*z[1]
    def parts(z):
        x,y = z
        X,Y = (x-left)*(right-x),(y-bottom)*(top-y)
        dx,dy = left+right-2*x,bottom+top-2*y
        q = ((x-cx)/a)**2+((y-cy)/b)**2-1
        qx,qy = 2*(x-cx)/a**2,2*(y-cy)/b**2
        return X,Y,dx,dy,q,qx,qy
    def exact(z):
        X,Y,_,_,q,_,_ = parts(z)
        return X*Y*q/scale+lift(z)
    def gradient(z):
        X,Y,dx,dy,q,qx,qy = parts(z)
        return jnp.array((Y*(dx*q+X*qx)/scale+.2,X*(dy*q+Y*qy)/scale-.1))
    def forcing(z):
        X,Y,dx,dy,q,qx,qy = parts(z)
        return -(Y*(-2*q+2*dx*qx+2*X/a**2)+X*(-2*q+2*dy*qy+2*Y/b**2))/scale
    return exact,gradient,forcing,lift


def independent_errors(plan,solution):
    exact,gradient,_,_ = manufactured(plan)
    q,w = np.polynomial.legendre.leggauss(plan.degree+4)
    def axis(n):return ((np.arange(n)[:,None]+(q+1)/2)/n).ravel(),np.tile(w/(2*n),n)
    r,wr = axis(plan.elements[0]);t,wt = axis(plan.elements[1])
    result = plan.evaluate(solution,r,t)
    points = result['points']
    values = jax.jit(jax.vmap(exact))(points.reshape(-1,2)).reshape(points.shape[:-1])
    gradients = jax.jit(jax.vmap(gradient))(points.reshape(-1,2)).reshape(points.shape)
    weights = result['jacobian_determinant']*jax.device_put(wr[:,None]*wt[None,:],plan.device)
    error = result['value']-values
    ge = result['gradient']-gradients
    boundary = plan.evaluate(solution,np.array((0.,1.)),np.linspace(0,1,193))
    bv = jax.jit(jax.vmap(exact))(boundary['points'].reshape(-1,2)).reshape(boundary['value'].shape)
    seam = plan.evaluate(solution,np.linspace(0,1,131),np.array((0.,1.)))['value']
    stats = dict(l2=jnp.sqrt(jnp.sum(weights*error**2)),
                 h1_seminorm=jnp.sqrt(jnp.sum(weights*jnp.sum(ge*ge,axis=-1))),
                 relative_l2=jnp.sqrt(jnp.sum(weights*error**2)/jnp.sum(weights*values**2)),
                 max_sample_error=jnp.max(jnp.abs(error)),
                 boundary_max_error=jnp.max(jnp.abs(boundary['value']-bv)),
                 seam_max_jump=jnp.max(jnp.abs(seam[:,:,-1]-jnp.roll(seam[:,:,0],-1,axis=0))),
                 independent_area=jnp.sum(weights))
    return {key:float(value) for key,value in jax.device_get(stats).items()}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--elements',type=int,nargs='+',default=[4,8,16,32])
    ap.add_argument('--degree',type=int,default=3)
    ap.add_argument('--out',type=Path,default=Path('build/mapped_spline_poisson'))
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    jax.config.update('jax_enable_x64',True)
    device=jax.devices('gpu')[0]
    print('DEVICE',device,flush=True)
    records=[]
    for n in args.elements:
        p=MappedPoissonPlan(elements=(n,n),degree=args.degree,device=device)
        _,_,forcing,lift=manufactured(p)
        solution=p.solve(forcing,lift=lift)
        warm_seconds=[p.solve(forcing,lift=lift).solve_seconds for _ in range(3)]
        errors=independent_errors(p,solution)
        record=dict(elements_per_direction_per_patch=n,degree=p.degree,dofs=p.dofs,
                    quadrature_points=int(np.prod(p.data['points'].shape[:-1])),
                    setup_seconds=p.setup_seconds,solve_seconds=solution.solve_seconds,
                    warm_solve_seconds=float(np.median(warm_seconds)),
                    iterations=solution.iterations,residual_norm=solution.residual_norm,
                    relative_residual=solution.residual_norm/solution.rhs_norm,
                    min_jacobian=p.min_jacobian,assembly_area=p.area,**errors)
        if records:
            ratio=n/records[-1]['elements_per_direction_per_patch']
            for key in ('l2','h1_seminorm'):
                record[key+'_rate']=float(np.log(records[-1][key]/record[key])/np.log(ratio))
        records.append(record);print(json.dumps(record),flush=True)
    payload=dict(device=str(device),backend='gpu',geometry='four exact ellipse-to-rectangle patches',
                 bounds=p.bounds,center=p.hole.center,axes=p.hole.axes,
                 exact_area=(p.bounds[1]-p.bounds[0])*(p.bounds[3]-p.bounds[2])-np.pi*np.prod(p.hole.axes),
                 cases=records)
    (args.out/'report.json').write_text(json.dumps(payload,indent=2)+'\n')
    result=jax.device_get(p.evaluate(solution,np.linspace(0,1,81),np.linspace(0,1,161)))
    np.savez_compressed(args.out/'solution.npz',**result,coefficients=jax.device_get(solution.coefficients))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    exact,_,_,_=manufactured(p)
    target=np.asarray(jax.vmap(exact)(jax.device_put(result['points'].reshape(-1,2),device))).reshape(result['value'].shape)
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,field,title in zip(axs,(result['value'],np.abs(result['value']-target)),('Mapped spline Poisson solution','Absolute error vs manufactured solution')):
        lo,hi=float(field.min()),float(field.max())
        for k in range(4):
            xy=result['points'][k]
            im=ax.pcolormesh(xy[...,0],xy[...,1],field[k],shading='gouraud',vmin=lo,vmax=hi,cmap='viridis')
            ax.plot(xy[:,0,0],xy[:,0,1],color='white',lw=.6)
        ax.set(title=title,aspect='equal',xlabel='x',ylabel='y');fig.colorbar(im,ax=ax)
    fig.savefig(args.out/'solution.png',dpi=160)


if __name__=='__main__':main()
