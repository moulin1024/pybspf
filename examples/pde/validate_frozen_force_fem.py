"""Continuous manufactured-flow and pressure-gradient calibration of reference FEM."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from frozen_force_fem import FrozenForceFEM


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--degree',type=int,choices=[3,4],default=3)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    jax.config.update('jax_enable_x64',True)
    def psi(z):
        x,y=z
        F=((x-.19)/.31)**2+((y+.13)/.23)**2-1
        return (x+1)**2*(5-x)**3*(1-y*y)**2*F*F*jnp.exp(-2*((x-.19)**2+(y+.13)**2))/500
    grad=jax.grad(psi);hess=jax.jacfwd(grad);third=jax.jacfwd(hess)
    @jax.jit
    def exact(points):
        g=jax.vmap(grad)(points);h=jax.vmap(hess)(points);t=jax.vmap(third)(points)
        uv=jnp.stack([g[:,1],-g[:,0]])
        du=jnp.stack([h[:,0,1],h[:,1,1],-h[:,0,0],-h[:,0,1]])
        lap=jnp.stack([t[:,1,0,0]+t[:,1,1,1],-t[:,0,0,0]-t[:,0,1,1]])
        return jnp.concatenate([uv,du]),uv-lap
    def force(p):return np.asarray(exact(p)[1])
    x=np.linspace(-.99,4.99,181);y=np.linspace(-.99,.99,101)
    xx,yy=np.meshgrid(x,y);pts=np.column_stack([xx.ravel(),yy.ravel()])
    pts=pts[((pts[:,0]-.19)/.31)**2+((pts[:,1]+.13)/.23)**2>1.0001]
    ref=np.asarray(exact(pts)[0]);records=[]
    for scale in (2.,1.):
        p=FrozenForceFEM(args.out/f'mms_{scale}.msh',scale,force,degree=args.degree,intorder=2*args.degree+2)
        a=p.solve(1.,1.);field=p.evaluate(a,pts)
        record=dict(scale=scale,linear_residual=p.solve_residual,
            velocity_relative_l2=float(np.linalg.norm(field[:2]-ref[:2])/np.linalg.norm(ref[:2])),
            gradient_relative_l2=float(np.linalg.norm(field[2:]-ref[2:])/np.linalg.norm(ref[2:])),
            vorticity_relative_l2=float(np.linalg.norm((field[4]-field[3])-(ref[4]-ref[3]))/np.linalg.norm(ref[4]-ref[3])))
        # Physical x is exactly represented by the P3 basis on quadratic geometry.
        v=p.basis.interpolate(p.basis.doflocs[0])
        record['coordinate_reproduction_max']=float(np.max(abs(v-p.basis.global_coordinates()[0])))
        records.append(record);print(json.dumps(record),flush=True)
    (args.out/'mms.json').write_text(json.dumps(records,indent=2)+'\n')
    assert records[-1]['gradient_relative_l2']<records[0]['gradient_relative_l2']/3
    assert records[-1]['coordinate_reproduction_max']<1e-10


if __name__=='__main__':main()
