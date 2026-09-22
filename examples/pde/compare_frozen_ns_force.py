"""Frozen initial convection: independent FEM versus actual and best BSPF responses.

Solve alpha*v - mu*Delta(v) + grad(p) = f0, div(v)=0, homogeneous velocity
on inlet/walls/hole and zero Laplacian traction at outlet. No time evolution.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter
import jax
import numpy as np
import scipy.linalg as la
from bspf_models.fluids.immersed_flow import ImmersedFlowPlan, channel_lift
from frozen_force_fem import FrozenForceFEM


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--scales',type=float,nargs='+',default=[1.,.7,.5])
    ap.add_argument('--degree',type=int,choices=[3,4],default=3)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    jax.config.update('jax_enable_x64',True)
    p=ImmersedFlowPlan(nx=73,ny=33,reynolds=20,wall_method='rational',
                      buffer_strength=0,quadrature_factor=4,basis_workers=4)
    print('PLAN',p.setup_seconds,flush=True)
    frozen=p.rational_lift.copy();frozen.flags.writeable=False
    def force(points):
        fields=[a+b for a,b in zip(channel_lift(points,p.bounds[2],p.peak),
                                  p.rational.evaluate(points,frozen))]
        _,u,v,ux,uy,vx=fields
        return -np.array([u*ux+v*uy,u*vx-v*ux])
    load=p.force_load(force(p.points).T)
    np.savez_compressed(args.out/'frozen_force.npz',coefficients=frozen,
        points=p.points,force=force(p.points),bounds=p.bounds,center=p.hole.center,axes=p.hole.axes)
    report=dict(parameters=dict(nx=73,ny=33,reynolds=20,quadrature_factor=4,
                    buffer_strength=0,wall_method='rational',basis_precision='mpfr'),
                frozen_force_coefficient_sha256=hashlib.sha256(frozen.tobytes()).hexdigest(),
                force_relative_difference_from_initial_ns_rhs=float(la.norm(load-p.explicit(p.stokes_state))/la.norm(load)),
                reference_levels=[],comparisons=[])
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    x,y=np.linspace(-1,5,401),np.linspace(-1,1,161);xx,yy=np.meshgrid(x,y)
    pts=np.column_stack((xx.ravel(),yy.ravel()))
    # Compare strictly interior samples; curved geometry is independently checked
    # by refinement. Volume-quadrature norms include the near-wall region.
    mask=(p.hole.level(pts)>1+1e-10)&(pts[:,0]>-1)&(pts[:,0]<5)&(abs(pts[:,1])<1)
    cases=[('stokes',0.,p.nu),('helmholtz',1.,.01*p.nu)]
    previous={};latest={}
    def norm(fields,alpha=1.,mu=1.):
        return float(np.sqrt(p.weights@(alpha*np.sum(fields[:2]**2,axis=0)+mu*np.sum(fields[2:]**2,axis=0))))
    def omega(fields):return fields[4]-fields[3]
    for scale in args.scales:
        start=perf_counter();tag=str(scale).replace('.','p')
        fem=FrozenForceFEM(args.out/f'reference_{tag}.msh',scale,force,
                           intorder=2*args.degree+2,degree=args.degree)
        print('FEM_MESH',json.dumps(fem.info),flush=True)
        row=dict(**fem.info,cases={})
        for name,alpha,mu in cases:
            state=fem.solve(alpha,mu)
            field=fem.evaluate(state,p.points)
            grid=np.full((6,len(pts)),np.nan);grid[:,mask]=fem.evaluate(state,pts[mask])
            rec=dict(linear_relative_residual=fem.solve_residual,
                divergence_l2=float(np.sqrt(p.weights@((field[2]+field[5])**2))),
                velocity_l2=norm(field,1,0),velocity_h1=norm(field),
                vorticity_l2=float(np.sqrt(p.weights@(omega(field)**2))))
            if name in previous:
                old=previous[name]
                rec['relative_change_from_previous']=dict(
                    velocity_l2=norm(field-old,1,0)/norm(field,1,0),
                    velocity_h1=norm(field-old)/norm(field),
                    vorticity_l2=float(np.sqrt(p.weights@((omega(field)-omega(old))**2))/np.sqrt(p.weights@(omega(field)**2))))
            np.savez_compressed(args.out/f'{name}_reference_{tag}.npz',state=state,
                                volume_fields=field,grid_fields=grid.reshape(6,len(y),len(x)),x=x,y=y)
            previous[name]=field;latest[name]=(field,grid.reshape(6,len(y),len(x)))
            row['cases'][name]=rec;print('FEM_CASE',name,json.dumps(rec),flush=True)
        row['seconds']=perf_counter()-start;report['reference_levels'].append(row)
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        del fem,state;gc.collect()
    for name,alpha,mu in cases:
        target,refgrid=latest[name]
        A=alpha*p.mass+mu*p.stiffness
        actual=la.solve(A,load,assume_a='pos')
        bu,bv,bux,buy,bvx=p.operators_fluid
        def projection_rhs(av,mv):
            return (av*(bu.T@(p.weights*target[0])+bv.T@(p.weights*target[1]))
                    +mv*(bux.T@(p.weights*(target[2]-target[5]))
                         +buy.T@(p.weights*target[3])+bvx.T@(p.weights*target[4])))
        best_h1=la.solve(p.mass+p.stiffness,projection_rhs(1,1),assume_a='pos')
        best_energy=la.solve(A,projection_rhs(alpha,mu),assume_a='pos')
        best_l2=la.cho_solve(p.mass_factor,projection_rhs(1,0))
        labels=['actual','best_h1','best_energy','best_l2']
        states=[actual,best_h1,best_energy,best_l2]
        grids=list(p.grid_many(np.array([np.zeros(p.dofs)]+states),x,y))
        base=grids[0];record=dict(case=name,alpha=alpha,mu=mu,metrics={})
        saved=dict(x=x,y=y,reference=refgrid)
        for label,a,g in zip(labels,states,grids[1:]):
            u,v,ux,uy,vx=[o@a for o in p.operators_fluid]
            field=np.array([u,v,ux,uy,vx,-ux]);error=field-target
            metric=dict(velocity_relative_l2=norm(error,1,0)/norm(target,1,0),
                velocity_relative_h1=norm(error)/norm(target),
                energy_relative_error=norm(error,alpha,mu)/norm(target,alpha,mu),
                vorticity_relative_l2=float(np.sqrt(p.weights@(omega(error)**2))/np.sqrt(p.weights@(omega(target)**2))))
            record['metrics'][label]=metric;saved[label+'_state']=a
            for key in ['u','v','vorticity']:
                saved[label+'_'+key]=np.where(mask.reshape(len(y),len(x)),g[key]-base[key],np.nan)
        record['actual_best_energy_relative_state_gap']=float(la.norm(actual-best_energy)/la.norm(actual))
        report['comparisons'].append(record)
        np.savez_compressed(args.out/f'{name}_comparison.npz',**saved)
        print('COMPARISON',json.dumps(record),flush=True)
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
