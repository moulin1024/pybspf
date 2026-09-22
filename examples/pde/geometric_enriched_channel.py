"""Channel evolution with weighted response enrichment and selectable lift."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import jax
import numpy as np
from bspf_models.fluids.immersed_flow import ImmersedFlowPlan
from weighted_response_space import WeightedResponsePlan,basis_evaluation_pool,refine_rational_background_stiffness
from validate_weighted_response import audit_plan
from immersed_channel_flow import independent_checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--re',type=float,default=200.)
    parser.add_argument('--dt',type=float,default=.01)
    parser.add_argument('--time',type=float,default=1.)
    parser.add_argument('--quadrature-factor',type=float,default=3.)
    parser.add_argument('--stiffness-quadrature-factor',type=float,
                        help='One-off rational background stiffness rule; nonlinear points stay fixed')
    parser.add_argument('--order',type=int,default=16)
    parser.add_argument('--tangent-factor',type=float,default=3.)
    parser.add_argument('--nonlinear-hole-angles',type=int,default=128)
    parser.add_argument('--wall-method',choices=('factor','rational'),default='factor',
                        help='Geometric compatible start, or LARS Stokes lift')
    parser.add_argument('--broad-lengths',type=float,nargs='*',default=[],
                        help='Optional broad scales; default retains only four thin scales')
    args=parser.parse_args()
    if min(args.dt,args.time,args.re)<=0 or not np.all(np.isfinite([args.dt,args.time,args.re])):
        parser.error('Require positive finite time, step and Reynolds number')
    steps=round(args.time/args.dt)
    if abs(steps*args.dt-args.time)>1e-12:parser.error('Time must be an integer number of steps')
    args.out.mkdir(parents=True,exist_ok=False)
    jax.config.update('jax_enable_x64',True)
    base=ImmersedFlowPlan(nx=73,ny=33,reynolds=args.re,wall_method=args.wall_method,
                         buffer_strength=0,quadrature_factor=args.quadrature_factor,basis_workers=4)
    print('BASE',base.setup_seconds,flush=True)
    stiffness_refinement=None
    with basis_evaluation_pool(base):
        if args.stiffness_quadrature_factor is not None:
            stiffness_refinement=refine_rational_background_stiffness(base,args.stiffness_quadrature_factor)
            print('BACKGROUND_STIFFNESS',json.dumps(stiffness_refinement),flush=True)
        plan=WeightedResponsePlan(base,order=args.order,broad_lengths=args.broad_lengths,tangent_factor=args.tangent_factor)
        if args.nonlinear_hole_angles!=128:
            plan.refine_nonlinear_hole_rule(args.nonlinear_hole_angles)
    initial=(plan.compatible_state() if base.factored_wall else np.zeros(plan.dofs))
    if base.factored_wall:
        assert base.rational is None
    assert 'stokes_state' not in base.__dict__ and 'stokes_state' not in plan.__dict__
    np.testing.assert_allclose(plan.linear_lift[:base.dofs],base.linear_lift,rtol=1e-10,atol=1e-12)
    probes=np.array([[-.7,-.3],[-.5,.6],[.8,.2],[1.7,-.4],[4.3,.7]])
    np.testing.assert_allclose(plan.evaluate(initial,probes),base.evaluate(initial[:base.dofs],probes),atol=1e-12,rtol=1e-12)
    report=dict(reynolds=args.re,dt=args.dt,final_time=args.time,nx=base.nx,ny=base.ny,
                wall_method=args.wall_method,initial_state='compatible' if base.factored_wall else 'stokes',
                rational=None if base.rational is None else base.rational.info,
                bounds=list(base.bounds),peak_inlet=base.peak,buffer_strength=0.,basis_precision='mpfr',
                quadrature_factor=args.quadrature_factor,quadrature_points=len(base.weights),
                background_stiffness_refinement=stiffness_refinement,
                setup_seconds=base.setup_seconds+plan.info['setup_seconds'],enrichment=plan.info)
    sources=[__file__,'examples/pde/weighted_response_space.py','examples/pde/weighted_response_assembly.py',
             'examples/pde/short_response_space.py','examples/pde/validate_weighted_response.py',
             'examples/pde/scaled_response_quadrature.py',
             'packages/models/src/bspf_models/fluids/immersed_flow.py']
    report['source_sha256']={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources}
    def save_report():
        (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    save_report()
    with basis_evaluation_pool(base):
        report['audit']=audit_plan(plan)
    report['setup_seconds']=base.setup_seconds+plan.info['setup_seconds']
    save_report()
    if not report['audit']['passed']:raise ArithmeticError('Independent integration audit failed')
    report['initial_checks']=independent_checks(plan,initial)
    x=np.linspace(-1,5,401);y=np.linspace(-1,1,161)
    a=initial.copy();times=[0.];states=[a.copy()];history=[]
    stepper=plan.stepper(args.dt)
    start=perf_counter()
    for k in range(steps):
        try:
            candidate=stepper.step(a,k*args.dt)
            if not np.all(np.isfinite(candidate)):raise FloatingPointError('Non-finite state')
        except (FloatingPointError,ValueError) as error:
            report['failure']=dict(time=(k+1)*args.dt,error=str(error));break
        a=candidate
        if (k+1)%max(1,round(.1/args.dt))==0 or k==steps-1:
            t=(k+1)*args.dt;times.append(t);states.append(a.copy())
            diag=dict(t=t,**plan.diagnostics(a));history.append(diag)
            np.savez_compressed(args.out/'states.npz',t=times,states=states)
            (args.out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
            print('NS',json.dumps(diag),flush=True)
    report['completed_time']=times[-1]
    report['evolution_seconds']=perf_counter()-start
    fields=np.array([[g[key] for key in ('u','v','vorticity','psi')]
                     for g in plan.grid_many(np.asarray(states),x,y)])
    np.savez_compressed(args.out/'fields.npz',x=x,y=y,t=times,fields=fields,
                        states=states,center=base.hole.center,axes=base.hole.axes)
    report['checks']=independent_checks(plan,a)
    report['final']=history[-1] if history else {}
    d4=np.diff(fields[:,2],n=4,axis=1)
    region=d4[:,(y[2:-2]>-.8)&(y[2:-2]<.8)][:,:,(x>-.85)&(x<-.45)]
    report['roughness']=[dict(t=t,rms=float(np.sqrt(np.mean(v*v)))) for t,v in zip(times,region)]
    save_report()
    print('FINAL',json.dumps(report['roughness']),flush=True)


if __name__=='__main__':main()
