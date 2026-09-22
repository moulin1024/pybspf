"""Compare geometric enrichment against frozen FEM, then evolve unfiltered NS."""
import argparse
import gc
import json
from functools import partial
from pathlib import Path
from time import perf_counter
import jax
import numpy as np
import scipy.linalg as la
from bspf_models.fluids.immersed_flow import ImmersedFlowPlan
from short_response_space import ResponseModes,EnrichedFlowPlan


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--reference',type=Path,default=Path('build/immersed_flow/frozen_force_p4_20260922'))
    ap.add_argument('--quadrature-factor',type=float,default=4.)
    ap.add_argument('--scaled-quadrature',action='store_true',help='Use fixed-budget thickness-scaled ray coordinates')
    ap.add_argument('--weighted-quadrature',action='store_true',help='Separate bulk/background and exponential-weighted thin blocks and nonlinear loads')
    ap.add_argument('--weighted-order',type=int,default=12)
    ap.add_argument('--weighted-angles',type=int,default=128)
    ap.add_argument('--audit-weighted',action='store_true',help='Compare bilinear forms and nonlinear actions with an independent positive ray rule before evolving')
    ap.add_argument('--reynolds',type=float,default=20.)
    ap.add_argument('--skip-reference',action='store_true',help='Run NS without the Re=20 frozen FEM comparison')
    ap.add_argument('--family',choices=['local','paired','hybrid','broad'],default='local')
    ap.add_argument('--outer-length',type=float,default=.2)
    ap.add_argument('--dt',type=float,default=.02)
    ap.add_argument('--check-dt-time',type=float,default=0.,help='Compare dt and dt/2 from the initial state up to this time in the same space')
    ap.add_argument('--time',type=float,default=1.)
    ap.add_argument('--frames',type=int,help='Save exactly this many equally spaced states, excluding t=0 and including final time')
    ap.add_argument('--levels',type=int,nargs='+',choices=[2,4],default=[2,4])
    ap.add_argument('--angular-order',type=int,default=16)
    ap.add_argument('--wall-order',type=int,default=24)
    args=ap.parse_args()
    if args.weighted_quadrature and (args.scaled_quadrature or args.family!='broad' or args.levels!=[4] or not args.skip_reference):
        ap.error('weighted quadrature requires --family broad --levels 4 --skip-reference and excludes --scaled-quadrature')
    if args.audit_weighted and not args.weighted_quadrature:
        ap.error('--audit-weighted requires --weighted-quadrature')
    if not np.isfinite(args.dt) or args.dt<=0 or not np.isfinite(args.time) or args.time<=0:
        ap.error('dt and time must be finite and positive')
    if abs(round(args.time/args.dt)*args.dt-args.time)>1e-12:
        ap.error('time must be a multiple of dt')
    steps=round(args.time/args.dt)
    if args.frames is not None and (args.frames<1 or args.frames>steps or steps%args.frames):
        ap.error('frames must be a positive divisor of the time-step count')
    if not np.isfinite(args.reynolds) or args.reynolds<=0:
        ap.error('reynolds must be finite and positive')
    if args.reynolds!=20 and not args.skip_reference:
        ap.error('the frozen FEM reference is for Re=20; use --skip-reference for other Reynolds numbers')
    if not np.isfinite(args.check_dt_time) or args.check_dt_time<0 or abs(round(args.check_dt_time/args.dt)*args.dt-args.check_dt_time)>1e-12:
        ap.error('check-dt-time must be a nonnegative multiple of dt')
    args.out.mkdir(parents=True,exist_ok=False)
    jax.config.update('jax_enable_x64',True)
    rule=None
    if args.scaled_quadrature:
        from scaled_response_quadrature import scaled_channel_quadrature
        design_ell=np.sqrt(.01*(2/3)*.46/args.reynolds)
        rule=partial(scaled_channel_quadrature,layer_lengths=design_ell*np.array([.5,1,2,4]))
    elif args.weighted_quadrature:
        from corner_background_quadrature import corner_channel_quadrature
        rule=corner_channel_quadrature
    p=ImmersedFlowPlan(nx=73,ny=33,reynolds=args.reynolds,wall_method='rational',
                      buffer_strength=0,quadrature_factor=args.quadrature_factor,
                      quadrature_rule=rule,basis_workers=4)
    print('BASE',p.setup_seconds,flush=True)
    if args.skip_reference:
        x=np.linspace(p.bounds[0],p.bounds[1],401)
        y=np.linspace(-p.bounds[2],p.bounds[2],161)
    else:
        data=np.load(args.reference/'frozen_force.npz')
        np.testing.assert_allclose(p.rational_lift,data['coefficients'],atol=0,rtol=0)
        _,u,v,ux,uy,vx=p.lift_fields
        force=-np.column_stack((u*ux+v*uy,u*vx-v*ux))
        ref=np.load(args.reference/'helmholtz_reference_0p5.npz')
        x,y=ref['x'],ref['y']
        if np.array_equal(p.points,data['points']):
            target=ref['volume_fields']
            np.testing.assert_allclose(force,data['force'].T,atol=1e-12,rtol=1e-12)
        else:
            from frozen_force_fem import FrozenForceFEM
            reader=FrozenForceFEM.load_for_evaluation(args.reference/'reference_0p5.msh')
            # Confirm DOF ordering against the already-frozen samples before reuse.
            np.testing.assert_allclose(reader.evaluate(ref['state'],data['points'][:100]),
                                       ref['volume_fields'][:,:100],atol=1e-13,rtol=1e-13)
            target=reader.evaluate(ref['state'],p.points)
            del reader;gc.collect()
        np.savez_compressed(args.out/'quadrature_reference.npz',points=p.points,weights=p.weights,target=target)
    report=dict(reynolds=args.reynolds,dt=args.dt,final_time=args.time,nu=p.nu,base_dofs=p.dofs,quadrature_factor=args.quadrature_factor,
                quadrature_rule='weighted-block' if args.weighted_quadrature else ('scaled-ray' if args.scaled_quadrature else 'standard'),quadrature_points=len(p.points),
                reference=None if args.skip_reference else str(args.reference),family=args.family,levels=[])
    def norm(f,av=1,mv=1):return float(np.sqrt(p.weights@(av*np.sum(f[:2]**2,axis=0)+mv*np.sum(f[2:]**2,axis=0))))
    def metric(q,a):
        u,v,ux,uy,vx=[o@a for o in q.operators_fluid]
        err=np.array([u,v,ux,uy,vx,-ux])-target
        return dict(h1=norm(err)/norm(target),l2=norm(err,1,0)/norm(target,1,0),
                    omega=float(np.sqrt(p.weights@((err[4]-err[3])**2))/np.sqrt(p.weights@((target[4]-target[3])**2))))
    for level in args.levels:
        start=perf_counter()
        # Geometry/time-scale choice only: no reference or frozen-force fitting.
        ell=np.sqrt(.01*p.nu)
        lengths=ell*np.array([.7,2.] if level==2 else [.5,1.,2.,4.])
        pairs=()
        if args.family=='paired':
            pairs=[(d,args.outer_length) for d in lengths]
            lengths=()
        elif args.family=='hybrid':
            pairs=[(ell,args.outer_length),(2*ell,args.outer_length)]
        elif args.family=='broad':
            lengths=np.r_[lengths,args.outer_length/2,args.outer_length]
        if args.weighted_quadrature:
            if args.angular_order!=16 or args.wall_order!=24 or args.outer_length!=.2:
                ap.error('weighted pair assembler currently requires angular=16, wall=24, outer=.2')
            from weighted_response_space import WeightedResponsePlan,basis_evaluation_pool
            if args.audit_weighted:
                from validate_weighted_response import audit_background
                with basis_evaluation_pool(p):
                    report['background_audit']=audit_background(p)
                (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
                if max(report['background_audit'].values())>1e-6:
                    raise ArithmeticError('Fixed background quadrature failed independent audit')
            with basis_evaluation_pool(p):
                q=WeightedResponsePlan(p,order=args.weighted_order,angles=args.weighted_angles)
            report['weighted_setup']=q.info
            report['base_setup_seconds']=p.setup_seconds
            (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            if args.audit_weighted:
                from validate_weighted_response import audit_plan
                with basis_evaluation_pool(p):
                    report['weighted_audit']=audit_plan(q)
                (args.out/'weighted_audit.json').write_text(json.dumps(report['weighted_audit'],indent=2)+'\n')
                if not report['weighted_audit']['passed']:
                    raise ArithmeticError('Weighted quadrature failed independent integration audit')
        else:
            q=EnrichedFlowPlan(p,ResponseModes(p.bounds,p.hole,lengths,args.angular_order,args.wall_order,pairs=pairs))
        if level==args.levels[-1]:
            # The wrapper now owns the volume matrices. Base rendering uses
            # its coefficient transform, so the duplicate volume storage can go.
            p.operators_fluid=()
        if args.skip_reference:
            rec=dict(**q.info,level=level,seconds=perf_counter()-start)
            report['levels'].append(rec)
            print('ENRICHED',json.dumps(rec),flush=True)
        else:
            A=q.mass+.01*p.nu*q.stiffness;load=q.force_load(force)
            actual=la.solve(A,load,assume_a='pos')
            u,v,ux,uy,vx=q.operators_fluid;w=q.weights
            def rhs(av,mv):
                return av*(u.T@(w*target[0])+v.T@(w*target[1]))+mv*(ux.T@(w*(target[2]-target[5]))+uy.T@(w*target[3])+vx.T@(w*target[4]))
            best=la.solve(q.mass+q.stiffness,rhs(1,1),assume_a='pos')
            energy=la.solve(A,rhs(1,.01*p.nu),assume_a='pos')
            rec=dict(**q.info,level=level,actual=metric(q,actual),best_h1=metric(q,best),best_energy=metric(q,energy),
                     actual_energy_gap_h1=norm(np.array([o@(actual-energy) for o in q.operators_fluid]+[-q.operators_fluid[2]@(actual-energy)]))/norm(target),
                     seconds=perf_counter()-start)
            report['levels'].append(rec)
            print('ENRICHED',json.dumps(rec),flush=True)
            grids=list(q.grid_many(np.array([q.stokes_state,actual,best]),x,y))
            np.savez_compressed(args.out/f'frozen_{level}.npz',x=x,y=y,reference=ref['grid_fields'],
                    actual_state=actual,best_state=best,
                    actual_omega=grids[1]['vorticity']-grids[0]['vorticity'],
                    best_omega=grids[2]['vorticity']-grids[0]['vorticity'])
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if level!=args.levels[-1]:
            del q
            if not args.skip_reference:del grids
            gc.collect()
    if args.check_dt_time:
        check_states=[]
        checked_coarse=[]
        for h in (args.dt,args.dt/2):
            state=q.stokes_state.copy();integrator=q.stepper(h)
            for k in range(round(args.check_dt_time/h)):
                state=integrator.step(state,k*h)
                if h==args.dt and args.check_dt_time==args.time:
                    checked_coarse.append(state.copy())
            if not np.all(np.isfinite(state)):
                raise RuntimeError(f'Non-finite state in short time-step check at dt={h}')
            check_states.append(state)
        delta=check_states[0]-check_states[1]
        correction=check_states[1]-q.stokes_state
        report['time_step_check']=dict(time=args.check_dt_time,coarse_dt=args.dt,fine_dt=args.dt/2,
            velocity_difference_l2=float(np.sqrt(max(delta@q.mass@delta,0))),
            relative_velocity_correction_l2=float(np.sqrt(max(delta@q.mass@delta,0)/max(correction@q.mass@correction,1e-300))),
            velocity_gradient_difference_l2=float(np.sqrt(max(delta@q.stiffness@delta,0))))
        check_grids=list(q.grid_many(np.array(check_states),x,y))
        np.savez_compressed(args.out/'time_step_check.npz',x=x,y=y,time=args.check_dt_time,
            dt=[args.dt,args.dt/2],states=check_states,
            fields=np.array([[g[key] for key in ['u','v','vorticity','psi']] for g in check_grids]))
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('DT_CHECK',json.dumps(report['time_step_check']),flush=True)
    t=0.;a=q.stokes_state.copy()
    states=[] if args.frames is not None else [a.copy()]
    times=[] if args.frames is not None else [0.]
    stepper=q.stepper(args.dt);steps=round(args.time/args.dt)
    save_every=steps//args.frames if args.frames is not None else max(1,steps//10)
    if abs(steps*args.dt-args.time)>1e-12:raise ValueError('time must be a multiple of dt')
    for k in range(steps):
        try:
            candidate=checked_coarse[k] if args.check_dt_time==args.time else stepper.step(a,t)
            if not np.all(np.isfinite(candidate)):
                raise FloatingPointError('Non-finite NS state')
        except (ValueError, FloatingPointError, la.LinAlgError) as error:
            report['failure']=dict(attempted_time=(k+1)*args.dt,last_finite_time=t,
                                   error=str(error))
            (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print('FAILED',json.dumps(report['failure']),flush=True)
            if not times or times[-1]!=t:
                states.append(a.copy());times.append(t)
            np.savez_compressed(args.out/'states.npz',t=times,states=states,
                                initial_state=q.stokes_state)
            break
        a=candidate
        t=(k+1)*args.dt
        if not np.all(np.isfinite(a)):
            raise RuntimeError(f'Non-finite NS state at t={t}')
        if (k+1)%save_every==0 or k==steps-1:
            states.append(a.copy());times.append(t)
            checkpoint=args.out/'states.tmp.npz'
            np.savez_compressed(checkpoint,t=times,states=states,initial_state=q.stokes_state)
            checkpoint.replace(args.out/'states.npz')
            print('NS',t,la.norm(a),flush=True)
    grids=list(q.grid_many(np.array(states),x,y))
    fields=np.array([[g[key] for key in ['u','v','vorticity','psi']] for g in grids])
    np.savez_compressed(args.out/'fields.npz',x=x,y=y,t=times,fields=fields,states=states,
                        initial_state=q.stokes_state,
                        center=p.hole.center,axes=p.hole.axes)
    d4=np.diff(fields[:,2],n=4,axis=1);sx=(x>-.85)&(x<-.45);sy=(y[2:-2]>-.8)&(y[2:-2]<.8)
    report['ns']=dict(samples=[dict(t=t,roughness=float(np.sqrt(np.mean(z[sy][:,sx]**2)))) for t,z in zip(times,d4)],
                      final=q.diagnostics(a))
    report['ns']['completed_time']=t
    # Independent wall probes, not assembly traces.
    theta=np.linspace(0,2*np.pi,401,endpoint=False)
    hole=np.column_stack((p.hole.center[0]+p.hole.axes[0]*np.cos(theta),p.hole.center[1]+p.hole.axes[1]*np.sin(theta)))
    wall=np.vstack([np.column_stack((np.linspace(-1,5,401),np.full(401,s))) for s in [-1,1]])
    inlet=np.column_stack((np.full(301,-1),np.linspace(-1,1,301)))
    hu,hv=q.evaluate(a,hole)[1:3];wu,wv=q.evaluate(a,wall)[1:3];iu,iv=q.evaluate(a,inlet)[1:3]
    report['ns']['checks']=dict(hole_speed=float(np.max(np.hypot(hu,hv))),wall_speed=float(np.max(np.hypot(wu,wv))),
                              inlet_error=float(np.max(np.hypot(iu-(1-inlet[:,1]**2),iv))))
    report['ns']['seconds']=perf_counter()-start
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FINAL',json.dumps(report['ns']),flush=True)


if __name__=='__main__':main()
