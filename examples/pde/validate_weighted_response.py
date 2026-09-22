"""Independent high-order positive-quadrature audit of the weighted NS adapter."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import jax
from bspf_models.fluids.immersed_flow import ImmersedFlowPlan,channel_lift,channel_quadrature,_stream_product
from scaled_response_quadrature import scaled_channel_quadrature
from weighted_response_space import WeightedResponsePlan, convection, basis_evaluation_pool
from corner_background_quadrature import corner_channel_quadrature


def sample_homogeneous(q, states, points):
    """Evaluate probes before rotation, never materialize Q-by-N transformed ops."""
    original=q.base
    ab,c=zip(*(q.split(s) for s in states))
    a=np.column_stack((original.transform@np.array(ab).T,original.lift_coefficients))
    factors=[]
    for axis in range(2):
        z,indices=np.unique(points[:,axis],return_inverse=True)
        factors.append(tuple(o[indices] for o in original._line_values(axis,z)))
    (x,dx,xx),(y,dy,yy)=factors
    coefficient=(a[:-1] if original.factored_wall else a).reshape(*original.shape,-1)
    def pair(first,second):
        return np.einsum('pi,ijk,pj->pk',first,coefficient,second,optimize=True)
    fields=(pair(x,y),pair(x,dy),-pair(dx,y),pair(dx,dy),pair(x,yy),-pair(xx,y))
    if original.factored_wall:
        jet=original.wall_factor(points)
        fields=_stream_product(fields,tuple(v[:,None] for v in jet))
        g,gx,gy,gxx,gxy,gyy=jet
        circulation=(1-g,-gy,gx,-gxy,-gyy,gxx)
        fields=tuple(f+v[:,None]*a[-1] for f,v in zip(fields,circulation))[1:]
        lift=tuple(f[:,-1]+l for f,l in zip(fields,original.base_fields(points)[1:]))
    else:
        rc=original.rational_modes@(original.rational_map@a)
        rc[:,-1]+=original.rational_lift
        correction=original.rational.evaluate(points,rc)[1:]
        fields=tuple(f+r for f,r in zip(fields[1:],correction))
        lift=tuple(f[:,-1]+l for f,l in zip(fields,channel_lift(points,original.bounds[2],original.peak)[1:]))
    background=tuple(f[:,:-1] for f in fields)
    if np.any(c):
        thin=q.modes.operators(points)[1:]
        extra=tuple(e@np.array(c).T for e in thin)
    else:
        extra=tuple(np.zeros_like(b) for b in background)
    fields=tuple(b+e for b,e in zip(background,extra))
    return fields,lift,background,extra


def audit_background(base, factor=4., probes=6):
    """Cheap independent smooth-background check before costly layer auditing."""
    rng=np.random.default_rng(20260922)
    states=rng.normal(size=(base.dofs,probes))
    states/=np.sqrt(np.einsum('ij,ij->j',states,(base.mass+base.stiffness)@states))
    wrapper=SimpleNamespace(base=base,split=lambda s:(s,np.empty(0)))
    p,w=corner_channel_quadrature(base.bounds,base.hole,base.nx,base.ny,factor,(base.buffer_start,),corner_order=32)
    m=np.zeros((probes,probes));k=np.zeros_like(m)
    for lo in range(0,len(w),4096):
        hi=min(lo+4096,len(w))
        fields,lift,_,_=sample_homogeneous(wrapper,states.T,p[lo:hi])
        if lo==0:
            np.testing.assert_allclose(np.array([f[:8,0]+l[:8] for f,l in zip(fields,lift)]),
                                       base.evaluate(states[:,0],p[:8])[1:],rtol=1e-9,atol=1e-9)
        for j,f in enumerate(fields):
            g=f.T@(w[lo:hi,None]*f)
            if j<2:m+=g
            else:k+=(2 if j==2 else 1)*g
    errors=dict(mass=float(np.linalg.norm(states.T@base.mass@states-m)/np.linalg.norm(m)),
                stiffness=float(np.linalg.norm(states.T@base.stiffness@states-k)/np.linalg.norm(k)))
    print('BACKGROUND_AUDIT',json.dumps(errors),flush=True)
    return errors


def nonlinear_parts(q,state):
    """Separate BB, BE, EB, EE loads using the production weighted rules."""
    ab,c=q.split(state);n=q.base.dofs
    cb=np.r_[ab,c[432:]];raw_coeff=np.r_[q.base.transform@ab,c[432:]]
    background=tuple(o@cb+l for o,l in zip(q.bulk,q.base.lift_fields[1:]))
    fb=np.zeros((len(cb),4));fe=np.zeros((432,4));raw_load=np.zeros((len(raw_coeff),4))
    fb[:,0]=-sum(o.T@(q.base.weights*f) for o,f in zip(q.bulk[:2],convection(background,background)))
    for local in q.local:
        w=local['weights'];ids=local['ids']
        b=tuple(o@raw_coeff+l for o,l in zip(local['raw'],local['lift']))
        e=tuple(o@c[:432] for o in local['thin'])
        eg=tuple(o@c[ids] for o in local['amplitude'])
        forces=[convection(b,b),convection(b,e),convection(e,b),convection(e,e)]
        fe[ids]=-sum(o.T@(w[:,None]*np.column_stack([f[k] for f in forces]))
                     for k,o in enumerate(local['amplitude'][:2]))
        corrections=[convection(b,eg),convection(eg,b),convection(eg,e)]
        raw_load[:,1:]-=sum(o.T@(w[:,None]*np.column_stack([f[k] for f in corrections]))
                            for k,o in enumerate(local['raw'][:2]))
    fb[:n]+=q.base.transform.T@raw_load[:q.base.ndofs]
    fb[n:]+=raw_load[q.base.ndofs:]
    outu,outv=[o@ab+l for o,l in zip(q.base.out_ops,q.out_lift)]
    incoming=np.minimum(outu,0)*q.out_weights
    fb[:n,0]+=sum(o.T@(incoming*f) for o,f in zip(q.base.out_ops,(outu,outv)))
    return np.vstack((fb[:n],q.scaling_rotation.T@np.vstack((fe,fb[n:]))-q.remove.T@fb[:n]))


def audit_plan(q, *, probes=6, seed=20260922, layer_order=72, bulk_order=128, minimum_tangent_order=20):
    start=perf_counter()
    rng=np.random.default_rng(seed)
    vectors=rng.normal(size=(q.dofs,probes))
    vectors/=np.sqrt(np.einsum('ij,ij->j',vectors,(q.mass+q.stiffness)@vectors))
    state=vectors[:,0]*(.2/np.sqrt(vectors[:,0]@q.mass@vectors[:,0]))
    original=q.base
    points,weights=scaled_channel_quadrature(q.bounds,q.hole,original.nx,original.ny,
        factor=2.5,x_breaks=(original.buffer_start,),layer_lengths=q.thin_modes.lengths,
        layer_order=layer_order,bulk_order=bulk_order,resolve_corners=True,minimum_tangent_order=minimum_tangent_order)
    mass=np.zeros((probes,probes));stiffness=np.zeros_like(mass)
    parts={kind:{part:np.zeros_like(mass) for part in ('background','cross','extra')}
           for kind in ('mass','stiffness')}
    # Includes zero-state Stokes convection and a finite perturbation so all
    # background/thin and thin/thin nonlinear products participate.
    loads=np.zeros((probes,2))
    nonlinear_reference=np.zeros((probes,4))
    stacked=np.column_stack((vectors,state)).T
    for start_index in range(0,len(weights),2048):
        stop=min(start_index+2048,len(weights));w=weights[start_index:stop]
        fields,lift,background,extra=sample_homogeneous(q,stacked,points[start_index:stop])
        if start_index==0:
            # Verify the fast coefficient evaluator against the original API.
            direct=q.evaluate(state,points[start_index:start_index+8])[1:]
            np.testing.assert_allclose(np.array([f[:8,-1]+l[:8] for f,l in zip(fields,lift)]),direct,
                                       rtol=1e-9,atol=1e-9)
        for k,f in enumerate(fields):
            matrix=f[:,:probes].T@(w[:,None]*f[:,:probes])
            if k<2: mass+=matrix
            else: stiffness+=(2 if k==2 else 1)*matrix
            kind='mass' if k<2 else 'stiffness';factor=2 if k==2 else 1
            b=background[k][:,:probes];e=extra[k][:,:probes]
            parts[kind]['background']+=factor*b.T@(w[:,None]*b)
            parts[kind]['cross']+=factor*b.T@(w[:,None]*e)
            parts[kind]['extra']+=factor*e.T@(w[:,None]*e)
        for j,total in enumerate((lift,tuple(f[:,-1]+l for f,l in zip(fields,lift)))):
            force=convection(total,total)
            loads[:,j]-=sum(f[:,:probes].T@(w*g) for f,g in zip(fields[:2],force))
        if q.background_modes is None:
            b=tuple(f[:,-1]+l for f,l in zip(background,lift));e=tuple(f[:,-1] for f in extra)
        else:
            # Production places broad responses in B, not in the thin E block.
            coefficients=q.split(state)[1][:432]
            e=tuple(o@coefficients for o in q.thin_modes.operators(points[start_index:stop])[1:])
            b=tuple(f[:,-1]+l-v for f,l,v in zip(fields,lift,e))
        for j,force in enumerate((convection(b,b),convection(b,e),convection(e,b),convection(e,e))):
            nonlinear_reference[:,j]-=sum(f[:,:probes].T@(w*g) for f,g in zip(fields[:2],force))
        if (start_index//2048)%16==15:
            print(f'AUDIT_POINTS {stop}/{len(weights)}',flush=True)
    numerical=[]
    for j,s in enumerate((np.zeros(q.dofs),state)):
        uv=[o@s+l for o,l in zip(q.out_ops,q.out_lift)]
        incoming=np.minimum(uv[0],0)*q.out_weights
        loads[:,j]+=sum((o@vectors).T@(incoming*f) for o,f in zip(q.out_ops,uv))
        if j==1:
            nonlinear_reference[:,0]+=sum((o@vectors).T@(incoming*f) for o,f in zip(q.out_ops,uv))
        numerical.append(vectors.T@(q.explicit(s)+q.linear_lift))
    def relative(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))
    errors=dict(mass=relative(vectors.T@q.mass@vectors,mass),
                stiffness=relative(vectors.T@q.stiffness@vectors,stiffness),
                stokes_nonlinear=relative(numerical[0],loads[:,0]),
                perturbed_nonlinear=relative(numerical[1],loads[:,1]))
    ab,cc=zip(*(q.split(s) for s in vectors.T));ab=np.array(ab).T;cc=np.array(cc).T
    component_errors={}
    for kind in ('mass','stiffness'):
        expected=dict(background=ab.T@getattr(q.base,kind)@ab,
                      cross=ab.T@getattr(q,'cross_'+kind)@cc,
                      extra=cc.T@getattr(q,'extra_'+kind)@cc)
        component_errors[kind]={k:relative(v,parts[kind][k]) for k,v in expected.items()}
    numerical_parts=vectors.T@nonlinear_parts(q,state)
    component_errors['nonlinear']={name:relative(numerical_parts[:,j],nonlinear_reference[:,j])
        for j,name in enumerate(('BB','BE','EB','EE'))}
    component_errors['nonlinear_norms']={name:float(np.linalg.norm(nonlinear_reference[:,j]))
        for j,name in enumerate(('BB','BE','EB','EE'))}
    component_errors['decomposition']=relative(numerical_parts.sum(axis=1),numerical[1])
    passed=errors['mass']<1e-6 and errors['stiffness']<1e-6 and max(errors['stokes_nonlinear'],errors['perturbed_nonlinear'])<1e-5
    result=dict(passed=bool(passed),relative_errors=errors,component_errors=component_errors,seed=seed,probes=probes,
                reference_points=len(weights),layer_order=layer_order,bulk_order=bulk_order,
                minimum_tangent_order=minimum_tangent_order,
                seconds=perf_counter()-start,thresholds=dict(bilinear=1e-6,nonlinear=1e-5))
    print('AUDIT',json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--nx',type=int,default=33)
    parser.add_argument('--ny',type=int,default=25)
    parser.add_argument('--order',type=int,default=12)
    args=parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    p=ImmersedFlowPlan(nx=args.nx,ny=args.ny,reynolds=200,wall_method='rational',
                       buffer_strength=0,quadrature_factor=2.5,quadrature_rule=corner_channel_quadrature,basis_workers=4)
    with basis_evaluation_pool(p):
        q=WeightedResponsePlan(p,order=args.order)
        result=dict(plan=q.info,audit=audit_plan(q))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    if not result['audit']['passed']:raise SystemExit(1)


if __name__=='__main__':main()
